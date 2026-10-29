"""Tests for the S04 deterministic soft objective (spec D.1.5)."""

import ast
import json
import logging
import math
from pathlib import Path

import pytest

from engine.internal_schema import Canvas, Design, Element
from engine.layout_objective import (
    ALIGN_DEVIATION_SCALE,
    ALIGN_SOURCE_TOLERANCE,
    BALANCE_AREA_CAP_FRACTION,
    GROUP_COMPACTNESS_RANGE,
    OBJECTIVE_VERSION,
    PAIR_DISTANCE_SCALE,
    PAIR_SLACK,
    PROVISIONAL_WEIGHTS,
    READ_HIERARCHY_WEIGHT,
    READ_REFERENCE_FONT_SIZE,
    READ_SIZE_WEIGHT,
    SCALE_LOG_RANGE,
    SOURCE_OVERLAP_IOU_THRESHOLD,
    SPACE_GAP_SMOOTHING,
    WEIGHT_ALIGN,
    WEIGHT_BALANCE,
    WEIGHT_GROUP,
    WEIGHT_MOVE,
    WEIGHT_PAIR,
    WEIGHT_READ,
    WEIGHT_SCALE,
    WEIGHT_SPACE,
    WEIGHT_UNWANTED_OVERLAP,
    ObjectiveScore,
    ObjectiveWeights,
    TermScores,
    evaluate_objective,
)
from engine.layout_search import baseline_proposal
from engine.layout_units import (
    ASSOCIATION_THRESHOLD,
    association,
    is_background_like,
)
from engine.text_fit import estimate_text_height

INPUT_DIR = Path(__file__).resolve().parent.parent / "input"
MODULE_PATH = Path(__file__).resolve().parent.parent / "engine" / "layout_objective.py"
SAMPLES = [
    "chameleon_sample_level1_warmup.json",
    "chameleon_sample_level2_core.json",
    "chameleon_sample_level3_advanced.json",
]


def shape(id, x, y, width, height, **extra):
    data = {
        "id": id, "type": "shape", "x": x, "y": y,
        "width": width, "height": height,
        "shape_type": "rectangle", "fill_color": "rgba(0,0,0,1)",
    }
    data.update(extra)
    return data


def text(id, x, y, width, height, content="hi", font_size=16, **extra):
    data = {
        "id": id, "type": "text", "x": x, "y": y,
        "width": width, "height": height, "content": content,
        "font_family": "Solstice Sans", "font_size": font_size,
        "color": "rgba(0,0,0,1)",
    }
    data.update(extra)
    return data


def design(canvas_width, canvas_height, elements):
    return Design.from_adobe({
        "canvas": {"width": canvas_width, "height": canvas_height},
        "elements": elements,
    })


def target_canvas(width, height):
    return Canvas.from_adobe({"width": width, "height": height})


def terms_list(terms):
    return (terms.move, terms.align, terms.pair, terms.scale,
            terms.unwanted_overlap, terms.balance, terms.read,
            terms.group, terms.space)


def test_provisional_weights_match_spec():
    assert (WEIGHT_MOVE, WEIGHT_ALIGN, WEIGHT_PAIR, WEIGHT_SCALE,
            WEIGHT_UNWANTED_OVERLAP, WEIGHT_BALANCE, WEIGHT_READ,
            WEIGHT_GROUP, WEIGHT_SPACE) == (
        0.05, 0.075, 0.125, 0.075, 0.125, 0.05, 0.25, 0.20, 0.05)
    assert (ALIGN_SOURCE_TOLERANCE, ALIGN_DEVIATION_SCALE,
            PAIR_SLACK, PAIR_DISTANCE_SCALE, SCALE_LOG_RANGE,
            SOURCE_OVERLAP_IOU_THRESHOLD, READ_REFERENCE_FONT_SIZE,
            READ_SIZE_WEIGHT, READ_HIERARCHY_WEIGHT,
            GROUP_COMPACTNESS_RANGE, SPACE_GAP_SMOOTHING,
            BALANCE_AREA_CAP_FRACTION) == (
        0.02, 0.1, 0.05, 0.25, 3.0, 0.02, 12.0, 0.8, 0.2, 3.0, 0.01, 0.25)
    assert PROVISIONAL_WEIGHTS == ObjectiveWeights()
    assert (WEIGHT_MOVE + WEIGHT_ALIGN + WEIGHT_PAIR + WEIGHT_SCALE
            + WEIGHT_UNWANTED_OVERLAP + WEIGHT_BALANCE + WEIGHT_READ
            + WEIGHT_GROUP + WEIGHT_SPACE) == pytest.approx(1.0)


def test_objective_version_is_provisional():
    assert OBJECTIVE_VERSION == "objective_v1_provisional"


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        ObjectiveWeights(move=0.5)
    with pytest.raises(ValueError, match="finite"):
        ObjectiveWeights(move=float("inf"), align=-float("inf"))


def test_no_ml_dependencies_by_ast():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "torch" not in source.lower()
    assert "sentence_transformers" not in source
    tree = ast.parse(source)
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    assert modules <= {
        "__future__", "logging", "math", "dataclasses",
        "engine.internal_schema", "engine.layout_units", "engine.text_fit",
    }


def test_python_310_syntax():
    for path in (MODULE_PATH, Path(__file__).resolve()):
        ast.parse(path.read_text(encoding="utf-8"), feature_version=(3, 10))


def test_identical_layout_zeroes_geometric_terms():
    source = design(1000, 1000, [
        shape("a", 100, 100, 100, 100, group_id="g"),
        shape("b", 300, 300, 100, 100, group_id="g"),
        text("t", 100, 500, 200, 60, content="hi", font_size=16),
    ])
    score = evaluate_objective(source, source, target_canvas(1000, 1000))
    assert score.terms.move == 0.0
    assert score.terms.align == 0.0
    assert score.terms.pair == 0.0
    assert score.terms.scale == 0.0
    assert score.terms.group == 0.0
    for value in terms_list(score.terms):
        assert 0.0 <= value <= 1.0
        assert math.isfinite(value)


def test_move_hand_computed():
    source = design(100, 100, [shape("a", 0, 0, 10, 10)])
    moved = design(100, 100, [shape("a", 10, 0, 10, 10)])
    score = evaluate_objective(source, moved, target_canvas(100, 100))
    assert score.terms.move == pytest.approx(0.1 / math.sqrt(2))


def test_move_capped_at_one():
    source = design(100, 100, [shape("a", 0, 0, 0, 0)])
    moved = design(100, 100, [shape("a", 100, 100, 0, 0)])
    score = evaluate_objective(source, moved, target_canvas(100, 100))
    assert score.terms.move == 1.0


def test_align_left_edges_hand_computed():
    source = design(1000, 1000, [
        shape("a", 100, 0, 100, 100),
        shape("b", 100, 500, 100, 100),
    ])
    moved = design(1000, 1000, [
        shape("a", 100, 0, 100, 100),
        shape("b", 150, 500, 100, 100),
    ])
    score = evaluate_objective(source, moved, target_canvas(1000, 1000))
    assert score.terms.align == pytest.approx(0.5)


def test_align_accepts_center_x_anchor():
    source = design(1000, 1000, [
        shape("a", 0, 0, 100, 100),
        shape("b", 25, 500, 50, 100),
    ])
    moved = design(1000, 1000, [
        shape("a", 0, 0, 100, 100),
        shape("b", 45, 500, 50, 100),
    ])
    score = evaluate_objective(source, moved, target_canvas(1000, 1000))
    assert score.terms.align == pytest.approx(0.2)


def test_align_ignores_background_pairs():
    source = design(1000, 1000, [
        shape("bg", 0, 0, 1000, 1000),
        shape("b", 0, 500, 100, 100),
    ])
    assert is_background_like(source.elements[0], source.source_canvas)
    moved = design(1000, 1000, [
        shape("bg", 0, 0, 1000, 1000),
        shape("b", 200, 500, 100, 100),
    ])
    score = evaluate_objective(source, moved, target_canvas(1000, 1000))
    assert score.terms.align == 0.0


def test_pair_separation_capped():
    source = design(1000, 1000, [
        shape("a", 100, 100, 300, 300, group_id="g"),
        shape("b", 150, 150, 50, 50, group_id="g"),
    ])
    canvas = source.source_canvas
    assert association(source.elements[0], source.elements[1], canvas) \
        >= ASSOCIATION_THRESHOLD
    moved = design(1000, 1000, [
        shape("a", 100, 100, 300, 300, group_id="g"),
        shape("b", 800, 800, 50, 50, group_id="g"),
    ])
    score = evaluate_objective(source, moved, target_canvas(1000, 1000))
    assert score.terms.pair == 1.0


def test_pair_approach_is_free():
    source = design(1000, 1000, [
        shape("a", 100, 100, 300, 300, group_id="g"),
        shape("b", 150, 150, 50, 50, group_id="g"),
    ])
    moved = design(1000, 1000, [
        shape("a", 100, 100, 300, 300, group_id="g"),
        shape("b", 225, 225, 50, 50, group_id="g"),
    ])
    score = evaluate_objective(source, moved, target_canvas(1000, 1000))
    assert score.terms.pair == 0.0


def test_pair_ignores_unassociated_pairs():
    source = design(1000, 1000, [
        shape("a", 0, 0, 50, 50),
        shape("b", 900, 900, 50, 50),
    ])
    canvas = source.source_canvas
    assert association(source.elements[0], source.elements[1], canvas) \
        < ASSOCIATION_THRESHOLD
    moved = design(1000, 1000, [
        shape("a", 0, 0, 50, 50),
        shape("b", 0, 900, 50, 50),
    ])
    score = evaluate_objective(source, moved, target_canvas(1000, 1000))
    assert score.terms.pair == 0.0


def test_scale_uniform_growth_near_zero():
    source = design(100, 100, [shape("a", 10, 10, 20, 20)])
    grown = design(200, 200, [shape("a", 20, 20, 40, 40)])
    score = evaluate_objective(source, grown, target_canvas(200, 200))
    expected = abs(math.log(1601 / 401) - math.log(40001 / 10001)) / 3.0
    assert score.terms.scale == pytest.approx(expected)
    assert score.terms.scale < 0.01


def test_scale_excludes_backgrounds():
    source = design(100, 100, [
        shape("a", 10, 10, 20, 20),
        shape("bg", 0, 0, 100, 100),
    ])
    assert is_background_like(source.elements[1], source.source_canvas)
    grown = design(200, 200, [
        shape("a", 20, 20, 40, 40),
        shape("bg", 0, 0, 200, 100),
    ])
    score = evaluate_objective(source, grown, target_canvas(200, 200))
    expected = abs(math.log(1601 / 401) - math.log(40001 / 10001)) / 3.0
    assert score.terms.scale == pytest.approx(expected)


def test_overlap_hand_computed():
    source = design(1000, 1000, [
        shape("a", 0, 0, 100, 100),
        shape("b", 200, 0, 100, 100),
    ])
    moved = design(1000, 1000, [
        shape("a", 0, 0, 100, 100),
        shape("b", 50, 0, 100, 100),
    ])
    score = evaluate_objective(source, moved, target_canvas(1000, 1000))
    assert score.terms.unwanted_overlap == pytest.approx(1 / 3)


def test_overlap_separated_pair_scores_zero():
    source = design(1000, 1000, [
        shape("a", 0, 0, 100, 100),
        shape("b", 200, 0, 100, 100),
    ])
    score = evaluate_objective(source, source, target_canvas(1000, 1000))
    assert score.terms.unwanted_overlap == 0.0


@pytest.mark.parametrize("variant", ["group", "source_overlap", "background", "zero_area"])
def test_overlap_exclusions(variant):
    first = shape("a", 0, 0, 100, 100)
    if variant == "group":
        first["group_id"] = "g"
        second_source = shape("b", 200, 0, 100, 100, group_id="g")
    elif variant == "source_overlap":
        second_source = shape("b", 50, 0, 100, 100)
    elif variant == "background":
        first = shape("a", 0, 0, 1000, 1000)
        second_source = shape("b", 200, 200, 100, 100)
    else:
        second_source = shape("b", 200, 0, 0, 100)
    second_moved = dict(second_source)
    second_moved["x"] = 50
    if variant == "background":
        second_moved["x"] = 400
        second_moved["y"] = 400
    source = design(1000, 1000, [first, second_source])
    moved = design(1000, 1000, [first, second_moved])
    score = evaluate_objective(source, moved, target_canvas(1000, 1000))
    assert score.terms.unwanted_overlap == 0.0


def test_balance_hand_computed():
    elements = [
        shape("a", 0, 400, 100, 100),
        shape("b", 900, 400, 100, 100),
    ]
    source = design(1000, 1000, elements)
    score = evaluate_objective(source, source, target_canvas(1000, 1000))
    expected = 50.0 / (math.hypot(1000, 1000) / 2)
    assert score.terms.balance == pytest.approx(expected)


def test_balance_ignores_backgrounds():
    plain = design(1000, 1000, [shape("a", 0, 400, 100, 100)])
    with_bg = design(1000, 1000, [
        shape("a", 0, 400, 100, 100),
        shape("bg", 0, 0, 1000, 1000),
    ])
    assert is_background_like(with_bg.elements[1], with_bg.source_canvas)
    target = target_canvas(1000, 1000)
    assert evaluate_objective(with_bg, with_bg, target).terms.balance == \
        evaluate_objective(plain, plain, target).terms.balance


def test_balance_caps_large_elements():
    source = design(1000, 1000, [
        shape("big", 0, 0, 900, 400),
        shape("d", 800, 800, 100, 100),
    ])
    assert not is_background_like(source.elements[0], source.source_canvas)
    score = evaluate_objective(source, source, target_canvas(1000, 1000))
    capped = BALANCE_AREA_CAP_FRACTION * 1000 * 1000
    small = 100 * 100
    center_x = (capped * 450 + small * 850) / (capped + small)
    center_y = (capped * 200 + small * 850) / (capped + small)
    expected = math.hypot(center_x - 500, center_y - 500) / (
        math.hypot(1000, 1000) / 2)
    assert score.terms.balance == pytest.approx(expected)


def test_balance_empty_without_foreground():
    source = design(1000, 1000, [shape("bg", 0, 0, 1000, 1000)])
    assert is_background_like(source.elements[0], source.source_canvas)
    score = evaluate_objective(source, source, target_canvas(1000, 1000))
    assert score.terms.balance == 0.0


def test_read_small_but_fitting_text():
    source = design(1000, 1000, [
        text("t", 0, 0, 500, 500, content="hi", font_size=6),
    ])
    score = evaluate_objective(source, source, target_canvas(1000, 1000))
    assert score.terms.read == pytest.approx(READ_SIZE_WEIGHT * 0.5)


def test_read_overflowing_text():
    source = design(1000, 1000, [
        text("t", 0, 0, 10, 100, content="a" * 100, font_size=16),
    ])
    score = evaluate_objective(source, source, target_canvas(1000, 1000))
    assert score.terms.read == pytest.approx(READ_SIZE_WEIGHT)


@pytest.mark.parametrize("box", [(0, 0, 0, 100), (0, 0, 100, 0)])
def test_read_degenerate_box_fully_overflows(box):
    source = design(1000, 1000, [
        text("t", *box, content="hi", font_size=16),
    ])
    score = evaluate_objective(source, source, target_canvas(1000, 1000))
    assert score.terms.read == pytest.approx(READ_SIZE_WEIGHT)


def test_read_skips_empty_content():
    source = design(1000, 1000, [
        text("t", 0, 0, 100, 100, content="", font_size=16),
    ])
    score = evaluate_objective(source, source, target_canvas(1000, 1000))
    assert score.terms.read == 0.0


def test_read_uses_shared_text_fit_estimate():
    source = design(1000, 1000, [
        text("t", 0, 0, 100, 40, content="hello world", font_size=20),
    ])
    score = evaluate_objective(source, source, target_canvas(1000, 1000))
    estimated = estimate_text_height("hello world", 100, 20, 1.2)
    assert estimated is not None
    overflow = max(0.0, min(1.0, (estimated - 40) / 40))
    assert overflow == pytest.approx(0.2)
    assert score.terms.read == pytest.approx(READ_SIZE_WEIGHT * overflow)


def test_read_hierarchy_reversal():
    source = design(1000, 1000, [
        text("big", 0, 0, 500, 500, content="hi", font_size=20),
        text("small", 0, 500, 500, 500, content="hi", font_size=10),
    ])
    swapped = design(1000, 1000, [
        text("big", 0, 0, 500, 500, content="hi", font_size=10),
        text("small", 0, 500, 500, 500, content="hi", font_size=20),
    ])
    score = evaluate_objective(source, swapped, target_canvas(1000, 1000))
    mean_size = ((1 - 10 / READ_REFERENCE_FONT_SIZE) + 0.0) / 2
    expected = READ_SIZE_WEIGHT * mean_size + READ_HIERARCHY_WEIGHT * 1.0
    assert score.terms.read == pytest.approx(expected)


def test_read_hierarchy_tie_counts_half():
    source = design(1000, 1000, [
        text("a", 0, 0, 500, 500, content="hi", font_size=20),
        text("b", 0, 500, 500, 500, content="hi", font_size=10),
    ])
    tied = design(1000, 1000, [
        text("a", 0, 0, 500, 500, content="hi", font_size=15),
        text("b", 0, 500, 500, 500, content="hi", font_size=15),
    ])
    score = evaluate_objective(source, tied, target_canvas(1000, 1000))
    assert score.terms.read == pytest.approx(READ_HIERARCHY_WEIGHT * 0.5)


def test_group_rigid_move_scores_zero():
    source = design(1000, 1000, [
        shape("a", 100, 100, 100, 100, group_id="g"),
        shape("b", 300, 300, 100, 100, group_id="g"),
    ])
    moved = design(1000, 1000, [
        shape("a", 150, 100, 100, 100, group_id="g"),
        shape("b", 350, 300, 100, 100, group_id="g"),
    ])
    score = evaluate_objective(source, moved, target_canvas(1000, 1000))
    assert score.terms.group == 0.0


def test_group_swap_hand_computed():
    source = design(1000, 1000, [
        shape("a", 100, 100, 100, 100, group_id="g"),
        shape("b", 300, 300, 100, 100, group_id="g"),
    ])
    swapped = design(1000, 1000, [
        shape("a", 300, 300, 100, 100, group_id="g"),
        shape("b", 100, 100, 100, 100, group_id="g"),
    ])
    score = evaluate_objective(source, swapped, target_canvas(1000, 1000))
    offset = min(1.0, math.hypot(2 / 3, 2 / 3))
    assert score.terms.group == pytest.approx((offset + 1.0 + 0.0) / 3)


def test_group_containment_escape_hand_computed():
    source = design(1000, 1000, [
        shape("outer", 0, 0, 400, 400, group_id="g"),
        shape("inner", 100, 100, 100, 100, group_id="g"),
    ])
    moved = design(1000, 1000, [
        shape("outer", 0, 0, 400, 400, group_id="g"),
        shape("inner", 500, 500, 100, 100, group_id="g"),
    ])
    score = evaluate_objective(source, moved, target_canvas(1000, 1000))
    offset = (math.hypot(1 / 6, 1 / 6) + math.hypot(13 / 24, 13 / 24)) / 2
    compact = abs(math.log(2.25)) / GROUP_COMPACTNESS_RANGE
    assert score.terms.group == pytest.approx((offset + compact + 1.0) / 3)


def test_group_containment_kept_scores_zero():
    source = design(1000, 1000, [
        shape("outer", 0, 0, 400, 400, group_id="g"),
        shape("inner", 100, 100, 100, 100, group_id="g"),
    ])
    moved = design(1000, 1000, [
        shape("outer", 50, 0, 400, 400, group_id="g"),
        shape("inner", 150, 100, 100, 100, group_id="g"),
    ])
    score = evaluate_objective(source, moved, target_canvas(1000, 1000))
    assert score.terms.group == 0.0


def test_group_ignores_singletons_and_ungrouped():
    source = design(1000, 1000, [
        shape("solo", 0, 0, 100, 100, group_id="solo"),
        shape("free", 500, 500, 100, 100),
    ])
    moved = design(1000, 1000, [
        shape("solo", 800, 800, 100, 100, group_id="solo"),
        shape("free", 0, 0, 100, 100),
    ])
    score = evaluate_objective(source, moved, target_canvas(1000, 1000))
    assert score.terms.group == 0.0


def test_space_even_gaps_score_zero():
    elements = [
        shape("a", 0, 400, 100, 100),
        shape("b", 200, 400, 100, 100),
        shape("c", 400, 400, 100, 100),
    ]
    source = design(1000, 1000, elements)
    score = evaluate_objective(source, source, target_canvas(1000, 1000))
    assert score.terms.space == 0.0


def test_space_uneven_gaps_hand_computed():
    elements = [
        shape("a", 0, 400, 100, 100),
        shape("b", 150, 400, 100, 100),
        shape("c", 500, 400, 100, 100),
    ]
    source = design(1000, 1000, elements)
    score = evaluate_objective(source, source, target_canvas(1000, 1000))
    assert score.terms.space == pytest.approx(0.1 / (0.15 + SPACE_GAP_SMOOTHING))


def test_space_uses_longer_target_axis():
    elements = [
        shape("a", 200, 0, 100, 50),
        shape("b", 200, 100, 100, 50),
        shape("c", 200, 500, 100, 50),
    ]
    source = design(500, 1000, elements)
    score = evaluate_objective(source, source, target_canvas(500, 1000))
    assert score.terms.space == pytest.approx(0.15 / (0.2 + SPACE_GAP_SMOOTHING))


def test_space_fewer_than_three_blocks_scores_zero():
    source = design(1000, 1000, [
        shape("a", 0, 400, 100, 100),
        shape("b", 500, 400, 100, 100),
    ])
    score = evaluate_objective(source, source, target_canvas(1000, 1000))
    assert score.terms.space == 0.0


def test_space_merges_shared_groups():
    source = design(1000, 1000, [
        shape("a", 0, 400, 100, 100, group_id="g"),
        shape("b", 200, 400, 100, 100, group_id="g"),
        shape("c", 500, 400, 100, 100),
    ])
    score = evaluate_objective(source, source, target_canvas(1000, 1000))
    assert score.terms.space == 0.0


def test_total_is_weighted_sum_of_terms():
    source = design(1000, 1000, [
        shape("a", 100, 100, 100, 100, group_id="g"),
        shape("b", 300, 300, 100, 100, group_id="g"),
    ])
    swapped = design(1000, 1000, [
        shape("a", 300, 300, 100, 100, group_id="g"),
        shape("b", 100, 100, 100, 100, group_id="g"),
    ])
    score = evaluate_objective(source, swapped, target_canvas(1000, 1000))
    assert score.weights is PROVISIONAL_WEIGHTS
    assert score.objective_version == OBJECTIVE_VERSION
    assert isinstance(score, ObjectiveScore)
    assert isinstance(score.terms, TermScores)
    expected = (
        PROVISIONAL_WEIGHTS.move * score.terms.move
        + PROVISIONAL_WEIGHTS.align * score.terms.align
        + PROVISIONAL_WEIGHTS.pair * score.terms.pair
        + PROVISIONAL_WEIGHTS.scale * score.terms.scale
        + PROVISIONAL_WEIGHTS.unwanted_overlap * score.terms.unwanted_overlap
        + PROVISIONAL_WEIGHTS.balance * score.terms.balance
        + PROVISIONAL_WEIGHTS.read * score.terms.read
        + PROVISIONAL_WEIGHTS.group * score.terms.group
        + PROVISIONAL_WEIGHTS.space * score.terms.space
    )
    assert score.total == pytest.approx(expected)


def test_custom_weights_isolate_one_term():
    source = design(100, 100, [shape("a", 0, 0, 10, 10)])
    moved = design(100, 100, [shape("a", 10, 0, 10, 10)])
    weights = ObjectiveWeights(
        move=1.0, align=0.0, pair=0.0, scale=0.0, unwanted_overlap=0.0,
        balance=0.0, read=0.0, group=0.0, space=0.0,
    )
    score = evaluate_objective(source, moved, target_canvas(100, 100),
                               weights=weights)
    assert score.weights is weights
    assert score.total == score.terms.move


def test_mismatched_ids_raise():
    source = design(1000, 1000, [
        shape("a", 0, 0, 100, 100),
        shape("b", 200, 0, 100, 100),
    ])
    target = target_canvas(1000, 1000)
    missing = design(1000, 1000, [shape("a", 0, 0, 100, 100)])
    with pytest.raises(ValueError, match="ids do not match"):
        evaluate_objective(source, missing, target)
    renamed = design(1000, 1000, [
        shape("a", 0, 0, 100, 100),
        shape("zzz", 200, 0, 100, 100),
    ])
    with pytest.raises(ValueError, match="ids do not match"):
        evaluate_objective(source, renamed, target)


def test_changed_type_raises():
    source = design(1000, 1000, [shape("a", 0, 0, 100, 100)])
    changed = design(1000, 1000, [
        text("a", 0, 0, 100, 100, content="hi", font_size=16),
    ])
    with pytest.raises(ValueError, match="changed type"):
        evaluate_objective(source, changed, target_canvas(1000, 1000))


def test_duplicate_ids_rejected_by_schema():
    canvas = Canvas(width=1000, height=1000)
    element = Element.from_adobe(shape("a", 0, 0, 100, 100))
    with pytest.raises(ValueError, match="duplicate ID"):
        Design(canvas=canvas, source_canvas=canvas, elements=(element, element))


def test_evaluate_is_deterministic():
    source = design(1000, 1000, [
        shape("a", 100, 100, 100, 100, group_id="g"),
        shape("b", 300, 300, 100, 100, group_id="g"),
        text("t", 100, 500, 200, 60, content="hello", font_size=14),
    ])
    moved = design(800, 1200, [
        shape("a", 80, 120, 90, 110, group_id="g"),
        shape("b", 260, 360, 90, 110, group_id="g"),
        text("t", 90, 600, 180, 70, content="hello", font_size=12),
    ])
    target = target_canvas(800, 1200)
    first = evaluate_objective(source, moved, target)
    second = evaluate_objective(source, moved, target)
    assert first == second


def test_overlap_eligibility_is_logged(caplog):
    source = design(1000, 1000, [
        shape("a", 0, 0, 100, 100),
        shape("b", 200, 0, 100, 100),
    ])
    with caplog.at_level(logging.DEBUG, logger="engine.layout_objective"):
        evaluate_objective(source, source, target_canvas(1000, 1000))
    assert "a&b" in caplog.text


def test_inputs_are_not_mutated():
    source = design(1000, 1000, [
        shape("a", 100, 100, 100, 100, group_id="g"),
        shape("b", 300, 300, 100, 100, group_id="g"),
    ])
    moved = design(1000, 1000, [
        shape("a", 150, 100, 100, 100, group_id="g"),
        shape("b", 350, 300, 100, 100, group_id="g"),
    ])
    target = target_canvas(1000, 1000)
    before = (source.to_adobe(), moved.to_adobe(), target.to_adobe())
    evaluate_objective(source, moved, target)
    assert (source.to_adobe(), moved.to_adobe(), target.to_adobe()) == before


@pytest.mark.parametrize("name", SAMPLES)
@pytest.mark.parametrize("attempt", [0, 1])
def test_sample_terms_bounded_and_finite(name, attempt):
    task = json.loads((INPUT_DIR / name).read_text(encoding="utf-8"))
    source = Design.from_adobe(task["source"])
    target = Canvas.from_adobe(task["target_canvas"])
    candidate = Design.from_adobe(baseline_proposal(source, target, attempt))
    score = evaluate_objective(source, candidate, target)
    for value in terms_list(score.terms):
        assert 0.0 <= value <= 1.0
        assert math.isfinite(value)
    assert 0.0 <= score.total <= 1.0
    assert math.isfinite(score.total)
