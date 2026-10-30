"""Tests for internal S04 layout units (grouping, background, association)."""

import ast
import copy
import json
import math
from pathlib import Path

import pytest

import engine.layout_units as layout_units
from engine.internal_schema import Canvas, Design, Element
from engine.layout_units import (
    ALIGNMENT_TOLERANCE,
    AMBIGUOUS_GROUP_ASSOCIATION,
    ASSOCIATION_PROXIMITY_SCALE,
    ASSOCIATION_THRESHOLD,
    ASSOCIATION_WEIGHT_ALIGNMENT,
    ASSOCIATION_WEIGHT_CONTAINS,
    ASSOCIATION_WEIGHT_PROXIMITY,
    ASSOCIATION_WEIGHT_SAME_GROUP,
    BACKGROUND_AREA_FRACTION,
    COVER_BACKGROUND_AREA_FRACTION,
    GROUP_MODE_COHESIVE,
    GROUP_MODE_SPLIT,
    MAX_UNIT_SIZE,
    SPLIT_MAX_CLUSTERS,
    SPLIT_MIN_CLUSTERS,
    SPLIT_SECOND_GAP_RATIO,
    LayoutUnit,
    association,
    box_contains,
    box_iou,
    build_layout_units,
    contains_any,
    covers_canvas_center,
    element_iou,
    group_forces_split,
    group_is_ambiguous,
    is_aligned,
    is_background_like,
    min_pair_association,
    normalized_center_distance,
    required_group_modes,
    same_explicit_group,
    split_unit,
)

INPUT_DIR = Path(__file__).resolve().parent.parent / "input"
MODULE_PATH = Path(__file__).resolve().parent.parent / "engine" / "layout_units.py"


def load_fixture(name):
    task = json.loads((INPUT_DIR / name).read_text(encoding="utf-8"))
    return (
        task,
        Design.from_adobe(task["source"]),
        Canvas.from_adobe(task["target_canvas"]),
    )


def make_shape(id, x, y, width, height, **extra):
    data = {
        "id": id, "type": "shape", "x": x, "y": y,
        "width": width, "height": height,
        "shape_type": "rectangle", "fill_color": "rgba(0,0,0,1)",
    }
    data.update(extra)
    return Element.from_adobe(data)


def make_image(id, x, y, width, height, **extra):
    data = {
        "id": id, "type": "image", "x": x, "y": y,
        "width": width, "height": height, "asset_id": "art.png",
    }
    data.update(extra)
    return Element.from_adobe(data)


def make_design(elements, width=1000, height=1000):
    canvas = Canvas(width=width, height=height)
    return Design(canvas=canvas, source_canvas=canvas,
                  elements=tuple(elements))


def unit_ids(units):
    return tuple(tuple(item.id for item in unit.members) for unit in units)


def test_named_threshold_values():
    assert BACKGROUND_AREA_FRACTION == 0.75
    assert COVER_BACKGROUND_AREA_FRACTION == 0.75
    assert ASSOCIATION_THRESHOLD == 0.75
    assert ASSOCIATION_WEIGHT_SAME_GROUP == 0.40
    assert ASSOCIATION_WEIGHT_CONTAINS == 0.25
    assert ASSOCIATION_WEIGHT_PROXIMITY == 0.20
    assert ASSOCIATION_WEIGHT_ALIGNMENT == 0.15
    assert ASSOCIATION_PROXIMITY_SCALE == 0.20
    assert ALIGNMENT_TOLERANCE == 0.02
    assert MAX_UNIT_SIZE == 6
    assert AMBIGUOUS_GROUP_ASSOCIATION == 0.75
    assert SPLIT_MIN_CLUSTERS == 2
    assert SPLIT_MAX_CLUSTERS == 3
    assert SPLIT_SECOND_GAP_RATIO == 0.5
    assert GROUP_MODE_COHESIVE == "cohesive"
    assert GROUP_MODE_SPLIT == "split"
    weights = (
        ASSOCIATION_WEIGHT_SAME_GROUP + ASSOCIATION_WEIGHT_CONTAINS
        + ASSOCIATION_WEIGHT_PROXIMITY + ASSOCIATION_WEIGHT_ALIGNMENT
    )
    assert weights == pytest.approx(1.0)


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
        "__future__", "math", "collections.abc", "dataclasses",
        "engine.candidate_validator", "engine.internal_schema",
    }


def test_python_310_syntax():
    for path in (MODULE_PATH, Path(__file__).resolve()):
        ast.parse(path.read_text(encoding="utf-8"), feature_version=(3, 10))


@pytest.mark.parametrize("name", [
    "chameleon_sample_level1_warmup.json",
    "chameleon_sample_level2_core.json",
    "chameleon_sample_level3_advanced.json",
])
def test_fixture_background_flags(name):
    _, source, _ = load_fixture(name)
    canvas = source.source_canvas
    by_id = source.elements_by_id
    assert is_background_like(by_id["bg"], canvas) is True
    assert is_background_like(by_id["logo"], canvas) is False
    assert all(
        is_background_like(item, canvas) is False
        for item in source.elements if item.type == "text"
    )


def test_background_branches():
    canvas = Canvas(width=1000, height=1000)
    assert is_background_like(
        make_shape("big", 50, 50, 900, 900), canvas) is True
    assert is_background_like(
        make_shape("small", 100, 100, 500, 500), canvas) is False
    assert is_background_like(
        make_shape("off", -2000, -2000, 1000, 1000), canvas) is False
    assert is_background_like(
        make_image("cover", 600, 600, 1000, 1000, fit="cover"),
        canvas) is True
    assert is_background_like(
        make_image("cover_small", 0, 0, 100, 100, fit="cover"),
        canvas) is False
    assert is_background_like(
        make_image("contain", 0, 0, 900, 900, fit="contain"),
        canvas) is True


def test_covers_canvas_center_edges():
    canvas = Canvas(width=100, height=100)
    assert covers_canvas_center(make_shape("p", 50, 50, 0, 0), canvas) is True
    assert covers_canvas_center(make_shape("e", 0, 0, 50, 50), canvas) is True
    assert covers_canvas_center(
        make_shape("m", 51, 51, 10, 10), canvas) is False


def test_same_explicit_group():
    canvas = Canvas(width=100, height=100)
    del canvas
    left = make_shape("a", 0, 0, 10, 10, group_id="g")
    right = make_shape("b", 20, 20, 10, 10, group_id="g")
    other = make_shape("c", 0, 0, 10, 10, group_id="h")
    null = make_shape("d", 0, 0, 10, 10, group_id=None)
    absent = make_shape("e", 0, 0, 10, 10)
    empty_left = make_shape("f", 0, 0, 10, 10, group_id="")
    empty_right = make_shape("g", 5, 5, 10, 10, group_id="")
    assert same_explicit_group(left, right) is True
    assert same_explicit_group(left, other) is False
    assert same_explicit_group(null, null) is False
    assert same_explicit_group(absent, absent) is False
    assert same_explicit_group(left, null) is False
    assert same_explicit_group(empty_left, empty_right) is True


def test_box_contains():
    outer = make_shape("o", 0, 0, 100, 100)
    inner = make_shape("i", 10, 10, 20, 20)
    edge = make_shape("e", 50, 50, 50, 50)
    twin = make_shape("t", 0, 0, 100, 100)
    overlap = make_shape("v", 40, 40, 70, 70)
    flat = make_shape("z", 0, 0, 0, 100)
    assert box_contains(outer, inner) is True
    assert box_contains(inner, outer) is False
    assert box_contains(outer, edge) is True
    assert box_contains(outer, twin) is True
    assert box_contains(twin, outer) is True
    assert contains_any(outer, overlap) is False
    assert contains_any(outer, flat) is False
    assert contains_any(flat, inner) is False


def test_normalized_center_distance():
    canvas = Canvas(width=1000, height=500)
    first = make_shape("a", 50, 50, 100, 100)
    second = make_shape("b", 550, 300, 100, 100)
    assert normalized_center_distance(first, second, canvas) == pytest.approx(
        math.hypot(0.5, 0.5),
    )
    assert normalized_center_distance(first, first, canvas) == 0.0
    assert normalized_center_distance(first, second, canvas) == \
        normalized_center_distance(second, first, canvas)


def test_alignment_accepts_left_center_and_right():
    canvas = Canvas(width=1000, height=1000)
    base = make_shape("a", 100, 100, 200, 200)
    left = make_shape("l", 120, 500, 50, 50)
    center = make_shape("c", 150, 500, 100, 100)
    right = make_shape("r", 200, 500, 100, 100)
    none = make_shape("n", 500, 500, 100, 100)
    outside = make_shape("o", 121, 500, 50, 50)
    assert is_aligned(base, left, canvas) is True
    assert is_aligned(base, center, canvas) is True
    assert is_aligned(base, right, canvas) is True
    assert is_aligned(base, none, canvas) is False
    assert is_aligned(base, outside, canvas) is False


def test_association_fixture_pairs():
    _, source, _ = load_fixture("chameleon_sample_level2_core.json")
    canvas = source.source_canvas
    by_id = source.elements_by_id
    badge = association(by_id["badge_bg"], by_id["badge_label"], canvas)
    assert badge == pytest.approx(1.0)
    cta = association(by_id["cta_bg"], by_id["cta_label"], canvas)
    assert cta > 0.99
    assert association(by_id["badge_bg"], by_id["badge_label"], canvas) == \
        association(by_id["badge_label"], by_id["badge_bg"], canvas)
    _, advanced, _ = load_fixture("chameleon_sample_level3_advanced.json")
    adv_canvas = advanced.source_canvas
    adv = advanced.elements_by_id
    ribbon = association(adv["ribbon_bg"], adv["ribbon_label"], adv_canvas)
    assert ribbon == pytest.approx(1.0)


def test_association_ungrouped_containment_caps_below_threshold():
    canvas = Canvas(width=1000, height=1000)
    outer = make_shape("o", 0, 0, 100, 100)
    inner = make_shape("i", 25, 25, 50, 50)
    assert association(outer, inner, canvas) == pytest.approx(0.60)


def test_explicit_group_stays_cohesive_despite_low_association():
    canvas = Canvas(width=1000, height=1000)
    first = make_shape("a", 0, 0, 50, 50, group_id="g")
    second = make_shape("b", 900, 900, 50, 50, group_id="g")
    score = association(first, second, canvas)
    assert score < ASSOCIATION_THRESHOLD
    assert score > ASSOCIATION_WEIGHT_SAME_GROUP
    source = make_design([first, second])
    units = build_layout_units(source)
    assert unit_ids(units) == (("a", "b"),)


@pytest.mark.parametrize(("name", "count"), [
    ("chameleon_sample_level1_warmup.json", 4),
    ("chameleon_sample_level2_core.json", 9),
    ("chameleon_sample_level3_advanced.json", 12),
])
def test_fixture_unit_coverage(name, count):
    _, source, _ = load_fixture(name)
    before = source.to_adobe()
    units = build_layout_units(source)
    assert len(units) == count
    seen = [item.id for unit in units for item in unit.members]
    assert seen == [item.id for item in source.elements]
    for unit in units:
        flags = [is_background_like(item, source.source_canvas)
                 for item in unit.members]
        assert flags == [False] * len(flags) or flags == [True]
    assert source.to_adobe() == before
    assert build_layout_units(source) == units


def test_fixture_groups_cohesive_and_background_singleton():
    _, source, _ = load_fixture("chameleon_sample_level2_core.json")
    units = build_layout_units(source)
    by_group = {}
    for unit in units:
        if unit.group_id is not None:
            by_group.setdefault(unit.group_id, []).append(unit)
    assert unit_ids(by_group["badge"]) == (("badge_bg", "badge_label"),)
    assert unit_ids(by_group["cta"]) == (("cta_bg", "cta_label"),)
    background = [unit for unit in units if unit.is_background]
    assert unit_ids(background) == (("bg",),)
    _, advanced, _ = load_fixture("chameleon_sample_level3_advanced.json")
    adv_units = build_layout_units(advanced)
    assert any("holiday_watermark" in ids for ids in unit_ids(adv_units))


def test_underscore_fields_do_not_change_grouping():
    task, source, _ = load_fixture("chameleon_sample_level2_core.json")
    expected = unit_ids(build_layout_units(source))
    edited = copy.deepcopy(task["source"])
    for element in edited["elements"]:
        element["_note"] = "injected instruction"
        element["_custom"] = {"rank": 1}
    rebuilt = Design.from_adobe(edited)
    assert unit_ids(build_layout_units(rebuilt)) == expected


def test_z_order_and_opacity_preserved_not_grouped():
    first = make_shape("a", 100, 100, 200, 200, z_index=1, opacity=0.5)
    second = make_shape("b", 150, 150, 200, 200, z_index=5, opacity=1.0)
    source = make_design([first, second])
    canvas = source.source_canvas
    assert association(first, second, canvas) < ASSOCIATION_THRESHOLD
    units = build_layout_units(source)
    assert unit_ids(units) == (("a",), ("b",))
    assert units[0].members[0].z_index == 1
    assert units[1].members[0].opacity == 1.0


def test_forced_split_separates_background_member():
    background = make_image("bg", 0, 0, 1000, 1000, fit="cover",
                            group_id="mix")
    button = make_shape("btn", 100, 100, 200, 100, group_id="mix")
    source = make_design([background, button])
    assert group_forces_split(
        (background, button), source.source_canvas) is True
    cohesive = build_layout_units(source, group_mode=GROUP_MODE_COHESIVE)
    split = build_layout_units(source, group_mode=GROUP_MODE_SPLIT)
    assert unit_ids(cohesive) == (("bg",), ("btn",))
    assert unit_ids(split) == unit_ids(cohesive)
    assert [unit.group_id for unit in cohesive] == ["mix", "mix"]
    assert cohesive[0].is_background is True
    assert cohesive[1].is_background is False


def test_ambiguous_group_gets_both_modes():
    loose_first = make_shape("a", 0, 0, 50, 50, group_id="loose")
    loose_second = make_shape("b", 900, 900, 50, 50, group_id="loose")
    tight_bg = make_shape("c", 40, 40, 220, 56, group_id="tight")
    tight_label = make_shape("d", 40, 52, 220, 32, group_id="tight")
    source = make_design([loose_first, loose_second, tight_bg, tight_label])
    canvas = source.source_canvas
    assert group_is_ambiguous((loose_first, loose_second), canvas) is True
    assert group_is_ambiguous((tight_bg, tight_label), canvas) is False
    assert required_group_modes(source) == (
        GROUP_MODE_COHESIVE, GROUP_MODE_SPLIT,
    )
    cohesive = build_layout_units(source, group_mode=GROUP_MODE_COHESIVE)
    split = build_layout_units(source, group_mode=GROUP_MODE_SPLIT)
    assert unit_ids(cohesive) == (("a", "b"), ("c", "d"))
    assert unit_ids(split) == (("a",), ("b",), ("c", "d"))


@pytest.mark.parametrize("name", [
    "chameleon_sample_level1_warmup.json",
    "chameleon_sample_level2_core.json",
    "chameleon_sample_level3_advanced.json",
])
def test_fixtures_need_only_cohesive_mode(name):
    _, source, _ = load_fixture(name)
    assert required_group_modes(source) == (GROUP_MODE_COHESIVE,)


def test_split_at_largest_gap_two_and_three_clusters():
    canvas = Canvas(width=1000, height=100)
    trio = LayoutUnit.of(
        (make_shape("a", 0, 0, 10, 10),
         make_shape("b", 20, 0, 10, 10),
         make_shape("c", 130, 0, 10, 10)),
        (0, 1, 2), canvas,
    )
    assert unit_ids(split_unit(trio, canvas)) == (("a", "b"), ("c",))
    quad = LayoutUnit.of(
        (make_shape("a", 0, 0, 10, 10),
         make_shape("b", 110, 0, 10, 10),
         make_shape("c", 130, 0, 10, 10),
         make_shape("d", 240, 0, 10, 10)),
        (0, 1, 2, 3), canvas,
    )
    assert unit_ids(split_unit(quad, canvas)) == (
        ("a",), ("b", "c"), ("d",),
    )


def test_split_axis_tie_prefers_x():
    canvas = Canvas(width=1000, height=1000)
    unit = LayoutUnit.of(
        (make_shape("a", 0, 110, 10, 10),
         make_shape("b", 110, 0, 10, 10),
         make_shape("c", 120, 110, 10, 10)),
        (0, 1, 2), canvas,
    )
    assert unit_ids(split_unit(unit, canvas)) == (("a",), ("b", "c"))


def test_split_overlapping_units():
    canvas = Canvas(width=1000, height=1000)
    pair = LayoutUnit.of(
        (make_shape("a", 0, 0, 100, 100),
         make_shape("b", 10, 10, 100, 100)),
        (0, 1), canvas,
    )
    assert unit_ids(split_unit(pair, canvas)) == (("a",), ("b",))
    trio = LayoutUnit.of(
        tuple(make_shape(f"m{k}", 0, 0, 100, 100) for k in range(3)),
        (0, 1, 2), canvas,
    )
    assert unit_ids(split_unit(trio, canvas)) == (
        ("m0",), ("m1",), ("m2",),
    )
    _, source, _ = load_fixture("chameleon_sample_level2_core.json")
    units = build_layout_units(source)
    badge = next(unit for unit in units if unit.group_id == "badge")
    assert unit_ids(split_unit(badge, source.source_canvas)) == (
        ("badge_bg",), ("badge_label",),
    )


def test_relative_geometry_and_degenerate_width():
    _, source, _ = load_fixture("chameleon_sample_level2_core.json")
    units = build_layout_units(source)
    badge = next(unit for unit in units if unit.group_id == "badge")
    assert badge.bbox == (40.0, 40.0, 220.0, 56.0)
    assert badge.relative_geometry[0] == (0.0, 0.0, 1.0, 1.0)
    assert badge.relative_geometry[1][0] == pytest.approx(0.0)
    assert badge.relative_geometry[1][1] == pytest.approx(12 / 56)
    assert badge.relative_geometry[1][2] == pytest.approx(1.0)
    assert badge.relative_geometry[1][3] == pytest.approx(32 / 56)
    canvas = Canvas(width=100, height=100)
    lines = LayoutUnit.of(
        (make_shape("a", 5, 0, 0, 10, group_id="ln"),
         make_shape("b", 5, 20, 0, 10, group_id="ln")),
        (0, 1), canvas,
    )
    assert lines.bbox == (5.0, 0.0, 0.0, 30.0)
    assert lines.relative_geometry[0][0] == 0.0
    assert lines.relative_geometry[0][2] == 0.0
    assert lines.relative_geometry[1][1] == pytest.approx(20 / 30)


def test_extents_include_rotation_and_stroke():
    _, advanced, _ = load_fixture("chameleon_sample_level3_advanced.json")
    units = build_layout_units(advanced)
    ribbon = next(unit for unit in units if unit.group_id == "ribbon")
    assert ribbon.extents[0] < ribbon.bbox[0]
    canvas = Canvas(width=1000, height=1000)
    stroked = LayoutUnit.of(
        (make_shape("a", 100, 100, 200, 100,
                    stroke_color="#000000", stroke_width=20),),
        (0,), canvas,
    )
    assert stroked.extents == (90.0, 90.0, 310.0, 210.0)


def test_greedy_cluster_cap_and_empty_edges():
    pairs = [(left, right) for left in range(7) for right in range(left + 1, 7)]
    clusters = layout_units._greedy_cluster(7, pairs, MAX_UNIT_SIZE)
    assert sorted(len(cluster) for cluster in clusters) == [1, 6]
    assert sorted(idx for cluster in clusters for idx in cluster) == list(
        range(7),
    )
    assert layout_units._greedy_cluster(3, [], MAX_UNIT_SIZE) == [
        (0,), (1,), (2,),
    ]
    with pytest.raises(ValueError, match="cap"):
        layout_units._greedy_cluster(2, [(0, 1)], 0)


def test_box_iou_values():
    assert box_iou((0, 0, 100, 100), (50, 50, 100, 100)) == pytest.approx(
        2500 / 17500,
    )
    assert box_iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0
    assert box_iou((0, 0, 10, 10), (20, 20, 10, 10)) == 0.0
    first = make_shape("a", 0, 0, 100, 100)
    second = make_shape("b", 50, 50, 100, 100)
    assert element_iou(first, second) == pytest.approx(
        box_iou((0, 0, 100, 100), (50, 50, 100, 100)),
    )


def test_min_pair_association_singleton_is_cohesive():
    canvas = Canvas(width=100, height=100)
    assert min_pair_association((), canvas) == 1.0
    assert min_pair_association(
        (make_shape("a", 0, 0, 10, 10),), canvas,
    ) == 1.0


def test_arbitrary_element_counts():
    single = make_design([make_shape("only", 10, 10, 20, 20)])
    assert unit_ids(build_layout_units(single)) == (("only",),)
    grid = [
        make_shape(f"e{r * 10 + c}", c * 100, r * 100, 50, 50)
        for r in range(7) for c in range(10)
    ]
    many = make_design(grid)
    units = build_layout_units(many)
    assert len(units) == 70
    assert [item.id for unit in units for item in unit.members] == [
        item.id for item in many.elements
    ]


def test_invalid_inputs_rejected():
    _, source, _ = load_fixture("chameleon_sample_level1_warmup.json")
    with pytest.raises(ValueError, match="group_mode"):
        build_layout_units(source, group_mode="bogus")
    canvas = Canvas(width=100, height=100)
    with pytest.raises(ValueError, match="aligned"):
        LayoutUnit.of((), (), canvas)
    with pytest.raises(ValueError, match="aligned"):
        LayoutUnit.of((make_shape("a", 0, 0, 10, 10),), (0, 1), canvas)
