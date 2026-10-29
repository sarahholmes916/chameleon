"""Tests for the S04 candidate generator (baseline + flow/stack families)."""

import ast
import json
from pathlib import Path

import pytest

from engine.candidate_generator import (
    BAND_GAP_FRACTION,
    FAMILY_HORIZONTAL_FLOW,
    FAMILY_SCALE_BASELINE,
    FAMILY_VERTICAL_STACK,
    FLOW_MAX_SINGLE_BAND_UNITS,
    FONT_GROWTH_CAP,
    GENERATOR_VERSION,
    MARGIN_FRACTION,
    SLOT_DIRECTION_HORIZONTAL,
    SLOT_DIRECTION_VERTICAL,
    SLOT_SAFETY_FACTOR,
    STACK_MAX_SINGLE_COLUMN_UNITS,
    Slot,
    allocate_slots,
    baseline_candidate,
    horizontal_flow_candidates,
    vertical_stack_candidates,
)
from engine.candidate_validator import validate_candidate
from engine.internal_schema import Canvas, Design
from engine.layout_search import baseline_proposal
from engine.layout_units import (
    GROUP_MODE_COHESIVE,
    GROUP_MODE_SPLIT,
    required_group_modes,
)
from engine.text_fit import estimate_text_height

INPUT_DIR = Path(__file__).resolve().parent.parent / "input"
MODULE_PATH = Path(__file__).resolve().parent.parent / "engine" / "candidate_generator.py"
SAMPLES = [
    "chameleon_sample_level1_warmup.json",
    "chameleon_sample_level2_core.json",
    "chameleon_sample_level3_advanced.json",
]
FAMILIES = {
    FAMILY_SCALE_BASELINE: lambda source, target: [baseline_candidate(source, target)],
    FAMILY_HORIZONTAL_FLOW: horizontal_flow_candidates,
    FAMILY_VERTICAL_STACK: vertical_stack_candidates,
}
MUTABLE_GEOMETRY = {"x", "y", "width", "height"}


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


def image(id, x, y, width, height, fit="contain", **extra):
    data = {
        "id": id, "type": "image", "x": x, "y": y,
        "width": width, "height": height, "asset_id": "art.png", "fit": fit,
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


def load_sample(name):
    task = json.loads((INPUT_DIR / name).read_text(encoding="utf-8"))
    return (
        Design.from_adobe(task["source"]),
        Canvas.from_adobe(task["target_canvas"]),
    )


def boxes_by_id(candidate):
    return {el["id"]: el for el in candidate["elements"]}


def same_json(first, second):
    if type(first) is not type(second):
        return False
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(
            same_json(first[key], second[key]) for key in first)
    if isinstance(first, list):
        return len(first) == len(second) and all(
            same_json(a, b) for a, b in zip(first, second))
    return first == second


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
        "__future__", "math", "dataclasses", "typing",
        "engine.internal_schema", "engine.layout_edits",
        "engine.layout_search", "engine.layout_units", "engine.text_fit",
    }


def test_python_310_syntax():
    for path in (MODULE_PATH, Path(__file__).resolve()):
        ast.parse(path.read_text(encoding="utf-8"), feature_version=(3, 10))


def test_generator_version_pinned():
    assert GENERATOR_VERSION == "s04_generator_v0.1"


def test_allocate_slots_tiles_single_band():
    slots = allocate_slots(3, target_canvas(1000, 500),
                           direction=SLOT_DIRECTION_HORIZONTAL)
    assert len(slots) == 3
    assert all(isinstance(slot, Slot) for slot in slots)
    assert (slots[0].x, slots[0].y) == pytest.approx((40.0, 20.0))
    assert (slots[0].width, slots[0].height) == pytest.approx((920 / 3, 460.0))
    for left, right in zip(slots, slots[1:]):
        assert left.x + left.width == pytest.approx(right.x)
        assert left.y == right.y and left.height == right.height
    assert slots[-1].x + slots[-1].width == pytest.approx(960.0)


def test_allocate_slots_splits_two_bands():
    count = FLOW_MAX_SINGLE_BAND_UNITS * 2
    slots = allocate_slots(count, target_canvas(1600, 400),
                           direction=SLOT_DIRECTION_HORIZONTAL)
    assert len(slots) == count
    band_h = (400 * (1 - 2 * MARGIN_FRACTION)
              * (1 - BAND_GAP_FRACTION)) / 2
    assert slots[0].y == pytest.approx(400 * MARGIN_FRACTION)
    assert slots[0].height == pytest.approx(band_h)
    assert slots[4].y == pytest.approx(400 * MARGIN_FRACTION + band_h
                                       + 400 * (1 - 2 * MARGIN_FRACTION)
                                       * BAND_GAP_FRACTION)
    assert slots[0].width == pytest.approx(1600 * (1 - 2 * MARGIN_FRACTION) / 4)


def test_allocate_slots_vertical_transposes():
    slots = allocate_slots(STACK_MAX_SINGLE_COLUMN_UNITS * 2,
                           target_canvas(400, 1600),
                           direction=SLOT_DIRECTION_VERTICAL)
    assert len(slots) == STACK_MAX_SINGLE_COLUMN_UNITS * 2
    assert slots[0].x == pytest.approx(400 * MARGIN_FRACTION)
    assert slots[4].x > slots[0].x + slots[0].width
    assert slots[0].y == pytest.approx(1600 * MARGIN_FRACTION)
    assert slots[1].y == pytest.approx(slots[0].y + slots[0].height)


def test_allocate_slots_rejects_bad_direction_and_count():
    with pytest.raises(ValueError, match="direction"):
        allocate_slots(2, target_canvas(100, 100), direction="diagonal")
    with pytest.raises(ValueError, match="non-negative"):
        allocate_slots(-1, target_canvas(100, 100),
                       direction=SLOT_DIRECTION_HORIZONTAL)
    assert allocate_slots(0, target_canvas(100, 100),
                          direction=SLOT_DIRECTION_HORIZONTAL) == []


def test_core_flow_walkthrough_positions():
    source, target = load_sample("chameleon_sample_level2_core.json")
    assert (target.width, target.height) == (1600, 400)
    assert required_group_modes(source) == (GROUP_MODE_COHESIVE,)
    candidates = horizontal_flow_candidates(source, target)
    assert len(candidates) == 1
    boxes = boxes_by_id(candidates[0])
    assert boxes["bg"]["x"] == 0.0 and boxes["bg"]["y"] == 0.0
    assert (boxes["bg"]["width"], boxes["bg"]["height"]) == (1600.0, 400.0)
    product = boxes["product"]
    assert (product["x"], product["y"]) == pytest.approx((1264.6, 16.0))
    assert (product["width"], product["height"]) == pytest.approx((174.8, 174.8))
    headline = boxes["headline"]
    placed_h = 140.0 * 368.0 / 900.0
    placed_y = 209.2 + (174.8 - placed_h) / 2
    assert (headline["x"], headline["width"]) == pytest.approx((432.0, 368.0))
    assert (headline["y"], headline["height"]) == pytest.approx(
        (placed_y, placed_h), abs=1e-6)


def test_family_slot_order_differs_by_direction():
    source, target = load_sample("chameleon_sample_level2_core.json")
    flow = boxes_by_id(horizontal_flow_candidates(source, target)[0])
    stack = boxes_by_id(vertical_stack_candidates(source, target)[0])
    assert flow["logo"]["x"] == pytest.approx(800.0)
    column_w = (1472.0 - 1472.0 * 0.05) / 2
    column_x = 64.0 + column_w + 1472.0 * 0.05
    logo_w = 320.0 * min(column_w / 320.0, 92.0 / 110.0)
    assert stack["logo"]["x"] == pytest.approx(
        column_x + (column_w - logo_w) / 2, abs=1e-6)
    assert stack["logo"]["x"] != pytest.approx(flow["logo"]["x"])


@pytest.mark.parametrize("name", SAMPLES)
@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_emission_preserves_canvas_ids_and_order(name, family):
    source, target = load_sample(name)
    wanted = tuple(el.id for el in source.elements)
    candidates = FAMILIES[family](source, target)
    assert 1 <= len(candidates) <= 2
    for candidate in candidates:
        assert set(candidate.keys()) == {"canvas", "elements"}
        assert candidate["canvas"] == {"width": target.width, "height": target.height}
        assert tuple(el["id"] for el in candidate["elements"]) == wanted


@pytest.mark.parametrize("name", SAMPLES)
@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_only_k6_fields_change(name, family):
    source, target = load_sample(name)
    originals = {el["id"]: el for el in task_source_elements(name)}
    for candidate in FAMILIES[family](source, target):
        for changed in candidate["elements"]:
            before = originals[changed["id"]]
            assert set(changed.keys()) == set(before.keys())
            for key, old in before.items():
                if key in MUTABLE_GEOMETRY:
                    continue
                if key == "font_size" and before["type"] == "text":
                    continue
                assert same_json(changed[key], old), (changed["id"], key)


def task_source_elements(name):
    task = json.loads((INPUT_DIR / name).read_text(encoding="utf-8"))
    return task["source"]["elements"]


@pytest.mark.parametrize("name", SAMPLES)
@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_validity_grid_all_fixtures_families_modes(name, family):
    source, target = load_sample(name)
    modes = required_group_modes(source)
    candidates = FAMILIES[family](source, target)
    if family == FAMILY_SCALE_BASELINE:
        assert len(candidates) == 1
    else:
        assert len(candidates) == len(modes)
    for mode, candidate in zip(modes if family != FAMILY_SCALE_BASELINE else modes[:1],
                               candidates):
        report = validate_candidate(source, candidate, target)
        assert report.is_valid, (name, family, mode, report.violations)


@pytest.mark.parametrize("name", SAMPLES)
@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_noncover_images_keep_exact_ratio(name, family):
    source, target = load_sample(name)
    originals = {el.id: el for el in source.elements}
    for candidate in FAMILIES[family](source, target):
        parsed = Design.from_adobe(candidate)
        for element in parsed.elements:
            if element.type != "image":
                continue
            first = originals[element.id]
            if first.effective_value("fit") == "cover":
                continue
            if not all(v > 0 for v in (first.width, first.height,
                                       element.width, element.height)):
                continue
            ratio = ((element.width * first.height)
                     / (element.height * first.width))
            assert ratio == pytest.approx(1.0, rel=1e-9)


@pytest.mark.parametrize("name", SAMPLES)
@pytest.mark.parametrize("family", [FAMILY_HORIZONTAL_FLOW, FAMILY_VERTICAL_STACK])
def test_background_fills_target_exactly(name, family):
    source, target = load_sample(name)
    for candidate in FAMILIES[family](source, target):
        background = boxes_by_id(candidate)["bg"]
        assert (background["x"], background["y"]) == (0.0, 0.0)
        assert (background["width"], background["height"]) == (
            float(target.width), float(target.height))


def test_canvas_fit_maximizes_bg_like_contain_image():
    source = design(1000, 1000, [
        image("big", 50, 50, 900, 900, fit="contain"),
        text("t", 100, 100, 200, 60, content="hi", font_size=20),
    ])
    target = target_canvas(1600, 400)
    candidates = horizontal_flow_candidates(source, target)
    assert len(candidates) == 1
    big = boxes_by_id(candidates[0])["big"]
    assert (big["x"], big["y"]) == pytest.approx((600.0, 0.0), abs=1e-6)
    assert (big["width"], big["height"]) == pytest.approx((400.0, 400.0))
    report = validate_candidate(source, candidates[0], target)
    assert report.is_valid, report.violations


def test_rotated_cover_shrinks_to_fit():
    source = design(1000, 1000, [
        image("bg", 0, 0, 1000, 1000, fit="cover", rotation=30),
        text("t", 100, 100, 200, 60, content="hi", font_size=20),
    ])
    target = target_canvas(1000, 1000)
    candidates = horizontal_flow_candidates(source, target)
    assert len(candidates) == 1
    background = boxes_by_id(candidates[0])["bg"]
    assert background["x"] > 0.0 and background["y"] > 0.0
    assert background["width"] < 1000.0 and background["height"] < 1000.0
    report = validate_candidate(source, candidates[0], target)
    assert report.is_valid, report.violations


def test_zero_area_fill_keeps_degeneracy():
    source = design(1000, 1000, [
        image("line", 100, 100, 0, 500, fit="cover"),
        text("t", 100, 100, 200, 60, content="hi", font_size=20),
    ])
    target = target_canvas(1600, 400)
    candidates = horizontal_flow_candidates(source, target)
    assert len(candidates) == 1
    line = boxes_by_id(candidates[0])["line"]
    assert line["width"] == 0.0
    assert line["height"] == 400.0
    report = validate_candidate(source, candidates[0], target)
    assert report.is_valid, report.violations


def test_headline_font_grows_to_ladder_top():
    source, target = load_sample("chameleon_sample_level2_core.json")
    candidate = horizontal_flow_candidates(source, target)[0]
    headline = boxes_by_id(candidate)["headline"]
    slot_scale = min(368.0 / 900.0, 174.8 / 140.0) * SLOT_SAFETY_FACTOR
    seed = 56.0 * slot_scale
    assert headline["font_size"] == pytest.approx(seed * 1.1 ** 4)
    assert headline["font_size"] <= seed * FONT_GROWTH_CAP


@pytest.mark.parametrize("name", SAMPLES)
@pytest.mark.parametrize("family", [FAMILY_HORIZONTAL_FLOW, FAMILY_VERTICAL_STACK])
def test_fonts_fit_or_floor(name, family):
    source, target = load_sample(name)
    by_id = source.elements_by_id
    for candidate in FAMILIES[family](source, target):
        for element in candidate["elements"]:
            if element["type"] != "text":
                continue
            assert element["font_size"] >= 1.0
            if element["font_size"] == 1.0:
                continue
            line_height = float(by_id[element["id"]].effective_value("line_height"))
            estimated = estimate_text_height(element["content"], element["width"],
                                             element["font_size"], line_height)
            assert estimated is not None and estimated <= element["height"]


def test_split_group_yields_two_candidates():
    source = design(1000, 1000, [
        shape("a", 0, 0, 50, 50, group_id="g"),
        shape("b", 900, 900, 50, 50, group_id="g"),
    ])
    assert required_group_modes(source) == (GROUP_MODE_COHESIVE, GROUP_MODE_SPLIT)
    target = target_canvas(1000, 1000)
    flow = horizontal_flow_candidates(source, target)
    stack = vertical_stack_candidates(source, target)
    assert len(flow) == 2 and len(stack) == 2
    assert flow[0] != flow[1]
    assert stack[0] != stack[1]
    for candidate in flow + stack:
        assert validate_candidate(source, candidate, target).is_valid


def test_ungrouped_design_yields_one_candidate():
    source, target = load_sample("chameleon_sample_level1_warmup.json")
    assert required_group_modes(source) == (GROUP_MODE_COHESIVE,)
    assert len(horizontal_flow_candidates(source, target)) == 1
    assert len(vertical_stack_candidates(source, target)) == 1


def test_generator_is_deterministic():
    source, target = load_sample("chameleon_sample_level2_core.json")
    assert horizontal_flow_candidates(source, target) == \
        horizontal_flow_candidates(source, target)
    assert vertical_stack_candidates(source, target) == \
        vertical_stack_candidates(source, target)
    assert baseline_candidate(source, target) == baseline_candidate(source, target)


def test_baseline_is_first_proportional_repair():
    source, target = load_sample("chameleon_sample_level2_core.json")
    assert baseline_candidate(source, target) == baseline_proposal(source, target, 1)
