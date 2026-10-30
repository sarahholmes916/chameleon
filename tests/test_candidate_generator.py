"""Tests for the S04 candidate generator (baseline, flow/stack, hero, zones)."""

import ast
import json
from pathlib import Path

import pytest

from engine.candidate_generator import (
    BAND_GAP_FRACTION,
    BLOCK_INNER_GAP_EM,
    BLOCK_OUTER_GAP_EM,
    DETACHED_OVERLAY_MAX_FRACTION,
    FAMILY_ASPECT_CUTOFF,
    FAMILY_BANNER_ZONES,
    FAMILY_HERO_LEFT,
    FAMILY_HERO_RIGHT,
    FAMILY_HORIZONTAL_FLOW,
    FAMILY_SCALE_BASELINE,
    FAMILY_VERTICAL_STACK,
    FILL_EDGE_TOLERANCE_PX,
    FLOW_MAX_SINGLE_BAND_UNITS,
    FONT_GROWTH_CAP,
    GENERATOR_VERSION,
    HERO_MAX_SINGLE_COLUMN_UNITS,
    MARGIN_FRACTION,
    OVERLAY_OPACITY_THRESHOLD,
    PORTRAIT_ASPECT_CUTOFF,
    SAFE_BOTTOM_MARGIN_FACTOR,
    SLOT_DIRECTION_HORIZONTAL,
    SLOT_DIRECTION_VERTICAL,
    SLOT_SAFETY_FACTOR,
    SMALL_ANCHOR_MAX_FRACTION,
    STACK_MAX_SINGLE_COLUMN_UNITS,
    Slot,
    _is_fill_member,
    _merge_text_blocks,
    allocate_slots,
    banner_zones_candidates,
    baseline_candidate,
    hero_left_candidates,
    hero_right_candidates,
    horizontal_flow_candidates,
    vertical_stack_candidates,
)
from engine.candidate_validator import validate_candidate
from engine.internal_schema import Canvas, Design
from engine.layout_search import baseline_proposal
from engine.layout_units import (
    GROUP_MODE_COHESIVE,
    GROUP_MODE_SPLIT,
    build_layout_units,
    is_background_like,
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
    FAMILY_HERO_LEFT: hero_left_candidates,
    FAMILY_HERO_RIGHT: hero_right_candidates,
    FAMILY_BANNER_ZONES: banner_zones_candidates,
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
    usable_h = 500 * (1 - MARGIN_FRACTION * (1 + SAFE_BOTTOM_MARGIN_FACTOR))
    assert (slots[0].width, slots[0].height) == pytest.approx((920 / 3, usable_h))
    for left, right in zip(slots, slots[1:]):
        assert left.x + left.width == pytest.approx(right.x)
        assert left.y == right.y and left.height == right.height
    assert slots[-1].x + slots[-1].width == pytest.approx(960.0)


def test_allocate_slots_splits_two_bands():
    count = FLOW_MAX_SINGLE_BAND_UNITS * 2
    slots = allocate_slots(count, target_canvas(1600, 400),
                           direction=SLOT_DIRECTION_HORIZONTAL)
    assert len(slots) == count
    usable_h = 400 * (1 - MARGIN_FRACTION * (1 + SAFE_BOTTOM_MARGIN_FACTOR))
    band_h = usable_h * (1 - BAND_GAP_FRACTION) / 2
    assert slots[0].y == pytest.approx(400 * MARGIN_FRACTION)
    assert slots[0].height == pytest.approx(band_h)
    assert slots[4].y == pytest.approx(400 * MARGIN_FRACTION + band_h
                                       + usable_h * BAND_GAP_FRACTION)
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
    usable_w = 1600 * (1 - 2 * MARGIN_FRACTION)
    usable_h = 400 * (1 - MARGIN_FRACTION * (1 + SAFE_BOTTOM_MARGIN_FACTOR))
    slot_w = usable_w / 4
    band_h = usable_h * (1 - BAND_GAP_FRACTION) / 2
    band2_top = 400 * MARGIN_FRACTION + band_h + usable_h * BAND_GAP_FRACTION
    product = boxes["product"]
    assert (product["x"], product["y"]) == pytest.approx(
        (64.0 + 3 * slot_w + (slot_w - band_h) / 2, 16.0))
    assert (product["width"], product["height"]) == pytest.approx((band_h, band_h))
    headline = boxes["headline"]
    placed_h = 140.0 * slot_w / 900.0
    placed_y = band2_top + (band_h - placed_h) / 2
    assert (headline["x"], headline["width"]) == pytest.approx((432.0, slot_w))
    assert (headline["y"], headline["height"]) == pytest.approx(
        (placed_y, placed_h), abs=1e-6)


def test_family_slot_order_differs_by_direction():
    source, target = load_sample("chameleon_sample_level2_core.json")
    flow = boxes_by_id(horizontal_flow_candidates(source, target)[0])
    stack = boxes_by_id(vertical_stack_candidates(source, target)[0])
    assert flow["logo"]["x"] == pytest.approx(800.0)
    column_w = (1472.0 - 1472.0 * 0.05) / 2
    column_x = 64.0 + column_w + 1472.0 * 0.05
    row_h = 400 * (1 - MARGIN_FRACTION * (1 + SAFE_BOTTOM_MARGIN_FACTOR)) / 4
    logo_w = 320.0 * min(column_w / 320.0, row_h / 110.0)
    assert stack["logo"]["x"] == pytest.approx(
        column_x + (column_w - logo_w) / 2, abs=1e-6)
    assert stack["logo"]["x"] != pytest.approx(flow["logo"]["x"])


@pytest.mark.parametrize("name", SAMPLES)
@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_emission_preserves_canvas_ids_and_order(name, family):
    source, target = load_sample(name)
    wanted = tuple(el.id for el in source.elements)
    candidates = FAMILIES[family](source, target)
    # Hero/zones multiply modes x orders x small-variants (at most 6 here);
    # exact per-fixture counts are pinned in test_hero_zones_candidate_counts.
    assert 1 <= len(candidates) <= 6
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
    elif family in (FAMILY_HORIZONTAL_FLOW, FAMILY_VERTICAL_STACK):
        assert len(candidates) == len(modes)
    else:
        assert len(candidates) >= len(modes)
    for candidate in candidates:
        report = validate_candidate(source, candidate, target)
        assert report.is_valid, (name, family, report.violations)


@pytest.mark.parametrize(("name", "family", "count"), [
    ("chameleon_sample_level1_warmup.json", FAMILY_HERO_LEFT, 3),
    ("chameleon_sample_level1_warmup.json", FAMILY_HERO_RIGHT, 3),
    ("chameleon_sample_level1_warmup.json", FAMILY_BANNER_ZONES, 3),
    ("chameleon_sample_level2_core.json", FAMILY_HERO_LEFT, 4),
    ("chameleon_sample_level2_core.json", FAMILY_HERO_RIGHT, 4),
    ("chameleon_sample_level2_core.json", FAMILY_BANNER_ZONES, 4),
    ("chameleon_sample_level3_advanced.json", FAMILY_HERO_LEFT, 4),
    ("chameleon_sample_level3_advanced.json", FAMILY_HERO_RIGHT, 4),
    ("chameleon_sample_level3_advanced.json", FAMILY_BANNER_ZONES, 4),
])
def test_hero_zones_candidate_counts(name, family, count):
    source, target = load_sample(name)
    assert len(FAMILIES[family](source, target)) == count


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
@pytest.mark.parametrize("family", [FAMILY_HORIZONTAL_FLOW, FAMILY_VERTICAL_STACK,
                                     FAMILY_HERO_LEFT, FAMILY_HERO_RIGHT])
def test_background_fills_target_exactly(name, family):
    source, target = load_sample(name)
    for candidate in FAMILIES[family](source, target):
        background = boxes_by_id(candidate)["bg"]
        assert (background["x"], background["y"]) == (0.0, 0.0)
        assert (background["width"], background["height"]) == (
            float(target.width), float(target.height))


def test_canvas_fit_maximizes_edge_touching_contain_image():
    source = design(1000, 1000, [
        image("big", 0, 0, 1000, 1000, fit="contain"),
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


def test_zero_area_slots_keep_degeneracy():
    source = design(1000, 1000, [
        image("line", 100, 100, 0, 500, fit="cover"),
        text("t", 100, 100, 200, 60, content="hi", font_size=20),
    ])
    target = target_canvas(1600, 400)
    candidates = horizontal_flow_candidates(source, target)
    assert len(candidates) == 1
    line = boxes_by_id(candidates[0])["line"]
    assert line["width"] == 0.0
    usable_h = 400 * (1 - MARGIN_FRACTION * (1 + SAFE_BOTTOM_MARGIN_FACTOR))
    assert (line["x"], line["y"], line["height"]) == pytest.approx(
        (432.0, 16.0, usable_h), abs=1e-6)
    report = validate_candidate(source, candidates[0], target)
    assert report.is_valid, report.violations


def test_fill_predicate_requires_all_edges():
    assert FILL_EDGE_TOLERANCE_PX == 1.0
    canvas = target_canvas(1000, 1000)

    def fills(data):
        element = design(1000, 1000, [data]).elements[0]
        return _is_fill_member(element, canvas)

    assert fills(shape("s", 0, 0, 1000, 1000)) is True
    assert fills(image("i", 0, 0, 1000, 1000, fit="cover")) is True
    assert fills(text("t", 0, 0, 1000, 1000, content="x", font_size=12)) is True
    assert fills(shape("s", -5, -5, 1010, 1010)) is True
    assert fills(shape("s", 1.1, 0, 998.9, 1000)) is False
    assert fills(shape("s", 0, 0, 998.9, 1000)) is False
    assert fills(image("i", 100, 100, 200, 200, fit="cover")) is False
    assert fills(image("i", 0, 0, 1000, 1000, fit="contain")) is False


def test_small_cover_photo_slots():
    source = design(1000, 1000, [
        image("photo", 100, 100, 200, 200, fit="cover"),
        text("t", 400, 400, 200, 60, content="hi", font_size=20),
    ])
    target = target_canvas(1600, 400)
    candidates = horizontal_flow_candidates(source, target)
    assert len(candidates) == 1
    photo = boxes_by_id(candidates[0])["photo"]
    assert photo["width"] < 1600.0 and photo["height"] < 400.0
    report = validate_candidate(source, candidates[0], target)
    assert report.is_valid, report.violations


def test_inset_panel_slots():
    source = design(1000, 1000, [
        shape("panel", 50, 50, 900, 900),
        text("t", 100, 100, 200, 60, content="hi", font_size=20),
    ])
    assert is_background_like(source.elements[0], source.source_canvas)
    target = target_canvas(1600, 400)
    candidates = horizontal_flow_candidates(source, target)
    assert len(candidates) == 1
    panel = boxes_by_id(candidates[0])["panel"]
    assert panel["width"] < 1600.0 and panel["height"] < 400.0
    report = validate_candidate(source, candidates[0], target)
    assert report.is_valid, report.violations


def test_no_background_design_slots_everything():
    source = design(1000, 1000, [
        shape("a", 100, 100, 200, 200),
        shape("b", 400, 400, 200, 200),
        text("t", 100, 500, 200, 60, content="hi", font_size=20),
        image("logo", 650, 100, 200, 100, fit="contain"),
    ])
    target = target_canvas(1600, 400)
    for candidates in (horizontal_flow_candidates(source, target),
                       vertical_stack_candidates(source, target)):
        assert len(candidates) == 1
        for element in candidates[0]["elements"]:
            assert (element["width"], element["height"]) != (1600.0, 400.0)
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


def test_core_badge_respects_headline_hierarchy():
    source, target = load_sample("chameleon_sample_level2_core.json")
    for candidates in (horizontal_flow_candidates(source, target),
                       vertical_stack_candidates(source, target)):
        boxes = boxes_by_id(candidates[0])
        assert boxes["badge_label"]["font_size"] <= boxes["headline"]["font_size"]


def test_hierarchy_cap_inversion_trap():
    source = design(1000, 1000, [
        text("big", 100, 100, 300, 60, content="word " * 40, font_size=40),
        text("small", 100, 500, 400, 100, content="hi", font_size=12),
    ])
    target = target_canvas(1600, 400)
    candidates = horizontal_flow_candidates(source, target)
    assert len(candidates) == 1
    boxes = boxes_by_id(candidates[0])
    assert boxes["big"]["font_size"] < 30.0
    assert boxes["small"]["font_size"] <= boxes["big"]["font_size"]
    assert boxes["small"]["font_size"] >= 1.0
    report = validate_candidate(source, candidates[0], target)
    assert report.is_valid, report.violations


def test_equal_source_sizes_share_cap():
    source = design(1000, 1000, [
        text("dense", 100, 100, 300, 60, content="word " * 40, font_size=20),
        text("roomy", 100, 500, 400, 100, content="hi", font_size=20),
    ])
    target = target_canvas(1600, 400)
    candidates = horizontal_flow_candidates(source, target)
    assert len(candidates) == 1
    boxes = boxes_by_id(candidates[0])
    assert boxes["roomy"]["font_size"] > boxes["dense"]["font_size"]
    assert boxes["dense"]["font_size"] >= 1.0
    report = validate_candidate(source, candidates[0], target)
    assert report.is_valid, report.violations


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
    hero_left = hero_left_candidates(source, target)
    hero_right = hero_right_candidates(source, target)
    assert len(hero_left) == 2 and len(hero_right) == 2
    for candidate in hero_left + hero_right:
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
    assert hero_left_candidates(source, target) == hero_left_candidates(source, target)
    assert hero_right_candidates(source, target) == hero_right_candidates(source, target)
    assert baseline_candidate(source, target) == baseline_candidate(source, target)


def test_baseline_is_first_proportional_repair():
    source, target = load_sample("chameleon_sample_level2_core.json")
    assert baseline_candidate(source, target) == baseline_proposal(source, target, 1)


def center_x(box):
    return box["x"] + box["width"] / 2


def center_y(box):
    return box["y"] + box["height"] / 2


def test_hero_picks_largest_image_and_mirrors():
    source, target = load_sample("chameleon_sample_level2_core.json")
    left = boxes_by_id(hero_left_candidates(source, target)[0])
    right = boxes_by_id(hero_right_candidates(source, target)[0])
    assert center_x(left["product"]) < target.width / 2
    assert center_x(right["product"]) > target.width / 2
    assert left["product"]["width"] == pytest.approx(right["product"]["width"])
    assert left["product"]["height"] == pytest.approx(right["product"]["height"])
    assert center_x(left["product"]) + center_x(right["product"]) == pytest.approx(
        target.width, abs=1e-6)
    for placed in (left, right):
        union_left = min(placed["headline"]["x"], placed["deco_circle_2"]["x"])
        union_right = max(placed["headline"]["x"] + placed["headline"]["width"],
                          placed["deco_circle_2"]["x"] + placed["deco_circle_2"]["width"])
        placed["headline_unit_cx"] = (union_left + union_right) / 2
    assert left["headline_unit_cx"] + right["headline_unit_cx"] == pytest.approx(
        target.width, abs=1e-6)


def test_hero_falls_back_to_largest_unit_without_images():
    source = design(1000, 1000, [
        text("big", 0, 0, 600, 200, content="big", font_size=20),
        text("small", 0, 300, 200, 50, content="small", font_size=12),
        shape("chip", 0, 400, 100, 100),
    ])
    target = target_canvas(1000, 1000)
    assert required_group_modes(source) == (GROUP_MODE_COHESIVE,)
    candidates = hero_left_candidates(source, target)
    assert len(candidates) == 1
    placed = boxes_by_id(candidates[0])
    assert center_x(placed["big"]) < target.width / 2
    assert center_x(placed["small"]) > target.width / 2
    assert center_x(placed["chip"]) > target.width / 2


def test_hero_small_only_image_anchors_while_text_heroes():
    source, target = load_sample("chameleon_sample_level1_warmup.json")
    assert target.width / target.height < PORTRAIT_ASPECT_CUTOFF
    for family, logo_left in ((hero_left_candidates, False),
                              (hero_right_candidates, True)):
        candidates = family(source, target)
        assert len(candidates) == 3
        top, bottom, center = (boxes_by_id(candidate)["logo"]
                               for candidate in candidates)
        if logo_left:
            assert top["x"] < 1.0 and bottom["x"] < 1.0
        else:
            assert top["x"] + top["width"] > target.width - 1.0
            assert bottom["x"] + bottom["width"] > target.width - 1.0
        assert top["y"] < 1.0
        assert bottom["y"] + bottom["height"] > target.height - 1.0
        assert center_x(center) == pytest.approx(target.width / 2)
        assert center["y"] < 1.0
        placed = boxes_by_id(candidates[0])
        if logo_left:
            assert center_x(placed["headline"]) > target.width / 2
        else:
            assert center_x(placed["headline"]) < target.width / 2


def test_hero_corner_variants_top_and_bottom():
    source, target = load_sample("chameleon_sample_level2_core.json")
    for family, logo_left in ((hero_left_candidates, False),
                              (hero_right_candidates, True)):
        candidates = family(source, target)
        assert len(candidates) == 4
        logos = [boxes_by_id(candidate)["logo"] for candidate in candidates]
        if logo_left:
            assert all(logo["x"] < 1.0 for logo in logos)
        else:
            assert all(logo["x"] + logo["width"] > target.width - 1.0
                       for logo in logos)
        tops = sorted(logo["y"] for logo in logos)
        assert tops[0] < 1.0 and tops[1] < 1.0
        assert all(299.0 < top < 300.0 for top in tops[2:])


def test_hero_small_stays_small_and_keeps_ratio():
    source, target = load_sample("chameleon_sample_level2_core.json")
    for family in (hero_left_candidates, hero_right_candidates):
        for candidate in family(source, target):
            logo = boxes_by_id(candidate)["logo"]
            assert logo["width"] <= target.width * SMALL_ANCHOR_MAX_FRACTION + 1.0
            assert logo["height"] <= target.height * SMALL_ANCHOR_MAX_FRACTION + 1.0
            assert logo["width"] / logo["height"] == pytest.approx(320 / 110, rel=1e-9)


def boxes_overlap(first, second):
    return (min(first["x"] + first["width"], second["x"] + second["width"])
            > max(first["x"], second["x"])
            and min(first["y"] + first["height"], second["y"] + second["height"])
            > max(first["y"], second["y"]))


TEXT_AND_PILL_IDS = ("headline", "subhead", "badge_bg", "badge_label",
                     "cta_bg", "cta_label")


def test_overlay_rides_only_image_unit_with_uniform_offsets():
    source, target = load_sample("chameleon_sample_level2_core.json")
    placed = boxes_by_id(hero_left_candidates(source, target)[0])
    product, deco = placed["product"], placed["deco_circle_1"]
    scale = product["width"] / 500
    assert center_x(deco) - center_x(product) == pytest.approx(
        -390 * scale, abs=1e-6)
    assert center_y(deco) - center_y(product) == pytest.approx(
        -360 * scale, abs=1e-6)
    assert boxes_overlap(product, deco)
    for other_id in TEXT_AND_PILL_IDS:
        assert not boxes_overlap(deco, placed[other_id]), other_id
    wash = placed["deco_circle_2"]
    cap = DETACHED_OVERLAY_MAX_FRACTION * min(target.width, target.height)
    assert max(wash["width"], wash["height"]) <= cap + 1.0
    for other_id in TEXT_AND_PILL_IDS:
        assert not boxes_overlap(wash, placed[other_id]), other_id


def test_overlay_without_target_stays_independent_and_wash_fills():
    assert 0.2 < OVERLAY_OPACITY_THRESHOLD
    source = design(1000, 1000, [
        image("photo", 100, 100, 400, 400),
        shape("wash", 250, 150, 200, 200, opacity=0.2),
        shape("loner", 700, 700, 100, 100, opacity=0.2),
        shape("veil", 0, 0, 1000, 1000, opacity=0.1),
    ])
    target = target_canvas(1000, 1000)
    candidates = hero_left_candidates(source, target)
    assert len(candidates) == 1
    placed = boxes_by_id(candidates[0])
    veil = placed["veil"]
    assert (veil["x"], veil["y"]) == (0.0, 0.0)
    assert (veil["width"], veil["height"]) == (1000.0, 1000.0)
    assert center_x(placed["loner"]) > target.width / 2
    photo, wash = placed["photo"], placed["wash"]
    assert center_x(wash) - center_x(photo) == pytest.approx(
        -(center_y(wash) - center_y(photo)), abs=1e-6)
    assert center_x(wash) - center_x(photo) > 0


def test_pill_survives_split_mode():
    source = design(1000, 1000, [
        shape("pill_bg", 0, 0, 100, 100, group_id="p"),
        text("pill_label", 500, 500, 100, 50, content="x", font_size=16,
             group_id="p"),
        text("title", 100, 100, 300, 60, content="title", font_size=20),
    ])
    assert required_group_modes(source) == (GROUP_MODE_COHESIVE, GROUP_MODE_SPLIT)
    target = target_canvas(1000, 1000)
    candidates = hero_left_candidates(source, target)
    assert len(candidates) == 2
    for candidate in candidates:
        placed = boxes_by_id(candidate)
        back, label = placed["pill_bg"], placed["pill_label"]
        assert (center_x(label) - center_x(back)) / 500 == pytest.approx(
            (center_y(label) - center_y(back)) / 475, rel=1e-9)
        assert center_x(placed["title"]) > target.width / 2


def test_similar_pills_separate_without_passing_neighbors():
    source = design(1000, 1000, [
        shape("a_bg", 0, 0, 200, 50, group_id="a"),
        text("a_label", 0, 5, 200, 40, content="a", font_size=14, group_id="a"),
        shape("b_bg", 0, 100, 200, 50, group_id="b"),
        text("b_label", 0, 105, 200, 40, content="b", font_size=14, group_id="b"),
        text("solo", 0, 200, 300, 60, content="hello", font_size=20),
        image("hero", 0, 300, 400, 400),
    ])
    assert required_group_modes(source) == (GROUP_MODE_COHESIVE,)
    target = target_canvas(1000, 1000)
    candidates = hero_left_candidates(source, target)
    assert len(candidates) == 1
    placed = boxes_by_id(candidates[0])
    assert placed["a_bg"]["y"] < placed["solo"]["y"] < placed["b_bg"]["y"]
    assert center_x(placed["hero"]) < target.width / 2


def test_headline_before_subhead_core_both_families():
    source, target = load_sample("chameleon_sample_level2_core.json")
    for family in (hero_left_candidates, hero_right_candidates):
        for candidate in family(source, target):
            placed = boxes_by_id(candidate)
            assert placed["headline"]["y"] < placed["subhead"]["y"]


def test_hero_column_count_follows_aspect():
    elements = [text(f"t{pos}", 0, pos * 100, 200, 60, content="w", font_size=16)
                for pos in range(6)]
    source = design(1000, 1000, elements)
    assert required_group_modes(source) == (GROUP_MODE_COHESIVE,)
    wide = target_canvas(1000, 400)
    assert wide.width / wide.height >= FAMILY_ASPECT_CUTOFF
    placed = boxes_by_id(hero_left_candidates(source, wide)[0])
    assert len({placed[f"t{pos}"]["x"] for pos in range(1, 6)}) == 1
    narrow = target_canvas(400, 1000)
    assert narrow.width / narrow.height < FAMILY_ASPECT_CUTOFF
    assert 5 > HERO_MAX_SINGLE_COLUMN_UNITS
    placed = boxes_by_id(hero_left_candidates(source, narrow)[0])
    left_x = [placed[f"t{pos}"]["x"] for pos in (1, 2, 3)]
    right_x = [placed[f"t{pos}"]["x"] for pos in (4, 5)]
    assert left_x[0] == left_x[1] == left_x[2]
    assert right_x[0] == right_x[1]
    assert left_x[0] < right_x[0]


def test_hero_column_packs_estimate_sized_rows():
    source, target = load_sample("chameleon_sample_level2_core.json")
    placed = boxes_by_id(hero_left_candidates(source, target)[0])
    usable_w = 1600 * (1 - 2 * MARGIN_FRACTION)
    side_w = (usable_w - usable_w * 0.05) / 2
    column_x = 64.0 + side_w + usable_w * 0.05
    headline, subhead = placed["headline"], placed["subhead"]
    assert (headline["x"], headline["width"]) == pytest.approx((column_x, side_w))
    assert (subhead["x"], subhead["width"]) == pytest.approx((column_x, side_w))
    inner = subhead["y"] - (headline["y"] + headline["height"])
    outer = placed["cta_bg"]["y"] - (subhead["y"] + subhead["height"])
    assert 0 < inner < outer
    assert placed["badge_bg"]["width"] <= 220.0 + 1.0
    usable_bottom = 400 * (1 - MARGIN_FRACTION * SAFE_BOTTOM_MARGIN_FACTOR)
    assert placed["cta_bg"]["y"] + placed["cta_bg"]["height"] < usable_bottom


def test_text_blocks_merge_stacked_aligned_pairs_only():
    source = design(1000, 1000, [
        text("a", 0, 0, 200, 60, content="a", font_size=16),
        text("b", 0, 60, 200, 60, content="b", font_size=16),
        text("far", 0, 500, 200, 60, content="c", font_size=16),
        shape("pill_bg", 0, 300, 100, 100, group_id="p"),
        text("pill_label", 0, 305, 100, 40, content="d", font_size=14,
             group_id="p"),
    ])
    units, block_keys = _merge_text_blocks(
        list(build_layout_units(source, group_mode=GROUP_MODE_COHESIVE)),
        source.source_canvas)
    assert len(block_keys) == 1
    assert sorted(next(iter(block_keys))) == [0, 1]
    assert len(units) == 3


def test_packed_block_uses_inner_gap_between_members():
    source = design(1000, 1000, [
        text("a", 0, 0, 200, 60, content="w", font_size=16),
        text("b", 0, 60, 200, 60, content="w", font_size=16),
        text("c", 0, 500, 200, 60, content="w", font_size=16),
        image("hero", 0, 700, 400, 200),
    ])
    target = target_canvas(1000, 1000)
    placed = boxes_by_id(hero_left_candidates(source, target)[0])
    inner = placed["b"]["y"] - (placed["a"]["y"] + placed["a"]["height"])
    outer = placed["c"]["y"] - (placed["b"]["y"] + placed["b"]["height"])
    assert inner / outer == pytest.approx(BLOCK_INNER_GAP_EM / BLOCK_OUTER_GAP_EM)


def test_third_small_overflows_into_column():
    source = design(1000, 1000, [
        image("big", 0, 0, 500, 500),
        image("s1", 0, 600, 100, 100),
        image("s2", 200, 600, 100, 100),
        image("s3", 400, 600, 100, 100),
    ])
    target = target_canvas(1000, 1000)
    placed = boxes_by_id(hero_left_candidates(source, target)[0])
    corners = [placed["s1"], placed["s2"]]
    assert all(box["y"] < 1.0 for box in corners)
    assert {box["x"] < 1.0 for box in corners} == {True, False}
    third = placed["s3"]
    assert 1.0 < third["x"] and third["x"] + third["width"] < 999.0


def test_label_block_swap_varies_core_column_order():
    source, target = load_sample("chameleon_sample_level2_core.json")
    candidates = hero_left_candidates(source, target)
    assert len(candidates) == 4
    base = boxes_by_id(candidates[0])
    assert base["headline"]["y"] < base["cta_bg"]["y"]
    swapped = boxes_by_id(candidates[2])
    assert swapped["cta_bg"]["y"] < swapped["headline"]["y"]
    assert candidates[0] != candidates[2]


def test_label_block_swaps_vary_l3_column_order():
    source, target = load_sample("chameleon_sample_level3_advanced.json")
    candidates = hero_left_candidates(source, target)
    assert len(candidates) == 4
    base = boxes_by_id(candidates[0])
    assert base["ribbon_bg"]["y"] < base["headline"]["y"] < base["badge1_bg"]["y"]
    swapped = boxes_by_id(candidates[2])
    assert swapped["headline"]["y"] < swapped["ribbon_bg"]["y"]
    assert candidates[0] != candidates[2]


def test_banner_zones_order_and_differ_from_hero():
    source, target = load_sample("chameleon_sample_level2_core.json")
    zones = banner_zones_candidates(source, target)
    assert len(zones) == 4
    placed = boxes_by_id(zones[0])
    assert (center_x(placed["badge_bg"]) < center_x(placed["product"])
            < center_x(placed["deco_circle_2"]) < center_x(placed["headline"])
            < center_x(placed["cta_bg"]))
    hero = boxes_by_id(hero_left_candidates(source, target)[0])
    assert zones[0] != hero


def test_overlay_without_image_target_stays_capped_and_disjoint():
    source = design(1000, 1000, [
        text("t", 100, 100, 400, 100, content="hello", font_size=20),
        shape("wash", 150, 120, 400, 400, opacity=0.2),
        image("photo", 600, 600, 300, 300),
    ])
    target = target_canvas(1000, 1000)
    placed = boxes_by_id(hero_left_candidates(source, target)[0])
    wash = placed["wash"]
    cap = DETACHED_OVERLAY_MAX_FRACTION * min(target.width, target.height)
    assert max(wash["width"], wash["height"]) <= cap + 1.0
    assert not boxes_overlap(wash, placed["t"])
