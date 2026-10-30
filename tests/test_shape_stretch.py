"""Shape roles, independent shape stretching, and pool integration (D.1.7).

Core suite: standard library plus S02/S04 modules only (no torch).
"""
import copy
import json
from pathlib import Path

import pytest

from engine import candidate_pool as pool_module
from engine.candidate_generator import GENERATOR_VERSION, baseline_candidate
from engine.candidate_pool import PoolConfig, build_candidate_pool
from engine.candidate_validator import validate_candidate
from engine.internal_schema import Canvas, Design
from engine.layout_edits import ProtectedFieldError, layout_edit
from engine.layout_objective import evaluate_objective
from engine.shape_roles import ShapeRole, classify_shapes
from engine.shape_stretch import (
    SHAPE_STRETCH_VERSION, VARIANT_AXIS_X, VARIANT_AXIS_Y, VARIANT_BALANCED,
    VARIANT_LABEL_FIT, VARIANT_PANEL_FILL, VARIANT_PROPORTIONAL, stretch_variants,
)

TARGETS = [(728, 90), (1600, 400), (90, 728), (300, 250), (1080, 1350)]


def shape(ident, x, y, w, h, **extra):
    data = dict(id=ident, type="shape", x=x, y=y, width=w, height=h,
                shape_type="rectangle", fill_color="rgba(10,20,30,1)")
    data.update(extra)
    return data


def text(ident, x, y, w, h, content, size, **extra):
    data = dict(id=ident, type="text", x=x, y=y, width=w, height=h, content=content,
                font_family="Solstice Sans", font_size=size, color="rgba(0,0,0,1)")
    data.update(extra)
    return data


def source_json():
    return dict(canvas=dict(width=1000, height=1000), elements=[
        shape("bg", 0, 0, 1000, 1000, z_index=0),
        shape("panel", 60, 60, 880, 300, z_index=1, fill_color="rgba(200,200,255,1)"),
        text("headline", 100, 100, 600, 80, "Big summer sale", 48, z_index=3),
        shape("rule", 100, 190, 600, 6, z_index=2),
        shape("divider", 800, 380, 4, 600, z_index=2),
        shape("dot", 850, 800, 100, 100, shape_type="ellipse", z_index=2),
        shape("cta_bg", 300, 800, 400, 80, corner_radius=40, group_id="cta", z_index=4),
        text("cta_label", 320, 815, 360, 50, "Shop now", 32, group_id="cta", z_index=5,
             text_align="center"),
        text("body", 100, 420, 500, 100, "Save on every item this week only", 24, z_index=3),
    ])


@pytest.fixture(scope="module")
def source():
    return Design.from_adobe(source_json())


def uniform_base(source, target, scale, origin=(0.0, 0.0)):
    """Hand-built proportional placement: every box scaled about the origin."""
    raw = source.to_adobe()
    raw["canvas"] = target.to_adobe()
    for element in raw["elements"]:
        element["x"] = origin[0] + element["x"] * scale
        element["y"] = origin[1] + element["y"] * scale
        element["width"] *= scale
        element["height"] *= scale
        if element["type"] == "text":
            element["font_size"] = max(1.0, element["font_size"] * scale)
    return raw


def boxes(raw):
    return {e["id"]: (e["x"], e["y"], e["width"], e["height"]) for e in raw["elements"]}


def by_id(raw):
    return {e["id"]: e for e in raw["elements"]}


def variants_by_name(source, target, base):
    return {v.name: v for v in stretch_variants(source, target, base)}


def inside(inner, outer, tol=1e-6):
    return (inner[0] >= outer[0] - tol and inner[1] >= outer[1] - tol
            and inner[0] + inner[2] <= outer[0] + outer[2] + tol
            and inner[1] + inner[3] <= outer[1] + outer[3] + tol)


# --- roles ----------------------------------------------------------------------
def test_roles_on_synthetic_source(source):
    roles = {k: v.role for k, v in classify_shapes(source).items()}
    assert roles == {
        "bg": ShapeRole.BACKGROUND, "panel": ShapeRole.PANEL, "rule": ShapeRole.BAR_H,
        "divider": ShapeRole.BAR_V, "dot": ShapeRole.PROPORTION_SENSITIVE,
        "cta_bg": ShapeRole.BUTTON,
    }
    assert classify_shapes(source)["cta_bg"].contained_ids == ("cta_label",)


def test_roles_ignore_ids_and_notes_and_default_to_fixed():
    raw = source_json()
    for element in raw["elements"]:
        element["id"] = {"bg": "logo", "cta_bg": "bg", "rule": "cta_bg"}.get(element["id"], element["id"])
        element["_note"] = "this is the background"
    raw["elements"][7]["group_id"] = "cta"
    renamed = Design.from_adobe(raw)
    roles = {k: v.role for k, v in classify_shapes(renamed).items()}
    assert roles["logo"] is ShapeRole.BACKGROUND and roles["bg"] is ShapeRole.BUTTON
    assert roles["cta_bg"] is ShapeRole.BAR_H
    ambiguous = Design.from_adobe(dict(canvas=dict(width=1000, height=1000), elements=[
        shape("a", 100, 100, 300, 200),      # mid-size, unlabeled, not thin: ambiguous
        shape("b", 500, 100, 120, 120),      # empty square: icon-like
        shape("c", 100, 500, 300, 60, rotation=15, group_id="g"),
        text("t", 110, 510, 280, 40, "Rotated", 20, rotation=15, group_id="g"),
    ]))
    found = {k: v.role for k, v in classify_shapes(ambiguous).items()}
    assert found == {"a": ShapeRole.FIXED, "b": ShapeRole.PROPORTION_SENSITIVE, "c": ShapeRole.FIXED}


# --- independent stretching ------------------------------------------------------
def test_wide_button_gets_wider_without_proportional_height():
    raw = source_json()
    raw["elements"][7]["content"] = "Subscribe to our weekly newsletter"
    long_label = Design.from_adobe(raw)
    target = Canvas(width=1000, height=500)
    base = uniform_base(long_label, target, 0.25)
    variant = variants_by_name(long_label, target, base)[VARIANT_LABEL_FIT]
    before, after = boxes(base)["cta_bg"], boxes(variant.adobe)["cta_bg"]
    width_ratio, height_ratio = after[2] / before[2], after[3] / before[3]
    assert width_ratio > 1.5 and height_ratio < width_ratio
    assert after[2] / after[3] > before[2] / before[3]
    assert validate_candidate(long_label, variant.adobe, target).is_valid


def test_label_fit_keeps_label_inside_button_with_balanced_padding(source):
    for width, height in TARGETS:
        target = Canvas(width=width, height=height)
        base = baseline_candidate(source, target)
        variant = variants_by_name(source, target, base).get(VARIANT_LABEL_FIT)
        assert variant is not None, (width, height)
        placed = boxes(variant.adobe)
        button, label = placed["cta_bg"], placed["cta_label"]
        assert inside(label, button)
        left = label[0] - button[0]
        right = button[0] + button[2] - label[0] - label[2]
        top = label[1] - button[1]
        bottom = button[1] + button[3] - label[1] - label[3]
        assert left == pytest.approx(right, abs=1e-6) and top == pytest.approx(bottom, abs=1e-6)
        assert left > 0 and top > 0
        assert variant.adobe["elements"][7]["font_size"] >= 1.0
        assert validate_candidate(source, variant.adobe, target).is_valid


def test_label_fit_never_outgrows_larger_source_text_or_cap(source):
    target = Canvas(width=1600, height=400)
    base = baseline_candidate(source, target)
    variant = variants_by_name(source, target, base)[VARIANT_LABEL_FIT]
    sizes = {e["id"]: e["font_size"] for e in variant.adobe["elements"] if e["type"] == "text"}
    base_sizes = {e["id"]: e["font_size"] for e in base["elements"] if e["type"] == "text"}
    assert sizes["cta_label"] <= base_sizes["cta_label"] * 1.5 + 1e-9
    assert sizes["cta_label"] <= base_sizes["headline"] + 1e-9
    assert {k: v for k, v in sizes.items() if k != "cta_label"} == \
        {k: v for k, v in base_sizes.items() if k != "cta_label"}


def test_background_rectangle_fills_target_region(source):
    for width, height in TARGETS:
        target = Canvas(width=width, height=height)
        base = baseline_candidate(source, target)
        variant = variants_by_name(source, target, base)[VARIANT_PANEL_FILL]
        placed = boxes(variant.adobe)
        assert placed["bg"] == pytest.approx((0, 0, width, height), abs=1e-3)
        panel = placed["panel"]
        assert panel[0] == pytest.approx(0.06 * width, rel=1e-6)
        assert panel[2] == pytest.approx(0.88 * width, rel=1e-6)
        assert validate_candidate(source, variant.adobe, target).is_valid


def test_panel_still_holds_its_content(source):
    for width, height in TARGETS:
        target = Canvas(width=width, height=height)
        base = baseline_candidate(source, target)
        variant = variants_by_name(source, target, base)[VARIANT_PANEL_FILL]
        placed = boxes(variant.adobe)
        for held in ("headline", "rule"):
            assert inside(placed[held], placed["panel"], tol=1e-4)


def test_horizontal_bar_lengthens_but_not_thicker(source):
    target = Canvas(width=1000, height=500)
    base = uniform_base(source, target, 0.5)
    base_boxes = boxes(base)
    # The heading was flowed wider than a uniform scale; its underline should follow.
    for element in base["elements"]:
        if element["id"] == "headline":
            element["width"] *= 1.6
    variant = variants_by_name(source, target, base)[VARIANT_AXIS_X]
    before, after = base_boxes["rule"], boxes(variant.adobe)["rule"]
    assert after[2] > before[2] * 1.4
    assert after[3] == before[3] and after[1] == before[1]
    headline = boxes(variant.adobe)["headline"]
    assert after[0] == pytest.approx(headline[0], abs=1e-6)
    assert after[0] + after[2] == pytest.approx(headline[0] + headline[2] * 600 / 600, abs=1e-6)
    assert validate_candidate(source, variant.adobe, target).is_valid


def test_vertical_divider_becomes_taller_but_not_wider(source):
    target = Canvas(width=500, height=1000)
    base = uniform_base(source, target, 0.5, origin=(0.0, 0.0))
    variant = variants_by_name(source, target, base)[VARIANT_AXIS_Y]
    before, after = boxes(base)["divider"], boxes(variant.adobe)["divider"]
    assert after[3] > before[3] * 1.05
    assert after[2] == before[2] and after[0] == before[0]
    assert after[3] == pytest.approx(source_json()["elements"][4]["height"] / 1000 * 1000, rel=1e-6)
    assert validate_candidate(source, variant.adobe, target).is_valid


def test_circle_stays_circular_in_every_variant_and_pool_candidate(source):
    original = by_id(source.to_adobe())["dot"]
    assert original["width"] == original["height"]
    for width, height in TARGETS:
        target = Canvas(width=width, height=height)
        pool = build_candidate_pool(source, target)
        assert any(r.variant != VARIANT_PROPORTIONAL for r in pool.candidates)
        for record in pool.candidates:
            if record.validation.is_valid and record.variant != VARIANT_PROPORTIONAL:
                dot = by_id(record.to_adobe())["dot"]
                assert dot["width"] == pytest.approx(dot["height"])
                if record.parent_candidate_id is None:  # a stretch variant never touches the circle
                    base_dot = by_id(next(r for r in pool.candidates
                                          if r.candidate_id == record.base_candidate_id).to_adobe())["dot"]
                    assert (dot["x"], dot["y"], dot["width"], dot["height"]) == \
                        (base_dot["x"], base_dot["y"], base_dot["width"], base_dot["height"])


def test_stretch_changes_only_k6_fields_and_preserves_identity(source):
    target = Canvas(width=728, height=90)
    base = baseline_candidate(source, target)
    variants = stretch_variants(source, target, base)
    assert {v.name for v in variants} >= {VARIANT_LABEL_FIT, VARIANT_PANEL_FILL, VARIANT_BALANCED}
    for variant in variants:
        assert [e["id"] for e in variant.adobe["elements"]] == [e["id"] for e in source.to_adobe()["elements"]]
        assert set(variant.adobe) == {"canvas", "elements"}
        for new, old in zip(variant.adobe["elements"], base["elements"]):
            changed = {k for k in new if new[k] != old.get(k)}
            allowed = {"x", "y", "width", "height"} | ({"font_size"} if new["type"] == "text" else set())
            assert changed <= allowed
            assert new.get("group_id") == old.get("group_id") and new.get("z_index") == old.get("z_index")
            if new["id"] not in variant.changed_ids:
                assert new == old
        assert validate_candidate(source, variant.adobe, target).is_valid
    assert base == baseline_candidate(source, target)  # the base was not mutated


def test_variants_are_deterministic_and_distinct(source):
    target = Canvas(width=1600, height=400)
    base = baseline_candidate(source, target)
    first, second = stretch_variants(source, target, base), stretch_variants(source, target, base)
    assert [(v.name, v.adobe, v.changed_ids) for v in first] == [(v.name, v.adobe, v.changed_ids) for v in second]
    assert len({json.dumps(v.adobe, sort_keys=True) for v in first}) == len(first)
    assert all(v.adobe != base for v in first)


def test_no_variants_when_nothing_can_stretch():
    fixed = Design.from_adobe(dict(canvas=dict(width=1000, height=1000), elements=[
        shape("dot", 100, 100, 100, 100, shape_type="ellipse"),
        text("t", 300, 300, 200, 40, "Hello", 24),
    ]))
    target = Canvas(width=728, height=90)
    assert stretch_variants(fixed, target, baseline_candidate(fixed, target)) == []
    pool = build_candidate_pool(fixed, target)
    assert all(r.variant == VARIANT_PROPORTIONAL for r in pool.candidates)


def test_blocked_stretch_is_rejected_not_forced(source):
    target = Canvas(width=1000, height=500)
    base = uniform_base(source, target, 0.5)
    for element in base["elements"]:
        if element["id"] == "body":
            # An unrelated text box sits right of the button, where widening would collide.
            element.update(x=boxes(base)["cta_bg"][0] + boxes(base)["cta_bg"][2] + 1,
                           y=boxes(base)["cta_bg"][1], width=60, height=20)
    for variant in stretch_variants(source, target, base):
        placed = boxes(variant.adobe)
        body, button = placed["body"], placed["cta_bg"]
        overlap = (min(body[0] + body[2], button[0] + button[2]) - max(body[0], button[0]) > 1e-6
                   and min(body[1] + body[3], button[1] + button[3]) - max(body[1], button[1]) > 1e-6)
        assert not overlap


# --- validator and edit contract ---------------------------------------------------
def test_validator_rejects_illegal_shape_edits(source):
    target = Canvas(width=728, height=90)
    variant = variants_by_name(source, target, baseline_candidate(source, target))[VARIANT_LABEL_FIT]
    assert validate_candidate(source, variant.adobe, target).is_valid
    for field, value in (("corner_radius", 3), ("fill_color", "rgba(0,0,0,1)"),
                         ("opacity", 0.5), ("rotation", 10), ("group_id", "other"), ("shape_type", "ellipse")):
        tampered = copy.deepcopy(variant.adobe)
        next(e for e in tampered["elements"] if e["id"] == "cta_bg")[field] = value
        assert not validate_candidate(source, tampered, target).is_valid, field
    collapsed = copy.deepcopy(variant.adobe)
    next(e for e in collapsed["elements"] if e["id"] == "rule")["height"] = 0
    assert not validate_candidate(source, collapsed, target).is_valid
    off = copy.deepcopy(variant.adobe)
    next(e for e in off["elements"] if e["id"] == "panel")["width"] = 10_000
    assert not validate_candidate(source, off, target).is_valid
    with pytest.raises(ProtectedFieldError):
        layout_edit(by_id(source.to_adobe())["cta_bg"], {"corner_radius": 3})


# --- pool integration ----------------------------------------------------------------
def test_pool_keeps_proportional_alternatives_and_provenance(source):
    target = Canvas(width=728, height=90)
    pool = build_candidate_pool(source, target)
    stretched = [r for r in pool.candidates if r.variant != VARIANT_PROPORTIONAL and r.parent_candidate_id is None]
    proportional = [r for r in pool.candidates if r.variant == VARIANT_PROPORTIONAL and r.parent_candidate_id is None]
    assert stretched and proportional
    ids = {r.candidate_id for r in pool.candidates}
    for record in stretched:
        assert record.base_candidate_id in ids and record.base_candidate_id != record.candidate_id
        assert record.shape_stretch_version == SHAPE_STRETCH_VERSION
        assert record.generator_version.startswith(GENERATOR_VERSION) and SHAPE_STRETCH_VERSION in record.generator_version
        assert set(record.to_adobe()) == {"canvas", "elements"}
        assert "variant" not in json.dumps(record.to_adobe())
    for record in proportional:
        assert record.base_candidate_id is None and record.shape_stretch_version is None
    assert GENERATOR_VERSION == "s04_generator_v0.3"


def test_pool_is_deterministic_and_respects_budgets(source):
    target = Canvas(width=728, height=90)
    config = PoolConfig(max_raw_candidates=16, max_retained_candidates=5)
    first, second = build_candidate_pool(source, target, config), build_candidate_pool(source, target, config)
    assert first == second
    assert len(first.candidates) <= 16 and len(first.retained) <= 5
    assert any(r.variant != VARIANT_PROPORTIONAL for r in first.candidates)
    capped = build_candidate_pool(source, target, PoolConfig(max_stretch_candidates=2))
    assert sum(r.variant != VARIANT_PROPORTIONAL and r.parent_candidate_id is None for r in capped.candidates) <= 2
    one_base = build_candidate_pool(source, target, PoolConfig(max_stretch_bases=1))
    assert len({r.base_candidate_id for r in one_base.candidates if r.base_candidate_id}) == 1
    off = build_candidate_pool(source, target, PoolConfig(max_stretch_bases=0))
    assert all(r.variant == VARIANT_PROPORTIONAL or r.parent_candidate_id for r in off.candidates)
    plain = build_candidate_pool(source, target, PoolConfig(max_stretch_bases=0, max_refine_bases=0))
    assert all(r.variant == VARIANT_PROPORTIONAL for r in plain.candidates)
    tiny = build_candidate_pool(source, target, PoolConfig(max_raw_candidates=3))
    assert len(tiny.candidates) <= 3
    default = build_candidate_pool(source, target)
    assert len(default.candidates) <= PoolConfig().max_raw_candidates
    assert sum(r.variant != VARIANT_PROPORTIONAL and r.parent_candidate_id is None
               for r in default.candidates) <= PoolConfig().max_stretch_candidates
    with pytest.raises(ValueError):
        PoolConfig(max_stretch_bases=-1)


def test_winner_is_chosen_by_scorer_not_by_stretching(source, monkeypatch):
    target = Canvas(width=728, height=90)

    def shape_area(design):
        return sum(e.width * e.height for e in design.elements if e.type == "shape")

    for sign, expect_stretched in ((1, True), (-1, False)):
        monkeypatch.setattr(pool_module, "score_deterministic",
                            lambda s, designs, t, sign=sign: [sign * shape_area(designs[0]) / 1e6])
        pool = build_candidate_pool(source, target)
        valid = [r for r in pool.candidates if r.validation.is_valid]
        assert pool.winner.score == max(r.score for r in valid)
        if expect_stretched:
            assert pool.winner.variant != VARIANT_PROPORTIONAL
        else:
            assert pool.winner.score >= max(r.score for r in valid if r.variant != VARIANT_PROPORTIONAL)


def test_stretch_variants_survive_in_retained_training_pool(source):
    target = Canvas(width=728, height=90)
    pool = build_candidate_pool(source, target, PoolConfig(max_retained_candidates=12))
    assert any(r.variant != VARIANT_PROPORTIONAL for r in pool.retained)
    assert any(r.variant == VARIANT_PROPORTIONAL for r in pool.retained)
    assert all(r.validation.is_valid for r in pool.retained)
    assert pool.winner is not None and pool.winner.validation.is_valid


@pytest.mark.parametrize("path", sorted(Path("input").glob("*.json")))
@pytest.mark.parametrize("target", [None, (728, 90), (90, 728)])
def test_official_samples_remain_valid_with_stretch(path, target):
    task = json.loads(path.read_text(encoding="utf-8"))
    design = Design.from_adobe(task["source"])
    canvas = Canvas.from_adobe(task["target_canvas"] if target is None else dict(width=target[0], height=target[1]))
    pool = build_candidate_pool(design, canvas)
    assert pool.winner is not None and not pool.failures
    ids = [e["id"] for e in task["source"]["elements"]]
    for record in pool.candidates:
        if record.variant != VARIANT_PROPORTIONAL:
            assert record.validation.is_valid, record.validation.violations
            adobe = record.to_adobe()
            assert [e["id"] for e in adobe["elements"]] == ids
            assert validate_candidate(design, adobe, canvas).is_valid
    assert validate_candidate(design, pool.winner.to_adobe(), canvas).is_valid


def test_samples_with_buttons_gain_a_label_fit_alternative():
    task = json.loads(Path("input/chameleon_sample_level2_core.json").read_text(encoding="utf-8"))
    design = Design.from_adobe(task["source"])
    pool = build_candidate_pool(design, Canvas(width=728, height=90))
    labels = [r for r in pool.candidates if r.variant == VARIANT_LABEL_FIT]
    assert labels
    for record in labels:
        placed = boxes(record.to_adobe())
        for button, label in (("badge_bg", "badge_label"), ("cta_bg", "cta_label")):
            assert inside(placed[label], placed[button], tol=1e-6)


# --- objective ----------------------------------------------------------------------
def test_objective_penalizes_distorted_circles_only(source):
    target = Canvas(width=1000, height=1000)
    reference = evaluate_objective(source, source, target).terms.scale
    # Area-preserving aspect changes isolate distortion from area drift.
    squashed = source.to_adobe()
    by_id(squashed)["dot"].update(width=200, height=50)
    assert evaluate_objective(source, Design.from_adobe(squashed), target).terms.scale > reference
    reshaped_bar = source.to_adobe()
    by_id(reshaped_bar)["rule"].update(width=300, height=12)
    assert evaluate_objective(source, Design.from_adobe(reshaped_bar), target).terms.scale == pytest.approx(reference)


def test_objective_penalizes_off_center_label_padding(source):
    target = Canvas(width=1000, height=1000)
    centered = source.to_adobe()
    skewed = copy.deepcopy(centered)
    label = by_id(skewed)["cta_label"]
    label["x"] = by_id(skewed)["cta_bg"]["x"] + 5
    label["width"] = 330
    assert (evaluate_objective(source, Design.from_adobe(skewed), target).terms.group
            > evaluate_objective(source, source, target).terms.group)


def test_backdrop_panel_cannot_buy_coverage_or_hide_gaps(source):
    target = Canvas(width=728, height=90)
    base = baseline_candidate(source, target)
    wide = copy.deepcopy(base)
    by_id(wide)["panel"].update(x=0, y=0, width=728, height=90)
    base_terms = evaluate_objective(source, Design.from_adobe(base), target).terms
    wide_terms = evaluate_objective(source, Design.from_adobe(wide), target).terms
    assert wide_terms.cover == pytest.approx(base_terms.cover)
    assert wide_terms.space == pytest.approx(base_terms.space)
    assert wide_terms.balance == pytest.approx(base_terms.balance)


def test_axis_variant_names_are_complete():
    from engine.shape_stretch import VARIANT_NAMES
    assert set(VARIANT_NAMES) == {VARIANT_PROPORTIONAL, VARIANT_LABEL_FIT, VARIANT_PANEL_FILL,
                                  VARIANT_AXIS_X, VARIANT_AXIS_Y, VARIANT_BALANCED}
