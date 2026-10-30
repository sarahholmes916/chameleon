"""Local layout refinement (D.1.8): geometry, optimization, compliance, pool integration.

Core suite: standard library plus S02/S04 modules only (no torch, no renderer).
"""
import math
import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from engine import candidate_pool as pool_module
from engine.candidate_pool import PoolConfig, build_candidate_pool, identity
from engine.candidate_validator import ValidationReport, validate_candidate
from engine.internal_schema import Canvas, Design
from engine.layout_objective import score_deterministic
from engine.layout_refinement import (
    REFINEMENT_VERSION, RefinementConfig, RefinementResult, _Search, diagnose, refine_candidate,
)
from engine.layout_typography import estimated_fit, fit_candidate_text
from scripts.source_templates import FAMILIES, make_source


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


def make(elements, size=(1000, 1000)):
    return Design.from_adobe(dict(canvas=dict(width=size[0], height=size[1]), elements=elements))


def candidate_of(source, target, edits=None):
    """Source geometry as a candidate on `target` (same size), with per-id field edits."""
    raw = source.to_adobe()
    raw["canvas"] = target.to_adobe()
    for element in raw["elements"]:
        element.update((edits or {}).get(element["id"], {}))
    return raw


def scorer(source, target):
    return lambda raw: score_deterministic(source, [Design.from_adobe(raw)], target)[0]


def refine(source, target, candidate, **kwargs):
    kwargs.setdefault("fit_text", fit_candidate_text)
    return refine_candidate(source, target, candidate, score=scorer(source, target),
                            validate=validate_candidate, **kwargs)


def boxes(raw):
    return {e["id"]: (e["x"], e["y"], e["width"], e["height"]) for e in raw["elements"]}


def inside(inner, outer, tol=1e-6):
    return (inner[0] >= outer[0] - tol and inner[1] >= outer[1] - tol
            and inner[0] + inner[2] <= outer[0] + outer[2] + tol
            and inner[1] + inner[3] <= outer[1] + outer[3] + tol)


@pytest.fixture(scope="module")
def cta_source():
    return make([
        text("sub", 100, 100, 400, 40, "A short subtitle line", 28, z_index=2),
        shape("cta_bg", 100, 200, 300, 70, corner_radius=35, group_id="cta", z_index=3),
        text("cta_label", 120, 215, 260, 40, "Shop now", 28, group_id="cta", z_index=4, text_align="center"),
        text("note", 600, 700, 300, 40, "Terms apply", 24, z_index=2),
    ])


# --- geometry -----------------------------------------------------------------------
def test_button_moves_away_from_unrelated_text_as_a_unit(cta_source):
    target = Canvas(width=1000, height=1000)
    crowded = candidate_of(cta_source, target, {"cta_bg": dict(y=141), "cta_label": dict(y=156)})
    before = boxes(crowded)
    gap_before = before["cta_bg"][1] - (before["sub"][1] + before["sub"][3])
    result = refine(cta_source, target, crowded)
    after = boxes(result.adobe)
    assert result.changed and result.score_after > result.score_before
    assert after["cta_bg"][1] - (after["sub"][1] + after["sub"][3]) > gap_before + 5
    # the button and its label travelled together: the label is still centered in its background
    shape_box, label_box = after["cta_bg"], after["cta_label"]
    assert inside(label_box, shape_box)
    assert (label_box[1] - shape_box[1]) == pytest.approx(shape_box[1] + shape_box[3] - label_box[1] - label_box[3], abs=1e-3)
    assert validate_candidate(cta_source, result.adobe, target).is_valid


def test_group_members_keep_their_offsets_under_pure_moves(cta_source):
    target = Canvas(width=1000, height=1000)
    crowded = candidate_of(cta_source, target, {"cta_bg": dict(y=141), "cta_label": dict(y=156)})
    search = _Search(cta_source, target, crowded, scorer(cta_source, target), validate_candidate,
                     fit_candidate_text, None, RefinementConfig())
    unit = search.units[search.unit_of["cta_bg"]]
    assert set(unit) == {"cta_bg", "cta_label"}
    moved = search.translate(search.start, unit, 13.0, -7.0)
    for ident in unit:
        before, after = search.start.boxes[ident], moved[ident]
        assert after[0] - before[0] == pytest.approx(13.0) and after[1] - before[1] == pytest.approx(-7.0)
        assert after[2:] == before[2:]


def test_backgrounds_and_panels_never_translate():
    source = make([shape("bg", 0, 0, 1000, 1000, z_index=0), shape("panel", 50, 50, 600, 300, z_index=1),
                   text("t", 100, 100, 300, 50, "Inside the panel", 30, z_index=2)])
    target = Canvas(width=1000, height=1000)
    search = _Search(source, target, candidate_of(source, target), scorer(source, target),
                     validate_candidate, fit_candidate_text, None, RefinementConfig())
    assert {"bg", "panel"} <= search.immovable and "t" not in search.immovable
    assert search.translate(search.start, ("bg",), 5.0, 5.0) is None
    assert search.translate(search.start, ("panel",), 5.0, 5.0) is None


def test_text_box_expansion_horizontal_and_vertical(cta_source):
    target = Canvas(width=1000, height=1000)
    source = make([text("t", 100, 100, 120, 24, "A fairly long headline for this box", 24)])
    candidate = candidate_of(source, target)
    search = _Search(source, target, candidate, scorer(source, target), validate_candidate,
                     fit_candidate_text, None, RefinementConfig())
    element = source.elements[0]
    box = search.start.boxes["t"]
    proposals = search.text_expand_proposals(search.start, element, 60.0, 30.0)
    wider = [p for p in proposals if p.boxes["t"][2] > box[2] and p.boxes["t"][3] == box[3]]
    taller = [p for p in proposals if p.boxes["t"][3] > box[3] and p.boxes["t"][2] == box[2]]
    assert wider and taller
    assert all(p.fonts == {} for p in proposals)            # expansion never touches the font size
    assert not search.fits("t", box, 24)                      # the start overflows


def test_refinement_enlarges_a_shrunken_text_by_expanding_its_box():
    source = make([text("t", 100, 100, 120, 24, "A fairly long headline for this box", 24)])
    target = Canvas(width=1000, height=1000)
    fitted = fit_candidate_text(candidate_of(source, target))   # what the pool hands to refinement
    shrunk = fitted["elements"][0]["font_size"]
    assert shrunk < 24 * 0.5
    result = refine(source, target, fitted)
    start, end = boxes(fitted)["t"], boxes(result.adobe)["t"]
    font = result.adobe["elements"][0]["font_size"]
    assert result.changed and font >= shrunk * 1.4 and end[2] * end[3] > start[2] * start[3]
    assert estimated_fit(dict(result.adobe["elements"][0]))["fits"]


def test_small_text_is_enlarged_by_growing_its_box_and_pushing_neighbors():
    source = make([
        text("head", 100, 100, 400, 60, "Big summer sale", 40, z_index=2),
        text("body", 100, 175, 400, 40, "Save on every item this week only", 24, z_index=2),
    ])
    target = Canvas(width=1000, height=1000)
    tiny = fit_candidate_text(candidate_of(source, target, {
        "head": dict(font_size=5, height=8, width=100), "body": dict(font_size=5, y=112, height=8, width=130)}))
    before = {e["id"]: e["font_size"] for e in tiny["elements"]}
    assert diagnose(source, target, tiny)["small_text"] == 2
    result = refine(source, target, tiny, config=RefinementConfig(max_evaluations=300, max_rounds=12))
    fonts = {e["id"]: e["font_size"] for e in result.adobe["elements"]}
    assert fonts["head"] > before["head"] * 1.5 and fonts["body"] > before["body"] * 1.2
    assert result.score_after > result.score_before
    assert diagnose(source, target, result.adobe)["small_text"] < 2


def test_shapes_keep_containment_and_labels_stay_inside_buttons(cta_source):
    target = Canvas(width=500, height=250)
    pool = build_candidate_pool(cta_source, target)
    for record in pool.candidates:
        if record.validation.is_valid:
            placed = boxes(record.to_adobe())
            if record.variant != "proportional" or record.parent_candidate_id:
                assert inside(placed["cta_label"], placed["cta_bg"], tol=1e-3), record.family


def test_panel_containment_is_restored():
    source = make([shape("panel", 50, 50, 600, 300, z_index=1),
                   text("t", 100, 100, 300, 50, "Inside the panel", 30, z_index=2),
                   text("u", 100, 200, 300, 50, "Also inside the panel", 30, z_index=2)])
    target = Canvas(width=1000, height=1000)
    broken = candidate_of(source, target, {"panel": dict(width=250, height=120)})
    search = _Search(source, target, broken, scorer(source, target), validate_candidate,
                     fit_candidate_text, None, RefinementConfig())
    assert search.broken_containers(search.start) == ["panel"]
    result = refine(source, target, broken)
    placed = boxes(result.adobe)
    assert inside(placed["t"], placed["panel"]) and inside(placed["u"], placed["panel"])
    assert result.score_after > result.score_before


def test_intentional_overlap_survives_refinement():
    source = make([
        dict(id="photo", type="image", x=300, y=300, width=300, height=200, asset_id="a", z_index=1),
        shape("badge", 500, 280, 140, 60, opacity=0.6, z_index=2),
        text("far", 50, 800, 300, 40, "Far away caption", 24, z_index=1),
    ])
    target = Canvas(width=1000, height=1000)
    candidate = candidate_of(source, target, {"far": dict(x=420, y=330, width=200, height=40)})
    overlap0 = _overlap(boxes(candidate)["photo"], boxes(candidate)["badge"])
    result = refine(source, target, candidate)
    placed = boxes(result.adobe)
    assert _overlap(placed["photo"], placed["badge"]) == pytest.approx(overlap0, rel=1e-6)
    assert (placed["badge"][0] - placed["photo"][0], placed["badge"][1] - placed["photo"][1]) == pytest.approx(
        (boxes(candidate)["badge"][0] - boxes(candidate)["photo"][0],
         boxes(candidate)["badge"][1] - boxes(candidate)["photo"][1]))


def _overlap(a, b):
    w = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    h = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


def test_image_aspect_ratio_and_circle_proportions_are_preserved():
    source = make([
        dict(id="photo", type="image", x=100, y=100, width=400, height=200, asset_id="a", z_index=1),
        shape("dot", 600, 600, 100, 100, shape_type="ellipse", z_index=1),
        text("t", 100, 400, 300, 40, "Caption text", 24, z_index=2),
    ])
    target = Canvas(width=728, height=300)
    pool = build_candidate_pool(source, target)
    for record in pool.candidates:
        if not record.validation.is_valid:
            continue
        placed = boxes(record.to_adobe())
        assert placed["photo"][2] / placed["photo"][3] == pytest.approx(2.0, rel=0.05)
        assert placed["dot"][2] == pytest.approx(placed["dot"][3])


# --- optimization --------------------------------------------------------------------------
def test_refinement_never_lowers_the_score_and_matches_the_scorer(cta_source):
    target = Canvas(width=728, height=90)
    for record in build_candidate_pool(cta_source, target, PoolConfig(max_refine_bases=0)).candidates[:6]:
        if not record.validation.is_valid:
            continue
        result = refine(cta_source, target, record.to_adobe())
        assert result.score_after >= result.score_before
        assert result.score_after == pytest.approx(scorer(cta_source, target)(result.adobe))
        assert validate_candidate(cta_source, result.adobe, target).is_valid


def test_invalid_proposals_are_rejected(cta_source):
    target = Canvas(width=1000, height=1000)
    crowded = candidate_of(cta_source, target, {"cta_bg": dict(y=141), "cta_label": dict(y=156)})

    def only_start_is_valid(source, raw, tgt):
        ok = json.dumps(raw, sort_keys=True) == json.dumps(crowded, sort_keys=True)
        return ValidationReport(ok, ok, ok, () if ok else ("rejected by test",))

    result = refine(cta_source, target, crowded, validate=only_start_is_valid) if False else \
        refine_candidate(cta_source, target, crowded, score=scorer(cta_source, target),
                         validate=only_start_is_valid, fit_text=fit_candidate_text)
    assert not result.changed and result.adobe == crowded
    assert result.rejected_invalid > 0 and result.accepted == 0


def test_original_is_returned_when_nothing_improves(cta_source):
    target = Canvas(width=1000, height=1000)
    candidate = candidate_of(cta_source, target)
    result = refine_candidate(cta_source, target, candidate, score=lambda raw: 0.5,
                              validate=validate_candidate, fit_text=fit_candidate_text)
    assert not result.changed and result.adobe == candidate and result.operators == ()
    assert result.score_before == result.score_after == 0.5 and result.stopped == "converged"


def test_search_terminates_under_its_budget(cta_source):
    target = Canvas(width=1000, height=1000)
    crowded = candidate_of(cta_source, target, {"cta_bg": dict(y=141), "cta_label": dict(y=156)})
    for budget in (0, 3, 9):
        result = refine(cta_source, target, crowded, config=RefinementConfig(max_evaluations=budget))
        assert result.evaluations <= budget
    capped = refine(cta_source, target, crowded, config=RefinementConfig(max_rounds=1))
    assert capped.accepted <= 1


def test_refinement_is_deterministic(cta_source):
    target = Canvas(width=728, height=90)
    base = build_candidate_pool(cta_source, target, PoolConfig(max_refine_bases=0)).candidates[0].to_adobe()
    first, second = refine(cta_source, target, base), refine(cta_source, target, base)
    assert first == second and first.adobe == second.adobe


def test_no_state_is_scored_twice(cta_source):
    target = Canvas(width=1000, height=1000)
    crowded = candidate_of(cta_source, target, {"cta_bg": dict(y=141), "cta_label": dict(y=156)})
    seen, real = [], scorer(cta_source, target)

    def recording(raw):
        seen.append(identity(raw))
        return real(raw)

    result = refine_candidate(cta_source, target, crowded, score=recording, validate=validate_candidate,
                              fit_text=fit_candidate_text)
    # every layout is scored at most once: no oscillation, no rework (the start is scored first)
    assert len(seen) == len(set(seen)) and result.evaluations > 0


def test_a_failing_proposal_does_not_discard_successful_ones(cta_source):
    target = Canvas(width=1000, height=1000)
    crowded = candidate_of(cta_source, target, {"cta_bg": dict(y=141), "cta_label": dict(y=156)})
    real, calls = scorer(cta_source, target), []

    def flaky(raw):
        calls.append(1)
        if len(calls) % 4 == 3:
            raise RuntimeError("scorer blew up on this proposal")
        return real(raw)

    result = refine_candidate(cta_source, target, crowded, score=flaky, validate=validate_candidate,
                              fit_text=fit_candidate_text)
    assert result.errors > 0 and result.changed and result.score_after > result.score_before


def test_score_ties_do_not_count_as_improvement_and_nonfinite_scores_are_rejected(cta_source):
    target = Canvas(width=1000, height=1000)
    candidate = candidate_of(cta_source, target)
    tied = refine_candidate(cta_source, target, candidate, score=lambda raw: 0.25,
                            validate=validate_candidate, fit_text=fit_candidate_text)
    assert not tied.changed
    values = iter([0.4] + [float("nan")] * 500)
    nan = refine_candidate(cta_source, target, candidate, score=lambda raw: next(values),
                           validate=validate_candidate, fit_text=fit_candidate_text)
    assert not nan.changed and nan.errors > 0


def test_font_is_not_shrunk_as_a_shortcut(cta_source):
    target = Canvas(width=1000, height=1000)
    crowded = candidate_of(cta_source, target, {"cta_bg": dict(y=141), "cta_label": dict(y=156)})
    start = {e["id"]: e["font_size"] for e in crowded["elements"] if e["type"] == "text"}
    result = refine(cta_source, target, crowded)
    for element in result.adobe["elements"]:
        if element["type"] == "text":
            assert element["font_size"] >= start[element["id"]] * 0.8 - 1e-9


def _search(cta_source, **config):
    from engine.layout_refinement import _Search
    target = Canvas(width=1000, height=1000)
    raw = candidate_of(cta_source, target, {})
    return _Search(cta_source, target, raw, lambda _: 0.0, validate_candidate, None, None,
                   RefinementConfig(**config))


def test_font_shrink_is_penalized_not_rejected(cta_source):
    search = _search(cta_source)
    base = {"cta_label": 20.0}

    def penalty(size):
        return search.shrink_penalty({"cta_label": size}, base)

    assert penalty(20.0) == 0.0 and penalty(30.0) == 0.0
    assert penalty(16.0) == 0.0  # exactly the 20% onset is free
    assert 0.0 < penalty(14.0) < penalty(10.0) < penalty(4.0)  # grows with the shrink
    assert math.isfinite(penalty(0.0))


def test_font_shrink_penalty_is_configurable(cta_source):
    base = {"cta_label": 20.0}
    free = _search(cta_source, font_shrink_weight=0.0).shrink_penalty({"cta_label": 2.0}, base)
    later = _search(cta_source, font_shrink_onset=0.5).shrink_penalty({"cta_label": 12.0}, base)
    assert free == 0.0 and later == 0.0


def test_config_validation():
    for bad in (dict(max_evaluations=-1), dict(step_fractions=()), dict(step_fractions=(0.0,)),
                dict(min_improvement=float("nan")), dict(font_shrink_onset=1.0),
                dict(font_shrink_weight=-1.0)):
        with pytest.raises(ValueError):
            RefinementConfig(**bad)


# --- compliance ----------------------------------------------------------------------------------
def test_refined_layout_preserves_identity_and_protected_fields_and_inputs(cta_source):
    target = Canvas(width=1000, height=1000)
    crowded = candidate_of(cta_source, target, {"cta_bg": dict(y=141), "cta_label": dict(y=156)})
    frozen_candidate, frozen_source = copy.deepcopy(crowded), cta_source.to_adobe()
    result = refine(cta_source, target, crowded)
    assert crowded == frozen_candidate and cta_source.to_adobe() == frozen_source
    assert [e["id"] for e in result.adobe["elements"]] == [e["id"] for e in frozen_source["elements"]]
    assert set(result.adobe) == {"canvas", "elements"} and result.adobe["canvas"] == target.to_adobe()
    for new, old in zip(result.adobe["elements"], frozen_candidate["elements"]):
        changed = {k for k in set(new) | set(old) if new.get(k) != old.get(k)}
        assert changed <= {"x", "y", "width", "height", "font_size"}
    assert validate_candidate(cta_source, result.adobe, target).is_valid


def test_invalid_refinement_cannot_reach_the_pool_or_resize_output(cta_source, monkeypatch):
    target = Canvas(width=728, height=90)

    def tamper(source, tgt, candidate, **kwargs):
        bad = copy.deepcopy(candidate)
        bad["elements"][1]["corner_radius"] = 3          # a protected field
        return RefinementResult(bad, True, 0.1, 0.9, ("tampered",), 1, 0, 0, 0, 0, 1, "converged")

    monkeypatch.setattr(pool_module, "refine_candidate", tamper)
    pool = build_candidate_pool(cta_source, target)
    assert not any(r.parent_candidate_id for r in pool.candidates)
    assert pool.winner is not None and validate_candidate(cta_source, pool.winner.to_adobe(), target).is_valid


def test_core_modules_import_without_torch():
    code = ("import sys; import engine.layout_refinement, engine.candidate_pool, engine.resize;"
            "bad=[m for m in sys.modules if m.split('.')[0] in ('torch','torch_geometric')];"
            "assert not bad, bad")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          cwd=str(Path(__file__).resolve().parents[1]))
    assert done.returncode == 0, done.stderr


# --- integration -------------------------------------------------------------------------------------
def _pool_with_refinement():
    source = make([
        text("head", 100, 100, 700, 80, "Big summer sale", 56, z_index=2),
        text("sub", 100, 200, 700, 40, "Save on every item this week only", 28, z_index=2),
        shape("cta_bg", 100, 300, 300, 70, corner_radius=35, group_id="cta", z_index=3),
        text("cta_label", 120, 315, 260, 40, "Shop now", 28, group_id="cta", z_index=4, text_align="center"),
        shape("rule", 100, 190, 600, 4, z_index=2),
    ])
    return source, Canvas(width=728, height=90)


def test_refined_candidates_enter_the_pool_with_provenance():
    source, target = _pool_with_refinement()
    pool = build_candidate_pool(source, target)
    refined = [r for r in pool.candidates if r.parent_candidate_id]
    assert refined, "a 728x90 banner has diagnosable problems, so something must be refined"
    by_id = {r.candidate_id: r for r in pool.candidates}
    for record in refined:
        parent = by_id[record.parent_candidate_id]
        assert record.refinement_version == REFINEMENT_VERSION and record.refinement_operators
        assert REFINEMENT_VERSION in record.generator_version
        assert record.score_before_refinement == parent.score and record.score > record.score_before_refinement
        assert record.source_design_id == parent.source_design_id and record.target_id == parent.target_id
        assert record.candidate_id == identity(record.to_adobe()) and record.validation.is_valid
        assert set(record.to_adobe()) == {"canvas", "elements"}
        assert "refinement" not in json.dumps(record.to_adobe())
    assert all(r.refinement_version is None and r.parent_candidate_id is None
               for r in pool.candidates if r not in refined)


def test_winner_is_the_highest_scoring_valid_candidate_and_retention_is_independent():
    source, target = _pool_with_refinement()
    many = build_candidate_pool(source, target, PoolConfig(max_retained_candidates=12))
    few = build_candidate_pool(source, target, PoolConfig(max_retained_candidates=1))
    valid = [r for r in many.candidates if r.validation.is_valid]
    assert many.winner.score == max(r.score for r in valid)
    assert many.winner == few.winner
    assert len(few.retained) == 1 and len(many.retained) > 1
    ids = [r.candidate_id for r in many.retained]
    assert len(ids) == len(set(ids))


def test_refinement_budgets_are_separate_and_bounded():
    source, target = _pool_with_refinement()
    off = build_candidate_pool(source, target, PoolConfig(max_refine_bases=0))
    assert not any(r.parent_candidate_id for r in off.candidates)
    one = build_candidate_pool(source, target, PoolConfig(max_refine_bases=1, max_refined_candidates=1))
    assert sum(1 for r in one.candidates if r.parent_candidate_id) <= 1
    tight = build_candidate_pool(source, target, PoolConfig(max_raw_candidates=16))
    assert len(tight.candidates) <= 16
    # refinement can never starve generation: the generator and stretch counts do not shrink
    # below what the same pool produces with refinement disabled, minus the reserved share
    plain = build_candidate_pool(source, target, PoolConfig(max_refine_bases=0))
    on = build_candidate_pool(source, target)
    assert sum(1 for r in on.candidates if not r.parent_candidate_id) >= len(plain.candidates) - 8
    with pytest.raises(ValueError):
        PoolConfig(max_refine_bases=-1)


def test_pool_with_refinement_is_deterministic():
    source, target = _pool_with_refinement()
    assert build_candidate_pool(source, target) == build_candidate_pool(source, target)


@pytest.mark.parametrize("name", ["chameleon_sample_level1_warmup", "chameleon_sample_level2_core",
                                  "chameleon_sample_level3_advanced"])
@pytest.mark.parametrize("size", [None, (728, 90), (90, 728)])
def test_official_samples_with_refinement(name, size):
    task = json.loads(Path(f"input/{name}.json").read_text(encoding="utf-8"))
    design = Design.from_adobe(task["source"])
    target = Canvas.from_adobe(task["target_canvas"] if size is None else dict(width=size[0], height=size[1]))
    pool = build_candidate_pool(design, target)
    plain = build_candidate_pool(design, target, PoolConfig(max_refine_bases=0))
    ids = [e["id"] for e in task["source"]["elements"]]
    assert pool.winner is not None and not pool.failures
    assert pool.winner.score >= plain.winner.score - 1e-12      # refinement can only add candidates
    for record in pool.candidates:
        if record.parent_candidate_id:
            adobe = record.to_adobe()
            assert [e["id"] for e in adobe["elements"]] == ids
            assert validate_candidate(design, adobe, target).is_valid


@pytest.mark.parametrize("family", sorted(FAMILIES))
@pytest.mark.parametrize("size", [(728, 90), (90, 728), (300, 250)])
def test_generated_sources_with_refinement(family, size):
    design = Design.from_adobe(make_source(family, 42))
    target = Canvas(width=size[0], height=size[1])
    pool = build_candidate_pool(design, target)
    plain = build_candidate_pool(design, target, PoolConfig(max_refine_bases=0))
    ids = [e.id for e in design.elements]
    assert pool.winner is not None and pool.winner.score >= plain.winner.score - 1e-12
    for record in pool.candidates:
        if record.parent_candidate_id:
            assert record.validation.is_valid and record.score > record.score_before_refinement
            assert [e["id"] for e in record.to_adobe()["elements"]] == ids


# --- blank-space filling ---------------------------------------------------------------
def test_fill_proposals_scale_units_up_without_exceeding_source_fonts():
    source = make([text("t", 300, 450, 400, 60, "A headline", 28)])
    target = Canvas(width=1000, height=1000)
    small = candidate_of(source, target, {"t": dict(x=450, y=480, width=100, height=30, font_size=12)})
    search = _Search(source, target, small, scorer(source, target), validate_candidate,
                     fit_candidate_text, None, RefinementConfig())
    proposals = search.fill_proposals(search.start)
    assert proposals and all(p.op == "fill_space" for p in proposals)
    start = search.start.boxes["t"]
    for proposal in proposals:
        box = proposal.boxes["t"]
        assert box[2] > start[2] and box[3] > start[3]
        assert box[0] >= 0 and box[1] >= 0 and box[0] + box[2] <= 1000 and box[1] + box[3] <= 1000
        assert proposal.fonts["t"] <= 28 + 1e-9


def test_blank_space_is_filled_by_refinement():
    source = make([text("t", 300, 450, 400, 60, "A headline", 28)])
    target = Canvas(width=1000, height=1000)
    small = candidate_of(source, target, {"t": dict(x=450, y=480, width=100, height=30, font_size=12)})
    result = refine(source, target, small)
    after, before = boxes(result.adobe)["t"], boxes(small)["t"]
    assert result.changed and result.score_after > result.score_before
    assert after[2] * after[3] > before[2] * before[3]
    assert validate_candidate(source, result.adobe, target).is_valid


def test_fill_never_moves_backdrops_or_breaks_containment():
    source = make([shape("bg", 0, 0, 1000, 1000, z_index=0), shape("panel", 100, 100, 600, 300, z_index=1),
                   text("t", 150, 150, 300, 50, "Inside the panel", 30, z_index=2)])
    target = Canvas(width=1000, height=1000)
    result = refine(source, target, candidate_of(source, target))
    after = boxes(result.adobe)
    assert after["bg"] == (0, 0, 1000, 1000) and after["panel"] == (100, 100, 600, 300)
    assert inside(after["t"], after["panel"])
