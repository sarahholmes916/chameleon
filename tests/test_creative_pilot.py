"""Creative pilot: sources are schema-valid, restructured and degraded candidates keep every protected field."""

import json
import random

from creative_pilot import degrade, families, metrics, operators as ops, splits
from creative_pilot.blocks import analyse
from creative_pilot.fitting import Measurer
from creative_pilot.layout import Fitter
from engine.candidate_validator import validate_candidate
from engine.internal_schema import Canvas, Design
from scripts import source_dataset as sd


class StubRenderer:
    """Deterministic stand-in for SourceRenderer.text_measurement (wrap by character count)."""

    def text_measurement(self, e):
        size, width = e["font_size"], e["width"]
        per_line = max(int((width - 2) / (.55 * size)), 1)
        words, lines, cur = e["content"].replace("\n", " ").split(" "), 1, 0
        for w in words:
            if cur and cur + 1 + len(w) > per_line:
                lines, cur = lines + 1, len(w)
            else:
                cur += len(w) + (1 if cur else 0)
        mw = min(max(len(w) for w in words), per_line if lines > 1 else cur) * .55 * size
        mh = size * 1.4 + (lines - 1) * size * e.get("line_height", 1.2)
        return dict(element_id=e["id"], lines=lines, measured_width=mw, measured_height=mh,
                    fits=mw <= width - 2 and mh <= e["height"] - 2)


def sample_source(name="stat_dashboard", seed=5):
    return families.make_creative_source(name, seed, 0)[0]


def test_every_family_is_schema_valid_and_has_a_difficulty():
    assert len(families.FAMILIES) == 22
    for name in families.FAMILIES:
        src = sample_source(name)
        report = sd.validate_source(src)
        assert report["schema_ok"], (name, report["issues"][:2])
        assert families.FAMILIES[name]["difficulty"] in (1, 2, 3)


def test_family_variants_are_deterministic():
    assert sample_source(seed=9) == sample_source(seed=9) and sample_source(seed=9) != sample_source(seed=10)


def test_restructured_candidates_are_valid_and_keep_protected_fields():
    meas = Measurer(StubRenderer())
    made = 0
    for name in ("stat_dashboard", "poster_split_diagonal", "price_tag_product"):
        src = sample_source(name)
        design, ctx = Design.from_adobe(src), analyse(src)
        for T in ((1600, 400), (728, 90), (90, 728)):
            for recipe in (ops.WIDE_RECIPES if T[0] > T[1] else ops.TALL_RECIPES):
                rng = random.Random(f"{name}{T}{recipe}")
                spec = ops.build_spec(ctx, T, recipe, ops.sample_params(recipe, T[0] > T[1], rng), rng)
                plan = ops.realize(ctx, T, Fitter(meas, ctx), spec, rng) if spec else None
                if not plan:
                    continue
                cand = metrics.assemble(src, plan.placements, T, meas)
                report = validate_candidate(design, cand, Canvas.from_adobe(dict(width=T[0], height=T[1])))
                assert report.is_valid, (name, T, recipe, report.violations[:2])
                assert [e["id"] for e in cand["elements"]] == [e["id"] for e in src["elements"]]
                assert set(cand) == {"canvas", "elements"}
                made += 1
    assert made > 20


def test_degradations_stay_valid_and_change_only_geometry_and_font():
    meas = Measurer(StubRenderer())
    src = sample_source("price_tag_product")
    design, ctx, T = Design.from_adobe(src), analyse(src), (1600, 400)
    rng = random.Random(1)
    spec = ops.build_spec(ctx, T, "cols", ops.sample_params("cols", True, rng), rng)
    parent = metrics.assemble(src, ops.realize(ctx, T, Fitter(meas, ctx), spec, rng).placements, T, meas)
    applied = 0
    for op in degrade.OPERATORS:
        child = degrade.degrade(parent, ctx, op, .8, "t", meas)
        if child is None:
            continue
        applied += 1
        assert validate_candidate(design, child, Canvas.from_adobe(dict(width=T[0], height=T[1]))).is_valid, op
        for a, b in zip(parent["elements"], child["elements"]):
            assert {k: v for k, v in a.items() if k not in ("x", "y", "width", "height", "font_size")} == \
                   {k: v for k, v in b.items() if k not in ("x", "y", "width", "height", "font_size")}
    assert applied >= 6


def test_structure_distance_ignores_tiny_perturbation_but_sees_real_restructuring():
    meas = Measurer(StubRenderer())
    src = sample_source("stat_dashboard")
    ctx, T = analyse(src), (1600, 400)
    cands = []
    for recipe in ("cols", "two_row", "cta_forward"):
        rng = random.Random(recipe)
        spec = ops.build_spec(ctx, T, recipe, ops.sample_params(recipe, True, rng), rng)
        plan = ops.realize(ctx, T, Fitter(meas, ctx), spec, rng) if spec else None
        if plan:
            cands.append(metrics.assemble(src, plan.placements, T, meas))
    d0 = metrics.descriptor(cands[0], src)
    nudged = json.loads(json.dumps(cands[0]))
    for e in nudged["elements"]:
        e["x"] = round(e["x"] + .2, 3) if e["x"] + e["width"] + .2 < T[0] else e["x"]
    assert metrics.structure_distance(d0, metrics.descriptor(nudged, src)) < .01
    assert len(cands) >= 2 and metrics.structure_distance(d0, metrics.descriptor(cands[1], src)) > .045


def test_splits_are_family_disjoint_and_deterministic():
    a = splits.assign_families()
    assert a == splits.assign_families() and set(a) == set(families.FAMILIES)
    assert sorted(set(a.values())) == ["test", "train", "val"]
    by_level = {}
    for fam, split in a.items():
        by_level.setdefault(families.FAMILIES[fam]["difficulty"], set()).add(split)
    assert all(s == {"train", "val", "test"} for s in by_level.values())  # every difficulty is represented in every split
