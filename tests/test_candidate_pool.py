"""Pool provenance, failure isolation, packing and heuristic regression contracts."""
import copy
import json
from pathlib import Path

import pytest

from engine import candidate_generator as gen
from engine.candidate_pool import PoolConfig, build_candidate_pool, identity, layout_distance
from engine.candidate_validator import validate_candidate
from engine.internal_schema import Canvas, Design
from engine.layout_objective import OBJECTIVE_VERSION, evaluate_objective
from engine.layout_units import LayoutUnit, build_layout_units


def text(i, x=10, y=10):
    return dict(id=str(i), type="text", x=x, y=y, width=100, height=20,
                content="Sample text", font_family="Solstice Sans", font_size=12, color="rgba(0,0,0,1)")


def design(elements):
    return Design.from_adobe(dict(canvas=dict(width=1000, height=1000), elements=elements))


def test_soft_clustering_positive_and_negative():
    source = design([text(0), text(1, y=35), text(2, y=700)])
    units = build_layout_units(source)
    assert [len(u.members) for u in units] == [2, 1]
    assert all(e.group_id == source.elements_by_id[e.id].group_id for u in units for e in u.members)
    overlay = design([text(0), text(1)])
    assert len(build_layout_units(overlay)) == 2
    unaligned = design([text(0), text(1, x=200, y=35)])
    assert len(build_layout_units(unaligned)) == 2


@pytest.mark.parametrize("count", [1, 5, 10, 20, 25, 50])
@pytest.mark.parametrize("target", [Canvas(width=728, height=90), Canvas(width=1, height=1000), Canvas(width=10000, height=1)])
def test_zone_dimensions_positive_and_inside(count, target):
    source = design([text(i, y=i * 25) for i in range(count)])
    units = [LayoutUnit.of((e,), (i,), source.canvas) for i, e in enumerate(source.elements)]
    slots = gen._zone_rects(units, target)
    assert len(slots) == count
    for slot in slots:
        assert slot.width > 0 and slot.height > 0
        assert slot.x >= 0 and slot.y >= 0
        assert slot.x + slot.width <= target.width + 1e-8
        assert slot.y + slot.height <= target.height + 1e-8


def test_dense_pack_reduces_gaps_without_collapsing_content():
    source = design([text(i, y=i * 30) for i in range(50)])
    units = [LayoutUnit.of((e,), (i,), source.canvas) for i, e in enumerate(source.elements)]
    rows = gen._rows_for_units(units, set())
    slots, scales = gen._pack_container(rows, gen.Slot(0, 0, 100, 10),
                                       {e.id: 12 for e in source.elements}, {}, Canvas(width=728, height=90))
    assert all(s.height > 0 for s in slots)
    assert all(scale > 0 for scale in scales)
    assert slots[-1].y + slots[-1].height <= 10 + 1e-8


def test_pool_identity_dedupe_diversity_and_snapshot():
    source, target = design([text(0)]), Canvas(width=1000, height=1000)
    a = source.to_adobe()
    b = copy.deepcopy(a)
    b["elements"][0]["x"] = 500
    pool = build_candidate_pool(source, target, families=[("A", lambda *_: [a, a]), ("B", lambda *_: [b])])
    assert len(pool.candidates) == 3 and pool.exact_duplicates == 1
    assert len(pool.retained) == 2 and {r.family for r in pool.retained} == {"A", "B"}
    assert pool.candidates[0].candidate_id == identity(a)
    assert all(r.objective_version == OBJECTIVE_VERSION for r in pool.candidates)
    assert layout_distance(*pool.retained, target) > 0
    a["elements"][0]["x"] = -100
    assert pool.candidates[0].to_adobe()["elements"][0]["x"] == 10
    assert pool.winner.score == max(r.score for r in pool.candidates)


def test_failure_isolation_including_partial_iterators_and_invalid_candidates(caplog):
    source, target = design([text(0)]), Canvas(width=1000, height=1000)
    def broken(*_):
        yield source.to_adobe()
        yield {}
        raise RuntimeError("family failed mid-stream")
    pool = build_candidate_pool(source, target, families=[("broken", broken), ("good", lambda *_: [source.to_adobe()])])
    assert len(pool.candidates) == 3
    assert pool.winner and all(r.validation.is_valid for r in pool.retained)
    assert any("broken[2]" in message for message in pool.failures)
    assert "family failed mid-stream" in caplog.text
    assert build_candidate_pool(source, target, families=[]).winner is None


def test_typography_only_alternatives_are_not_exact_duplicates():
    source, target = design([text(0)]), Canvas(width=1000, height=1000)
    a, b = source.to_adobe(), source.to_adobe()
    b["elements"][0]["font_size"] = 6
    pool = build_candidate_pool(source, target, PoolConfig(max_refine_bases=0), families=[("A", lambda *_: [a, b])])
    assert len(pool.retained) == 2


@pytest.mark.parametrize("path", sorted(Path("input").glob("*.json")))
def test_sample_pool_budget_determinism_and_valid_output(path):
    task = json.loads(path.read_text(encoding="utf-8"))
    source, target = Design.from_adobe(task["source"]), Canvas.from_adobe(task["target_canvas"])
    config = PoolConfig(max_raw_candidates=15, max_retained_candidates=4)
    a, b = build_candidate_pool(source, target, config), build_candidate_pool(source, target, config)
    assert a == b and len(a.candidates) <= 15 and len(a.retained) <= 4
    assert a.winner and validate_candidate(source, a.winner.to_adobe(), target).is_valid
    for r in a.retained:
        assert r.validation.is_valid and set(r.to_adobe()) == {"canvas", "elements"}


def test_text_occluded_by_large_image_has_full_overlap_penalty():
    source = design([text(0), dict(id="image", type="image", x=500, y=0, width=400, height=400, asset_id="existing")])
    raw = source.to_adobe()
    raw["elements"][1]["x"] = 0
    score = evaluate_objective(source, Design.from_adobe(raw), source.canvas)
    assert score.terms.unwanted_overlap == 1


@pytest.mark.parametrize("family", [gen.two_column_candidates, gen.controlled_overlay_candidates])
@pytest.mark.parametrize("kind", ["text", "images", "background", "rotation", "zero"])
def test_new_families_preserve_properties_and_handle_edge_cases(family, kind):
    if kind == "images":
        elements = [dict(id=str(i), type="image", x=i*100, y=30, width=40, height=80, asset_id="bundled.png") for i in range(8)]
    elif kind == "background":
        elements = [dict(id="bg", type="shape", x=0, y=0, width=1000, height=1000, shape_type="rectangle", fill_color="red")]
    else:
        elements = [text(0), text(1, x=40, y=20)]
        if kind == "rotation":
            elements.append(dict(id="shape", type="shape", x=100, y=100, width=100, height=30, rotation=35,
                                 shape_type="rectangle", fill_color="red", stroke_color="black", stroke_width=4))
        if kind == "zero":
            elements[1]["width"] = 0
    source, target = design(elements), Canvas(width=728, height=90)
    before = source.to_adobe()
    outputs = family(source, target)
    assert outputs and source.to_adobe() == before
    assert all(validate_candidate(source, output, target).is_valid for output in outputs)


def test_word_fit_handles_long_unbreakable_words_without_rewriting():
    from engine.layout_typography import estimated_fit, fit_candidate_text
    raw = design([dict(text(0), content="SUPERCALIFRAGILISTIC", width=75, height=35, font_size=24)]).to_adobe()
    result = fit_candidate_text(raw)
    assert result["elements"][0]["font_size"] < 24
    assert estimated_fit(result["elements"][0])["fits"]
    assert raw["elements"][0]["font_size"] == 24
    assert validate_candidate(Design.from_adobe(raw), result, Canvas(width=1000, height=1000)).is_valid


def test_injected_measurement_records_distinct_provenance():
    source = design([text(0)])
    def measure(e):
        return dict(fits=e["font_size"] <= 8, lines=1)
    pool = build_candidate_pool(source, source.canvas, families=[("A", lambda *_: [source.to_adobe()])], text_measurement=measure)
    assert pool.winner.to_adobe()["elements"][0]["font_size"] <= 8
    assert pool.winner.generator_version.endswith("+measured_text")


def test_near_duplicates_and_family_balancing_do_not_change_winner():
    source = design([text(0)])
    variants = []
    for x in (10, 10.01, 300, 600):
        raw = source.to_adobe()
        raw["elements"][0]["x"] = x
        variants.append(raw)
    pool = build_candidate_pool(source, source.canvas, PoolConfig(max_retained_candidates=2),
                                families=[("A", lambda *_: variants[:3]), ("B", lambda *_: variants[3:])])
    assert pool.near_duplicates == 1 and pool.exact_duplicates == 0
    assert {r.family for r in pool.retained} == {"A", "B"}
    assert pool.winner.score == max(r.score for r in pool.candidates)


def test_score_failure_keeps_valid_candidates(monkeypatch):
    import engine.candidate_pool as module
    source = design([text(0)])
    def broken(*_):
        raise RuntimeError("scorer failed")
    monkeypatch.setattr(module, "score_deterministic", broken)
    pool = build_candidate_pool(source, source.canvas, families=[("A", lambda *_: [source.to_adobe()])])
    assert pool.winner.validation.is_valid and pool.winner.score is None
    assert "scoring failed" in pool.failures[0]


@pytest.mark.parametrize("kwargs", [dict(max_raw_candidates=0), dict(max_retained_candidates=-1),
                                   dict(near_duplicate_distance=float("nan")), dict(seed=True)])
def test_invalid_configuration_fails_before_generation(kwargs):
    with pytest.raises(ValueError):
        PoolConfig(**kwargs)
