"""Real renderer and benchmark failure contracts; run in the sources CI environment."""
import json
from pathlib import Path

import pytest

from annotation.candidates import check_record
from engine.candidate_pool import PoolConfig, build_candidate_pool
from engine.internal_schema import Canvas, Design
from engine.layout_typography import estimated_fit
from scripts.evaluate_s04 import evaluate_task, render_checked
from scripts.source_dataset import SourceRenderer


@pytest.mark.parametrize("path", sorted(Path("input").glob("*.json")))
def test_real_renderer_s04_winner(path, tmp_path):
    task = json.loads(path.read_text(encoding="utf-8"))
    report = evaluate_task(task, tmp_path / "run", SourceRenderer(), PoolConfig(max_raw_candidates=12), measured_fit=True)
    assert report["winner"] and not report["render_failures"]
    assert (tmp_path / "run/candidate_000.png").is_file()
    assert (tmp_path / "run/candidate_manifest.jsonl").stat().st_size > 0


@pytest.mark.parametrize("mode", ["missing", "exception"])
def test_missing_or_failed_render_is_not_success(mode, tmp_path):
    class Broken:
        def render(self, *_):
            if mode == "exception":
                raise RuntimeError("renderer failed")
            return []
    with pytest.raises((OSError, RuntimeError)):
        render_checked(Broken(), {"canvas": {"width": 20, "height": 20}}, tmp_path / "missing.png")


@pytest.fixture(scope="module")
def renderer():
    return SourceRenderer()


def sample_tasks():
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path("input").glob("*.json"))]


def test_word_fit_estimate_tracks_actual_rendered_text(renderer):
    """The production (renderer-free) estimate must not under-measure real rendered text."""
    width_ratios, height_ratios = [], []
    for task in sample_tasks():
        for element in task["source"]["elements"]:
            if element["type"] != "text":
                continue
            for size in (10, 14, 20, 32, 48):
                probe = dict(element, font_size=size, width=1e6, height=1e6, content=element["content"].split(chr(10))[0])
                real, estimate = renderer.text_measurement(probe), estimated_fit(probe)
                if real["measured_width"] > 0:
                    width_ratios.append(estimate["measured_width"] / real["measured_width"])
                height_ratios.append(estimate["measured_height"] / real["measured_height"])
    assert len(width_ratios) > 30
    # underestimates beyond 3% would reintroduce clipping the calibration removed; gross
    # overestimates (>1.4x on average) would waste room and hide readable sizes
    assert min(width_ratios) >= 0.97 and min(height_ratios) >= 0.99
    assert sum(width_ratios) / len(width_ratios) < 1.4 and sum(height_ratios) / len(height_ratios) < 1.3


@pytest.mark.parametrize("measured", [False, True])
@pytest.mark.parametrize("size", [None, (728, 90), (90, 728)])
def test_winner_text_fits_when_actually_rendered(renderer, measured, size):
    """Production word-fit and the injected measured path: every pool-winner text fits when measured
    by the real renderer, unless it is at the schema floor: the renderer truncates sizes to whole pixels, so below 2px it draws
        1px text whose hinted advances no box estimate can predict."""
    misses = []
    for task in sample_tasks():
        source = Design.from_adobe(task["source"])
        target = Canvas.from_adobe(task["target_canvas"] if size is None else dict(width=size[0], height=size[1]))
        pool = build_candidate_pool(source, target, text_measurement=renderer.text_measurement if measured else None)
        for element in pool.winner.to_adobe()["elements"]:
            if element["type"] == "text" and not renderer.text_measurement(element)["fits"] and element["font_size"] >= 2.0:
                misses.append((target.to_adobe(), element["id"], element["font_size"]))
    assert not misses, misses


def test_refinement_with_renderer_measurement_keeps_text_fitting(renderer):
    source = Design.from_adobe(sample_tasks()[1]["source"])
    target = Canvas(width=728, height=90)
    pool = build_candidate_pool(source, target, text_measurement=renderer.text_measurement)
    refined = [r for r in pool.candidates if r.parent_candidate_id]
    assert refined, "the 728x90 core sample has tiny text, so refinement must act"
    for record in refined:
        assert record.validation.is_valid
        for element in record.to_adobe()["elements"]:
            if element["type"] == "text":
                assert renderer.text_measurement(element)["fits"] or element["font_size"] <= 1.0


def test_manifest_rows_with_refinement_provenance_import(tmp_path, renderer):
    task = sample_tasks()[1]
    task = dict(task, target_canvas=dict(width=728, height=90))
    evaluate_task(task, tmp_path / "run", renderer, PoolConfig(max_raw_candidates=24), render_pool=True, measured_fit=True)
    rows = [json.loads(line) for line in (tmp_path / "run/candidate_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows and all(not check_record(row) for row in rows)
    assert all("refinement_version" in row["provenance"] and "parent_candidate_id" in row["provenance"] for row in rows)
    assert all("score" not in json.dumps(row) for row in rows)
