"""Tests for the S06 resize boundary (batch selection + legacy fallback)."""

import ast
import copy
import json
from pathlib import Path

import pytest

import engine.resize as resize_module
from engine.candidate_validator import validate_candidate
from engine.internal_schema import Canvas, Design
from engine.layout_search import baseline_proposal, find_valid_layout
from engine.resize import _best_index, resize

INPUT_DIR = Path(__file__).resolve().parent.parent / "input"
MODULE_PATH = Path(__file__).resolve().parent.parent / "engine" / "resize.py"
SAMPLES = [
    "chameleon_sample_level1_warmup.json",
    "chameleon_sample_level2_core.json",
    "chameleon_sample_level3_advanced.json",
]
FAMILY_FUNCTIONS = (
    "baseline_candidate",
    "horizontal_flow_candidates",
    "vertical_stack_candidates",
    "hero_left_candidates",
    "hero_right_candidates",
    "banner_zones_candidates", "two_column_candidates", "controlled_overlay_candidates",
)


def load_task(name):
    task = json.loads((INPUT_DIR / name).read_text(encoding="utf-8"))
    return (
        task,
        Design.from_adobe(task["source"]),
        Canvas.from_adobe(task["target_canvas"]),
    )


def legacy_output(source, target):
    design = Design.from_adobe(source, target_canvas=target)
    canvas = Canvas.from_adobe(target)
    return find_valid_layout(
        design, canvas,
        lambda attempt, previous: baseline_proposal(design, canvas, attempt),
    )


def break_batch(monkeypatch, replacement):
    for name in FAMILY_FUNCTIONS:
        monkeypatch.setattr(resize_module, name, replacement)


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
        "logging", "collections.abc", "typing",
        "engine.candidate_generator", "engine.candidate_validator",
        "engine.internal_schema", "engine.layout_objective",
        "engine.layout_search", "engine.candidate_pool",
    }


def test_python_310_syntax():
    for path in (MODULE_PATH, Path(__file__).resolve()):
        ast.parse(path.read_text(encoding="utf-8"), feature_version=(3, 10))


@pytest.mark.parametrize("name", SAMPLES)
def test_resize_returns_valid_design(name):
    task, source, target = load_task(name)
    output = resize(task["source"], task["target_canvas"])
    assert set(output.keys()) == {"canvas", "elements"}
    assert output["canvas"] == {"width": target.width, "height": target.height}
    assert tuple(el["id"] for el in output["elements"]) == tuple(
        el.id for el in source.elements)
    assert validate_candidate(source, output, target).is_valid


def test_core_background_fills_canvas():
    task, _, target = load_task("chameleon_sample_level2_core.json")
    output = resize(task["source"], task["target_canvas"])
    background = next(el for el in output["elements"] if el["id"] == "bg")
    assert (background["x"], background["y"]) == (0.0, 0.0)
    assert (background["width"], background["height"]) == (1600.0, 400.0)


def test_winner_has_best_valid_score():
    from engine.candidate_pool import build_candidate_pool
    from engine.layout_objective import score_deterministic
    task, source, target = load_task("chameleon_sample_level2_core.json")
    output = resize(task["source"], task["target_canvas"])
    pool = build_candidate_pool(source, target)
    score = score_deterministic(source, [Design.from_adobe(output)], target)[0]
    assert score == max(r.score for r in pool.candidates if r.score is not None)
    assert validate_candidate(source, output, target).is_valid


def test_zones_skipped_below_aspect_cutoff(monkeypatch):
    task, _, _ = load_task("chameleon_sample_level1_warmup.json")
    calls = []
    monkeypatch.setattr(
        resize_module, "banner_zones_candidates",
        lambda *args, **kwargs: calls.append(args) or [])
    output = resize(task["source"], task["target_canvas"])
    assert calls == []
    assert set(output.keys()) == {"canvas", "elements"}


def test_zones_generated_above_aspect_cutoff(monkeypatch):
    task, _, _ = load_task("chameleon_sample_level2_core.json")
    calls = []
    real = resize_module.banner_zones_candidates
    monkeypatch.setattr(
        resize_module, "banner_zones_candidates",
        lambda *args, **kwargs: calls.append(args) or real(*args, **kwargs))
    resize(task["source"], task["target_canvas"])
    assert len(calls) == 1


def test_best_index_prefers_family_rank_on_exact_ties():
    assert _best_index([0.5, 0.5], [1, 0]) == 1
    assert _best_index([0.7, 0.7, 0.7], [2, 1, 0]) == 2


def test_best_index_prefers_earlier_position_within_family():
    assert _best_index([0.4, 0.4], [1, 1]) == 0


def test_best_index_score_beats_rank_and_position():
    assert _best_index([0.4, 0.9, 0.5], [0, 2, 1]) == 1


def _raise(*args, **kwargs):
    raise RuntimeError("synthetic generator failure")


@pytest.mark.parametrize("name", SAMPLES)
def test_fallback_when_generator_raises(monkeypatch, name):
    task, _, _ = load_task(name)
    break_batch(monkeypatch, _raise)
    output = resize(task["source"], task["target_canvas"])
    assert output == legacy_output(task["source"], task["target_canvas"])


@pytest.mark.parametrize("name", SAMPLES)
def test_other_families_survive_one_family_failure(monkeypatch, name):
    task, source, target = load_task(name)
    monkeypatch.setattr(resize_module, "horizontal_flow_candidates", _raise)
    monkeypatch.setattr(resize_module, "find_valid_layout", lambda *_a, **_k: pytest.fail("fallback discarded valid candidates"))
    output = resize(task["source"], task["target_canvas"])
    assert validate_candidate(source, output, target).is_valid


@pytest.mark.parametrize("name", SAMPLES)
def test_fallback_when_none_validate(monkeypatch, name):
    task, _, _ = load_task(name)
    invalid = [{"canvas": dict(task["target_canvas"]), "elements": []}]
    break_batch(monkeypatch, lambda *_: [dict(c) for c in invalid])
    output = resize(task["source"], task["target_canvas"])
    assert output == legacy_output(task["source"], task["target_canvas"])


@pytest.mark.parametrize("name", SAMPLES)
def test_fallback_when_batch_empty(monkeypatch, name):
    task, _, _ = load_task(name)
    break_batch(monkeypatch, lambda *_: [])
    output = resize(task["source"], task["target_canvas"])
    assert output == legacy_output(task["source"], task["target_canvas"])


@pytest.mark.parametrize("name", SAMPLES)
def test_resize_is_deterministic(name):
    task, _, _ = load_task(name)
    first = resize(task["source"], task["target_canvas"])
    second = resize(task["source"], task["target_canvas"])
    assert first == second


@pytest.mark.parametrize("name", SAMPLES)
def test_resize_does_not_mutate_inputs(name):
    task, _, _ = load_task(name)
    before = copy.deepcopy(task)
    resize(task["source"], task["target_canvas"])
    assert task == before
