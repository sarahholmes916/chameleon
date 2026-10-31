"""The learned-selection seam must fail closed to the deterministic winner and must not need Torch (runs in the core job)."""

import ast
import dataclasses
import importlib
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

import engine.learned_selection as L
import engine.resize as resize_module
from engine.candidate_pool import build_candidate_pool
from engine.internal_schema import Canvas, Design
from engine.resize import resize

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "input" / "chameleon_sample_level2_core.json"


@pytest.fixture(autouse=True)
def isolated_state(monkeypatch):
    monkeypatch.delenv(L.ENV_SWITCH, raising=False)
    monkeypatch.setattr(L, "_loaded", L._UNSET)


def cand(position, score):
    return SimpleNamespace(position=position, score=score)


def fake_state(monkeypatch, scores=None, error=None):
    def score_candidates(design, candidates, target, checkpoint):
        if error:
            raise error
        return scores
    monkeypatch.setattr(L, "_loaded", (SimpleNamespace(score_candidates=score_candidates), object()))


def pool_of(*candidates):
    return SimpleNamespace(retained=tuple(candidates))


def test_module_imports_no_ml_packages_at_top_level():
    tree = ast.parse(Path(L.__file__).read_text(encoding="utf-8"))
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert imported <= {"__future__", "hashlib", "importlib", "logging", "math", "os", "pathlib", "typing"}


def test_highest_model_score_wins_with_deterministic_tie_breaks(monkeypatch):
    pool = pool_of(cand(0, 0.5), cand(1, 0.9), cand(2, 0.1))
    fake_state(monkeypatch, [0.2, 0.7, 0.4])
    assert L.choose_candidate(None, None, pool).position == 1
    fake_state(monkeypatch, [0.7, 0.7, 0.7])                         # exact tie: better deterministic score, then earlier position
    assert L.choose_candidate(None, None, pool).position == 1
    tied = pool_of(cand(3, 0.5), cand(2, 0.5))
    fake_state(monkeypatch, [1.0, 1.0])
    assert L.choose_candidate(None, None, tied).position == 2


@pytest.mark.parametrize("scores", [[1.0], [1.0, math.nan], [1.0, math.inf], []])
def test_bad_scores_fall_back(monkeypatch, scores):
    fake_state(monkeypatch, scores)
    assert L.choose_candidate(None, None, pool_of(cand(0, 1.0), cand(1, 0.5))) is None


def test_scoring_exception_falls_back(monkeypatch):
    fake_state(monkeypatch, error=RuntimeError("boom"))
    assert L.choose_candidate(None, None, pool_of(cand(0, 1.0), cand(1, 0.5))) is None


def test_switch_off_and_single_candidate_never_load_the_model(monkeypatch):
    monkeypatch.setattr(L, "_load", lambda: pytest.fail("must not load"))
    monkeypatch.setenv(L.ENV_SWITCH, "off")
    assert L.choose_candidate(None, None, pool_of(cand(0, 1.0), cand(1, 0.5))) is None
    monkeypatch.delenv(L.ENV_SWITCH)
    assert L.choose_candidate(None, None, pool_of(cand(0, 1.0))) is None


def test_missing_ml_packages_fall_back_once(monkeypatch):
    calls = []

    def fail(name, *args):
        calls.append(name)
        raise ImportError(f"no module named {name}")
    monkeypatch.setattr(importlib, "import_module", fail)
    pool = pool_of(cand(0, 1.0), cand(1, 0.5))
    assert L.choose_candidate(None, None, pool) is None
    assert L.choose_candidate(None, None, pool) is None
    assert calls == ["torch"] and L._loaded is None                 # failure is remembered, not retried per design


def test_checkpoint_hash_mismatch_falls_back(monkeypatch):
    monkeypatch.setattr(L, "CHECKPOINT_SHA256", "0" * 64)
    assert L.choose_candidate(None, None, pool_of(cand(0, 1.0), cand(1, 0.5))) is None
    assert L._loaded is None


def test_missing_checkpoint_file_falls_back(monkeypatch, tmp_path):
    monkeypatch.setattr(L, "CHECKPOINT_PATH", tmp_path / "absent.pt")
    assert L.choose_candidate(None, None, pool_of(cand(0, 1.0), cand(1, 0.5))) is None


def sample_task():
    import json
    task = json.loads(SAMPLE.read_text(encoding="utf-8"))
    return task, Design.from_adobe(task["source"], target_canvas=task["target_canvas"]), Canvas.from_adobe(task["target_canvas"])


def test_resize_uses_the_preferred_validated_candidate(monkeypatch):
    task, design, target = sample_task()
    pool = build_candidate_pool(design, target)
    other = next(c for c in pool.retained if c.candidate_id != pool.winner.candidate_id)
    monkeypatch.setattr(resize_module, "choose_candidate", lambda d, t, p: other)
    assert resize(task["source"], task["target_canvas"]) == other.to_adobe()


def test_resize_keeps_the_winner_when_the_preferred_candidate_is_invalid(monkeypatch):
    task, design, target = sample_task()
    pool = build_candidate_pool(design, target)
    broken = dataclasses.replace(pool.retained[-1], adobe_json='{"canvas": {"width": 1, "height": 1}, "elements": []}')
    monkeypatch.setattr(resize_module, "choose_candidate", lambda d, t, p: broken)
    assert resize(task["source"], task["target_canvas"]) == pool.winner.to_adobe()
