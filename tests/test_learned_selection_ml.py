"""The shipped checkpoint through the real engine path (CPU).  Software contracts only, not evidence of better designs."""

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

import engine.learned_selection as L  # noqa: E402
from engine.candidate_pool import build_candidate_pool  # noqa: E402
from engine.candidate_validator import validate_candidate  # noqa: E402
from engine.internal_schema import Canvas, Design  # noqa: E402
from engine.resize import resize  # noqa: E402

INPUT = Path(__file__).resolve().parents[1] / "input"
SAMPLES = sorted(p.name for p in INPUT.glob("chameleon_sample_*.json"))


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    monkeypatch.delenv(L.ENV_SWITCH, raising=False)
    monkeypatch.setattr(L, "_loaded", L._UNSET)


def task(name):
    value = json.loads((INPUT / name).read_text(encoding="utf-8"))
    return value, Design.from_adobe(value["source"], target_canvas=value["target_canvas"]), Canvas.from_adobe(value["target_canvas"])


def test_shipped_checkpoint_loads_and_is_pinned():
    state = L._load()
    assert state is not None
    _, checkpoint = state
    assert checkpoint.training_metadata["status"] == "trained"


@pytest.mark.parametrize("name", SAMPLES)
def test_preferred_candidate_is_a_validated_pool_member_and_output_is_deterministic(name):
    value, design, target = task(name)
    pool = build_candidate_pool(design, target)
    chosen = L.choose_candidate(design, target, pool)
    assert chosen is not None and chosen.candidate_id in {c.candidate_id for c in pool.retained}
    assert chosen.validation.is_valid
    first = resize(value["source"], value["target_canvas"])
    assert first == resize(value["source"], value["target_canvas"])
    assert validate_candidate(design, first, target).is_valid


@pytest.mark.parametrize("name", SAMPLES)
def test_switch_off_restores_the_deterministic_winner(name, monkeypatch):
    value, design, target = task(name)
    monkeypatch.setenv(L.ENV_SWITCH, "off")
    assert resize(value["source"], value["target_canvas"]) == build_candidate_pool(design, target).winner.to_adobe()


def test_tampered_checkpoint_is_rejected(tmp_path, monkeypatch):
    copy = tmp_path / "ranker.pt"
    shutil.copy(L.CHECKPOINT_PATH, copy)
    with open(copy, "ab") as handle:
        handle.write(b"x")
    monkeypatch.setattr(L, "CHECKPOINT_PATH", copy)
    value, design, target = task(SAMPLES[0])
    assert L.choose_candidate(design, target, build_candidate_pool(design, target)) is None
    assert L._loaded is None
