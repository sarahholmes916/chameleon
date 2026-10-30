"""Real renderer and benchmark failure contracts; run in the sources CI environment."""
import json
from pathlib import Path

import pytest

from engine.candidate_pool import PoolConfig
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
