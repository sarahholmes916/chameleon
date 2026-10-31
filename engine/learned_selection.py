"""Optional learned choice among the candidate pool's already-validated, distinct layouts.

This is the only place the submission engine touches the GATv2 ranker. It never generates, edits, validates or serializes a layout:
it only returns which retained candidate of a ``CandidatePool`` to prefer. Anything that goes wrong (the ML packages or checkpoint
are missing, the checkpoint hash or version is wrong, a graph cannot be built, scoring raises, scores are not finite) returns
``None`` and the caller keeps the deterministic winner, so the baseline path never depends on this module working. The ML packages
are imported lazily, so importing this module needs neither Torch nor PyG. Set ``CHAMELEON_RANKER=off`` to disable it.
"""

from __future__ import annotations

import hashlib
import importlib
import logging
import math
import os
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)

CHECKPOINT_PATH = Path(__file__).resolve().parent / "checkpoints" / "ranker_synth_weight_0.25_best.pt"
# Pinned so a swapped or corrupted file is rejected instead of silently used.
CHECKPOINT_SHA256 = "7e6790f94fb4b99215c3c115027d5e7bd93e248654fe8a871bbf838322787472"
CHECKPOINT_DROPOUT = 0.10  # the training configuration recorded in the checkpoint manifest
ENV_SWITCH = "CHAMELEON_RANKER"

_UNSET = object()
_loaded: Any = _UNSET  # (ranker module, checkpoint) once loaded, None after a failed load (not retried)


def enabled() -> bool:
    return os.environ.get(ENV_SWITCH, "on").strip().lower() not in {"off", "0", "false", "no"}


def _load() -> Any:
    global _loaded
    if _loaded is not _UNSET:
        return _loaded
    try:
        if hashlib.sha256(CHECKPOINT_PATH.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
            raise ValueError("checkpoint file does not match the pinned SHA-256")
        torch = importlib.import_module("torch")
        ranker = importlib.import_module("engine.ranker")
        torch.set_num_threads(1)  # one thread: reproducible, and no oversubscription when scoring runs in parallel
        checkpoint = ranker.load_ranker_checkpoint(
            CHECKPOINT_PATH, expected_config=ranker.RankerConfig(dropout=CHECKPOINT_DROPOUT), device="cpu")
        if checkpoint.training_metadata["status"] != "trained":
            raise ValueError("checkpoint is not marked trained")
        _loaded = (ranker, checkpoint)
    except Exception as exc:  # ImportError, missing/corrupt/incompatible checkpoint, ...
        _logger.warning("learned ranker unavailable (%s: %s); using the deterministic winner", type(exc).__name__, exc)
        _loaded = None
    return _loaded


def choose_candidate(design: Any, target: Any, pool: Any) -> Any | None:
    """The retained pool candidate the ranker prefers, or None to keep the deterministic winner."""
    candidates = list(pool.retained)
    if len(candidates) < 2 or not enabled():
        return None
    state = _load()
    if state is None:
        return None
    ranker, checkpoint = state
    try:
        scores = ranker.score_candidates(design, candidates, target, checkpoint=checkpoint)
        if len(scores) != len(candidates) or not all(math.isfinite(s) for s in scores):
            raise ValueError("ranker did not return one finite score per candidate")
        # Higher model score wins; exact ties fall back to the deterministic score, then pool order.
        best = min(range(len(candidates)), key=lambda i: (
            -scores[i], -(candidates[i].score if candidates[i].score is not None else -math.inf), candidates[i].position, i))
        return candidates[best]
    except Exception as exc:
        _logger.warning("learned ranking failed (%s: %s); using the deterministic winner", type(exc).__name__, exc)
        return None
