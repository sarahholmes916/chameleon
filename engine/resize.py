"""
THIS IS THE FILE YOU IMPLEMENT.

Everything else in this repo (input loading, output writing, rendering,
the Dockerfile) is plumbing so you can focus on exactly one function:
`resize()`. See RULES.md and the design schema for the full contract.
"""

import logging
from collections.abc import Sequence
from typing import Any

from engine.candidate_generator import (
    FAMILY_ASPECT_CUTOFF,
    FAMILY_BANNER_ZONES,
    FAMILY_HERO_LEFT,
    FAMILY_HERO_RIGHT,
    FAMILY_HORIZONTAL_FLOW,
    FAMILY_SCALE_BASELINE,
    FAMILY_VERTICAL_STACK,
    FAMILY_TWO_COLUMN,
    FAMILY_CONTROLLED_OVERLAY,
    two_column_candidates,
    controlled_overlay_candidates,
    banner_zones_candidates,
    baseline_candidate,
    hero_left_candidates,
    hero_right_candidates,
    horizontal_flow_candidates,
    vertical_stack_candidates,
)
from engine.candidate_pool import build_candidate_pool
from engine.candidate_validator import validate_candidate
from engine.internal_schema import Canvas, Design
from engine.layout_search import baseline_proposal, find_valid_layout

_logger = logging.getLogger(__name__)

# Tie-break order after score: the baseline floor first, so the designed
# families override legacy-feasible output only when strictly better.
_FAMILY_RANK = {
    FAMILY_SCALE_BASELINE: 0,
    FAMILY_HORIZONTAL_FLOW: 1,
    FAMILY_VERTICAL_STACK: 2,
    FAMILY_HERO_LEFT: 3,
    FAMILY_HERO_RIGHT: 4,
    FAMILY_BANNER_ZONES: 5,
}


def _best_index(scores: Sequence[float], family_ranks: Sequence[int]) -> int:
    """Position of the winning candidate: higher score, family rank, order."""
    return min(range(len(scores)),
               key=lambda here: (-scores[here], family_ranks[here], here))


def _select_batch_candidate(design: Design, target: Canvas) -> dict[str, Any] | None:
    """Best validated batch candidate, or None when the batch yields nothing."""
    families = [
        (FAMILY_SCALE_BASELINE, lambda s, t: [baseline_candidate(s, t)]),
        (FAMILY_HORIZONTAL_FLOW, horizontal_flow_candidates),
        (FAMILY_VERTICAL_STACK, vertical_stack_candidates),
        (FAMILY_HERO_LEFT, hero_left_candidates),
        (FAMILY_HERO_RIGHT, hero_right_candidates),
        (FAMILY_TWO_COLUMN, two_column_candidates),
        (FAMILY_CONTROLLED_OVERLAY, controlled_overlay_candidates),
    ]
    if target.width / target.height > FAMILY_ASPECT_CUTOFF:
        families.append((FAMILY_BANNER_ZONES, banner_zones_candidates))
    pool = build_candidate_pool(design, target, families=families)
    if pool.winner is None:
        return None
    selected = pool.winner.to_adobe()
    return selected if validate_candidate(design, selected, target).is_valid else None



def resize(source: dict[str, Any], target_canvas: dict[str, Any]) -> dict[str, Any]:
    """
    Args:
        source: {"canvas": {"width": int, "height": int}, "elements": [...]}
                 — the design at its original size. See
                 chameleon_design_schema_v0.9.json for the full element spec
                 (text / image / shape, position, size, rotation, opacity,
                 z_index, group_id, and type-specific fields).
        target_canvas: {"width": int, "height": int} — the shape you're
                 resizing into.

    Returns:
        A design dict with the SAME shape as `source`:
        {"canvas": target_canvas, "elements": [...]}

        Hard requirements (see RULES.md for the full list):
        - Every element `id` in `source` must appear exactly once in your
          output. Do not add, remove, or invent elements.
        - Do not regenerate or redraw any asset's pixel content — you are
          repositioning and resizing the elements you were given, nothing
          more. This is a disqualifying rule, not a style preference.
        - Non-background image elements should be kept in proportion
          (matching the source aspect ratio within ~5%) unless they are a
          full-bleed / `fit: "cover"` element, which is expected to crop.
        - `font_size` may change to help text fit. `font_family` and
          `content` should not, for this stage of scoring (see RULES.md,
          "Adaptive typography and copy").
        - Nothing should end up outside `target_canvas`'s bounds.

    Raises NoFeasibleLayout if all repair attempts fail validation; no output
    design is returned on failure. Validation does not certify visual quality.

    This is where your actual approach goes: constraint solving, a learned
    layout model, heuristics, whatever you choose. Latency is not a scoring
    concern (see RULES.md) — spend your effort on quality, not speed.
    """

    # The output boundary never returns an invalid best-effort draft.
    design = Design.from_adobe(source, target_canvas=target_canvas)
    target = Canvas.from_adobe(target_canvas)
    try:
        selected = _select_batch_candidate(design, target)
    except Exception:
        _logger.debug("batch selection failed; using legacy repair", exc_info=True)
        selected = None
    if selected is not None:
        return selected
    return find_valid_layout(
        design, target,
        lambda attempt, previous_report: baseline_proposal(design, target, attempt),
    )
