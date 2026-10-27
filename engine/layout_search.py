"""Retry candidate generation until validation succeeds; never return a bad draft."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from engine.candidate_validator import ValidationReport, element_bounds, validate_candidate
from engine.internal_schema import Canvas, Design


class NoFeasibleLayout(RuntimeError):
    """Search exhausted its budget; this is not a proof of infeasibility."""

    def __init__(self, reports: tuple[ValidationReport, ...]):
        self.reports = reports
        super().__init__(f"No valid layout after {len(reports)} attempts: "
                         + "; ".join(reports[-1].violations))


Proposal = Callable[[int, ValidationReport | None], Design | Mapping[str, Any]]


def find_valid_layout(
    source: Design, target: Canvas, propose: Proposal, *, max_attempts: int = 8,
) -> dict[str, Any]:
    """Feed failures back to the proposer; only validated JSON may leave here.

    A finite budget prevents a broken or impossible generator from hanging.
    Exhaustion raises, returning neither the last draft nor a least-bad draft.
    Unexpected generator exceptions propagate rather than being swallowed.
    """
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    reports: list[ValidationReport] = []
    for attempt in range(max_attempts):
        candidate = propose(attempt, reports[-1] if reports else None)
        # Snapshot before checking so the exported object is exactly what passed.
        try:
            snapshot = (candidate if isinstance(candidate, Design) else Design.from_adobe(candidate)).to_adobe()
        except (ValueError, TypeError, OverflowError) as exc:
            reports.append(ValidationReport(False, False, False, (f"schema: {exc}",)))
            continue
        report = validate_candidate(source, snapshot, target)
        if report.is_valid:
            return snapshot
        reports.append(report)
    raise NoFeasibleLayout(tuple(reports))


def baseline_proposal(source: Design, target: Canvas, attempt: int) -> dict[str, Any]:
    """First try the old stretch; then fit the whole composition proportionally.

    Repair includes negative source coordinates, rotation and fixed-size strokes.
    It preserves relative element positions, explicit groups and image ratios.
    This is a geometric fallback, not a claim of attractive or legible reflow.
    """
    result = source.to_adobe()
    result["canvas"] = target.to_adobe()
    if attempt == 0:
        sx = target.width / source.canvas.width
        sy = target.height / source.canvas.height
        dx = dy = 0.0
    else:
        # Measure at a trial scale to account for strokes which cannot change.
        scale = min(target.width / source.canvas.width, target.height / source.canvas.height)
        scale *= 0.8 ** (attempt - 1)
        for _ in range(4):
            trial = source.to_adobe()
            for element in trial["elements"]:
                for field in ("x", "y", "width", "height"):
                    element[field] *= scale
            bounds = [element_bounds(e) for e in Design.from_adobe(trial).elements]
            left, top = min(b[0] for b in bounds), min(b[1] for b in bounds)
            right, bottom = max(b[2] for b in bounds), max(b[3] for b in bounds)
            width, height = right - left, bottom - top
            factor = min(1.0, (target.width * 0.999999) / max(width, 1e-300),
                         (target.height * 0.999999) / max(height, 1e-300))
            if factor >= 1:
                break
            scale *= factor
        # Re-measure the actual final scale, including fixed-width strokes.
        trial = source.to_adobe()
        for element in trial["elements"]:
            for field in ("x", "y", "width", "height"):
                element[field] *= scale
        bounds = [element_bounds(e) for e in Design.from_adobe(trial).elements]
        left, top = min(b[0] for b in bounds), min(b[1] for b in bounds)
        right, bottom = max(b[2] for b in bounds), max(b[3] for b in bounds)
        sx = sy = scale
        dx = (target.width - (right - left)) / 2 - left
        dy = (target.height - (bottom - top)) / 2 - top
    for element in result["elements"]:
        element["x"] = element["x"] * sx + dx
        element["y"] = element["y"] * sy + dy
        element["width"] *= sx
        element["height"] *= sy
        if element["type"] == "text":
            element["font_size"] = max(1.0, element["font_size"] * min(sx, sy))
    return result
