"""Shared text-fit estimate (docs/reference/D_candidate_generation.md, D.1.5).

Standard library only, so the deterministic scorer can reuse it without PyTorch. This is an
estimate from stored fields, not measured glyph layout: it ignores letter spacing, bold, font
family and word-boundary wrapping, and its glyph advance is an uncalibrated placeholder (D.1.6).
Changing the constants or the formula changes graph_v4 feature values, so it requires a new
feature/ranker version.
"""

from __future__ import annotations

from fractions import Fraction
import math


TEXT_FIT_ESTIMATOR_VERSION = "text_fit_v1"
# Average glyph advance in em; placeholder pending calibration against the renderer.
GLYPH_ADVANCE_EM = Fraction(1, 2)


def estimate_text_height(
    content: str, width: int | float | Fraction, font_size: int | float | Fraction,
    line_height: int | float | Fraction,
) -> Fraction | None:
    """Estimated text block height in canvas pixels, or None when width <= 0.

    lines = sum over "\\n"-separated paragraphs of max(1, ceil(len(par) * k * font_size / width));
    height = lines * font_size * line_height. Exact rational arithmetic keeps the ceiling
    deterministic and free of float overflow for extreme finite inputs.
    """
    width = Fraction(width)
    if width <= 0:
        return None
    size = Fraction(font_size)
    lines = sum(
        max(1, math.ceil(Fraction(len(paragraph)) * GLYPH_ADVANCE_EM * size / width))
        for paragraph in content.split("\n")
    )
    return lines * size * Fraction(line_height)
