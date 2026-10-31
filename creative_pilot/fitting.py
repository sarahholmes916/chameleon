"""Renderer-measured text fitting. Uses the same wrapping/Skia metrics as the preview renderer.

The production S04 estimator has under-measured actual renderer widths (docs/S04_EVALUATION.md); every size
chosen by the creative pilot is therefore checked with `SourceRenderer.text_measurement`. Fits are
monotone-searched (largest size that fits, smallest width that fits) and cached.
"""

from __future__ import annotations

INSET = 2.0  # SourceRenderer.text_measurement requires width-2 / height-2 of slack
SLACK = 1.0  # extra px so a tight box never rounds into a 1px miss
MIN_SIZE = 1.0


class Measurer:
    def __init__(self, renderer):
        self.renderer = renderer
        self.cache: dict = {}
        self.calls = 0

    def measure(self, element: dict, width: float, size: float) -> tuple[int, float, float]:
        """(lines, measured_width, measured_height) for `element`'s text at this width and font size."""
        # Quantise BEFORE measuring so the searched value, the cached value and the stored value are the same number
        # (output boxes keep 3 decimals, font sizes 2). Content-addressed key: element ids repeat across sources.
        width, size = round(width, 3), round(size, 2)
        key = (element["content"], element["font_family"], element.get("font_weight"), element.get("font_style"),
               element.get("line_height"), element.get("letter_spacing"), width, size)
        hit = self.cache.get(key)
        if hit is None:
            probe = dict(element, width=max(width, 1.0), height=1e9, font_size=size)
            m = self.renderer.text_measurement(probe)
            hit = (m["lines"], m["measured_width"], m["measured_height"])
            self.cache[key] = hit
            self.calls += 1
        return hit

    def need(self, element: dict, width: float, size: float) -> tuple[float, float]:
        """Box (w, h) a text needs at `size` when wrapped into `width`, including the renderer's inset."""
        _, mw, mh = self.measure(element, width, size)
        return mw + INSET + SLACK, mh + INSET + SLACK

    def fits(self, element: dict, width: float, height: float, size: float) -> bool:
        w, h = self.need(element, width, size)
        return w <= width + 1e-6 and h <= height + 1e-6

    def height(self, element: dict, width: float, size: float) -> float:
        return self.need(element, width, size)[1]

    def max_size(self, element: dict, width: float, height: float, cap: float, floor: float = MIN_SIZE) -> float:
        """Largest font size <= cap whose wrapped text fits (width, height); `floor` if nothing fits."""
        cap = max(cap, floor)
        if self.fits(element, width, height, cap):
            return cap
        lo, hi = floor, cap
        if not self.fits(element, width, height, lo):
            return floor
        for _ in range(14):
            mid = (lo + hi) / 2
            if self.fits(element, width, height, mid):
                lo = mid
            else:
                hi = mid
        return round(lo, 2)

    def min_width(self, element: dict, size: float, max_height: float, max_width: float) -> float | None:
        """Smallest width <= max_width where the text fits max_height at `size`; None if even max_width fails."""
        if not self.fits(element, max_width, max_height, size):
            return None
        lo, hi = 4.0, max_width
        if self.fits(element, lo, max_height, size):
            return lo
        for _ in range(12):
            mid = (lo + hi) / 2
            if self.fits(element, mid, max_height, size):
                hi = mid
            else:
                lo = mid
        return hi

    def fit_report(self, element: dict) -> dict:
        """Measured fit of a placed text element (as it will render)."""
        _, mw, mh = self.measure(element, element["width"], element["font_size"])
        return dict(fits=mw <= element["width"] - INSET + 1e-6 and mh <= element["height"] - INSET + 1e-6,
                    measured_width=mw, measured_height=mh)
