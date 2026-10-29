"""Deterministic output gate, independent of learned aesthetic scores."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Any, Mapping

from engine.internal_schema import Canvas, Design, Element


def mutable_fields(element_type: str) -> frozenset[str]:
    """The only fields a candidate may change (K6, docs/TECHNICAL_SPEC.md section 8.1.1).

    x, y, width and height on every element, plus font_size on text elements. This is the single
    definition: the validator uses it, and generator-side helpers derive from it.
    """
    fields = {"x", "y", "width", "height"}
    if element_type == "text":
        fields.add("font_size")
    return frozenset(fields)


@dataclass(frozen=True)
class ValidationReport:
    schema_ok: bool
    policy_ok: bool
    geometry_ok: bool
    violations: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return self.schema_ok and self.policy_ok and self.geometry_ok and not self.violations


def element_bounds(element: Element) -> tuple[float, float, float, float]:
    """Union of the stored box and rotated box; include visible shape strokes.

    This is a conservative geometry check, not a glyph-fit/rendering test.
    """
    x, y, width, height = (float(getattr(element, key)) for key in ("x", "y", "width", "height"))
    angle = math.radians(float(element.effective_value("rotation") % 360))
    cosine, sine = abs(math.cos(angle)), abs(math.sin(angle))
    stroke = 0.0
    if element.type == "shape" and element.effective_value("stroke_color") is not None:
        stroke = max(0.0, float(element.effective_value("stroke_width"))) / 2
    padding = stroke * (cosine + sine)
    half_w = max(width / 2, cosine * width / 2 + sine * height / 2 + padding)
    half_h = max(height / 2, sine * width / 2 + cosine * height / 2 + padding)
    cx, cy = x + width / 2, y + height / 2
    bounds = (cx - half_w, cy - half_h, cx + half_w, cy + half_h)
    if not all(math.isfinite(value) for value in bounds):
        raise ValueError("non-finite geometric extent")
    return bounds


def _same_json(a: Any, b: Any) -> bool:
    # Python equates True with 1 and 1 with 1.0; immutable JSON must not.
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same_json(a[key], b[key]) for key in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_same_json(x, y) for x, y in zip(a, b))
    return a == b


def validate_candidate(
    source: Design, candidate: Design | Mapping[str, Any], target: Canvas,
) -> ValidationReport:
    """Check schema, retained fields, target bounds and source image proportions.

    Only x/y/width/height and text font_size may change. Cover images are exempt
    from the box-ratio check; other positive-area images use <=5% relative error.
    No background exemption is inferred from IDs or appearance.
    """
    try:
        raw = candidate.to_adobe() if isinstance(candidate, Design) else candidate
        parsed = Design.from_adobe(raw)
    except (ValueError, TypeError, OverflowError) as exc:
        return ValidationReport(False, False, False, (f"schema: {exc}",))

    policy: list[str] = []
    geometry: list[str] = []
    if parsed.canvas != target:
        geometry.append("canvas: candidate must match the requested target")
    if tuple(e.id for e in parsed.elements) != tuple(e.id for e in source.elements):
        policy.append("identity: preserve every original ID exactly once and in original order")
    originals = source.elements_by_id
    for element in parsed.elements:
        original = originals.get(element.id)
        if original is not None:
            allowed = mutable_fields(original.type)
            before = {k: v for k, v in original.to_adobe().items() if k not in allowed}
            after = {k: v for k, v in element.to_adobe().items() if k not in allowed}
            if not _same_json(before, after):
                policy.append(f"{element.id}: immutable fields or optional-field presence changed")
            for dimension in ("width", "height"):
                if (getattr(original, dimension) == 0) != (getattr(element, dimension) == 0):
                    geometry.append(f"{element.id}: {dimension} collapsed or changed degeneracy")
            if (original.type == "image" and original.effective_value("fit") != "cover"
                    and all(v > 0 for v in (original.width, original.height, element.width, element.height))):
                ratio_change = (Fraction(element.width) * Fraction(original.height)
                                / (Fraction(element.height) * Fraction(original.width)))
                if abs(ratio_change - 1) > Fraction(1, 20):
                    geometry.append(f"{element.id}: image aspect ratio differs by more than 5%")
        try:
            left, top, right, bottom = element_bounds(element)
            if left < 0 or top < 0 or right > target.width or bottom > target.height:
                geometry.append(f"{element.id}: box, rotation or stroke extends off-canvas")
        except (ValueError, OverflowError) as exc:
            geometry.append(f"{element.id}: {exc}")

    return ValidationReport(True, not policy, not geometry, tuple(policy + geometry))
