"""Deterministic shape-role classification for S04 shape stretching.

Roles are search hints derived from SOURCE geometry, shape_type and the
relationships between elements. They never justify removal, reordering or any
change outside the K6 allowlist (x, y, width, height, text font_size), and ID
spellings, `_note` text and underscore-prefixed fields are never read.

Ambiguity defaults to ShapeRole.FIXED (keep uniform scaling): a wrong
"stretchable" label visibly distorts a design, a wrong "fixed" label only
forgoes an optimization. Ellipses (circles included) and icon-like squares are
PROPORTION_SENSITIVE and stay proportional unless a future task supplies strong
evidence. Rotated shapes are FIXED because stretching a rotated box does not
stretch the drawn shape along its own axes.

Roles (see docs/reference/D_candidate_generation.md, D.1.7):
- BACKGROUND: background-like rectangle (D.1.3 heuristic); fills its region.
- PANEL: rectangle containing other elements, or covering a large share of the
  canvas; fills a region around its content.
- BUTTON: small rectangle whose contained elements are exactly one text label
  (CTA, badge, ribbon); resized around the label.
- BAR_H / BAR_V: thin rectangles or lines; stretch along their long axis only.
- PROPORTION_SENSITIVE / FIXED: never stretched independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from engine.internal_schema import Design, Element
from engine.layout_units import box_contains, is_background_like

__all__ = [
    "BAR_ASPECT_MIN",
    "BUTTON_AREA_FRACTION_MAX",
    "BUTTON_LABEL_HEIGHT_FRACTION_MIN",
    "ICON_SQUARE_ASPECT_RANGE",
    "PANEL_AREA_FRACTION_MIN",
    "ShapeInfo",
    "ShapeRole",
    "classify_shapes",
    "is_proportion_sensitive",
]

# Rectangles at least this many times longer than thick are bars/separators.
BAR_ASPECT_MIN = 4.0
# A button/badge covers at most this share of the source canvas; larger
# containers are panels.
BUTTON_AREA_FRACTION_MAX = 0.25
# The label must be at least this fraction of the shape height; a tiny caption
# inside a large rectangle is a panel with a caption, not a button label.
BUTTON_LABEL_HEIGHT_FRACTION_MIN = 0.25
# A rectangle that is not a bar and covers at least this share of the canvas
# reads as a panel even when it contains nothing.
PANEL_AREA_FRACTION_MIN = 0.15
# Near-square rectangles with no content are icon-like and keep proportions.
ICON_SQUARE_ASPECT_RANGE = (0.8, 1.25)
# Tolerance (px) for source containment of a label in its shape.
_CONTAIN_TOLERANCE = 1.0


class ShapeRole(str, Enum):
    BACKGROUND = "background"
    PANEL = "panel"
    BUTTON = "button"
    BAR_H = "bar_h"
    BAR_V = "bar_v"
    PROPORTION_SENSITIVE = "proportion_sensitive"
    FIXED = "fixed"


@dataclass(frozen=True, slots=True)
class ShapeInfo:
    """Role of one source shape plus the elements it visually holds."""

    role: ShapeRole
    contained_ids: tuple[str, ...] = ()


def _contains(outer: Element, inner: Element) -> bool:
    tol = _CONTAIN_TOLERANCE
    return (
        outer.width > 0 and outer.height > 0
        and inner.width > 0 and inner.height > 0
        and outer.x - tol <= inner.x and outer.y - tol <= inner.y
        and outer.x + outer.width + tol >= inner.x + inner.width
        and outer.y + outer.height + tol >= inner.y + inner.height
    )


def _rotated(element: Element) -> bool:
    return float(element.effective_value("rotation")) % 360 != 0


def is_proportion_sensitive(element: Element) -> bool:
    """Positive-area ellipses (circles included) keep their aspect ratio.

    Context-free, so the objective can reuse it. Icon-like squares need
    context (do they hold content?) and are decided by classify_shapes.
    """
    return (element.type == "shape"
            and element.effective_value("shape_type") == "ellipse"
            and element.width > 0 and element.height > 0)


def classify_shapes(source: Design) -> dict[str, ShapeInfo]:
    """Role for every shape element of the source, keyed by element ID."""
    canvas = source.source_canvas
    canvas_area = float(canvas.width) * float(canvas.height)
    elements = source.elements
    roles: dict[str, ShapeInfo] = {}
    for element in elements:
        if element.type != "shape":
            continue
        shape_type = element.effective_value("shape_type")
        width, height = float(element.width), float(element.height)
        if is_proportion_sensitive(element):
            roles[element.id] = ShapeInfo(ShapeRole.PROPORTION_SENSITIVE)
            continue
        if _rotated(element):
            roles[element.id] = ShapeInfo(ShapeRole.FIXED)
            continue
        if shape_type == "line" or width == 0 or height == 0:
            if width == 0 and height == 0:
                roles[element.id] = ShapeInfo(ShapeRole.FIXED)
            elif height == 0 or (width > 0 and width >= height):
                roles[element.id] = ShapeInfo(ShapeRole.BAR_H)
            else:
                roles[element.id] = ShapeInfo(ShapeRole.BAR_V)
            continue
        if shape_type != "rectangle" and shape_type is not None:
            roles[element.id] = ShapeInfo(ShapeRole.FIXED)
            continue
        contained = tuple(
            other.id for other in elements
            if other.id != element.id and (box_contains(element, other)
                                           or _contains(element, other))
        )
        others = [source.elements_by_id[ident] for ident in contained]
        area = width * height
        area_fraction = area / canvas_area if canvas_area > 0 else 0.0
        if is_background_like(element, canvas):
            roles[element.id] = ShapeInfo(ShapeRole.BACKGROUND, contained)
            continue
        labels = [other for other in others if other.type == "text"]
        if (len(labels) == 1 and len(others) == 1
                and area_fraction <= BUTTON_AREA_FRACTION_MAX
                and float(labels[0].height) >= BUTTON_LABEL_HEIGHT_FRACTION_MIN * height
                and not _rotated(labels[0])):
            roles[element.id] = ShapeInfo(ShapeRole.BUTTON, contained)
            continue
        aspect = width / height
        thin = aspect >= BAR_ASPECT_MIN or aspect <= 1 / BAR_ASPECT_MIN
        if others or (area_fraction >= PANEL_AREA_FRACTION_MIN and not thin):
            roles[element.id] = ShapeInfo(ShapeRole.PANEL, contained)
            continue
        if aspect >= BAR_ASPECT_MIN:
            roles[element.id] = ShapeInfo(ShapeRole.BAR_H)
        elif aspect <= 1 / BAR_ASPECT_MIN:
            roles[element.id] = ShapeInfo(ShapeRole.BAR_V)
        elif ICON_SQUARE_ASPECT_RANGE[0] <= aspect <= ICON_SQUARE_ASPECT_RANGE[1]:
            roles[element.id] = ShapeInfo(ShapeRole.PROPORTION_SENSITIVE)
        else:
            roles[element.id] = ShapeInfo(ShapeRole.FIXED)
    return roles
