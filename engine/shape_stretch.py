"""Context-dependent shape stretching for S04 candidates (D.1.7).

The family generators place every layout unit with ONE uniform scale, so on
extreme targets (728x90) buttons, badges and bars shrink to slivers and the
canvas is left mostly empty. This module takes an already-placed candidate and
derives variants in which eligible SHAPES resize independently on each axis
(scale_x = new_w / old_w, scale_y = new_h / old_h). It is a post-placement
pass: it never changes the base candidate, never changes a unit's relative
group geometry beyond the shapes it names, and the base stays available as the
"proportional" alternative.

Only K6 fields change (x, y, width, height on shapes and their labels, plus
label font_size) and every change is routed through layout_edit. Roles come
from engine.shape_roles; anything ambiguous, rotated, elliptical or
icon-like is left exactly as the base placed it. Every move is checked against
the base layout and rejected, not repaired, when it would leave the canvas,
newly collide with an unrelated element, or newly cover something that stacks
below it. validate_candidate remains the authority; this module is pure and
deterministic (no RNG, ties break by source order).

Variants (each emitted only if it differs from the base):
- label_fit:   buttons resized around their label (consistent padding, label
               centered, widen before shrinking the font, bounded font growth).
- panel_fill:  panels/backgrounds fill the canvas region they occupy in the
               source (per-axis fractions) while still holding their content.
- axis_x:      horizontal bars/separators lengthen, thickness unchanged.
- axis_y:      vertical bars/separators lengthen, width unchanged.
- balanced:    all of the above at half strength (log-interpolated toward the
               full stretch) so aspect-ratio changes stay conservative.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from engine.internal_schema import ABSENT, Canvas, Design, Element
from engine.layout_edits import layout_edit
from engine.layout_typography import SINGLE_LINE_EM, estimated_fit
from engine.shape_roles import ShapeInfo, ShapeRole, classify_shapes

__all__ = [
    "BALANCED_STRENGTH",
    "SHAPE_STRETCH_VERSION",
    "VARIANT_AXIS_X",
    "VARIANT_AXIS_Y",
    "VARIANT_BALANCED",
    "VARIANT_LABEL_FIT",
    "VARIANT_NAMES",
    "VARIANT_PANEL_FILL",
    "VARIANT_PROPORTIONAL",
    "StretchVariant",
    "stretch_variants",
]

SHAPE_STRETCH_VERSION = "shape_stretch_v1"

VARIANT_PROPORTIONAL = "proportional"
VARIANT_LABEL_FIT = "label_fit"
VARIANT_PANEL_FILL = "panel_fill"
VARIANT_AXIS_X = "axis_x"
VARIANT_AXIS_Y = "axis_y"
VARIANT_BALANCED = "balanced"
VARIANT_NAMES = (VARIANT_PROPORTIONAL, VARIANT_LABEL_FIT, VARIANT_PANEL_FILL,
                 VARIANT_AXIS_X, VARIANT_AXIS_Y, VARIANT_BALANCED)

# Log-interpolation strength of the balanced variant toward the full stretch.
BALANCED_STRENGTH = 0.5
# Fractions of the requested stretch tried, largest first, so a bar or panel
# grows until it is blocked rather than being rejected outright.
_GROWTH_LADDER = (1.0, 0.75, 0.5, 0.25)
# Keep boxes strictly inside the canvas: element_bounds recomputes edges from
# centre +- half-size and float rounding must not cross the boundary.
_EDGE_EPS = 1e-6
# Moves smaller than this many pixels on every side are not stretches.
_MIN_MOVE_PX = 0.5
# A bar must grow by at least this factor to count as lengthened.
_MIN_BAR_GROWTH = 1.05
# Overlap area (px^2) at or below this is not an overlap (float dust).
_OVERLAP_EPS = 1e-6
# Label box slack over the natural single-line width, and its fixed pixel
# part (the word-fit estimate reserves 2px on each axis).
_LABEL_WIDTH_SLACK = 1.04
_LABEL_PIXEL_SLACK = 2.01
_LABEL_LINE_EM = SINGLE_LINE_EM * 1.001
# Source padding of a button label, in ems of the label font, clipped to
# sensible bounds; padding may shrink to the floor before the font does.
_PAD_X_EM_RANGE = (0.4, 2.5)
_PAD_Y_EM_RANGE = (0.15, 1.2)
_PAD_FACTORS = (1.0, 0.75, 0.5)
# Label font growth: ladder step, absolute cap relative to the base font.
_FONT_GROWTH_STEP = 1.1
_FONT_GROWTH_CAP = 1.5
# A button's aspect may grow or shrink by at most this factor vs the source.
_BUTTON_ASPECT_CHANGE_MAX = 3.0
# A label is single-line in the source when its box is at most this many line
# heights tall; multi-line labels are left to the base placement.
_SINGLE_LINE_MAX_LINES = 1.8
# A free bar (no associated element) stretches with the canvas only when it
# spans at least this fraction of the source canvas along its long axis.
_FREE_BAR_CANVAS_FRACTION = 0.5
# A bar is associated with an element it runs parallel to when they overlap by
# this fraction of the longer extent along the long axis and sit within this canvas fraction apart.
_BAR_OVERLAP_MIN = 0.6
_BAR_GAP_CANVAS_FRACTION = 0.15

Box = tuple[float, float, float, float]
Measure = Callable[[dict], Mapping[str, Any]]


@dataclass(frozen=True)
class StretchVariant:
    """One stretched alternative to a base candidate (Adobe JSON untouched)."""

    name: str
    adobe: dict[str, Any]
    changed_ids: tuple[str, ...]


def _overlap(a: Box, b: Box) -> bool:
    width = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    height = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return width > 0 and height > 0 and width * height > _OVERLAP_EPS


def _same_box(a: Box, b: Box) -> bool:
    return all(abs(p - q) <= _MIN_MOVE_PX for p, q in zip(a, b))


def _clip(value: float, bounds: tuple[float, float]) -> float:
    return min(bounds[1], max(bounds[0], value))


def _stroke_pad(element: Element) -> float:
    if element.type == "shape" and element.effective_value("stroke_color") is not None:
        return max(0.0, float(element.effective_value("stroke_width"))) / 2
    return 0.0


def _stack_key(element: Element, index: int) -> tuple[float, int]:
    z = element.z_index
    return (0.0 if z is ABSENT else float(z), index)


class _Context:
    """Mutable working copy of one base candidate's placement."""

    def __init__(self, source: Design, target: Canvas, base: Mapping[str, Any],
                 measure: Measure | None) -> None:
        self.source = source
        self.target = target
        self.measure = measure or estimated_fit
        self.roles = classify_shapes(source)
        self.elements = list(source.elements)
        self.index = {element.id: pos for pos, element in enumerate(self.elements)}
        placed = {item["id"]: item for item in base["elements"]}
        self.base_items = placed
        self.boxes: dict[str, Box] = {
            ident: (float(item["x"]), float(item["y"]),
                    float(item["width"]), float(item["height"]))
            for ident, item in placed.items()
        }
        self.start_boxes = dict(self.boxes)
        self.fonts: dict[str, float] = {
            ident: float(item["font_size"]) for ident, item in placed.items()
            if item["type"] == "text"
        }
        self.start_fonts = dict(self.fonts)

    # --- geometry helpers -------------------------------------------------
    def inside(self, box: Box, pad: float) -> bool:
        lo = pad + _EDGE_EPS / 2
        return (box[2] >= 0 and box[3] >= 0
                and box[0] >= lo and box[1] >= lo
                and box[0] + box[2] <= self.target.width - lo
                and box[1] + box[3] <= self.target.height - lo)

    def fit_to_canvas(self, box: Box, pad: float) -> Box:
        lo = pad + _EDGE_EPS
        left = max(lo, box[0])
        top = max(lo, box[1])
        right = min(self.target.width - lo, box[0] + box[2])
        bottom = min(self.target.height - lo, box[1] + box[3])
        return (left, top, max(0.0, right - left), max(0.0, bottom - top))

    def newly_overlaps(self, shape_id: str, new: Box, ignore: set[str], *,
                       below_only: bool = False) -> bool:
        """True when `new` collides with something the old box did not touch.

        Backgrounds and panels are layering context, never collisions. With
        below_only (panels), only elements that stack beneath the shape count:
        a panel may cover what is drawn above it but must not hide the rest.
        """
        old = self.boxes[shape_id]
        mine = self.elements[self.index[shape_id]]
        for other in self.elements:
            if other.id == shape_id or other.id in ignore:
                continue
            info = self.roles.get(other.id)
            if info is not None and info.role in (ShapeRole.BACKGROUND, ShapeRole.PANEL):
                continue
            box = self.boxes[other.id]
            if not _overlap(new, box) or _overlap(old, box):
                continue
            if below_only and _stack_key(other, self.index[other.id]) > _stack_key(
                    mine, self.index[shape_id]):
                continue
            return True
        return False

    def natural_width(self, label_id: str, size: float) -> float:
        probe = dict(self.base_items[label_id])
        probe.update(font_size=size, width=1e6, height=1e6)
        return float(self.measure(probe)["measured_width"])

    def source_natural_width(self, label: Element) -> float:
        probe = label.to_adobe()
        probe.update(width=1e6, height=1e6)
        return float(self.measure(probe)["measured_width"])

    def hierarchy_cap(self, label: Element) -> float:
        """Never let a label outgrow a text that was larger in the source."""
        caps = [self.fonts[other.id] for other in self.elements
                if other.type == "text" and other.id != label.id
                and float(other.font_size) > float(label.font_size)]
        return min(caps) if caps else math.inf


# --- buttons ----------------------------------------------------------------
def _fit_button(ctx: _Context, shape: Element, info: ShapeInfo) -> bool:
    label = ctx.source.elements_by_id[info.contained_ids[0]]
    size_src = float(label.font_size)
    line_height = float(label.effective_value("line_height"))
    if (not label.content or "\n" in label.content
            or float(label.height) > _SINGLE_LINE_MAX_LINES * size_src * line_height
            or label.effective_value("max_lines") not in (None, 1)
            or float(shape.width) <= 0 or float(shape.height) <= 0):
        return False
    nat_src = ctx.source_natural_width(label)
    pad_x_em = _clip((float(shape.width) - nat_src) / 2 / size_src, _PAD_X_EM_RANGE)
    pad_y_em = _clip((float(shape.height) - _LABEL_LINE_EM * size_src) / 2 / size_src,
                     _PAD_Y_EM_RANGE)
    radius = shape.effective_value("corner_radius")
    pill = radius > 0 and float(radius) >= float(shape.height) / 2 - 0.5
    aspect_src = float(shape.width) / float(shape.height)

    base_shape = ctx.boxes[shape.id]
    base_font = ctx.fonts[label.id]
    scale = min(base_shape[2] / float(shape.width), base_shape[3] / float(shape.height))
    top = min(base_font * _FONT_GROWTH_CAP, ctx.hierarchy_cap(label),
              max(size_src, base_font))
    rungs = {base_font}
    reference = size_src * scale
    if base_font < reference <= top:
        rungs.add(reference)
    rung = base_font * _FONT_GROWTH_STEP
    while rung <= top:
        rungs.add(rung)
        rung *= _FONT_GROWTH_STEP
    pad = _stroke_pad(shape)
    center_x = base_shape[0] + base_shape[2] / 2
    center_y = base_shape[1] + base_shape[3] / 2
    for size in sorted(rungs, reverse=True):
        nat = ctx.natural_width(label.id, size)
        label_w = nat * _LABEL_WIDTH_SLACK + _LABEL_PIXEL_SLACK
        label_h = size * _LABEL_LINE_EM + _LABEL_PIXEL_SLACK
        for factor in _PAD_FACTORS:
            pad_x = max(pad_x_em * factor, _PAD_X_EM_RANGE[0]) * size
            pad_y = max(pad_y_em * factor, _PAD_Y_EM_RANGE[0]) * size
            width = max(nat + 2 * pad_x, label_w + 0.4 * size)
            height = label_h + 2 * pad_y
            if pill and height > 2 * float(radius):
                continue
            aspect = width / height
            if not (aspect_src / _BUTTON_ASPECT_CHANGE_MAX
                    <= aspect <= aspect_src * _BUTTON_ASPECT_CHANGE_MAX):
                continue
            top_y = center_y - height / 2
            for left in (center_x - width / 2, base_shape[0],
                         base_shape[0] + base_shape[2] - width):
                box = (left, top_y, width, height)
                if not ctx.inside(box, pad):
                    continue
                if ctx.newly_overlaps(shape.id, box, {label.id}):
                    continue
                if _same_box(box, base_shape) and abs(size - base_font) < 1e-9:
                    return False
                ctx.boxes[shape.id] = box
                ctx.boxes[label.id] = (left + (width - label_w) / 2,
                                       top_y + (height - label_h) / 2,
                                       label_w, label_h)
                ctx.fonts[label.id] = size
                return True
    return False


# --- panels and backgrounds ---------------------------------------------------
def _blend(base: float, full: float, strength: float) -> float:
    """Log-interpolate positive sizes; linear when either end is not positive."""
    if base > 0 and full > 0:
        return math.exp((1 - strength) * math.log(base) + strength * math.log(full))
    return (1 - strength) * base + strength * full


def _fill_panel(ctx: _Context, shape: Element, info: ShapeInfo,
                strength: float) -> bool:
    if info.role is ShapeRole.BACKGROUND:
        strength = 1.0  # a half-filled background reads as a mistake, not a compromise
    source = ctx.source.source_canvas
    scale_x = float(ctx.target.width) / float(source.width)
    scale_y = float(ctx.target.height) / float(source.height)
    full = (float(shape.x) * scale_x, float(shape.y) * scale_y,
            float(shape.width) * scale_x, float(shape.height) * scale_y)
    pad = _stroke_pad(shape)
    left, top = full[0], full[1]
    right, bottom = full[0] + full[2], full[1] + full[3]
    if info.contained_ids:
        # Hold the content: source padding, scaled with the content unit.
        held = [ctx.boxes[ident] for ident in info.contained_ids]
        src_held = [ctx.source.elements_by_id[ident] for ident in info.contained_ids]
        src_left = min(float(e.x) for e in src_held)
        src_top = min(float(e.y) for e in src_held)
        src_right = max(float(e.x + e.width) for e in src_held)
        src_bottom = max(float(e.y + e.height) for e in src_held)
        h_left = min(b[0] for b in held)
        h_top = min(b[1] for b in held)
        h_right = max(b[0] + b[2] for b in held)
        h_bottom = max(b[1] + b[3] for b in held)
        unit = 1.0
        if src_right > src_left and src_bottom > src_top:
            unit = min((h_right - h_left) / (src_right - src_left),
                       (h_bottom - h_top) / (src_bottom - src_top))
        left = min(left, h_left - max(0.0, src_left - float(shape.x)) * unit)
        top = min(top, h_top - max(0.0, src_top - float(shape.y)) * unit)
        right = max(right, h_right + max(
            0.0, float(shape.x + shape.width) - src_right) * unit)
        bottom = max(bottom, h_bottom + max(
            0.0, float(shape.y + shape.height) - src_bottom) * unit)
    target_box = ctx.fit_to_canvas((left, top, right - left, bottom - top), pad)
    old = ctx.boxes[shape.id]
    ignore = set(info.contained_ids)
    for fraction in _GROWTH_LADDER:
        reach = strength * fraction
        width = _blend(old[2], target_box[2], reach)
        height = _blend(old[3], target_box[3], reach)
        center_x = old[0] + old[2] / 2 + reach * (
            target_box[0] + target_box[2] / 2 - old[0] - old[2] / 2)
        center_y = old[1] + old[3] / 2 + reach * (
            target_box[1] + target_box[3] / 2 - old[1] - old[3] / 2)
        box = ctx.fit_to_canvas((center_x - width / 2, center_y - height / 2,
                                 width, height), pad)
        if info.contained_ids:
            # Never lose content to a partial blend: grow to hold it.
            held = [ctx.boxes[ident] for ident in info.contained_ids]
            union_left = min(box[0], *(b[0] for b in held))
            union_top = min(box[1], *(b[1] for b in held))
            union_right = max(box[0] + box[2], *(b[0] + b[2] for b in held))
            union_bottom = max(box[1] + box[3], *(b[1] + b[3] for b in held))
            box = ctx.fit_to_canvas((union_left, union_top, union_right - union_left,
                                     union_bottom - union_top), pad)
        if _same_box(box, old) or not ctx.inside(box, pad):
            continue
        if ctx.newly_overlaps(shape.id, box, ignore, below_only=True):
            continue
        ctx.boxes[shape.id] = box
        return True
    return False


# --- bars and separators ------------------------------------------------------
def _bar_target_extent(ctx: _Context, shape: Element, horizontal: bool
                       ) -> tuple[float, float] | None:
    """Wanted (start, end) along the bar's long axis in target pixels."""
    source = ctx.source.source_canvas
    start_src = float(shape.x if horizontal else shape.y)
    length_src = float(shape.width if horizontal else shape.height)
    cross_src = float(shape.y if horizontal else shape.x)
    thick_src = float(shape.height if horizontal else shape.width)
    canvas_long = float(source.width if horizontal else source.height)
    canvas_cross = float(source.height if horizontal else source.width)
    best: tuple[float, int, Element] | None = None
    for position, other in enumerate(ctx.elements):
        if other.id == shape.id:
            continue
        info = ctx.roles.get(other.id)
        if info is not None and info.role in (
                ShapeRole.BAR_H, ShapeRole.BAR_V, ShapeRole.BACKGROUND, ShapeRole.PANEL):
            continue
        o_start = float(other.x if horizontal else other.y)
        o_len = float(other.width if horizontal else other.height)
        o_cross = float(other.y if horizontal else other.x)
        o_thick = float(other.height if horizontal else other.width)
        if o_len <= 0 or length_src <= 0:
            continue
        shared = min(start_src + length_src, o_start + o_len) - max(start_src, o_start)
        # Parallel runs share most of the LONGER extent (an underline matches its
        # heading); a small element merely lying along a long bar is not its partner.
        if shared / max(length_src, o_len) < _BAR_OVERLAP_MIN:
            continue
        gap = max(o_cross - (cross_src + thick_src), cross_src - (o_cross + o_thick), 0.0)
        if gap > _BAR_GAP_CANVAS_FRACTION * canvas_cross:
            continue
        if best is None or (gap, position) < best[:2]:
            best = (gap, position, other)
    if best is not None:
        other = best[2]
        o_start = float(other.x if horizontal else other.y)
        o_len = float(other.width if horizontal else other.height)
        placed = ctx.boxes[other.id]
        p_start, p_len = (placed[0], placed[2]) if horizontal else (placed[1], placed[3])
        if p_len <= 0:
            return None
        return (p_start + (start_src - o_start) / o_len * p_len,
                p_start + (start_src + length_src - o_start) / o_len * p_len)
    if length_src >= _FREE_BAR_CANVAS_FRACTION * canvas_long:
        ratio = (float(ctx.target.width) if horizontal else float(ctx.target.height)) / canvas_long
        return (start_src * ratio, (start_src + length_src) * ratio)
    return None


def _stretch_bar(ctx: _Context, shape: Element, horizontal: bool,
                 strength: float) -> bool:
    wanted = _bar_target_extent(ctx, shape, horizontal)
    if wanted is None or wanted[1] <= wanted[0]:
        return False
    old = ctx.boxes[shape.id]
    pad = _stroke_pad(shape)
    old_start, old_len = (old[0], old[2]) if horizontal else (old[1], old[3])
    full_len = wanted[1] - wanted[0]
    for fraction in _GROWTH_LADDER:
        reach = strength * fraction
        length = _blend(old_len, full_len, reach)
        center = old_start + old_len / 2 + reach * (
            (wanted[0] + wanted[1]) / 2 - old_start - old_len / 2)
        if horizontal:
            box = (center - length / 2, old[1], length, old[3])
        else:
            box = (old[0], center - length / 2, old[2], length)
        box = ctx.fit_to_canvas(box, pad)
        new_len = box[2] if horizontal else box[3]
        # A stretch variant only ever lengthens bars, and visibly so.
        if new_len < old_len * _MIN_BAR_GROWTH + _MIN_MOVE_PX or not ctx.inside(box, pad):
            continue
        if ctx.newly_overlaps(shape.id, box, set()):
            continue
        ctx.boxes[shape.id] = box
        return True
    return False


# --- assembly -------------------------------------------------------------------
def _apply(ctx: _Context, *, buttons: bool, panels: float, bars_h: float,
           bars_v: float) -> tuple[str, ...]:
    for element in ctx.elements:
        info = ctx.roles.get(element.id)
        if info is not None and panels > 0 and info.role in (
                ShapeRole.PANEL, ShapeRole.BACKGROUND):
            _fill_panel(ctx, element, info, panels)
    for element in ctx.elements:
        info = ctx.roles.get(element.id)
        if info is None:
            continue
        if buttons and info.role is ShapeRole.BUTTON:
            _fit_button(ctx, element, info)
        elif bars_h > 0 and info.role is ShapeRole.BAR_H:
            _stretch_bar(ctx, element, True, bars_h)
        elif bars_v > 0 and info.role is ShapeRole.BAR_V:
            _stretch_bar(ctx, element, False, bars_v)
    return tuple(
        element.id for element in ctx.elements
        if not _same_box(ctx.boxes[element.id], ctx.start_boxes[element.id])
        or ctx.fonts.get(element.id) != ctx.start_fonts.get(element.id)
    )


def _emit(ctx: _Context, base: Mapping[str, Any], changed: tuple[str, ...]
          ) -> dict[str, Any]:
    touched = set(changed)
    elements = []
    for original in base["elements"]:
        ident = original["id"]
        if ident not in touched:
            elements.append(layout_edit(original, {}))
            continue
        box = ctx.boxes[ident]
        changes: dict[str, Any] = {"x": box[0], "y": box[1],
                                   "width": box[2], "height": box[3]}
        if original["type"] == "text":
            changes["font_size"] = ctx.fonts[ident]
        elements.append(layout_edit(original, changes))
    return {"canvas": dict(base["canvas"]), "elements": elements}


_RECIPES: tuple[tuple[str, dict[str, Any]], ...] = (
    (VARIANT_LABEL_FIT, dict(buttons=True, panels=0.0, bars_h=0.0, bars_v=0.0)),
    (VARIANT_PANEL_FILL, dict(buttons=False, panels=1.0, bars_h=0.0, bars_v=0.0)),
    (VARIANT_AXIS_X, dict(buttons=False, panels=0.0, bars_h=1.0, bars_v=0.0)),
    (VARIANT_AXIS_Y, dict(buttons=False, panels=0.0, bars_h=0.0, bars_v=1.0)),
    (VARIANT_BALANCED, dict(buttons=True, panels=BALANCED_STRENGTH,
                            bars_h=BALANCED_STRENGTH, bars_v=BALANCED_STRENGTH)),
)


def stretch_variants(source: Design, target: Canvas, base: Mapping[str, Any], *,
                     measure: Measure | None = None) -> list[StretchVariant]:
    """Stretched alternatives to `base`, in fixed recipe order.

    `base` is an Adobe-shaped candidate for `source` on `target` (it need not
    be validated, but only a valid base yields meaningful variants). `measure`
    may be the renderer's text measurement; the default is the S04 word-fit
    estimate. Variants identical to the base or to an earlier variant are
    dropped, so an empty list means no shape in this layout can stretch.
    Nothing here validates or scores; callers pass variants through the same
    gate as every other candidate.
    """
    if not any(info.role not in (ShapeRole.FIXED, ShapeRole.PROPORTION_SENSITIVE)
               for info in classify_shapes(source).values()):
        return []
    seen: set[str] = set()
    variants: list[StretchVariant] = []
    for name, recipe in _RECIPES:
        ctx = _Context(source, target, base, measure)
        changed = _apply(ctx, **recipe)
        if not changed:
            continue
        adobe = _emit(ctx, base, changed)
        key = repr([(e["id"], e["x"], e["y"], e["width"], e["height"],
                     e.get("font_size")) for e in adobe["elements"]])
        if key in seen:
            continue
        seen.add(key)
        variants.append(StretchVariant(name, adobe, changed))
    return variants
