"""S04 candidate generator: proportional baseline + flow/stack families.

Implements the discrete composition stage of spec D.1.3/D.1.4 (no continuous
refinement yet): layout units from engine.layout_units are assigned to slots
and emitted as Adobe dicts. Every emitted change passes through layout_edit,
so only K6 fields (x, y, width, height, text font_size) can change. The
generator is pure: it never calls validate_candidate and never raises on an
infeasible proposal (spec 5.4); validation and repair stay downstream. Same
inputs give byte-identical outputs; every ordering tie-breaks by source
index. Stdlib plus S02/S04 helpers only: no ML-framework imports.

Three placement paths, chosen per unit:
- Fill: units whose members all touch every canvas edge in the source
  (within FILL_EDGE_TOLERANCE_PX) stretch to the full target canvas.
  Cover images are K9-exempt and shapes/text boxes are free, so only
  edge-bleeders qualify; area alone never fills. Zero-rotation,
  stroke-free singletons land on the exact box; anything rotated,
  stroked, or multi-member takes the analytic shrink below.
- Canvas fit: a lone edge-bleeding non-cover image scales uniformly
  to the largest centered box inside the canvas (K9 forbids stretching it).
- Slot: every other unit scales uniformly into its slot (member-relative
  geometry preserved, image ratios exact) with the analytic shrink below.

Rotation-aware shrink: after centering a unit, each member's stored and
rotated-plus-stroke bounds (the K8 union, mirroring element_bounds) give
linear upper bounds on the uniform scale; the minimum over members, sides,
and both branches is exact in one pass because max-constraints decompose.
Bounds that fail even at scale zero (absurd fixed strokes) fall back to
MIN_UNIT_SCALE as a best effort for downstream repair to judge.

Text font size is searched per placed box with estimate_text_height: a
bounded ladder above and below the proportional seed, largest fitting rung
wins, floor 1.0 (K7). Growth rungs never exceed any source-larger text's
decided size (one descending-size pass with a running cap); the seed and
shrink rungs stay eligible so small boxes still fit-or-floor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from engine.internal_schema import Canvas, Design, Element
from engine.layout_edits import layout_edit
from engine.layout_search import baseline_proposal
from engine.layout_units import (
    LayoutUnit,
    build_layout_units,
    required_group_modes,
)
from engine.text_fit import estimate_text_height

__all__ = [
    "BAND_GAP_FRACTION",
    "COLUMN_GAP_FRACTION",
    "FAMILY_HORIZONTAL_FLOW",
    "FAMILY_SCALE_BASELINE",
    "FAMILY_VERTICAL_STACK",
    "FILL_EDGE_TOLERANCE_PX",
    "FLOW_MAX_SINGLE_BAND_UNITS",
    "FONT_GROWTH_CAP",
    "FONT_GROWTH_STEP",
    "FONT_GROWTH_STEPS",
    "FONT_SHRINK_STEP",
    "GENERATOR_VERSION",
    "MARGIN_FRACTION",
    "MIN_UNIT_SCALE",
    "SLOT_DIRECTION_HORIZONTAL",
    "SLOT_DIRECTION_VERTICAL",
    "SLOT_SAFETY_FACTOR",
    "STACK_MAX_SINGLE_COLUMN_UNITS",
    "Slot",
    "allocate_slots",
    "baseline_candidate",
    "horizontal_flow_candidates",
    "vertical_stack_candidates",
]

FAMILY_SCALE_BASELINE = "SCALE_BASELINE"
FAMILY_HORIZONTAL_FLOW = "HORIZONTAL_FLOW"
FAMILY_VERTICAL_STACK = "VERTICAL_STACK"
GENERATOR_VERSION = "s04_generator_v0.1"

SLOT_DIRECTION_HORIZONTAL = "horizontal"
SLOT_DIRECTION_VERTICAL = "vertical"

# Usable-rect margin as a fraction of each target axis (D.1.3).
MARGIN_FRACTION = 0.04
# A source box "touches" a canvas edge when within this many pixels of it
# (overhang counts as touching). Separates exact bleed (0px, every fixture
# background) from any real inset; area alone never qualifies for fill.
FILL_EDGE_TOLERANCE_PX = 1.0
# Above this many slotted units, flow splits into two bands / stack into two
# columns (D.1.3 names one or two bands; the threshold is ours).
FLOW_MAX_SINGLE_BAND_UNITS = 4
STACK_MAX_SINGLE_COLUMN_UNITS = 4
# Gap between the two bands/columns, as a fraction of the usable extent.
BAND_GAP_FRACTION = 0.05
COLUMN_GAP_FRACTION = 0.05
# Uniform shrink absorbing float rounding so placed bounds sit strictly
# inside the canvas (uniform: image ratios unaffected).
SLOT_SAFETY_FACTOR = 1.0 - 1e-9
# Best-effort scale when rotated bounds cannot fit at any scale (absurd
# fixed strokes); downstream validation decides the candidate.
MIN_UNIT_SCALE = 1e-6
# Font ladder around the proportional seed (D.1.4): bounded growth, shrink
# descent to the floor, largest fitting rung wins, floor 1.0.
FONT_GROWTH_STEP = 1.1
FONT_GROWTH_STEPS = 4
FONT_GROWTH_CAP = 1.5
FONT_SHRINK_STEP = 0.9


@dataclass(frozen=True, slots=True)
class Slot:
    """One tiled sub-rectangle of the usable target area, in target pixels."""

    x: float
    y: float
    width: float
    height: float


def allocate_slots(unit_count: int, target: Canvas, *,
                   direction: str) -> list[Slot]:
    """Tile the usable target rect into unit_count ordered slot rects.

    The single swappable slot policy: margin inset, one band (or column)
    up to the threshold, else two with the gap fraction between them.
    Slots tile their band exactly and come back in unit order.
    """
    if direction not in (SLOT_DIRECTION_HORIZONTAL, SLOT_DIRECTION_VERTICAL):
        raise ValueError(f"direction must be {SLOT_DIRECTION_HORIZONTAL!r} or "
                         f"{SLOT_DIRECTION_VERTICAL!r}")
    if unit_count < 0:
        raise ValueError("unit_count must be non-negative")
    if unit_count == 0:
        return []
    horizontal = direction == SLOT_DIRECTION_HORIZONTAL
    margin = MARGIN_FRACTION
    full_w = float(target.width)
    full_h = float(target.height)
    origin_x = full_w * margin
    origin_y = full_h * margin
    usable_w = full_w * (1 - 2 * margin)
    usable_h = full_h * (1 - 2 * margin)
    if horizontal:
        threshold = FLOW_MAX_SINGLE_BAND_UNITS
        gap_fraction = BAND_GAP_FRACTION
    else:
        threshold = STACK_MAX_SINGLE_COLUMN_UNITS
        gap_fraction = COLUMN_GAP_FRACTION
    if unit_count <= threshold:
        sizes = [unit_count]
    else:
        first = (unit_count + 1) // 2
        sizes = [first, unit_count - first]
    slots = []
    if horizontal:
        gap = usable_h * gap_fraction if len(sizes) == 2 else 0.0
        band_h = (usable_h - gap) / len(sizes)
        top = origin_y
        for size in sizes:
            slot_w = usable_w / size
            for position in range(size):
                slots.append(Slot(origin_x + position * slot_w, top,
                                  slot_w, band_h))
            top += band_h + gap
    else:
        gap = usable_w * gap_fraction if len(sizes) == 2 else 0.0
        column_w = (usable_w - gap) / len(sizes)
        left = origin_x
        for size in sizes:
            slot_h = usable_h / size
            for position in range(size):
                slots.append(Slot(left, origin_y + position * slot_h,
                                  column_w, slot_h))
            left += column_w + gap
    return slots


def _touches_every_edge(element: Element, canvas: Canvas) -> bool:
    """True when the stored source box reaches all four canvas edges."""
    tolerance = FILL_EDGE_TOLERANCE_PX
    width = float(canvas.width)
    height = float(canvas.height)
    left = float(element.x)
    top = float(element.y)
    return (left <= tolerance
            and top <= tolerance
            and left + float(element.width) >= width - tolerance
            and top + float(element.height) >= height - tolerance)


def _is_fill_member(element: Element, canvas: Canvas) -> bool:
    """Edge-bleeding cover images and shapes/text stretch to full canvas.

    Non-cover images are never fill members even when edge-bleeding:
    K9 forbids changing their ratio (they canvas-fit instead).
    """
    if not _touches_every_edge(element, canvas):
        return False
    if element.type == "image":
        return element.effective_value("fit") == "cover"
    return True


def _is_canvas_fit_unit(unit: LayoutUnit, canvas: Canvas) -> bool:
    """A lone edge-bleeding non-cover image: maximal uniform fit."""
    return (len(unit.members) == 1
            and unit.members[0].type == "image"
            and unit.members[0].effective_value("fit") != "cover"
            and _touches_every_edge(unit.members[0], canvas))


def _rotation_terms(element: Element) -> tuple[float, float, float]:
    """(abs cos, abs sin, stroke pad) mirroring element_bounds (K8)."""
    angle = math.radians(float(element.effective_value("rotation") % 360))
    cosine, sine = abs(math.cos(angle)), abs(math.sin(angle))
    pad = 0.0
    if element.type == "shape" and element.effective_value("stroke_color") is not None:
        pad = max(0.0, float(element.effective_value("stroke_width"))) / 2
        pad *= cosine + sine
    return cosine, sine, pad


def _max_fit_scale(specs: list[tuple[float, float, float, float, float, float, float]],
                   center_x: float, center_y: float,
                   canvas_w: float, canvas_h: float) -> float | None:
    """Largest uniform scale about the center keeping every bound in-canvas.

    Each spec is (dux, duy, stored_hw, stored_hh, rotated_hw, rotated_hh,
    pad): center offset from the shrink origin and half-extents, all at
    scale 1. Every side of each branch (stored and rotated) is one linear
    upper bound; the minimum is exact. None means infeasible at scale zero.
    """
    limit = math.inf
    for dux, duy, shw, shh, rhw, rhh, pad in specs:
        for half_w, half_h, fixed in ((shw, shh, 0.0), (rhw, rhh, pad)):
            for distance, coef in ((center_x - fixed, half_w - dux),
                                   (canvas_w - center_x - fixed, half_w + dux),
                                   (center_y - fixed, half_h - duy),
                                   (canvas_h - center_y - fixed, half_h + duy)):
                if distance < 0:
                    return None
                if coef > 0:
                    candidate = distance / coef
                    if candidate < limit:
                        limit = candidate
    return limit


def _member_specs(unit: LayoutUnit) -> list[tuple[float, float, float, float, float, float, float]]:
    """Bound specs of unit members relative to the unit-box center at scale 1."""
    box_x, box_y, box_w, box_h = unit.bbox
    middle_x = box_x + box_w / 2
    middle_y = box_y + box_h / 2
    specs = []
    for element in unit.members:
        width = float(element.width)
        height = float(element.height)
        center_x = float(element.x) + width / 2
        center_y = float(element.y) + height / 2
        cosine, sine, pad = _rotation_terms(element)
        specs.append((center_x - middle_x, center_y - middle_y,
                      width / 2, height / 2,
                      cosine * width / 2 + sine * height / 2,
                      sine * width / 2 + cosine * height / 2,
                      pad))
    return specs


def _slot_scale_for_box(box_w: float, box_h: float,
                        slot_w: float, slot_h: float) -> float:
    scales = []
    if box_w > 0:
        scales.append(slot_w / box_w)
    if box_h > 0:
        scales.append(slot_h / box_h)
    return min(scales) if scales else 1.0


def _place_slotted(unit: LayoutUnit, slot: Slot,
                   target: Canvas) -> tuple[dict[str, tuple[float, float, float, float]], float]:
    """Place a unit uniformly in its slot; returns member boxes and scale."""
    box_x, box_y, box_w, box_h = unit.bbox
    slot_scale = _slot_scale_for_box(box_w, box_h, slot.width, slot.height)
    center_x = slot.x + slot.width / 2
    center_y = slot.y + slot.height / 2
    room = _max_fit_scale(_member_specs(unit), center_x, center_y,
                          float(target.width), float(target.height))
    if room is None or room <= 0:
        scale = MIN_UNIT_SCALE
    else:
        scale = min(slot_scale, room) * SLOT_SAFETY_FACTOR
    placed_x = center_x - box_w * scale / 2
    placed_y = center_y - box_h * scale / 2
    boxes = {}
    for element, (fx, fy, fw, fh) in zip(unit.members, unit.relative_geometry):
        boxes[element.id] = (placed_x + fx * box_w * scale,
                             placed_y + fy * box_h * scale,
                             fw * box_w * scale, fh * box_h * scale)
    return boxes, scale


def _place_fill(unit: LayoutUnit, target: Canvas) -> dict[str, tuple[float, float, float, float]]:
    """Stretch a fill unit over the whole canvas, shrinking only if bounds demand."""
    canvas_w = float(target.width)
    canvas_h = float(target.height)
    if len(unit.members) == 1:
        only = unit.members[0]
        cosine, sine, pad = _rotation_terms(only)
        if cosine == 1.0 and sine == 0.0 and pad == 0.0:
            return {only.id: (0.0, 0.0,
                              canvas_w if only.width > 0 else 0.0,
                              canvas_h if only.height > 0 else 0.0)}
    specs = []
    for element, (fx, fy, fw, fh) in zip(unit.members, unit.relative_geometry):
        cosine, sine, pad = _rotation_terms(element)
        half_w = fw * canvas_w / 2
        half_h = fh * canvas_h / 2
        specs.append(((fx + fw / 2 - 0.5) * canvas_w,
                      (fy + fh / 2 - 0.5) * canvas_h,
                      half_w, half_h,
                      cosine * half_w + sine * half_h,
                      sine * half_w + cosine * half_h,
                      pad))
    room = _max_fit_scale(specs, canvas_w / 2, canvas_h / 2, canvas_w, canvas_h)
    if room is None or room <= 0:
        scale = MIN_UNIT_SCALE
    elif room < 1.0:
        scale = room * SLOT_SAFETY_FACTOR
    else:
        scale = 1.0
    boxes = {}
    for element, (fx, fy, fw, fh) in zip(unit.members, unit.relative_geometry):
        boxes[element.id] = (canvas_w / 2 + (fx * canvas_w - canvas_w / 2) * scale,
                             canvas_h / 2 + (fy * canvas_h - canvas_h / 2) * scale,
                             fw * canvas_w * scale, fh * canvas_h * scale)
    return boxes


def _search_font_size(content: str, box_w: float, box_h: float, seed: float,
                      line_height: float, cap: float) -> float:
    """Largest fitting rung with growth capped, else 1.0 (D.1.4).

    Only growth rungs above the seed honor the cap; the seed and the
    shrink descent stay eligible so small boxes fit-or-floor. The descent
    continues geometrically to the floor (it always terminates: each step
    multiplies by a factor below one), so a fitting size is never skipped
    for lack of ladder rungs.
    """
    rungs = []
    for step in range(FONT_GROWTH_STEPS, 0, -1):
        rung = seed * FONT_GROWTH_STEP ** step
        if rung <= seed * FONT_GROWTH_CAP and rung <= cap:
            rungs.append(rung)
    rungs.append(seed)
    rung = seed
    while True:
        rung *= FONT_SHRINK_STEP
        if rung < 1.0:
            break
        rungs.append(rung)
    rungs.append(1.0)
    for rung in dict.fromkeys(rungs):
        estimated = estimate_text_height(content, box_w, rung, line_height)
        if estimated is not None and estimated <= box_h:
            return rung
    return 1.0


def _decide_font_sizes(source: Design,
                       boxes: dict[str, tuple[float, float, float, float]],
                       scales: dict[str, float]) -> dict[str, float]:
    """Text sizes in descending source-size order under a running cap.

    Equal source sizes share one cap update, so source ties stay
    unconstrained; every decision feeds the cap for smaller texts.
    """
    ordered = sorted(
        ((element, position)
         for position, element in enumerate(source.elements)
         if element.type == "text"),
        key=lambda pair: (-float(pair[0].font_size), pair[1]),
    )
    fonts: dict[str, float] = {}
    cap = math.inf
    start = 0
    while start < len(ordered):
        size = float(ordered[start][0].font_size)
        stop = start
        while stop < len(ordered) and float(ordered[stop][0].font_size) == size:
            stop += 1
        decided = []
        for element, _ in ordered[start:stop]:
            placed = boxes[element.id]
            seed = max(1.0, float(element.font_size) * scales[element.id])
            choice = _search_font_size(
                element.content, placed[2], placed[3], seed,
                float(element.effective_value("line_height")), cap,
            )
            fonts[element.id] = choice
            decided.append(choice)
        cap = min(cap, min(decided))
        start = stop
    return fonts


def _order_key_flow(unit: LayoutUnit) -> tuple[float, float, int]:
    return (unit.bbox[1], unit.bbox[0], min(unit.source_indices))


def _order_key_stack(unit: LayoutUnit) -> tuple[float, float, int]:
    return (unit.bbox[0], unit.bbox[1], min(unit.source_indices))


def _family_candidates(source: Design, target: Canvas, *,
                       direction: str) -> list[dict[str, Any]]:
    """One Adobe dict per group mode, slotted along the family direction."""
    order = _order_key_flow if direction == SLOT_DIRECTION_HORIZONTAL else _order_key_stack
    canvas = source.source_canvas
    target_w = float(target.width)
    target_h = float(target.height)
    results = []
    for mode in required_group_modes(source):
        units = build_layout_units(source, group_mode=mode)
        fills: list[LayoutUnit] = []
        fits: list[LayoutUnit] = []
        slotted: list[LayoutUnit] = []
        for unit in units:
            if all(_is_fill_member(member, canvas) for member in unit.members):
                fills.append(unit)
            elif _is_canvas_fit_unit(unit, canvas):
                fits.append(unit)
            else:
                slotted.append(unit)
        ordered = sorted(slotted, key=order)
        slots = allocate_slots(len(ordered), target, direction=direction)
        boxes: dict[str, tuple[float, float, float, float]] = {}
        scales: dict[str, float] = {}
        for unit, slot in zip(ordered, slots):
            placed, scale = _place_slotted(unit, slot, target)
            boxes.update(placed)
            for element in unit.members:
                scales[element.id] = scale
        for unit in fills:
            boxes.update(_place_fill(unit, target))
            for element in unit.members:
                scales[element.id] = 1.0
        canvas_slot = Slot(0.0, 0.0, target_w, target_h)
        for unit in fits:
            placed, scale = _place_slotted(unit, canvas_slot, target)
            boxes.update(placed)
            for element in unit.members:
                scales[element.id] = scale
        fonts = _decide_font_sizes(source, boxes, scales)
        elements = []
        for original in source.to_adobe()["elements"]:
            placed = boxes[original["id"]]
            changes: dict[str, Any] = {
                "x": placed[0], "y": placed[1],
                "width": placed[2], "height": placed[3],
            }
            if original["type"] == "text":
                changes["font_size"] = fonts[original["id"]]
            elements.append(layout_edit(original, changes))
        results.append({"canvas": target.to_adobe(), "elements": elements})
    return results


def baseline_candidate(source: Design, target: Canvas) -> dict[str, Any]:
    """The SCALE_BASELINE batch member: the first proportional repair.

    Attempt 0 is the deliberate stretch probe; attempt 1 is the uniform
    whole-composition fit that the batch carries for feasibility.
    """
    return baseline_proposal(source, target, 1)


def horizontal_flow_candidates(source: Design, target: Canvas) -> list[dict[str, Any]]:
    """HORIZONTAL_FLOW: units ordered by source row, slotted left to right."""
    return _family_candidates(source, target, direction=SLOT_DIRECTION_HORIZONTAL)


def vertical_stack_candidates(source: Design, target: Canvas) -> list[dict[str, Any]]:
    """VERTICAL_STACK: units ordered by source column, slotted top to bottom."""
    return _family_candidates(source, target, direction=SLOT_DIRECTION_VERTICAL)
