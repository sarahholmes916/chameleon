"""S04 candidate generator: proportional baseline + flow/stack/hero families.

Implements the discrete composition stage of spec D.1.3/D.1.4 (no continuous
refinement yet): layout units from engine.layout_units are assigned to slots
and emitted as Adobe dicts. Every emitted change passes through layout_edit,
so only K6 fields (x, y, width, height, text font_size) can change. The
generator is pure: it never calls validate_candidate and never raises on an
infeasible proposal (spec 5.4); validation and repair stay downstream. Same
inputs give byte-identical outputs; every ordering tie-breaks by source
index. Stdlib plus S02/S04 helpers only: no ML-framework imports.

The hero families place the largest non-fill image unit (by stored-box area)
in a side panel and the rest in a reading-ordered text column; small images
anchor to corners, low-opacity shapes ride their overlapped unit, and exact
2-member shape+text groups stay merged. HERO_LEFT heroes left, HERO_RIGHT
mirrors.

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
    ALIGNMENT_TOLERANCE,
    LayoutUnit,
    box_iou,
    build_layout_units,
    contains_any,
    required_group_modes,
)
from engine.text_fit import estimate_text_height

__all__ = [
    "BAND_GAP_FRACTION",
    "BLOCK_INNER_GAP_EM",
    "BLOCK_OUTER_GAP_EM",
    "COLUMN_GAP_FRACTION",
    "DETACHED_OVERLAY_MAX_FRACTION",
    "FAMILY_ASPECT_CUTOFF",
    "FAMILY_BANNER_ZONES",
    "FAMILY_HERO_LEFT",
    "FAMILY_HERO_RIGHT",
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
    "HERO_MAX_SINGLE_COLUMN_UNITS",
    "HERO_SIDE_FRACTION",
    "MARGIN_FRACTION",
    "MAX_PACK_PASSES",
    "MIN_UNIT_SCALE",
    "OVERLAY_ATTACH_IOU_THRESHOLD",
    "OVERLAY_OPACITY_THRESHOLD",
    "PILL_ASPECT_TOLERANCE",
    "PORTRAIT_ASPECT_CUTOFF",
    "SAFE_BOTTOM_MARGIN_FACTOR",
    "SLOT_DIRECTION_HORIZONTAL",
    "SLOT_DIRECTION_VERTICAL",
    "SLOT_SAFETY_FACTOR",
    "SMALL_ANCHOR_MAX_FRACTION",
    "SMALL_IMAGE_AREA_FRACTION",
    "STACK_MAX_SINGLE_COLUMN_UNITS",
    "TEXT_BLOCK_GAP_FACTOR",
    "TEXT_LABEL_HEIGHT_EM",
    "TEXT_ROW_PADDING_EM",
    "Slot",
    "allocate_slots",
    "banner_zones_candidates",
    "baseline_candidate",
    "hero_left_candidates",
    "hero_right_candidates",
    "horizontal_flow_candidates",
    "vertical_stack_candidates",
]

FAMILY_SCALE_BASELINE = "SCALE_BASELINE"
FAMILY_HORIZONTAL_FLOW = "HORIZONTAL_FLOW"
FAMILY_VERTICAL_STACK = "VERTICAL_STACK"
FAMILY_HERO_LEFT = "HERO_LEFT"
FAMILY_HERO_RIGHT = "HERO_RIGHT"
FAMILY_BANNER_ZONES = "BANNER_ZONES"
# The one wide/narrow regime boundary (log-symmetric 2.0): hero columns wrap
# below it, banner zones apply above it, and resize batch assembly skips
# VERTICAL_STACK above it / HORIZONTAL_FLOW below its reciprocal.
FAMILY_ASPECT_CUTOFF = 2.0
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
# Hero panel share of the usable width (D.1.2: dominant region plus the rest;
# the even split is ours -- no generic signal picks a lopsided share).
HERO_SIDE_FRACTION = 0.5
# Above this many hero-column units, the column wraps into two sub-columns
# (mirrors the flow/stack two-band rule; the threshold is ours).
HERO_MAX_SINGLE_COLUMN_UNITS = 4
# Bottom usable margin as a multiple of the side/top margin (optical bottom
# weight: the last row never sits on the margin line).
SAFE_BOTTOM_MARGIN_FACTOR = 1.5
# Targets with aspect below this are portrait (definitional taller-than-wide);
# hero smalls gain a top-center variant there.
PORTRAIT_ASPECT_CUTOFF = 1.0
# Two text units merge into one stacked block when their vertical gap is below
# this factor of the larger line-height: the midpoint between line-tiling
# (gap ~0, one paragraph) and paragraph breaks (~1 line).
TEXT_BLOCK_GAP_FACTOR = 0.5
# Packed text rows pad this many ems (at the decided size) above and below
# the estimated text height (quarter-em inset).
TEXT_ROW_PADDING_EM = 0.25
# Packed gaps as multiples of the larger adjacent decided font: inner rows of
# one block match the padding micro-grid; outer gaps are full-em breaks.
BLOCK_INNER_GAP_EM = 0.25
BLOCK_OUTER_GAP_EM = 1.0
# A singleton text unit reads as a label (swappable around text blocks) when
# its source height is at most this many ems: one line (~1.2em) plus slack,
# well below two lines.
TEXT_LABEL_HEIGHT_EM = 1.5
# An overlay with no image unit to ride stays independent, capped so its
# larger side fits this fraction of the target's shorter (binding) axis:
# subordinate decorations share the small-marks quarter-scale.
DETACHED_OVERLAY_MAX_FRACTION = 0.25
# Layout/room refinement passes for packed stacks (initial plus refinements);
# bounded and deterministic, the grid verifies every emission.
MAX_PACK_PASSES = 3
# Image-bearing units below this fraction of the source canvas area anchor
# to corners instead of taking column rows (generic smallness separating
# logos and badges from content images).
SMALL_IMAGE_AREA_FRACTION = 0.05
# Corner anchors fit inside this fraction of each target axis and never
# upscale past 1.0, so small units stay small on any target.
SMALL_ANCHOR_MAX_FRACTION = 0.25
# Shapes below this opacity are decorative overlays riding their overlapped
# unit; the midpoint separates washes from solids.
OVERLAY_OPACITY_THRESHOLD = 0.5
# An overlay attaches to the max-IoU non-fill unit when the unit-box IoU
# exceeds this (any genuine area overlap qualifies; edge grazes with no
# shared area stay below it).
OVERLAY_ATTACH_IOU_THRESHOLD = 0.02
# Two pill units read as near-identical when their relative aspect differs
# by at most this (same-component tolerance, not a fixture fit).
PILL_ASPECT_TOLERANCE = 0.10


@dataclass(frozen=True, slots=True)
class Slot:
    """One tiled sub-rectangle of the usable target area, in target pixels."""

    x: float
    y: float
    width: float
    height: float


def _usable_rect(target: Canvas) -> tuple[float, float, float, float]:
    """Usable rect: side/top margin plus the heavier optical bottom margin."""
    full_w = float(target.width)
    full_h = float(target.height)
    margin = MARGIN_FRACTION
    return (full_w * margin, full_h * margin,
            full_w * (1 - 2 * margin),
            full_h * (1 - margin - margin * SAFE_BOTTOM_MARGIN_FACTOR))


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
    origin_x, origin_y, usable_w, usable_h = _usable_rect(target)
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


def _member_specs_origin(unit: LayoutUnit, origin_x: float,
                         origin_y: float) -> list[tuple[float, float, float, float, float, float, float]]:
    """Bound specs of unit members relative to an explicit origin at scale 1."""
    specs = []
    for element in unit.members:
        width = float(element.width)
        height = float(element.height)
        center_x = float(element.x) + width / 2
        center_y = float(element.y) + height / 2
        cosine, sine, pad = _rotation_terms(element)
        specs.append((center_x - origin_x, center_y - origin_y,
                      width / 2, height / 2,
                      cosine * width / 2 + sine * height / 2,
                      sine * width / 2 + cosine * height / 2,
                      pad))
    return specs


def _member_specs(unit: LayoutUnit) -> list[tuple[float, float, float, float, float, float, float]]:
    """Bound specs of unit members relative to the unit-box center at scale 1."""
    box_x, box_y, box_w, box_h = unit.bbox
    return _member_specs_origin(unit, box_x + box_w / 2, box_y + box_h / 2)


def _slot_scale_for_box(box_w: float, box_h: float,
                        slot_w: float, slot_h: float) -> float:
    scales = []
    if box_w > 0:
        scales.append(slot_w / box_w)
    if box_h > 0:
        scales.append(slot_h / box_h)
    return min(scales) if scales else 1.0


def _place_slotted(unit: LayoutUnit, slot: Slot,
                   target: Canvas, *,
                   cap: float = math.inf) -> tuple[dict[str, tuple[float, float, float, float]], float]:
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
        scale = min(slot_scale, room, cap) * SLOT_SAFETY_FACTOR
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


def _partition_units(units: list[LayoutUnit],
                     canvas: Canvas) -> tuple[list[LayoutUnit], list[LayoutUnit], list[LayoutUnit]]:
    """Split units into fill, canvas-fit, and slotted placement paths."""
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
    return fills, fits, slotted


def _emit_candidate(source: Design, target: Canvas,
                    boxes: dict[str, tuple[float, float, float, float]],
                    scales: dict[str, float],
                    fonts: dict[str, float]) -> dict[str, Any]:
    """Adobe dict with every change routed through layout_edit (K6 only)."""
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
    return {"canvas": target.to_adobe(), "elements": elements}


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
        fills, fits, slotted = _partition_units(units, canvas)
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
        results.append(_emit_candidate(source, target, boxes, scales, fonts))
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


def _is_pill_unit(unit: LayoutUnit) -> bool:
    """An exact 2-member shape+text explicit group (badge/button signature)."""
    return (unit.group_id is not None
            and len(unit.members) == 2
            and {member.type for member in unit.members} == {"shape", "text"})


def _merge_pills(units: list[LayoutUnit], canvas: Canvas) -> list[LayoutUnit]:
    """Re-merge pill groups that split mode separated, in source order."""
    parts: dict[str, list[LayoutUnit]] = {}
    for unit in units:
        if unit.group_id is not None:
            parts.setdefault(unit.group_id, []).append(unit)
    repaired: dict[str, LayoutUnit] = {}
    for label, pieces in parts.items():
        members = [member for unit in pieces for member in unit.members]
        indices = [index for unit in pieces for index in unit.source_indices]
        if (len(pieces) > 1 and len(members) == 2
                and {member.type for member in members} == {"shape", "text"}):
            ordered = sorted(zip(indices, members), key=lambda pair: pair[0])
            repaired[label] = LayoutUnit.of(
                [member for _, member in ordered],
                [index for index, _ in ordered], canvas)
    if not repaired:
        return list(units)
    merged = []
    done: set[str] = set()
    for unit in units:
        label = unit.group_id
        if label in repaired:
            if label not in done:
                merged.append(repaired[label])
                done.add(label)
        else:
            merged.append(unit)
    return merged


def _unit_line_height(unit: LayoutUnit) -> float:
    """Largest member line-height in px (font size times multiplier)."""
    return max(float(member.font_size)
               * float(member.effective_value("line_height"))
               for member in unit.members)


def _units_aligned(first: LayoutUnit, second: LayoutUnit, canvas: Canvas) -> bool:
    """Left/center/right unit-box edge agreement within tolerance."""
    width = float(canvas.width)
    if width <= 0:
        return False
    first_x, _, first_w, _ = first.bbox
    second_x, _, second_w, _ = second.bbox
    left = abs(second_x - first_x) / width
    center = abs((second_x + second_w / 2) - (first_x + first_w / 2)) / width
    right = abs((second_x + second_w) - (first_x + first_w)) / width
    return min(left, center, right) <= ALIGNMENT_TOLERANCE


def _block_continues(first: LayoutUnit, second: LayoutUnit, canvas: Canvas) -> bool:
    """Second stacks directly under first: same column, aligned, close."""
    first_x, first_y, first_w, first_h = first.bbox
    second_x, second_y, second_w, second_h = second.bbox
    gap = second_y - (first_y + first_h)
    if gap < 0:
        return False
    if min(first_x + first_w, second_x + second_w) - max(first_x, second_x) <= 0:
        return False
    if not _units_aligned(first, second, canvas):
        return False
    limit = TEXT_BLOCK_GAP_FACTOR * max(_unit_line_height(first),
                                        _unit_line_height(second))
    return gap < limit


def _merge_text_blocks(units: list[LayoutUnit], canvas: Canvas
                       ) -> tuple[list[LayoutUnit], set[tuple[int, ...]]]:
    """Merge stacked ungrouped text units into blocks; keys mark merged ones.

    Candidates are all-text units without an explicit group, scanned in
    reading order; each run whose consecutive pairs share the column, agree
    on an edge, and sit within half a line becomes one block unit in source
    order. Explicit groups are never merged here.
    """
    mergeable = [unit for unit in units
                 if unit.group_id is None
                 and all(member.type == "text" for member in unit.members)]
    runs: list[list[LayoutUnit]] = []
    for unit in sorted(mergeable, key=_order_key_flow):
        if runs and _block_continues(runs[-1][-1], unit, canvas):
            runs[-1].append(unit)
        else:
            runs.append([unit])
    merged: dict[int, LayoutUnit] = {}
    block_keys: set[tuple[int, ...]] = set()
    for run in runs:
        if len(run) < 2:
            continue
        pairs = sorted(
            ((index, member) for unit in run
             for index, member in zip(unit.source_indices, unit.members)),
            key=lambda pair: pair[0])
        block = LayoutUnit.of([member for _, member in pairs],
                              [index for index, _ in pairs], canvas)
        block_keys.add(block.source_indices)
        for unit in run:
            merged[id(unit)] = block
    if not merged:
        return list(units), block_keys
    out = []
    done: set[int] = set()
    for unit in units:
        replacement = merged.get(id(unit))
        if replacement is None:
            out.append(unit)
        elif id(replacement) not in done:
            out.append(replacement)
            done.add(id(replacement))
    return out, block_keys


def _is_translucent(element: Element) -> bool:
    """Below the overlay opacity threshold (absent opacity reads 1.0)."""
    opacity = element.effective_value("opacity")
    return (isinstance(opacity, (int, float))
            and opacity < OVERLAY_OPACITY_THRESHOLD)


def _is_overlay_unit(unit: LayoutUnit) -> bool:
    """Every member a translucent shape: decorative wash, never content."""
    return all(member.type == "shape" and _is_translucent(member)
               for member in unit.members)


def _is_fill_unit(unit: LayoutUnit, canvas: Canvas) -> bool:
    """Every member edge-bleeds as fill (cover image, shape, or text)."""
    return all(_is_fill_member(member, canvas) for member in unit.members)


def _overlay_evidence(overlay: LayoutUnit, candidate: LayoutUnit) -> bool:
    """Containment either way, or genuine unit-box area overlap."""
    if any(contains_any(outer, inner)
           for outer in overlay.members for inner in candidate.members):
        return True
    return box_iou(overlay.bbox, candidate.bbox) > OVERLAY_ATTACH_IOU_THRESHOLD


def _attach_overlays(units: list[LayoutUnit], canvas: Canvas
                     ) -> list[tuple[LayoutUnit, LayoutUnit]]:
    """Pair each overlay with its max-IoU image unit; the rest stay independent.

    Targets must contain an image member and pass overlap evidence: pills,
    text, and structural shapes never receive overlays, so riders never wrap
    them. Full-bleed overlays stay independent so they fill downstream.
    Overlays attach in source order; IoU ties break by source index. Units
    are returned unmerged: riders follow their target's placement instead of
    expanding its box.
    """
    riders: list[tuple[LayoutUnit, LayoutUnit]] = []
    overlays = sorted((unit for unit in units if _is_overlay_unit(unit)),
                      key=lambda unit: min(unit.source_indices))
    for overlay in overlays:
        if _is_fill_unit(overlay, canvas):
            continue
        targets = [unit for unit in units
                   if unit is not overlay
                   and _has_image(unit)
                   and not _is_fill_unit(unit, canvas)
                   and _overlay_evidence(overlay, unit)]
        if not targets:
            continue
        best = max(targets, key=lambda unit: (
            box_iou(overlay.bbox, unit.bbox), -min(unit.source_indices)))
        riders.append((overlay, best))
    return riders


def _placed_unit_center(unit: LayoutUnit,
                        boxes: dict[str, tuple[float, float, float, float]]
                        ) -> tuple[float, float]:
    """Center of the unit's placed member boxes."""
    left = min(boxes[member.id][0] for member in unit.members)
    top = min(boxes[member.id][1] for member in unit.members)
    right = max(boxes[member.id][0] + boxes[member.id][2] for member in unit.members)
    bottom = max(boxes[member.id][1] + boxes[member.id][3] for member in unit.members)
    return (left + right) / 2, (top + bottom) / 2


def _rider_room(overlay: LayoutUnit, unit: LayoutUnit,
                placed_x: float, placed_y: float, target: Canvas) -> float | None:
    """Largest rider scale about the unit's placed center staying in-canvas.

    The anchor is the unit center (which maps to the fixed slot/row center),
    so the bound stays valid when the target rescales; member corners would
    drift under recentering.
    """
    box_x, box_y, box_w, box_h = unit.bbox
    specs = _member_specs_origin(
        overlay, float(box_x) + float(box_w) / 2, float(box_y) + float(box_h) / 2)
    return _max_fit_scale(specs, placed_x, placed_y,
                          float(target.width), float(target.height))


def _place_rider(overlay: LayoutUnit, unit: LayoutUnit,
                 placed_x: float, placed_y: float,
                 scale: float) -> dict[str, tuple[float, float, float, float]]:
    """Rider boxes under the target unit's uniform transform (offsets kept)."""
    box_x, box_y, box_w, box_h = (float(value) for value in unit.bbox)
    origin_x = box_x + box_w / 2
    origin_y = box_y + box_h / 2
    rider_x, rider_y, rider_w, rider_h = (float(value) for value in overlay.bbox)
    boxes = {}
    for element, (fx, fy, fw, fh) in zip(overlay.members, overlay.relative_geometry):
        boxes[element.id] = (
            placed_x + (rider_x + fx * rider_w - origin_x) * scale,
            placed_y + (rider_y + fy * rider_h - origin_y) * scale,
            fw * rider_w * scale, fh * rider_h * scale)
    return boxes


def _pill_similar(first: LayoutUnit, second: LayoutUnit) -> bool:
    """Both pills and near-identical aspect (relative difference)."""
    if not (_is_pill_unit(first) and _is_pill_unit(second)):
        return False
    first_w, first_h = first.bbox[2], first.bbox[3]
    second_w, second_h = second.bbox[2], second.bbox[3]
    if first_w <= 0 or first_h <= 0 or second_w <= 0 or second_h <= 0:
        return False
    aspects = sorted((first_w / first_h, second_w / second_h))
    return (aspects[1] - aspects[0]) / aspects[1] <= PILL_ASPECT_TOLERANCE


def _separate_pills(order: list[LayoutUnit]) -> list[LayoutUnit]:
    """One left-to-right pass separating similar adjacent pill pairs.

    On a similar adjacent pair, the second swaps with its right neighbor
    (when one exists) and the scan skips ahead, so every unit moves at most
    one slot and no unit passes its nearest source-order neighbor. A similar
    pair at the tail has no right neighbor and stays adjacent.
    """
    order = list(order)
    index = 0
    while index + 1 < len(order):
        if (_pill_similar(order[index], order[index + 1])
                and index + 2 < len(order)):
            order[index + 1], order[index + 2] = order[index + 2], order[index + 1]
            index += 3
        else:
            index += 1
    return order


def _hero_rects(target: Canvas, *, mirror: bool,
                has_hero: bool) -> tuple[Slot | None, Slot]:
    """Hero panel plus text column tiling the usable rect (mirrored on ask).

    Without a hero the column takes the whole usable rect.
    """
    origin_x, origin_y, usable_w, usable_h = _usable_rect(target)
    if not has_hero:
        return None, Slot(origin_x, origin_y, usable_w, usable_h)
    gap = usable_w * COLUMN_GAP_FRACTION
    side_w = (usable_w - gap) * HERO_SIDE_FRACTION
    if mirror:
        hero = Slot(origin_x + usable_w - side_w, origin_y, side_w, usable_h)
        column = Slot(origin_x, origin_y, usable_w - side_w - gap, usable_h)
    else:
        hero = Slot(origin_x, origin_y, side_w, usable_h)
        column = Slot(origin_x + side_w + gap, origin_y,
                      usable_w - side_w - gap, usable_h)
    return hero, column


@dataclass(frozen=True, slots=True)
class _PackedRow:
    """One packed row: a block member text, or a whole unit placed uniformly."""

    unit: LayoutUnit
    member: Element | None


def _is_upright_text(member: Element) -> bool:
    """Text with no effective rotation: reflowable in packed rows."""
    return (member.type == "text"
            and float(member.effective_value("rotation")) % 360 == 0)


def _rows_for_units(units: list[LayoutUnit],
                    block_keys: set[tuple[int, ...]]) -> list[_PackedRow]:
    """Member rows for reflowable texts, unit rows for locked compositions.

    Merged blocks and ungrouped singleton texts reflow one row per member
    (top order) when every text is upright; explicit groups, pills, images,
    shapes, overlays, and any rotated text keep their internal geometry in
    a single uniform row.
    """
    rows = []
    for unit in units:
        reflowable = (
            unit.source_indices in block_keys
            or (unit.group_id is None and len(unit.members) == 1
                and unit.members[0].type == "text")
        ) and all(_is_upright_text(member) for member in unit.members
                  if member.type == "text")
        if reflowable and any(member.type == "text" for member in unit.members):
            members = sorted((member for member in unit.members
                              if member.type == "text"),
                             key=lambda item: (item.y, item.x))
            rows.extend(_PackedRow(unit, member) for member in members)
        else:
            rows.append(_PackedRow(unit, None))
    return rows


def _row_font(row: _PackedRow, fonts: dict[str, float]) -> float:
    """Largest decided font among the row's texts, else 0.0."""
    if row.member is not None:
        return fonts[row.member.id]
    return max((fonts[member.id] for member in row.unit.members
                if member.type == "text"),
               default=0.0)


def _row_nominal_scale(row: _PackedRow, width: float, target: Canvas) -> float:
    """Width-driven row scale before rider caps (pill/overlay caps included)."""
    unit = row.unit
    unit_w = float(unit.bbox[2])
    if row.member is not None:
        member_w = float(row.member.width)
        return width / member_w if member_w > 0 else 1.0
    if _is_pill_unit(unit):
        return min(1.0, width / unit_w) if unit_w > 0 else 1.0
    if _is_overlay_unit(unit):
        nominal = width / unit_w if unit_w > 0 else 1.0
        longest = max(unit_w, float(unit.bbox[3]))
        if longest > 0:
            cap = (DETACHED_OVERLAY_MAX_FRACTION
                   * min(float(target.width), float(target.height)) / longest)
            nominal = min(nominal, cap)
        return nominal
    return width / unit_w if unit_w > 0 else 1.0


def _pack_container(rows: list[_PackedRow], rect: Slot,
                    fonts: dict[str, float], caps: dict[int, float],
                    target: Canvas) -> tuple[list[Slot], list[float]]:
    """Packed rows in rect: text rows estimate-sized, rest uniform, centered.

    Text rows take the full width with height from estimate_text_height plus
    padding; pills render at natural size at most; independent overlays are
    capped to the detached fraction; images and shapes width-fit. Inner gaps
    separate block siblings, outer gaps separate blocks and units, and the
    stack reserves a rhythm pad at the bottom, clamping uniformly when the
    column is short. Returns row slots and scales in row order.
    """
    if not rows:
        return [], []
    col_w = float(rect.width)
    scales = []
    for row in rows:
        nominal = _row_nominal_scale(row, col_w, target)
        scales.append(min(nominal, caps.get(id(row.unit), math.inf)))
    heights = []
    for row, scale in zip(rows, scales):
        if row.member is not None:
            member = row.member
            font = fonts[member.id]
            estimated = estimate_text_height(
                member.content, col_w, font,
                float(member.effective_value("line_height")))
            heights.append(float(estimated or 0) + 2 * TEXT_ROW_PADDING_EM * font)
        else:
            heights.append(float(row.unit.bbox[3]) * scale)
    row_fonts = [_row_font(row, fonts) for row in rows]
    gaps = []
    for first, second, font_first, font_second in zip(rows, rows[1:],
                                                      row_fonts, row_fonts[1:]):
        factor = BLOCK_INNER_GAP_EM if first.unit is second.unit else BLOCK_OUTER_GAP_EM
        gaps.append(factor * max(font_first, font_second))
    stack_max = max(row_fonts, default=0.0)
    last_font = row_fonts[-1] if row_fonts[-1] > 0 else stack_max
    bottom_pad = BLOCK_OUTER_GAP_EM * last_font
    total = sum(heights) + sum(gaps)
    avail = max(0.0, float(rect.height) - bottom_pad)
    if total > avail and total > 0:
        clamp = avail / total
    else:
        clamp = 1.0
    heights = [height * clamp for height in heights]
    gaps = [gap * clamp for gap in gaps]
    scales = [scale * clamp for scale in scales]
    slots = []
    cursor = float(rect.y) + max(0.0, avail - sum(heights) - sum(gaps)) / 2
    for row, height, scale, index in zip(rows, heights, scales, range(len(rows))):
        if row.member is not None:
            slots.append(Slot(float(rect.x), cursor, col_w, height))
        else:
            width = float(row.unit.bbox[2]) * scale
            slots.append(Slot(float(rect.x) + (col_w - width) / 2, cursor,
                              width, height))
        cursor += height
        if index < len(gaps):
            cursor += gaps[index]
    return slots, scales


def _place_anchored(unit: LayoutUnit, target: Canvas, *,
                    top: bool, left: bool,
                    cap: float = math.inf) -> tuple[dict[str, tuple[float, float, float, float]], float]:
    """Pin a small unit's box corner to a canvas corner, shrinking to fit.

    The unit fits inside the anchor fraction of each axis and never upscales
    past 1.0; the analytic shrink about the pinned corner keeps rotated and
    stroked bounds in-canvas the same way centered slots do. The pin sits one
    safety factor inside the true corner (sub-pixel): an exact pin would land
    member edges on the canvas boundary where float addition drifts past it.
    """
    canvas_w = float(target.width)
    canvas_h = float(target.height)
    box_x, box_y, box_w, box_h = unit.bbox
    origin_x = box_x if left else box_x + box_w
    origin_y = box_y if top else box_y + box_h
    inset_x = canvas_w * (1 - SLOT_SAFETY_FACTOR)
    inset_y = canvas_h * (1 - SLOT_SAFETY_FACTOR)
    corner_x = inset_x if left else canvas_w - inset_x
    corner_y = inset_y if top else canvas_h - inset_y
    fit = _slot_scale_for_box(box_w, box_h,
                              canvas_w * SMALL_ANCHOR_MAX_FRACTION,
                              canvas_h * SMALL_ANCHOR_MAX_FRACTION)
    room = _max_fit_scale(_member_specs_origin(unit, origin_x, origin_y),
                          corner_x, corner_y, canvas_w, canvas_h)
    if room is None or room <= 0:
        scale = MIN_UNIT_SCALE
    else:
        scale = min(fit, 1.0, room, cap) * SLOT_SAFETY_FACTOR
    boxes = {}
    for element, (fx, fy, fw, fh) in zip(unit.members, unit.relative_geometry):
        boxes[element.id] = (
            corner_x + (box_x + fx * box_w - origin_x) * scale,
            corner_y + (box_y + fy * box_h - origin_y) * scale,
            fw * box_w * scale, fh * box_h * scale)
    return boxes, scale


def _has_image(unit: LayoutUnit) -> bool:
    """True when any member is an image (hero and small signals)."""
    return any(member.type == "image" for member in unit.members)


def _assembly_units(source: Design, group_mode: str
                    ) -> tuple[list[LayoutUnit], set[tuple[int, ...]],
                               list[tuple[LayoutUnit, LayoutUnit]]]:
    """Units for hero/zones: pills and text blocks merged, overlays attached."""
    canvas = source.source_canvas
    merged = _merge_pills(
        list(build_layout_units(source, group_mode=group_mode)), canvas)
    units, block_keys = _merge_text_blocks(merged, canvas)
    return units, block_keys, _attach_overlays(units, canvas)


def _is_label_unit(unit: LayoutUnit) -> bool:
    """Pill, or a singleton text short enough to read as a label."""
    if _is_pill_unit(unit):
        return True
    if len(unit.members) != 1 or unit.members[0].type != "text":
        return False
    only = unit.members[0]
    return float(only.height) <= TEXT_LABEL_HEIGHT_EM * float(only.font_size)


def _order_variants(order: list[LayoutUnit],
                    block_keys: set[tuple[int, ...]]) -> list[list[LayoutUnit]]:
    """Base order plus one single-swap candidate per block/label adjacency."""
    variants = [list(order)]
    for index in range(len(order) - 1):
        first, second = order[index], order[index + 1]
        first_block = first.source_indices in block_keys
        second_block = second.source_indices in block_keys
        if ((first_block and _is_label_unit(second))
                or (second_block and _is_label_unit(first))):
            swapped = list(order)
            swapped[index], swapped[index + 1] = swapped[index + 1], swapped[index]
            variants.append(swapped)
    return variants


def _distinct_orders(orders: list[list[LayoutUnit]]) -> list[list[LayoutUnit]]:
    """Drop orders the anti pass converged back to an earlier sequence."""
    seen: set[tuple[int, ...]] = set()
    distinct = []
    for order in orders:
        key = tuple(id(unit) for unit in order)
        if key not in seen:
            seen.add(key)
            distinct.append(order)
    return distinct


def _split_subcolumns(ordered: list[LayoutUnit],
                      target: Canvas) -> list[list[LayoutUnit]]:
    """Ceil-first halves past the threshold on narrow targets, else one group."""
    aspect = (float(target.width) / float(target.height)
              if target.height > 0 else math.inf)
    if len(ordered) <= HERO_MAX_SINGLE_COLUMN_UNITS or aspect >= FAMILY_ASPECT_CUTOFF:
        return [list(ordered)]
    first = (len(ordered) + 1) // 2
    return [list(ordered[:first]), list(ordered[first:])]


def _subcolumn_rects(column: Slot, count: int) -> list[Slot]:
    """One rect, or two with the column gap between them."""
    if count <= 1:
        return [column]
    gap = column.width * COLUMN_GAP_FRACTION
    sub_w = (column.width - gap) / 2
    return [Slot(column.x, column.y, sub_w, column.height),
            Slot(column.x + sub_w + gap, column.y, sub_w, column.height)]


def _assign_small_corners(smalls: list[LayoutUnit], prefs: list[bool]
                          ) -> tuple[list[tuple[LayoutUnit, bool]], list[LayoutUnit]]:
    """Preferred corner, flipped side when taken, column overflow when both fill."""
    taken: set[bool] = set()
    anchored = []
    overflow = []
    for unit, prefer in zip(smalls, prefs):
        side = prefer
        if side in taken:
            side = not side
        if side in taken:
            overflow.append(unit)
        else:
            taken.add(side)
            anchored.append((unit, side))
    return anchored, overflow


def _small_variants(target: Canvas) -> tuple[str, ...]:
    """Top/bottom corner rows, plus top-center on portrait targets."""
    aspect = (float(target.width) / float(target.height)
              if target.height > 0 else math.inf)
    if aspect < PORTRAIT_ASPECT_CUTOFF:
        return ("top", "bottom", "center")
    return ("top", "bottom")


def _place_centered(smalls: list[LayoutUnit], target: Canvas,
                    caps: dict[int, float]) -> tuple[dict[str, tuple[float, float, float, float]], dict[str, float]]:
    """Portrait variant: smalls stacked touching, centered, from the top inset."""
    canvas_w = float(target.width)
    canvas_h = float(target.height)
    anchor_w = canvas_w * SMALL_ANCHOR_MAX_FRACTION
    anchor_h = canvas_h * SMALL_ANCHOR_MAX_FRACTION
    inset_y = canvas_h * (1 - SLOT_SAFETY_FACTOR)
    scales = []
    for unit in smalls:
        fit = _slot_scale_for_box(float(unit.bbox[2]), float(unit.bbox[3]),
                                  anchor_w, anchor_h)
        scales.append(min(fit, 1.0, caps.get(id(unit), math.inf)) * SLOT_SAFETY_FACTOR)
    for _ in range(2):
        cursor = inset_y
        shrunk = False
        restacked = []
        for unit, scale in zip(smalls, scales):
            box_h = float(unit.bbox[3]) * scale
            room = _max_fit_scale(_member_specs(unit), canvas_w / 2,
                                  cursor + box_h / 2, canvas_w, canvas_h)
            if room is None or room <= 0:
                restacked.append(MIN_UNIT_SCALE)
                shrunk = True
            elif room < scale:
                restacked.append(room * SLOT_SAFETY_FACTOR)
                shrunk = True
            else:
                restacked.append(scale)
            cursor += box_h
        scales = restacked
        if not shrunk:
            break
    boxes: dict[str, tuple[float, float, float, float]] = {}
    out_scales: dict[str, float] = {}
    cursor = inset_y
    for unit, scale in zip(smalls, scales):
        box_w = float(unit.bbox[2]) * scale
        box_h = float(unit.bbox[3]) * scale
        left = (canvas_w - box_w) / 2
        for element, (fx, fy, fw, fh) in zip(unit.members, unit.relative_geometry):
            boxes[element.id] = (left + fx * box_w, cursor + fy * box_h,
                                 fw * box_w, fh * box_h)
            out_scales[element.id] = scale
        cursor += box_h
    return boxes, out_scales


def _run_pack_passes(
    source: Design, target: Canvas, *,
    pack_jobs: list[tuple[list[_PackedRow], Slot]],
    fit_jobs: list[tuple[LayoutUnit, Slot, float]],
    anchor_jobs: list[tuple[LayoutUnit, bool, bool]],
    center_units: list[LayoutUnit],
    fill_units: list[LayoutUnit],
    riders: list[tuple[LayoutUnit, LayoutUnit]],
) -> tuple[dict[str, tuple[float, float, float, float]], dict[str, float], dict[str, float]]:
    """Bounded layout/room loop; returns (boxes, scales, fonts).

    Each pass decides fonts in provisional tall boxes, lays out packed rows
    with clamping, places fits/anchors/fills, and caps rider-bound and
    rotation-bound targets; the loop stops early when caps stabilize. Final
    fonts are decided in the final boxes.
    """
    canvas_w = float(target.width)
    canvas_h = float(target.height)
    caps: dict[int, float] = {}
    boxes: dict[str, tuple[float, float, float, float]] = {}
    scales: dict[str, float] = {}
    for _ in range(MAX_PACK_PASSES):
        tall: dict[str, tuple[float, float, float, float]] = {}
        seeds: dict[str, float] = {}
        for rows, rect in pack_jobs:
            for row in rows:
                scale = min(_row_nominal_scale(row, float(rect.width), target),
                            caps.get(id(row.unit), math.inf))
                if row.member is not None:
                    tall[row.member.id] = (rect.x, rect.y, rect.width, rect.height)
                    seeds[row.member.id] = scale
                else:
                    box_w = float(row.unit.bbox[2])
                    box_h = float(row.unit.bbox[3])
                    for element, (fx, fy, fw, fh) in zip(row.unit.members,
                                                         row.unit.relative_geometry):
                        if element.type == "text":
                            tall[element.id] = (
                                float(rect.x) + fx * box_w * scale,
                                float(rect.y) + fy * box_h * scale,
                                fw * box_w * scale, fh * box_h * scale)
                            seeds[element.id] = scale
        for unit, slot, base_cap in fit_jobs:
            placed, scale = _place_slotted(
                unit, slot, target,
                cap=min(base_cap, caps.get(id(unit), math.inf)))
            for element in unit.members:
                if element.type == "text":
                    tall[element.id] = placed[element.id]
                    seeds[element.id] = scale
        for unit, top, left in anchor_jobs:
            placed, scale = _place_anchored(
                unit, target, top=top, left=left,
                cap=caps.get(id(unit), math.inf))
            for element in unit.members:
                if element.type == "text":
                    tall[element.id] = placed[element.id]
                    seeds[element.id] = scale
        if center_units:
            placed, center_scales = _place_centered(center_units, target, caps)
            for unit in center_units:
                for element in unit.members:
                    if element.type == "text":
                        tall[element.id] = placed[element.id]
                        seeds[element.id] = center_scales[element.id]
        for unit in fill_units:
            placed = _place_fill(unit, target)
            for element in unit.members:
                if element.type == "text":
                    tall[element.id] = placed[element.id]
                    seeds[element.id] = 1.0
        fonts = _decide_font_sizes(source, tall, seeds)
        boxes = {}
        scales = {}
        row_slots: list[tuple[_PackedRow, Slot, float]] = []
        for rows, rect in pack_jobs:
            slots, row_scales = _pack_container(rows, rect, fonts, caps, target)
            for row, slot, scale in zip(rows, slots, row_scales):
                row_slots.append((row, slot, scale))
                if row.member is not None:
                    boxes[row.member.id] = (slot.x, slot.y, slot.width, slot.height)
                    scales[row.member.id] = scale
                else:
                    box_w = float(row.unit.bbox[2])
                    box_h = float(row.unit.bbox[3])
                    for element, (fx, fy, fw, fh) in zip(row.unit.members,
                                                         row.unit.relative_geometry):
                        boxes[element.id] = (
                            slot.x + fx * box_w * scale,
                            slot.y + fy * box_h * scale,
                            fw * box_w * scale, fh * box_h * scale)
                        scales[element.id] = scale
        for unit, slot, base_cap in fit_jobs:
            placed, scale = _place_slotted(
                unit, slot, target,
                cap=min(base_cap, caps.get(id(unit), math.inf)))
            boxes.update(placed)
            for element in unit.members:
                scales[element.id] = scale
        for unit, top, left in anchor_jobs:
            placed, scale = _place_anchored(
                unit, target, top=top, left=left,
                cap=caps.get(id(unit), math.inf))
            boxes.update(placed)
            for element in unit.members:
                scales[element.id] = scale
        if center_units:
            placed, center_scales = _place_centered(center_units, target, caps)
            boxes.update(placed)
            scales.update(center_scales)
        for unit in fill_units:
            boxes.update(_place_fill(unit, target))
            for element in unit.members:
                scales[element.id] = 1.0
        new_caps: dict[int, float] = {}
        for row, slot, _ in row_slots:
            if row.member is not None:
                continue
            specs = _member_specs(row.unit)
            room = _max_fit_scale(specs, slot.x + slot.width / 2,
                                  slot.y + slot.height / 2, canvas_w, canvas_h)
            if room is None or room <= 0:
                cap = MIN_UNIT_SCALE
            else:
                cap = room * SLOT_SAFETY_FACTOR
            key = id(row.unit)
            new_caps[key] = min(new_caps.get(key, math.inf), cap)
        for overlay, overlay_target in riders:
            center_x, center_y = _placed_unit_center(overlay_target, boxes)
            room = _rider_room(overlay, overlay_target, center_x, center_y, target)
            if room is None or room <= 0:
                cap = MIN_UNIT_SCALE
            else:
                cap = room * SLOT_SAFETY_FACTOR
            key = id(overlay_target)
            new_caps[key] = min(new_caps.get(key, math.inf), cap)
        if new_caps == caps:
            break
        caps = new_caps
    rider_target_scale: dict[int, float] = {}
    for _, overlay_target in riders:
        key = id(overlay_target)
        if key not in rider_target_scale:
            rider_target_scale[key] = scales[overlay_target.members[0].id]
    for overlay, overlay_target in riders:
        center_x, center_y = _placed_unit_center(overlay_target, boxes)
        scale = rider_target_scale[id(overlay_target)]
        boxes.update(_place_rider(overlay, overlay_target, center_x, center_y, scale))
        for element in overlay.members:
            scales[element.id] = scale
    fonts = _decide_font_sizes(source, boxes, scales)
    return boxes, scales, fonts


def _small_prefs(smalls: list[LayoutUnit], canvas: Canvas, *,
                 mirror: bool, has_hero: bool) -> list[bool]:
    """Preferred corner sides: opposite the hero, else the source side."""
    prefs = []
    for unit in smalls:
        if has_hero:
            prefs.append(bool(mirror))
        else:
            center_x = unit.bbox[0] + unit.bbox[2] / 2
            prefs.append(center_x <= float(canvas.width) / 2)
    return prefs


def _hero_candidates(source: Design, target: Canvas, *,
                     mirror: bool) -> list[dict[str, Any]]:
    """Hero panel plus packed reading-ordered column, per mode/order/variant.

    The largest-area non-small non-fill image unit heroes (the largest
    slotted unit steps in when no image unit qualifies); smalls anchor
    opposite the hero with top/bottom rows (plus top-center on portrait
    targets); the rest pack in reading order with similar pills separated
    and label/block orders varied. Fill and canvas-fit units keep paths.
    """
    canvas = source.source_canvas
    canvas_area = float(canvas.width) * float(canvas.height)
    target_w = float(target.width)
    target_h = float(target.height)
    small_limit = SMALL_IMAGE_AREA_FRACTION * canvas_area
    results = []
    for mode in required_group_modes(source):
        units, block_keys, riders = _assembly_units(source, mode)
        fills, fits, slotted = _partition_units(units, canvas)
        rider_ids = {id(overlay) for overlay, _ in riders}
        free = [unit for unit in slotted if id(unit) not in rider_ids]
        image_units = [unit for unit in free
                       if _has_image(unit)
                       and unit.bbox[2] * unit.bbox[3] >= small_limit]
        pool = image_units or free
        if pool:
            hero = max(pool, key=lambda unit: (
                unit.bbox[2] * unit.bbox[3], -min(unit.source_indices)))
        else:
            hero = None
        has_hero = hero is not None
        rest = [unit for unit in free if unit is not hero]
        smalls = sorted(
            (unit for unit in rest
             if _has_image(unit)
             and unit.bbox[2] * unit.bbox[3] < small_limit),
            key=lambda unit: min(unit.source_indices))
        anchored, _ = _assign_small_corners(
            smalls, _small_prefs(smalls, canvas, mirror=mirror, has_hero=has_hero))
        anchored_ids = {id(unit) for unit, _ in anchored}
        column_base = [unit for unit in rest if id(unit) not in anchored_ids]
        base = sorted(column_base, key=_order_key_flow)
        orders = _distinct_orders(
            [_separate_pills(variant)
             for variant in _order_variants(base, block_keys)])
        hero_slot, column_slot = _hero_rects(
            target, mirror=mirror, has_hero=has_hero)
        canvas_slot = Slot(0.0, 0.0, target_w, target_h)
        fit_jobs = ([(hero, hero_slot, math.inf)]
                    if hero is not None and hero_slot is not None else [])
        fit_jobs += [(unit, canvas_slot, math.inf) for unit in fits]
        variants = _small_variants(target) if smalls else (None,)
        for order in orders:
            groups = _split_subcolumns(order, target)
            rects = _subcolumn_rects(column_slot, len(groups))
            pack_jobs = [(_rows_for_units(group, block_keys), rect)
                         for group, rect in zip(groups, rects)]
            for variant in variants:
                if variant == "center":
                    anchor_jobs: list[tuple[LayoutUnit, bool, bool]] = []
                    center_units = [unit for unit, _ in anchored]
                elif variant is None:
                    anchor_jobs = []
                    center_units = []
                else:
                    anchor_jobs = [(unit, variant == "top", side)
                                   for unit, side in anchored]
                    center_units = []
                boxes, scales, fonts = _run_pack_passes(
                    source, target, pack_jobs=pack_jobs, fit_jobs=fit_jobs,
                    anchor_jobs=anchor_jobs, center_units=center_units,
                    fill_units=fills, riders=riders)
                results.append(_emit_candidate(source, target, boxes, scales, fonts))
    return results


def _zone_rects(ordered: list[LayoutUnit], target: Canvas) -> list[Slot]:
    """Band zones in order, widths proportional to source unit widths."""
    origin_x, origin_y, usable_w, usable_h = _usable_rect(target)
    count = len(ordered)
    if count == 0:
        return []
    gap = usable_w * COLUMN_GAP_FRACTION
    total = sum(float(unit.bbox[2]) for unit in ordered)
    net = usable_w - gap * (count - 1)
    if total > 0:
        widths = [net * float(unit.bbox[2]) / total for unit in ordered]
    else:
        widths = [net / count] * count
    rects = []
    left = origin_x
    for width in widths:
        rects.append(Slot(left, origin_y, width, usable_h))
        left += width + gap
    return rects


def banner_zones_candidates(source: Design, target: Canvas) -> list[dict[str, Any]]:
    """BANNER_ZONES: one band of content-sized zones in reading order.

    Every slotted non-small unit takes a zone sized by its source width with
    packed rows inside; smalls anchor (source sides, no hero to oppose);
    label/block orders vary like hero columns. Resize applies this family on
    targets wider than FAMILY_ASPECT_CUTOFF.
    """
    canvas = source.source_canvas
    canvas_area = float(canvas.width) * float(canvas.height)
    target_w = float(target.width)
    target_h = float(target.height)
    small_limit = SMALL_IMAGE_AREA_FRACTION * canvas_area
    results = []
    for mode in required_group_modes(source):
        units, block_keys, riders = _assembly_units(source, mode)
        fills, fits, slotted = _partition_units(units, canvas)
        rider_ids = {id(overlay) for overlay, _ in riders}
        free = [unit for unit in slotted if id(unit) not in rider_ids]
        smalls = sorted(
            (unit for unit in free
             if _has_image(unit)
             and unit.bbox[2] * unit.bbox[3] < small_limit),
            key=lambda unit: min(unit.source_indices))
        anchored, _ = _assign_small_corners(
            smalls, _small_prefs(smalls, canvas, mirror=False, has_hero=False))
        anchored_ids = {id(unit) for unit, _ in anchored}
        zone_base = [unit for unit in free if id(unit) not in anchored_ids]
        base = sorted(zone_base, key=_order_key_flow)
        orders = _distinct_orders(
            [_separate_pills(variant)
             for variant in _order_variants(base, block_keys)])
        canvas_slot = Slot(0.0, 0.0, target_w, target_h)
        fit_jobs = [(unit, canvas_slot, math.inf) for unit in fits]
        variants = _small_variants(target) if smalls else (None,)
        for order in orders:
            rects = _zone_rects(order, target)
            pack_jobs = [(_rows_for_units([unit], block_keys), rect)
                         for unit, rect in zip(order, rects)]
            for variant in variants:
                if variant == "center":
                    anchor_jobs: list[tuple[LayoutUnit, bool, bool]] = []
                    center_units = [unit for unit, _ in anchored]
                elif variant is None:
                    anchor_jobs = []
                    center_units = []
                else:
                    anchor_jobs = [(unit, variant == "top", side)
                                   for unit, side in anchored]
                    center_units = []
                boxes, scales, fonts = _run_pack_passes(
                    source, target, pack_jobs=pack_jobs, fit_jobs=fit_jobs,
                    anchor_jobs=anchor_jobs, center_units=center_units,
                    fill_units=fills, riders=riders)
                results.append(_emit_candidate(source, target, boxes, scales, fonts))
    return results


def hero_left_candidates(source: Design, target: Canvas) -> list[dict[str, Any]]:
    """HERO_LEFT: hero panel left, reading-ordered column right."""
    return _hero_candidates(source, target, mirror=False)


def hero_right_candidates(source: Design, target: Canvas) -> list[dict[str, Any]]:
    """HERO_RIGHT: mirror of HERO_LEFT (hero panel right, column left)."""
    return _hero_candidates(source, target, mirror=True)
