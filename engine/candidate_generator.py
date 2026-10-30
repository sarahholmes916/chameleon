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
    LayoutUnit,
    box_iou,
    build_layout_units,
    contains_any,
    required_group_modes,
)
from engine.text_fit import estimate_text_height

__all__ = [
    "BAND_GAP_FRACTION",
    "COLUMN_GAP_FRACTION",
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
    "HERO_TWO_COLUMN_ASPECT",
    "MARGIN_FRACTION",
    "MIN_UNIT_SCALE",
    "OVERLAY_ATTACH_IOU_THRESHOLD",
    "OVERLAY_OPACITY_THRESHOLD",
    "PILL_ASPECT_TOLERANCE",
    "SLOT_DIRECTION_HORIZONTAL",
    "SLOT_DIRECTION_VERTICAL",
    "SLOT_SAFETY_FACTOR",
    "SMALL_ANCHOR_MAX_FRACTION",
    "SMALL_IMAGE_AREA_FRACTION",
    "STACK_MAX_SINGLE_COLUMN_UNITS",
    "Slot",
    "allocate_slots",
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
# Targets at least this wide keep one hero column (rows are already wide);
# squarer or taller targets wrap past the threshold into two.
HERO_TWO_COLUMN_ASPECT = 2.0
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


def _attach_overlays(units: list[LayoutUnit], canvas: Canvas) -> list[LayoutUnit]:
    """Merge each overlay into its max-IoU non-fill overlapped unit.

    Overlays attach in source order against the current unit list; full-bleed
    overlays stay independent so they fill downstream instead of dragging a
    canvas-sized box into a slot. Pill units are eligible targets like any
    other unit: the merge keeps their members together. IoU ties break by
    source index.
    """
    working = list(units)
    overlays = sorted((unit for unit in units if _is_overlay_unit(unit)),
                      key=lambda unit: min(unit.source_indices))
    for overlay in overlays:
        if not any(unit is overlay for unit in working):
            continue
        if _is_fill_unit(overlay, canvas):
            continue
        targets = [unit for unit in working
                   if unit is not overlay
                   and not _is_fill_unit(unit, canvas)
                   and _overlay_evidence(overlay, unit)]
        if not targets:
            continue
        best = max(targets, key=lambda unit: (
            box_iou(overlay.bbox, unit.bbox), -min(unit.source_indices)))
        pairs = sorted(zip(overlay.source_indices + best.source_indices,
                           overlay.members + best.members),
                       key=lambda pair: pair[0])
        merged = LayoutUnit.of([member for _, member in pairs],
                               [index for index, _ in pairs], canvas)
        working = [merged if unit is best else unit for unit in working]
        working = [unit for unit in working if unit is not overlay]
    return working


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
    full_w = float(target.width)
    full_h = float(target.height)
    margin = MARGIN_FRACTION
    origin_x = full_w * margin
    origin_y = full_h * margin
    usable_w = full_w * (1 - 2 * margin)
    usable_h = full_h * (1 - 2 * margin)
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


def _column_slots(column: Slot, heights: list[float],
                  target: Canvas) -> list[Slot]:
    """Row slots tiling the column, heights proportional to unit extents.

    Past the threshold on a narrow target the column wraps into two
    sub-columns (ceil-first membership); wide targets keep one column of
    already-wide rows. Slots come back in unit order.
    """
    count = len(heights)
    if count == 0:
        return []
    aspect = (float(target.width) / float(target.height)
              if target.height > 0 else math.inf)
    if count <= HERO_MAX_SINGLE_COLUMN_UNITS or aspect >= HERO_TWO_COLUMN_ASPECT:
        groups = [list(range(count))]
        spans = [(column.x, column.width)]
    else:
        first = (count + 1) // 2
        groups = [list(range(first)), list(range(first, count))]
        gap = column.width * COLUMN_GAP_FRACTION
        sub_w = (column.width - gap) / 2
        spans = [(column.x, sub_w), (column.x + sub_w + gap, sub_w)]
    slots: list[Slot | None] = [None] * count
    for (left, width), members in zip(spans, groups):
        total = sum(heights[index] for index in members)
        if total <= 0:
            rows = [column.height / len(members)] * len(members)
        else:
            rows = [column.height * heights[index] / total for index in members]
        top = column.y
        for index, height in zip(members, rows):
            slots[index] = Slot(left, top, width, height)
            top += height
    return [slot for slot in slots if slot is not None]


def _place_anchored(unit: LayoutUnit, target: Canvas, *,
                    top: bool, left: bool) -> tuple[dict[str, tuple[float, float, float, float]], float]:
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
        scale = min(fit, 1.0, room) * SLOT_SAFETY_FACTOR
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


def _hero_candidates(source: Design, target: Canvas, *,
                     mirror: bool) -> list[dict[str, Any]]:
    """Hero panel plus reading-ordered column, per group mode and variant.

    The largest-area non-small non-fill image unit heroes (small units are
    never dominant enough to hero; the largest slotted unit steps in when no
    image unit qualifies); small image units anchor to corners with top and
    bottom variants; the rest stack in reading order with similar pills
    separated. Fill and canvas-fit units keep their usual paths.
    """
    canvas = source.source_canvas
    canvas_area = float(canvas.width) * float(canvas.height)
    target_w = float(target.width)
    target_h = float(target.height)
    results = []
    for mode in required_group_modes(source):
        units = _attach_overlays(
            _merge_pills(build_layout_units(source, group_mode=mode), canvas),
            canvas)
        fills, fits, slotted = _partition_units(units, canvas)
        small_limit = SMALL_IMAGE_AREA_FRACTION * canvas_area
        image_units = [unit for unit in slotted
                       if _has_image(unit)
                       and unit.bbox[2] * unit.bbox[3] >= small_limit]
        pool = image_units or slotted
        if pool:
            hero = max(pool, key=lambda unit: (
                unit.bbox[2] * unit.bbox[3], -min(unit.source_indices)))
        else:
            hero = None
        rest = [unit for unit in slotted if unit is not hero]
        smalls = sorted(
            (unit for unit in rest
             if _has_image(unit)
             and unit.bbox[2] * unit.bbox[3] < small_limit),
            key=lambda unit: min(unit.source_indices))
        small_ids = {id(unit) for unit in smalls}
        column_units = [unit for unit in rest if id(unit) not in small_ids]
        ordered = _separate_pills(sorted(column_units, key=_order_key_flow))
        hero_slot, column_slot = _hero_rects(
            target, mirror=mirror, has_hero=hero is not None)
        row_slots = _column_slots(
            column_slot, [unit.bbox[3] for unit in ordered], target)
        variants = ((True, False) if smalls else (True,))
        for variant_top in variants:
            boxes: dict[str, tuple[float, float, float, float]] = {}
            scales: dict[str, float] = {}
            if hero is not None and hero_slot is not None:
                placed, scale = _place_slotted(hero, hero_slot, target)
                boxes.update(placed)
                for member in hero.members:
                    scales[member.id] = scale
            for unit, slot in zip(ordered, row_slots):
                placed, scale = _place_slotted(unit, slot, target)
                boxes.update(placed)
                for member in unit.members:
                    scales[member.id] = scale
            for unit in smalls:
                center_x = unit.bbox[0] + unit.bbox[2] / 2
                placed, scale = _place_anchored(
                    unit, target, top=variant_top,
                    left=center_x <= float(canvas.width) / 2)
                boxes.update(placed)
                for member in unit.members:
                    scales[member.id] = scale
            for unit in fills:
                boxes.update(_place_fill(unit, target))
                for member in unit.members:
                    scales[member.id] = 1.0
            canvas_slot = Slot(0.0, 0.0, target_w, target_h)
            for unit in fits:
                placed, scale = _place_slotted(unit, canvas_slot, target)
                boxes.update(placed)
                for member in unit.members:
                    scales[member.id] = scale
            fonts = _decide_font_sizes(source, boxes, scales)
            results.append(_emit_candidate(source, target, boxes, scales, fonts))
    return results


def hero_left_candidates(source: Design, target: Canvas) -> list[dict[str, Any]]:
    """HERO_LEFT: hero panel left, reading-ordered column right."""
    return _hero_candidates(source, target, mirror=False)


def hero_right_candidates(source: Design, target: Canvas) -> list[dict[str, Any]]:
    """HERO_RIGHT: mirror of HERO_LEFT (hero panel right, column left)."""
    return _hero_candidates(source, target, mirror=True)
