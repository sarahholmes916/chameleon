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
    "FONT_RATIO_TOLERANCE",
    "FONT_SHRINK_STEP",
    "GENERATOR_VERSION",
    "HERO_FLOOR_FRACTION",
    "HERO_MAX_SINGLE_COLUMN_UNITS",
    "HERO_SIDE_FRACTION",
    "MARGIN_FRACTION",
    "MARK_INSET_FRACTION",
    "MAX_PACK_PASSES",
    "MAX_RIDER_FIT_PASSES",
    "MIN_UNIT_SCALE",
    "OVERLAY_OPACITY_THRESHOLD",
    "PILL_ASPECT_TOLERANCE",
    "PILL_LABEL_PADDING_FRACTION",
    "PORTRAIT_ASPECT_CUTOFF",
    "RESERVE_DUST_MARGIN_PX",
    "SAFE_BOTTOM_MARGIN_FACTOR",
    "SLOT_DIRECTION_HORIZONTAL",
    "SLOT_DIRECTION_VERTICAL",
    "SLOT_SAFETY_FACTOR",
    "SMALL_ANCHOR_MAX_FRACTION",
    "SMALL_IMAGE_AREA_FRACTION",
    "STACK_MAX_SINGLE_COLUMN_UNITS",
    "TEXT_BLOCK_GAP_FACTOR",
    "TEXT_LABEL_HEIGHT_EM",
    "TYPE_SCALE_TOLERANCE",
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
# Measurement slack for exact source-ratio caps (tests only; the generator's
# own caps are exact, so verification keeps the full margin for float
# reassociation): far below one ladder rung (10%), never a visible step.
FONT_RATIO_TOLERANCE = 1e-9
# Measurement slack for exact geometric proportion caps (tests only; the
# generator's own caps are exact): float-reassociation dust only.
TYPE_SCALE_TOLERANCE = 1e-9
# Inset of corner-mark reservation trims past the mark edge: pack-cursor float
# accumulation (~1e-7 observed) must not bridge an exact-touch trim boundary
# on clamped stacks; 1e-3 px is invisible with a thousandfold margin.
RESERVE_DUST_MARGIN_PX = 1e-3
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
# Packed gaps as multiples of the larger adjacent decided font: inner rows of
# one block sit tight at a quarter em; outer gaps are full-em breaks.
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
# Center-clamp/size refinement passes when fitting one rider (fixtures settle
# in at most three; bounded and deterministic regardless).
MAX_RIDER_FIT_PASSES = 8
# Image-bearing units below this fraction of the source canvas area anchor
# to corners instead of taking column rows (generic smallness separating
# logos and badges from content images).
SMALL_IMAGE_AREA_FRACTION = 0.05
# Corner anchors fit inside this fraction of each target axis and never
# upscale past 1.0, so small units stay small on any target.
SMALL_ANCHOR_MAX_FRACTION = 0.25
# Corner-mark inset, equal pixels on both axes as a fraction of the shorter
# target side: one margin module (the MARGIN_FRACTION unit) on the binding
# axis, so marks share the layout's edge rhythm with corner symmetry.
MARK_INSET_FRACTION = 0.04
# Shapes below this opacity are decorative overlays riding their overlapped
# unit; the midpoint separates washes from solids.
OVERLAY_OPACITY_THRESHOLD = 0.5
# A hero unit's smaller placed side spans at least this fraction of the
# target's shorter side wherever slot-fit reaches it (dominance floor for
# heroes; narrow fallback heroes whose slot cannot reach it are exempt).
# The floor holds because riders cap themselves instead of shrinking targets.
HERO_FLOOR_FRACTION = 0.6
# Two pill units read as near-identical when their relative aspect differs
# by at most this (same-component tolerance, not a fixture fit).
PILL_ASPECT_TOLERANCE = 0.10
# Per-side inset of the label ink box inside its pill shape, as a fraction of
# each shape axis: scale-invariant minimum air so label ink never touches the
# pill edge at any target size.
PILL_LABEL_PADDING_FRACTION = 0.1


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
    unconstrained; every decision feeds the cap for smaller texts. Past the
    largest group, each text is additionally capped at its exact source-size
    ratio to the largest text times the largest decided size, so no text can
    outgrow its source proportion (the seed honors the ratio cap too; the
    rank cap keeps its growth-only semantics).
    """
    ordered = sorted(
        ((element, position)
         for position, element in enumerate(source.elements)
         if element.type == "text"),
        key=lambda pair: (-float(pair[0].font_size), pair[1]),
    )
    fonts: dict[str, float] = {}
    cap = math.inf
    largest_size = float(ordered[0][0].font_size) if ordered else 0.0
    largest_decided: float | None = None
    start = 0
    while start < len(ordered):
        size = float(ordered[start][0].font_size)
        stop = start
        while stop < len(ordered) and float(ordered[stop][0].font_size) == size:
            stop += 1
        group_cap = cap
        ratio_cap: float | None = None
        if largest_decided is not None and largest_size > 0:
            ratio_cap = (size / largest_size) * largest_decided
            group_cap = min(group_cap, ratio_cap)
        decided = []
        for element, _ in ordered[start:stop]:
            placed = boxes[element.id]
            seed = max(1.0, float(element.font_size) * scales[element.id])
            if ratio_cap is not None:
                seed = min(seed, max(ratio_cap, 1.0))
            choice = _search_font_size(
                element.content, placed[2], placed[3], seed,
                float(element.effective_value("line_height")), group_cap,
            )
            fonts[element.id] = choice
            decided.append(choice)
        if largest_decided is None:
            largest_decided = min(decided)
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
        chrome_cap, image_cap = _type_scale_caps(source, boxes)
        if chrome_cap < math.inf or image_cap < math.inf:
            for unit, slot in zip(ordered, slots):
                if _has_non_text(unit):
                    bound = (image_cap if _has_image(unit) else chrome_cap)
                    placed, scale = _place_slotted(
                        unit, slot, target, cap=bound)
                    boxes.update(placed)
                    for element in unit.members:
                        scales[element.id] = scale
        fonts = _decide_font_sizes(source, boxes, scales)
        _center_pill_labels(
            [unit for unit in list(ordered) + fits if _is_pill_unit(unit)],
            boxes, fonts)
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


def _center_pill_labels(pill_units: list[LayoutUnit],
                        boxes: dict[str, tuple[float, float, float, float]],
                        fonts: dict[str, float]) -> None:
    """Center each pill label in its shape; fit its font with padding.

    The label keeps its size and moves to the shape center on both axes when
    it fits inside; its font only shrinks, to fit both the label box and the
    padded shape inset. Hierarchy caps survive (shrinks only), and riders
    stay clear (the label never leaves the shape avoid-box they cleared).
    """
    for unit in pill_units:
        shape = next(member for member in unit.members
                     if member.type == "shape")
        label = next(member for member in unit.members
                     if member.type == "text")
        shape_box = boxes[shape.id]
        label_box = boxes[label.id]
        shape_w, shape_h = shape_box[2], shape_box[3]
        label_w, label_h = label_box[2], label_box[3]
        if label_w <= shape_w and label_h <= shape_h:
            boxes[label.id] = (
                shape_box[0] + (shape_w - label_w) / 2,
                shape_box[1] + (shape_h - label_h) / 2,
                label_w, label_h)
        inset = 1 - 2 * PILL_LABEL_PADDING_FRACTION
        fit_w = min(label_w, shape_w * inset)
        fit_h = min(label_h, shape_h * inset)
        if fit_w > 0 and fit_h > 0:
            seed = fonts[label.id]
            fonts[label.id] = _search_font_size(
                label.content, fit_w, fit_h, seed,
                float(label.effective_value("line_height")), seed)


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
    """Containment either way, or any positive unit-box area overlap."""
    if any(contains_any(outer, inner)
           for outer in overlay.members for inner in candidate.members):
        return True
    return box_iou(overlay.bbox, candidate.bbox) > 0


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


def _rider_half_extents(overlay: LayoutUnit) -> tuple[float, float]:
    """Max rotated member extents about the overlay unit's center."""
    box_x, box_y, box_w, box_h = (float(value) for value in overlay.bbox)
    middle_x = box_x + box_w / 2
    middle_y = box_y + box_h / 2
    half_x = 0.0
    half_y = 0.0
    for member in overlay.members:
        width = float(member.width)
        height = float(member.height)
        cosine, sine, pad = _rotation_terms(member)
        reach_x = cosine * width / 2 + sine * height / 2 + pad
        reach_y = sine * width / 2 + cosine * height / 2 + pad
        half_x = max(half_x, abs(float(member.x) + width / 2 - middle_x) + reach_x)
        half_y = max(half_y, abs(float(member.y) + height / 2 - middle_y) + reach_y)
    return half_x, half_y


def _fit_rider(overlay: LayoutUnit, hint_x: float, hint_y: float,
               target_scale: float,
               avoid: list[tuple[float, float, float, float]],
               target: Canvas) -> tuple[float, float, float]:
    """Rider center and scale: mapped offset, clamped, clearing content.

    The rider keeps its source center-offset scaled by the target's own
    scale; the center clamps minimally into the canvas and the size caps so
    the rotated bounds stay in-canvas (dust-proof margins) and disjoint from
    every avoided box. Bounded refinement, deterministic; a center trapped
    inside an avoided box degrades to a minimal valid mark.
    """
    half_x, half_y = _rider_half_extents(overlay)
    canvas_w = float(target.width)
    canvas_h = float(target.height)
    if half_x <= 0 and half_y <= 0:
        return (min(max(hint_x, 0.0), canvas_w),
                min(max(hint_y, 0.0), canvas_h), target_scale)
    margin_w = canvas_w * (1 - SLOT_SAFETY_FACTOR)
    margin_h = canvas_h * (1 - SLOT_SAFETY_FACTOR)
    scale = target_scale
    center_x, center_y = hint_x, hint_y
    for _ in range(MAX_RIDER_FIT_PASSES):
        low_x = half_x * scale + margin_w
        high_x = canvas_w - half_x * scale - margin_w
        low_y = half_y * scale + margin_h
        high_y = canvas_h - half_y * scale - margin_h
        if low_x > high_x or low_y > high_y:
            fits = [scale]
            if half_x > 0:
                fits.append((canvas_w - 2 * margin_w) / (2 * half_x))
            if half_y > 0:
                fits.append((canvas_h - 2 * margin_h) / (2 * half_y))
            scale = min(fits)
            if scale <= 0:
                break
            continue
        center_x = min(max(hint_x, low_x), high_x)
        center_y = min(max(hint_y, low_y), high_y)
        bound = scale
        for left, top, width, height in avoid:
            if half_x > 0:
                axis_x = (abs(center_x - (left + width / 2)) - width / 2) / half_x
            else:
                axis_x = math.inf if abs(center_x - (left + width / 2)) >= width / 2 else -math.inf
            if half_y > 0:
                axis_y = (abs(center_y - (top + height / 2)) - height / 2) / half_y
            else:
                axis_y = math.inf if abs(center_y - (top + height / 2)) >= height / 2 else -math.inf
            bound = min(bound, max(axis_x, axis_y))
        if bound >= scale:
            break
        scale = bound
        if scale <= 0:
            break
    if scale <= 0:
        scale = MIN_UNIT_SCALE
        center_x = min(max(hint_x, margin_w), canvas_w - margin_w)
        center_y = min(max(hint_y, margin_h), canvas_h - margin_h)
    return center_x, center_y, scale * SLOT_SAFETY_FACTOR


def _place_rider(overlay: LayoutUnit, center_x: float, center_y: float,
                 scale: float) -> dict[str, tuple[float, float, float, float]]:
    """Rider member boxes centered uniformly (ratios exact)."""
    box_x, box_y, box_w, box_h = (float(value) for value in overlay.bbox)
    middle_x = box_x + box_w / 2
    middle_y = box_y + box_h / 2
    boxes = {}
    for element in overlay.members:
        width = float(element.width)
        height = float(element.height)
        member_x = float(element.x) + width / 2
        member_y = float(element.y) + height / 2
        placed_w = width * scale
        placed_h = height * scale
        boxes[element.id] = (
            center_x + (member_x - middle_x) * scale - placed_w / 2,
            center_y + (member_y - middle_y) * scale - placed_h / 2,
            placed_w, placed_h)
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

    Text rows take the full width with height exactly estimate_text_height
    (ink-tight, so box gaps are visual gaps); pills render at natural size at
    most; independent overlays are capped to the detached fraction; images and
    shapes width-fit. Inner gaps separate block siblings, outer gaps separate
    blocks and units, all exact and never clamped; the stack centers exactly
    in the rect inset by the edge floors (the bottom margin plus an inner em
    in the end rows' sizes at each end), shrinking rows — never gaps, floors,
    or marks — when short. Returns row slots and scales in row order.
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
            estimated = estimate_text_height(
                member.content, col_w, fonts[member.id],
                float(member.effective_value("line_height")))
            heights.append(float(estimated or 0))
        else:
            heights.append(float(row.unit.bbox[3]) * scale)
    row_fonts = [_row_font(row, fonts) for row in rows]
    gaps = []
    for first, second, font_first, font_second in zip(rows, rows[1:],
                                                      row_fonts, row_fonts[1:]):
        factor = BLOCK_INNER_GAP_EM if first.unit is second.unit else BLOCK_OUTER_GAP_EM
        gaps.append(factor * max(font_first, font_second))
    edge = float(target.height) * MARGIN_FRACTION * SAFE_BOTTOM_MARGIN_FACTOR
    space_top = max(float(rect.y),
                    edge + BLOCK_INNER_GAP_EM * row_fonts[0])
    space_bottom = min(float(rect.y + rect.height),
                       float(target.height) - edge
                       - BLOCK_INNER_GAP_EM * row_fonts[-1])
    space = max(0.0, space_bottom - space_top)
    total_gaps = sum(gaps)
    total_rows = sum(heights)
    if total_rows > 0 and total_rows + total_gaps > space:
        factor = max(0.0, (space - total_gaps) / total_rows)
    else:
        factor = 1.0
    heights = [height * factor for height in heights]
    scales = [scale * factor for scale in scales]
    slots = []
    cursor = space_top + max(0.0, space - sum(heights) - total_gaps) / 2
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
    mark inset inside the true corner, equal pixels on both axes.
    """
    canvas_w = float(target.width)
    canvas_h = float(target.height)
    box_x, box_y, box_w, box_h = unit.bbox
    origin_x = box_x if left else box_x + box_w
    origin_y = box_y if top else box_y + box_h
    inset = min(canvas_w, canvas_h) * MARK_INSET_FRACTION
    corner_x = inset if left else canvas_w - inset
    corner_y = inset if top else canvas_h - inset
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


def _has_non_text(unit: LayoutUnit) -> bool:
    """True when any member is not text (the type scale cap applies)."""
    return any(member.type != "text" for member in unit.members)


def _type_scale_caps(source: Design,
                     boxes: dict[str, tuple[float, float, float, float]]
                     ) -> tuple[float, float]:
    """Largest-text decided box scales; non-text packed units scale past never.

    The largest text is the maximum font size with first source index breaking
    ties. Returns (chrome cap, image cap): text-chrome units (pills, shapes)
    cap at the smaller of the decided box width and height scales, keeping
    both source proportions to the headline box; image units cap at the width
    scale only (width presence, not a text-box height coupling that would
    forbid picture-scale heroes). Fit units span their slots by regime
    instead, and fills, riders, and anchored smalls keep their own regimes.
    Missing boxes or non-positive source sides lift a cap (infinity).
    """
    texts = [(position, element) for position, element in enumerate(source.elements)
             if element.type == "text"]
    if not texts:
        return (math.inf, math.inf)
    _, largest = max(texts, key=lambda pair: (float(pair[1].font_size),
                                              -pair[0]))
    placed = boxes.get(largest.id)
    if placed is None:
        return (math.inf, math.inf)
    ratios = []
    if float(largest.width) > 0 and placed[2] > 0:
        ratios.append(placed[2] / float(largest.width))
    width_cap = min(ratios) if ratios else math.inf
    if float(largest.height) > 0 and placed[3] > 0:
        ratios.append(placed[3] / float(largest.height))
    if not ratios:
        return (math.inf, math.inf)
    return (min(ratios), width_cap)


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


def _lay_pack_jobs(pack_jobs: list[tuple[list[_PackedRow], Slot]],
                   fonts: dict[str, float], caps: dict[int, float],
                   target: Canvas
                   ) -> tuple[list[tuple[_PackedRow, Slot, float]],
                              dict[str, tuple[float, float, float, float]],
                              dict[str, float]]:
    """Pack-job rows to slots; returns (row slots, boxes, scales)."""
    row_slots: list[tuple[_PackedRow, Slot, float]] = []
    boxes: dict[str, tuple[float, float, float, float]] = {}
    scales: dict[str, float] = {}
    for rows, rect in pack_jobs:
        slots, row_scales = _pack_container(rows, rect, fonts, caps, target)
        for row, slot, scale in zip(rows, slots, row_scales):
            row_slots.append((row, slot, scale))
            if row.member is not None:
                boxes[row.member.id] = (slot.x, slot.y, slot.width,
                                        slot.height)
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
    return row_slots, boxes, scales


def _refresh_type_caps(source: Design,
                       pack_jobs: list[tuple[list[_PackedRow], Slot]],
                       boxes: dict[str, tuple[float, float, float, float]],
                       fonts: dict[str, float],
                       caps: dict[int, float]) -> None:
    """Re-merge type caps with the headline height at its final estimate.

    Loop-box headline heights use provisional fonts, but the relay lays text
    rows at estimate(final fonts); the height ratio is recomputed from that
    estimate so the relay's pills cap against the shipped headline box (the
    relay's row factor scales headline and pills together within one rect).
    Widths are slot-fixed and read from boxes. Falls back to loop boxes when
    the largest text is not a packed member row.
    """
    texts = [(position, element) for position, element in enumerate(source.elements)
             if element.type == "text"]
    if not texts:
        return
    _, largest = max(texts, key=lambda pair: (float(pair[1].font_size),
                                              -pair[0]))
    placed = boxes.get(largest.id)
    if placed is None:
        return
    head_w = placed[2]
    head_h = placed[3]
    for rows, rect in pack_jobs:
        for row in rows:
            if row.member is not None and row.member.id == largest.id:
                estimated = estimate_text_height(
                    largest.content, float(rect.width), fonts[largest.id],
                    float(largest.effective_value("line_height")))
                head_h = float(estimated or 0)
    ratios = []
    if float(largest.width) > 0 and head_w > 0:
        ratios.append(head_w / float(largest.width))
    width_cap = min(ratios) if ratios else math.inf
    if float(largest.height) > 0 and head_h > 0:
        ratios.append(head_h / float(largest.height))
    if not ratios:
        return
    chrome_cap, image_cap = min(ratios), width_cap
    if chrome_cap < math.inf or image_cap < math.inf:
        for rows, _ in pack_jobs:
            for row in rows:
                if row.member is None and _has_non_text(row.unit):
                    bound = (image_cap if _has_image(row.unit)
                             else chrome_cap)
                    key = id(row.unit)
                    caps[key] = min(caps.get(key, math.inf), bound)


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
    with clamping, places fits/anchors/fills, and caps rotation-bound rows;
    the loop stops early when caps stabilize. Non-text packed units cap at
    the largest text's decided box scales (chrome both-axis, images width;
    fit units span their slots by regime; fills, riders, and anchored smalls
    keep their own regimes). Riders never shrink targets:
    after the final layout each rider keeps its scaled center-offset with its
    own size capped to stay in-canvas and clear of text and pill boxes.
    Final fonts are decided in the final boxes then pack jobs relay once with
    them, so shipped gaps match shipped sizes before riders place.
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
        row_slots, boxes, scales = _lay_pack_jobs(
            pack_jobs, fonts, caps, target)
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
        chrome_cap, image_cap = _type_scale_caps(source, boxes)
        if chrome_cap < math.inf or image_cap < math.inf:
            for rows, _ in pack_jobs:
                for row in rows:
                    if row.member is None and _has_non_text(row.unit):
                        key = id(row.unit)
                        bound = (image_cap if _has_image(row.unit)
                                 else chrome_cap)
                        new_caps[key] = min(new_caps.get(key, math.inf),
                                            bound)
        if new_caps == caps:
            break
        caps = new_caps
    fill_ids = {member.id for unit in fill_units for member in unit.members}
    avoid_ids = {element.id for element in source.elements
                 if element.type == "text"} - fill_ids
    for rows, _ in pack_jobs:
        for row in rows:
            if _is_pill_unit(row.unit):
                avoid_ids.update(member.id for member in row.unit.members)
    for unit, _, _ in fit_jobs:
        if _is_pill_unit(unit):
            avoid_ids.update(member.id for member in unit.members)
    fonts = _decide_font_sizes(source, boxes, scales)
    _refresh_type_caps(source, pack_jobs, boxes, fonts, caps)
    _, pack_boxes, pack_scales = _lay_pack_jobs(
        pack_jobs, fonts, caps, target)
    boxes.update(pack_boxes)
    scales.update(pack_scales)
    for overlay, overlay_target in riders:
        target_x, target_y = _placed_unit_center(overlay_target, boxes)
        scale = scales[overlay_target.members[0].id]
        unit_x, unit_y, unit_w, unit_h = (float(value) for value in overlay_target.bbox)
        rider_x, rider_y, rider_w, rider_h = (float(value) for value in overlay.bbox)
        hint_x = target_x + ((rider_x + rider_w / 2) - (unit_x + unit_w / 2)) * scale
        hint_y = target_y + ((rider_y + rider_h / 2) - (unit_y + unit_h / 2)) * scale
        avoid = [boxes[element_id] for element_id in sorted(avoid_ids)]
        center_x, center_y, rider_scale = _fit_rider(
            overlay, hint_x, hint_y, scale, avoid, target)
        boxes.update(_place_rider(overlay, center_x, center_y, rider_scale))
        for element in overlay.members:
            scales[element.id] = rider_scale
    pill_units: dict[int, LayoutUnit] = {}
    for rows, _ in pack_jobs:
        for row in rows:
            if _is_pill_unit(row.unit):
                pill_units[id(row.unit)] = row.unit
    for unit, _, _ in fit_jobs:
        if _is_pill_unit(unit):
            pill_units[id(unit)] = unit
    _center_pill_labels(list(pill_units.values()), boxes, fonts)
    return boxes, scales, fonts


def _unit_outer_box(placed: dict[str, tuple[float, float, float, float]],
                    unit: LayoutUnit) -> tuple[float, float, float, float]:
    """Bounding box over a unit's placed member boxes."""
    left = min(placed[member.id][0] for member in unit.members)
    top = min(placed[member.id][1] for member in unit.members)
    right = max(placed[member.id][0] + placed[member.id][2]
                for member in unit.members)
    bottom = max(placed[member.id][1] + placed[member.id][3]
                 for member in unit.members)
    return (left, top, right - left, bottom - top)


def _reserve_marks(rects: list[Slot],
                   anchor_boxes: list[tuple[float, float, float, float]]
                   ) -> list[Slot]:
    """Trim rect tops/bottoms past corner marks overlapping their x-range.

    A mark centered above a rect's middle pushes the top down past the mark,
    otherwise the bottom rises past it; reservation is spatial (the corner
    belongs to the mark), never negotiated per row. A mark covering a rect
    leaves a zero-height sliver the validator judges.
    """
    reserved = []
    for rect in rects:
        top = float(rect.y)
        bottom = float(rect.y + rect.height)
        middle = float(rect.y + rect.height / 2)
        for left, mark_top, width, height in anchor_boxes:
            if left + width <= rect.x or left >= rect.x + rect.width:
                continue
            mark_bottom = mark_top + height
            if mark_bottom <= top or mark_top >= bottom:
                continue
            if mark_top + height / 2 < middle:
                top = max(top, mark_bottom + RESERVE_DUST_MARGIN_PX)
            else:
                bottom = min(bottom, mark_top - RESERVE_DUST_MARGIN_PX)
        if bottom < top:
            bottom = top
        reserved.append(Slot(rect.x, top, rect.width, bottom - top))
    return reserved


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


def _anchor_reservations(anchor_jobs: list[tuple[LayoutUnit, bool, bool]],
                         target: Canvas) -> list[tuple[float, float, float, float]]:
    """Outer boxes of corner-anchored smalls (cap-free, hence final-exact)."""
    reserved = []
    for unit, top, left in anchor_jobs:
        placed, _ = _place_anchored(unit, target, top=top, left=left)
        reserved.append(_unit_outer_box(placed, unit))
    return reserved


def _hero_candidates(source: Design, target: Canvas, *,
                     mirror: bool) -> list[dict[str, Any]]:
    """Hero panel plus packed reading-ordered column, per mode/order/variant.

    The largest-area non-small non-fill image unit heroes (the largest
    slotted unit steps in when no image unit qualifies); smalls anchor
    opposite the hero with top/bottom rows (plus top-center on portrait
    targets); the rest pack in reading order with similar pills separated
    and label/block orders varied. Fill and canvas-fit units keep paths.
    Anchored smalls reserve their corner boxes; packed rects trim past them.
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
        # The hero takes full slot-fit: riders cap themselves instead of
        # shrinking it, which is what keeps HERO_FLOOR_FRACTION attainable.
        fit_jobs = ([(hero, hero_slot, math.inf)]
                    if hero is not None and hero_slot is not None else [])
        fit_jobs += [(unit, canvas_slot, math.inf) for unit in fits]
        variants = _small_variants(target) if smalls else (None,)
        for order in orders:
            groups = _split_subcolumns(order, target)
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
                rects = _reserve_marks(
                    _subcolumn_rects(column_slot, len(groups)),
                    _anchor_reservations(anchor_jobs, target))
                pack_jobs = [(_rows_for_units(group, block_keys), rect)
                             for group, rect in zip(groups, rects)]
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
    """BANNER_ZONES: one band of content-sized zones, brand first, CTA last.

    The smallest-area image unit opens the band (brand zone), the pill
    holding the last source-order text closes it, and content zones keep
    source order between them; every zone is sized by its source width with
    packed rows inside. Remaining smalls anchor (source sides, no hero to
    oppose); label/block orders vary like hero columns. Resize applies this
    family on targets wider than FAMILY_ASPECT_CUTOFF.
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
        image_units = [unit for unit in free if _has_image(unit)]
        if image_units:
            brand = min(image_units, key=lambda unit: (
                unit.bbox[2] * unit.bbox[3], min(unit.source_indices)))
        else:
            brand = None
        last_text_index = -1
        for unit in free:
            for member, index in zip(unit.members, unit.source_indices):
                if member.type == "text":
                    last_text_index = max(last_text_index, index)
        call_to_action = None
        if last_text_index >= 0:
            for unit in free:
                if (_is_pill_unit(unit)
                        and last_text_index in unit.source_indices):
                    call_to_action = unit
        smalls = sorted(
            (unit for unit in free
             if _has_image(unit)
             and unit.bbox[2] * unit.bbox[3] < small_limit
             and unit is not brand),
            key=lambda unit: min(unit.source_indices))
        anchored, _ = _assign_small_corners(
            smalls, _small_prefs(smalls, canvas, mirror=False, has_hero=False))
        anchored_ids = {id(unit) for unit, _ in anchored}
        zone_base = [unit for unit in free if id(unit) not in anchored_ids]
        base = sorted(zone_base, key=lambda unit: (
            0 if unit is brand else 2 if unit is call_to_action else 1,
            min(unit.source_indices)))
        orders = _distinct_orders(
            [_separate_pills(variant)
             for variant in _order_variants(base, block_keys)])
        canvas_slot = Slot(0.0, 0.0, target_w, target_h)
        fit_jobs = [(unit, canvas_slot, math.inf) for unit in fits]
        variants = _small_variants(target) if smalls else (None,)
        for order in orders:
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
                rects = _reserve_marks(
                    _zone_rects(order, target),
                    _anchor_reservations(anchor_jobs, target))
                pack_jobs = [(_rows_for_units([unit], block_keys), rect)
                             for unit, rect in zip(order, rects)]
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
