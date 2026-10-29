"""Deterministic soft objective for S04 candidate layouts (spec D.1.5).

Nine-term weighted penalty J(L) over a (source, candidate, target) triple;
lower is better. Weights and constants are the spec's PROVISIONAL starting
values (D.1.6): untested, not final, and not a validated aesthetic metric.
Any change to a weight, constant, or term definition must bump
OBJECTIVE_VERSION. Stdlib plus S02/S04 helpers only, with no ML-framework,
encoder, or checkpoint imports, so this runs in the baseline environment
(spec 8.2 item 7).

The scorer operates on validated candidates but never validates: passing an
unvalidated design is a caller error. Elements are matched by exact ID, never
by position. Every term is bounded to [0, 1] (0 best); with weights summing
to 1.0 the total is in [0, 1] as well.

Mechanical readings adopted where D.1.5 under-specifies (all deterministic):
- Role and eligibility evidence (background-like, containment, group
  sharing, source overlap) is read from the SOURCE layout; only measured
  quantities (distances, IoU, areas, font sizes) come from the candidate.
- "Confirmed decorative/overlay pairs" (E_align) means a pair where either
  member is background-like (D.1.3) or the source boxes overlap (IoU above
  SOURCE_OVERLAP_IOU_THRESHOLD) or one contains the other. Same-group pairs
  are not excluded from E_align.
- "Pairs selected as soft associations" (E_pair) means source pairs with
  association >= ASSOCIATION_THRESHOLD, reusing engine.layout_units.
- E_scale covers source positive-area boxes that are not background-like.
- E_read hierarchy (e_hier) covers all text-element pairs with different
  source sizes ("source text pairs", D.1.5).
- E_group (d) covers within-group source containment pairs whose outer box
  is not background-like (spec 8.1.5); a fully collapsed group box scores 1.
- E_space uses population std, the x-axis when the target is square, and
  min-source-index tie-breaks.
- BALANCE_AREA_CAP_FRACTION is an implementation choice: D.1.5 orders a
  per-object cap but gives no value.
- Inputs that would make a logarithm undefined (zero/underflowed ratios)
  score 1.0 for that comparison, never NaN.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from engine.internal_schema import Canvas, Design, Element
from engine.layout_units import (
    ASSOCIATION_THRESHOLD,
    association,
    box_contains,
    contains_any,
    element_iou,
    is_background_like,
    normalized_center_distance,
    same_explicit_group,
)
from engine.text_fit import estimate_text_height

__all__ = [
    "ALIGN_DEVIATION_SCALE",
    "ALIGN_SOURCE_TOLERANCE",
    "BALANCE_AREA_CAP_FRACTION",
    "GROUP_COMPACTNESS_RANGE",
    "OBJECTIVE_VERSION",
    "PAIR_DISTANCE_SCALE",
    "PAIR_SLACK",
    "PROVISIONAL_WEIGHTS",
    "READ_HIERARCHY_WEIGHT",
    "READ_REFERENCE_FONT_SIZE",
    "READ_SIZE_WEIGHT",
    "SCALE_LOG_RANGE",
    "SOURCE_OVERLAP_IOU_THRESHOLD",
    "SPACE_GAP_SMOOTHING",
    "WEIGHT_ALIGN",
    "WEIGHT_BALANCE",
    "WEIGHT_GROUP",
    "WEIGHT_MOVE",
    "WEIGHT_PAIR",
    "WEIGHT_READ",
    "WEIGHT_SCALE",
    "WEIGHT_SPACE",
    "WEIGHT_UNWANTED_OVERLAP",
    "ObjectiveScore",
    "ObjectiveWeights",
    "TermScores",
    "evaluate_objective",
]

_logger = logging.getLogger(__name__)

# Version of the weights, constants, and term definitions below. Bump on any
# change (D.1.6 step 5). Provisional until calibrated against human labels.
OBJECTIVE_VERSION = "objective_v1_provisional"

# Provisional nine-term weights (D.1.5, sum 1.0). Untested starting values.
WEIGHT_MOVE = 0.05
WEIGHT_ALIGN = 0.075
WEIGHT_PAIR = 0.125
WEIGHT_SCALE = 0.075
WEIGHT_UNWANTED_OVERLAP = 0.125
WEIGHT_BALANCE = 0.05
WEIGHT_READ = 0.25
WEIGHT_GROUP = 0.20
WEIGHT_SPACE = 0.05

# E_align: source anchor agreement tolerance and candidate deviation scale.
ALIGN_SOURCE_TOLERANCE = 0.02
ALIGN_DEVIATION_SCALE = 0.1
# E_pair: slack before separation counts, and distance normalization.
PAIR_SLACK = 0.05
PAIR_DISTANCE_SCALE = 0.25
# E_scale: log-range normalization.
SCALE_LOG_RANGE = 3.0
# Source-overlap boundary (D.1.3): IoU at or below this counts as not
# overlapping. Used for overlap eligibility, E_align overlay exclusion, and
# E_space block merging.
SOURCE_OVERLAP_IOU_THRESHOLD = 0.02
# E_read: placeholder reference size in target px, and size/hierarchy split.
READ_REFERENCE_FONT_SIZE = 12.0
READ_SIZE_WEIGHT = 0.8
READ_HIERARCHY_WEIGHT = 0.2
# E_group compactness log-range normalization.
GROUP_COMPACTNESS_RANGE = 3.0
# E_space gap-mean smoothing.
SPACE_GAP_SMOOTHING = 0.01
# E_balance: each element's area weight is capped at this fraction of the
# target area. Unspecified by D.1.5; implementation choice.
BALANCE_AREA_CAP_FRACTION = 0.25


@dataclass(frozen=True, slots=True)
class ObjectiveWeights:
    """One float per D.1.5 term; defaults are the provisional weights."""

    move: float = WEIGHT_MOVE
    align: float = WEIGHT_ALIGN
    pair: float = WEIGHT_PAIR
    scale: float = WEIGHT_SCALE
    unwanted_overlap: float = WEIGHT_UNWANTED_OVERLAP
    balance: float = WEIGHT_BALANCE
    read: float = WEIGHT_READ
    group: float = WEIGHT_GROUP
    space: float = WEIGHT_SPACE

    def __post_init__(self) -> None:
        values = (
            self.move, self.align, self.pair, self.scale,
            self.unwanted_overlap, self.balance, self.read,
            self.group, self.space,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("objective weights must be finite")
        if not math.isclose(sum(values), 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("objective weights must sum to 1.0 (spec D.1.6)")


PROVISIONAL_WEIGHTS = ObjectiveWeights()


@dataclass(frozen=True, slots=True)
class TermScores:
    """Per-term penalties in [0, 1]; 0 is best for every term."""

    move: float
    align: float
    pair: float
    scale: float
    unwanted_overlap: float
    balance: float
    read: float
    group: float
    space: float


@dataclass(frozen=True, slots=True)
class ObjectiveScore:
    """Per-term values, weighted total, and the weights/version used."""

    terms: TermScores
    total: float
    weights: ObjectiveWeights
    objective_version: str


def evaluate_objective(
    source: Design,
    candidate: Design,
    target: Canvas,
    *,
    weights: ObjectiveWeights | None = None,
) -> ObjectiveScore:
    """Score one candidate layout against its source on the target canvas.

    Pure and deterministic: same inputs give the same score. Lower total is
    better. Raises ValueError when candidate IDs do not match the source
    (missing, unexpected, or duplicate) or an element type changed.
    """
    active = PROVISIONAL_WEIGHTS if weights is None else weights
    matched = _match_by_id(source, candidate)
    source_width = float(source.source_canvas.width)
    source_height = float(source.source_canvas.height)
    target_width = float(target.width)
    target_height = float(target.height)
    terms = TermScores(
        move=_e_move(matched, source_width, source_height,
                     target_width, target_height),
        align=_e_align(matched, source.source_canvas, target),
        pair=_e_pair(matched, source.source_canvas, target),
        scale=_e_scale(matched, source.source_canvas,
                       source_width, source_height,
                       target_width, target_height),
        unwanted_overlap=_e_unwanted_overlap(matched, source.source_canvas),
        balance=_e_balance(matched, source.source_canvas,
                           target_width, target_height),
        read=_e_read(matched),
        group=_e_group(matched, source.source_canvas,
                       source_width, source_height,
                       target_width, target_height),
        space=_e_space(matched, source.source_canvas, target),
    )
    total = (
        active.move * terms.move + active.align * terms.align
        + active.pair * terms.pair + active.scale * terms.scale
        + active.unwanted_overlap * terms.unwanted_overlap
        + active.balance * terms.balance + active.read * terms.read
        + active.group * terms.group + active.space * terms.space
    )
    return ObjectiveScore(terms=terms, total=total, weights=active,
                          objective_version=OBJECTIVE_VERSION)


def _match_by_id(
    source: Design, candidate: Design,
) -> list[tuple[Element, Element]]:
    """Pair source and candidate elements by exact ID in source order."""
    seen: dict[str, Element] = {}
    for element in candidate.elements:
        if element.id in seen:
            raise ValueError(f"duplicate candidate id {element.id!r}")
        seen[element.id] = element
    source_ids = [element.id for element in source.elements]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("duplicate source element id")
    if set(source_ids) != set(seen):
        raise ValueError("candidate ids do not match the source")
    matched = []
    for element in source.elements:
        proposed = seen[element.id]
        if proposed.type != element.type:
            raise ValueError(
                f"element {element.id!r} changed type "
                f"{element.type!r} to {proposed.type!r}"
            )
        matched.append((element, proposed))
    return matched


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _center(element: Element) -> tuple[float, float]:
    return (float(element.x) + float(element.width) / 2,
            float(element.y) + float(element.height) / 2)


def _union_box(boxes: list[tuple[float, float, float, float]]
               ) -> tuple[float, float, float, float]:
    left = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    right = max(box[0] + box[2] for box in boxes)
    bottom = max(box[1] + box[3] for box in boxes)
    return (left, top, right - left, bottom - top)


def _stored_box(element: Element) -> tuple[float, float, float, float]:
    return (float(element.x), float(element.y),
            float(element.width), float(element.height))


def _intersection_area(first: tuple[float, float, float, float],
                       second: tuple[float, float, float, float]) -> float:
    width = max(0.0, min(first[0] + first[2], second[0] + second[2])
                - max(first[0], second[0]))
    height = max(0.0, min(first[1] + first[3], second[1] + second[3])
                 - max(first[1], second[1]))
    return width * height


def _positive_area(element: Element) -> bool:
    return element.width > 0 and element.height > 0


def _e_move(matched: list[tuple[Element, Element]],
            source_width: float, source_height: float,
            target_width: float, target_height: float) -> float:
    penalties = []
    for original, proposed in matched:
        source_x, source_y = _center(original)
        target_x, target_y = _center(proposed)
        distance = math.hypot(target_x / target_width - source_x / source_width,
                              target_y / target_height - source_y / source_height)
        penalties.append(min(1.0, distance / math.sqrt(2)))
    return _mean(penalties)


def _confirmed_overlay_pair(first: Element, second: Element,
                            canvas: Canvas) -> bool:
    """Source-evidence test for the E_align decorative/overlay exclusion."""
    return (
        is_background_like(first, canvas)
        or is_background_like(second, canvas)
        or element_iou(first, second) > SOURCE_OVERLAP_IOU_THRESHOLD
        or contains_any(first, second)
    )


def _anchors(element: Element) -> tuple[tuple[float, float, float],
                                       tuple[float, float, float]]:
    box = _stored_box(element)
    return ((box[0], box[0] + box[2] / 2, box[0] + box[2]),
            (box[1], box[1] + box[3] / 2, box[1] + box[3]))


def _e_align(matched: list[tuple[Element, Element]],
             source_canvas: Canvas, target: Canvas) -> float:
    source_sizes = (float(source_canvas.width), float(source_canvas.height))
    target_sizes = (float(target.width), float(target.height))
    penalties = []
    for left in range(len(matched)):
        for right in range(left + 1, len(matched)):
            original_a, proposed_a = matched[left]
            original_b, proposed_b = matched[right]
            if _confirmed_overlay_pair(original_a, original_b, source_canvas):
                continue
            source_anchors = (_anchors(original_a), _anchors(original_b))
            target_anchors = (_anchors(proposed_a), _anchors(proposed_b))
            for axis in (0, 1):
                for slot in range(3):
                    gap = abs(source_anchors[0][axis][slot]
                              - source_anchors[1][axis][slot])
                    if gap / source_sizes[axis] > ALIGN_SOURCE_TOLERANCE:
                        continue
                    drift = abs(target_anchors[0][axis][slot]
                                - target_anchors[1][axis][slot])
                    penalties.append(min(
                        1.0, drift / target_sizes[axis] / ALIGN_DEVIATION_SCALE,
                    ))
    return _mean(penalties)


def _e_pair(matched: list[tuple[Element, Element]],
            source_canvas: Canvas, target: Canvas) -> float:
    penalties = []
    for left in range(len(matched)):
        for right in range(left + 1, len(matched)):
            original_a, proposed_a = matched[left]
            original_b, proposed_b = matched[right]
            if association(original_a, original_b, source_canvas) < ASSOCIATION_THRESHOLD:
                continue
            before = normalized_center_distance(original_a, original_b,
                                                source_canvas)
            after = normalized_center_distance(proposed_a, proposed_b, target)
            penalties.append(min(
                1.0, max(0.0, after - before - PAIR_SLACK) / PAIR_DISTANCE_SCALE,
            ))
    return _mean(penalties)


def _e_scale(matched: list[tuple[Element, Element]], source_canvas: Canvas,
             source_width: float, source_height: float,
             target_width: float, target_height: float) -> float:
    expected = math.log((target_width * target_height + 1.0)
                        / (source_width * source_height + 1.0))
    penalties = []
    for original, proposed in matched:
        if not _positive_area(original):
            continue
        if is_background_like(original, source_canvas):
            continue
        area_before = float(original.width) * float(original.height)
        area_after = float(proposed.width) * float(proposed.height)
        actual = math.log((area_after + 1.0) / (area_before + 1.0))
        drift = actual - expected
        if drift != drift or math.isinf(drift):
            penalties.append(1.0)
        else:
            penalties.append(min(1.0, abs(drift) / SCALE_LOG_RANGE))
    return _mean(penalties)


def _overlap_eligible(first: Element, second: Element,
                      canvas: Canvas) -> bool:
    """D.1.3 eligibility: non-background, positive-area, source-separated."""
    return (
        not is_background_like(first, canvas)
        and not is_background_like(second, canvas)
        and _positive_area(first)
        and _positive_area(second)
        and element_iou(first, second) <= SOURCE_OVERLAP_IOU_THRESHOLD
        and not contains_any(first, second)
        and not same_explicit_group(first, second)
    )


def _e_unwanted_overlap(matched: list[tuple[Element, Element]],
                        source_canvas: Canvas) -> float:
    penalties = []
    eligible: list[str] = []
    for left in range(len(matched)):
        for right in range(left + 1, len(matched)):
            original_a, proposed_a = matched[left]
            original_b, proposed_b = matched[right]
            if not _overlap_eligible(original_a, original_b, source_canvas):
                continue
            eligible.append(f"{original_a.id}&{original_b.id}")
            penalties.append(min(1.0, element_iou(proposed_a, proposed_b)))
    _logger.debug("overlap-eligible pairs: %s",
                  ",".join(eligible) if eligible else "none")
    return _mean(penalties)


def _e_balance(matched: list[tuple[Element, Element]], source_canvas: Canvas,
               target_width: float, target_height: float) -> float:
    cap = BALANCE_AREA_CAP_FRACTION * target_width * target_height
    total_weight = 0.0
    weighted_x = 0.0
    weighted_y = 0.0
    for original, proposed in matched:
        if is_background_like(original, source_canvas):
            continue
        weight = min(float(proposed.width) * float(proposed.height), cap)
        center_x, center_y = _center(proposed)
        total_weight += weight
        weighted_x += weight * center_x
        weighted_y += weight * center_y
    if total_weight <= 0:
        return 0.0
    offset = math.hypot(weighted_x / total_weight - target_width / 2,
                        weighted_y / total_weight - target_height / 2)
    half_diagonal = math.hypot(target_width, target_height) / 2
    return min(1.0, offset / half_diagonal)


def _overflow_penalty(content: str, proposed: Element) -> float:
    """Fractional overflow of a text box, or 1.0 when degenerate."""
    width = float(proposed.width)
    height = float(proposed.height)
    if width <= 0 or height <= 0:
        return 1.0
    estimated = estimate_text_height(
        content, proposed.width, proposed.font_size,
        proposed.effective_value("line_height"),
    )
    if estimated is None:
        return 1.0
    return _clip01((estimated - height) / height)


def _e_read(matched: list[tuple[Element, Element]]) -> float:
    texts = [(original, proposed) for original, proposed in matched
             if original.type == "text"]
    per_element = []
    for _, proposed in texts:
        content = proposed.content
        if content == "":
            continue
        small = _clip01(1.0 - float(proposed.font_size)
                        / READ_REFERENCE_FONT_SIZE)
        per_element.append(max(small, _overflow_penalty(content, proposed)))
    flips = []
    for left in range(len(texts)):
        for right in range(left + 1, len(texts)):
            before = (float(texts[left][0].font_size)
                      - float(texts[right][0].font_size))
            if before == 0:
                continue
            after = (float(texts[left][1].font_size)
                     - float(texts[right][1].font_size))
            if after == 0:
                flips.append(0.5)
            elif after * before < 0:
                flips.append(1.0)
            else:
                flips.append(0.0)
    return READ_SIZE_WEIGHT * _mean(per_element) + READ_HIERARCHY_WEIGHT * _mean(flips)


def _group_offset(center_x: float, center_y: float,
                  box: tuple[float, float, float, float]
                  ) -> tuple[float, float]:
    """Member-center offset from the group-box center in group-box units."""
    middle_x = box[0] + box[2] / 2
    middle_y = box[1] + box[3] / 2
    across = (center_x - middle_x) / box[2] if box[2] != 0 else 0.0
    down = (center_y - middle_y) / box[3] if box[3] != 0 else 0.0
    return (across, down)


def _group_components(members: list[tuple[Element, Element]],
                      source_canvas: Canvas,
                      source_width: float, source_height: float,
                      target_width: float, target_height: float
                      ) -> list[float]:
    source_box = _union_box([_stored_box(original) for original, _ in members])
    target_box = _union_box([_stored_box(proposed) for _, proposed in members])
    offsets = []
    for original, proposed in members:
        before = _group_offset(*_center(original), source_box)
        after = _group_offset(*_center(proposed), target_box)
        offsets.append(min(1.0, math.hypot(after[0] - before[0],
                                           after[1] - before[1])))
    components = [_mean(offsets)]
    flips = []
    for left in range(len(members)):
        for right in range(left + 1, len(members)):
            original_a, proposed_a = members[left]
            original_b, proposed_b = members[right]
            if contains_any(original_a, original_b):
                continue
            center_a = _center(original_a)
            center_b = _center(original_b)
            moved_a = _center(proposed_a)
            moved_b = _center(proposed_b)
            across = (moved_b[0] - moved_a[0]) * (center_b[0] - center_a[0])
            down = (moved_b[1] - moved_a[1]) * (center_b[1] - center_a[1])
            flips.append(1.0 if across < 0 or down < 0 else 0.0)
    if flips:
        components.append(_mean(flips))
    area_before = source_box[2] * source_box[3]
    area_after = target_box[2] * target_box[3]
    if area_before == 0:
        components.append(0.0)
    else:
        ratio = ((area_after / area_before)
                 / ((target_width * target_height)
                    / (source_width * source_height)))
        if ratio > 0:
            components.append(min(1.0, abs(math.log(ratio))
                                  / GROUP_COMPACTNESS_RANGE))
        else:
            components.append(1.0)
    losses = []
    for outer in range(len(members)):
        for inner in range(len(members)):
            if outer == inner:
                continue
            holder, kept = members[outer][0], members[inner][0]
            if not box_contains(holder, kept):
                continue
            if is_background_like(holder, source_canvas):
                continue
            inner_box = _stored_box(members[inner][1])
            outer_box = _stored_box(members[outer][1])
            inner_area = inner_box[2] * inner_box[3]
            if inner_area <= 0:
                losses.append(0.0)
            else:
                kept_fraction = _clip01(_intersection_area(inner_box, outer_box)
                                        / inner_area)
                losses.append(1.0 - kept_fraction)
    if losses:
        components.append(_mean(losses))
    return components


def _e_group(matched: list[tuple[Element, Element]], source_canvas: Canvas,
             source_width: float, source_height: float,
             target_width: float, target_height: float) -> float:
    grouped: dict[str, list[int]] = {}
    for position, (original, _) in enumerate(matched):
        if isinstance(original.group_id, str):
            grouped.setdefault(original.group_id, []).append(position)
    penalties = []
    for positions in grouped.values():
        if len(positions) < 2:
            continue
        members = [matched[position] for position in positions]
        penalties.append(_mean(_group_components(
            members, source_canvas, source_width, source_height,
            target_width, target_height,
        )))
    return _mean(penalties)


def _space_blocks(matched: list[tuple[Element, Element]],
                  source_canvas: Canvas) -> list[list[int]]:
    """Union non-background elements into blocks over source group/overlap."""
    eligible = [position for position, (original, _) in enumerate(matched)
                if not is_background_like(original, source_canvas)]
    parent = list(range(len(matched)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for left_pos, left in enumerate(eligible):
        for right in eligible[left_pos + 1:]:
            original_a = matched[left][0]
            original_b = matched[right][0]
            if (same_explicit_group(original_a, original_b)
                    or element_iou(original_a, original_b)
                    > SOURCE_OVERLAP_IOU_THRESHOLD
                    or contains_any(original_a, original_b)):
                root_left, root_right = find(left), find(right)
                if root_left != root_right:
                    parent[root_right] = root_left
    blocks: dict[int, list[int]] = {}
    for position in eligible:
        blocks.setdefault(find(position), []).append(position)
    return list(blocks.values())


def _e_space(matched: list[tuple[Element, Element]], source_canvas: Canvas,
             target: Canvas) -> float:
    blocks = _space_blocks(matched, source_canvas)
    if len(blocks) < 3:
        return 0.0
    horizontal = float(target.width) >= float(target.height)
    axis_length = float(target.width) if horizontal else float(target.height)
    spans = []
    for block in blocks:
        box = _union_box([_stored_box(matched[position][1])
                          for position in block])
        start = box[0] if horizontal else box[1]
        size = box[2] if horizontal else box[3]
        spans.append((start + size / 2, start, start + size, min(block)))
    spans.sort(key=lambda span: (span[0], span[3]))
    gaps = [max(0.0, spans[k + 1][1] - spans[k][2]) / axis_length
            for k in range(len(spans) - 1)]
    mean_gap = sum(gaps) / len(gaps)
    variance = sum((gap - mean_gap) ** 2 for gap in gaps) / len(gaps)
    return min(1.0, math.sqrt(variance) / (mean_gap + SPACE_GAP_SMOOTHING))
