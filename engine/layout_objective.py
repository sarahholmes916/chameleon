"""Deterministic soft objective for S04 candidate layouts (spec D.1.5).

Ten-term weighted penalty J(L) over a (source, candidate, target) triple;
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
- E_group / E_cover (objective_v6_relations_blank): E_group also rewards source
  relations (layered or aligned-and-near pairs) staying together; E_cover also
  penalizes blank space beyond the source's own. Both are soft preferences.
- E_read (objective_v5_readability): the size part is 0.7 * mean + 0.3 * worst
  text (READ_WORST_TEXT_SHARE), so a single unreadably small text is no longer
  diluted by fine ones. The 12px reference stays a provisional quality
  preference, not an Adobe validity rule.
- E_scale also penalizes aspect-ratio distortion of proportion-sensitive
  shapes (ellipses and circles, engine.shape_roles.is_proportion_sensitive):
  the element penalty is max(area drift, |ln(aspect ratio change)| over
  SHAPE_DISTORTION_RANGE). Independently stretched buttons, panels and bars
  are not penalized for their aspect change; only their area drift counts.
- Backdrop panels (source rectangles that hold other elements, role PANEL in
  engine.shape_roles) are layering context, not content: E_cover, E_balance
  and the E_space block boxes ignore them, so stretching a panel over the
  canvas can neither buy coverage nor hide gaps by overlapping its neighbors.
  Their stretch still counts toward E_scale. Without this exclusion a
  panel-fill variant wins on coverage alone (observed before the change).
- E_group also scores UNGROUPED containment: a shape that held an element in the
  source (non-background-like holder, different or absent group) should still hold
  it; the mean loss counts as one more group-like penalty. Without it, repairing a
  panel that lost its content cannot register as an improvement for the refiner.
- E_group containment losses also include padding asymmetry: when a label
  stays inside its holder, an asymmetry of its padding (left/right over holder
  width, top/bottom over holder height) beyond the source's own asymmetry
  scores up to PADDING_ASYMMETRY_GAIN-scaled penalty, so off-center or
  clipped-looking button labels lose to centered ones. No new term or weight
  is added; objective_v4_shape changes the semantics of E_scale and E_group.
- E_cover is one minus the union area of candidate boxes for
  source-non-background-like elements over the target area (coordinate
  compression; an empty eligible set covers nothing and scores 1).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

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
from engine.shape_roles import ShapeRole, classify_shapes, is_proportion_sensitive
from engine.text_fit import estimate_text_height

__all__ = [
    "ALIGN_DEVIATION_SCALE",
    "ALIGN_SOURCE_TOLERANCE",
    "BACKDROP_AREA_RATIO",
    "BALANCE_AREA_CAP_FRACTION",
    "CandidateWithDesign",
    "GROUP_COMPACTNESS_RANGE",
    "OBJECTIVE_VERSION",
    "PAIR_DISTANCE_SCALE",
    "PADDING_ASYMMETRY_GAIN",
    "PAIR_SLACK",
    "PROVISIONAL_WEIGHTS",
    "READ_HIERARCHY_WEIGHT",
    "READ_REFERENCE_FONT_SIZE",
    "READ_SIZE_WEIGHT",
    "READ_WORST_TEXT_SHARE",
    "SCALE_LOG_RANGE",
    "SHAPE_DISTORTION_RANGE",
    "SOURCE_OVERLAP_IOU_THRESHOLD",
    "SPACE_GAP_SMOOTHING",
    "WEIGHT_ALIGN",
    "WEIGHT_BALANCE",
    "WEIGHT_COVER",
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
    "score_deterministic",
]

_logger = logging.getLogger(__name__)

# Version of the weights, constants, and term definitions below. Bump on any
# change (D.1.6 step 5). Provisional until calibrated against human labels.
OBJECTIVE_VERSION = "objective_v6_relations_blank"

# Provisional ten-term weights (D.1.5, sum 1.0). The nine v1 weights are each
# multiplied by 0.9 (relative proportions unchanged, mirroring the D.1.5
# halving precedent); coverage takes the freed 0.10. Untested starting values.
WEIGHT_MOVE = 0.045
WEIGHT_ALIGN = 0.0675
WEIGHT_PAIR = 0.1125
WEIGHT_SCALE = 0.0675
WEIGHT_UNWANTED_OVERLAP = 0.1125
WEIGHT_BALANCE = 0.045
WEIGHT_READ = 0.225
WEIGHT_GROUP = 0.18
WEIGHT_SPACE = 0.045
WEIGHT_COVER = 0.10

# E_align: source anchor agreement tolerance and candidate deviation scale.
ALIGN_SOURCE_TOLERANCE = 0.02
ALIGN_DEVIATION_SCALE = 0.1
# E_pair: slack before separation counts, and distance normalization.
PAIR_SLACK = 0.05
PAIR_DISTANCE_SCALE = 0.25
# E_scale: log-range normalization.
SCALE_LOG_RANGE = 3.0
# E_scale: a proportion-sensitive shape whose aspect ratio changes by this
# factor (2:1) scores the full distortion penalty.
SHAPE_DISTORTION_RANGE = math.log(2.0)
# E_group: padding asymmetry (fraction of holder size, averaged over axes)
# beyond the source's own that scores the full penalty is 1 / this gain.
PADDING_ASYMMETRY_GAIN = 4.0
# A holder is a backdrop only when its area is at least this multiple of its
# largest contained element (same-size overlays are content, not backdrops).
BACKDROP_AREA_RATIO = 1.5
# Source-overlap boundary (D.1.3): IoU at or below this counts as not
# overlapping. Used for overlap eligibility, E_align overlay exclusion, and
# E_space block merging.
SOURCE_OVERLAP_IOU_THRESHOLD = 0.02
# E_read: placeholder reference size in target px, and size/hierarchy split.
READ_REFERENCE_FONT_SIZE = 12.0
READ_SIZE_WEIGHT = 0.8
READ_HIERARCHY_WEIGHT = 0.2
# E_read: share of the size part taken from the WORST text instead of the mean. A mean
# lets one unreadable text hide among many fine ones (1 of 8 texts at 1.9px moved the
# term ~0.1); a quality preference, provisional like every constant here.
READ_WORST_TEXT_SHARE = 0.3
# E_group: share of the term given to SOURCE relations (layered or aligned-and-near pairs
# that are not explicitly grouped) relative to the existing group components (weight 1).
# A soft preference: keeping such pairs together is rewarded, breaking them is never invalid.
RELATION_GROUP_SHARE = 1.0
# A source pair counts as "near" when its canvas-normalized center distance is at most this.
RELATION_NEAR_DISTANCE = 0.35
# E_cover: the raw uncovered fraction grows toward 1 by this share of the blank space
# BEYOND the source's own (area plus empty bands per axis); no extra blank = no change.
COVER_EXCESS_SHARE = 0.5
# Floor on the source's covered fraction when normalizing blank-space excess.
COVER_EXCESS_FLOOR = 0.05
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
    cover: float = WEIGHT_COVER

    def __post_init__(self) -> None:
        values = (
            self.move, self.align, self.pair, self.scale,
            self.unwanted_overlap, self.balance, self.read,
            self.group, self.space, self.cover,
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
    cover: float


@dataclass(frozen=True, slots=True)
class ObjectiveScore:
    """Per-term values, weighted total, and the weights/version used."""

    terms: TermScores
    total: float
    weights: ObjectiveWeights
    objective_version: str


class CandidateWithDesign(Protocol):
    """A candidate wrapper exposing its layout as `.design`.

    Structural mirror of the Protocol of the same name in
    engine.ranker (deliberately not imported: that module requires
    ML frameworks, which the baseline path must not need).
    """

    @property
    def design(self) -> Design: ...


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
    backdrop = _backdrop_ids(source)
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
                           target_width, target_height, backdrop),
        read=_e_read(matched),
        group=_e_group(matched, source.source_canvas,
                       source_width, source_height,
                       target_width, target_height, target),
        space=_e_space(matched, source.source_canvas, target, backdrop),
        cover=_e_cover(matched, source.source_canvas,
                       target_width, target_height, backdrop),
    )
    total = (
        active.move * terms.move + active.align * terms.align
        + active.pair * terms.pair + active.scale * terms.scale
        + active.unwanted_overlap * terms.unwanted_overlap
        + active.balance * terms.balance + active.read * terms.read
        + active.group * terms.group + active.space * terms.space
        + active.cover * terms.cover
    )
    return ObjectiveScore(terms=terms, total=total, weights=active,
                          objective_version=OBJECTIVE_VERSION)


def score_deterministic(
    source: Design,
    candidates: Sequence[Design | CandidateWithDesign],
    target: Canvas,
    *,
    weights: ObjectiveWeights | None = None,
) -> list[float]:
    """Score validated candidates without any learned model; higher is better.

    Each score is 1.0 - J(L) (spec §6.2 says only "higher is better"; this
    mapping keeps scores in [0, 1] whenever the total is). Scores come back
    in input order with no sorting, repair, or fallback. An empty candidate
    list returns []. A malformed candidate (IDs not matching the source, or
    a changed element type) raises ValueError and no partial list is ever
    returned. Pure and deterministic.
    """
    if not candidates:
        return []
    scores = []
    for candidate in candidates:
        design = candidate if isinstance(candidate, Design) else candidate.design
        total = evaluate_objective(source, design, target, weights=weights).total
        scores.append(1.0 - total)
    return scores


def _backdrop_ids(source: Design) -> frozenset[str]:
    """Panels that visibly hold smaller content (not same-size overlays)."""
    found = set()
    for ident, info in classify_shapes(source).items():
        if info.role is not ShapeRole.PANEL or not info.contained_ids:
            continue
        holder = source.elements_by_id[ident]
        largest = max(float(source.elements_by_id[inner].width)
                      * float(source.elements_by_id[inner].height)
                      for inner in info.contained_ids)
        if float(holder.width) * float(holder.height) >= BACKDROP_AREA_RATIO * largest:
            found.add(ident)
    return frozenset(found)


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
            continue
        penalty = min(1.0, abs(drift) / SCALE_LOG_RANGE)
        if (is_proportion_sensitive(original) and proposed.width > 0
                and proposed.height > 0):
            change = math.log((float(proposed.width) / float(proposed.height))
                              / (float(original.width) / float(original.height)))
            penalty = max(penalty, min(1.0, abs(change) / SHAPE_DISTORTION_RANGE))
        penalties.append(penalty)
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
            penalty = element_iou(proposed_a, proposed_b)
            # IoU hides occlusion of a small text box by a large image. Measure
            # the fraction of text obscured as well, without penalizing source overlays.
            intersection = _intersection_area(_stored_box(proposed_a), _stored_box(proposed_b))
            for proposed in (proposed_a, proposed_b):
                area = float(proposed.width) * float(proposed.height)
                if proposed.type == "text" and area > 0:
                    penalty = max(penalty, intersection / area)
            penalties.append(min(1.0, penalty))
    _logger.debug("overlap-eligible pairs: %s",
                  ",".join(eligible) if eligible else "none")
    return _mean(penalties)


def _e_balance(matched: list[tuple[Element, Element]], source_canvas: Canvas,
               target_width: float, target_height: float,
               backdrop: frozenset[str] = frozenset()) -> float:
    cap = BALANCE_AREA_CAP_FRACTION * target_width * target_height
    total_weight = 0.0
    weighted_x = 0.0
    weighted_y = 0.0
    for original, proposed in matched:
        if is_background_like(original, source_canvas) or original.id in backdrop:
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
    size = ((1.0 - READ_WORST_TEXT_SHARE) * _mean(per_element)
            + READ_WORST_TEXT_SHARE * max(per_element, default=0.0))
    return READ_SIZE_WEIGHT * size + READ_HIERARCHY_WEIGHT * _mean(flips)


def _group_offset(center_x: float, center_y: float,
                  box: tuple[float, float, float, float]
                  ) -> tuple[float, float]:
    """Member-center offset from the group-box center in group-box units."""
    middle_x = box[0] + box[2] / 2
    middle_y = box[1] + box[3] / 2
    across = (center_x - middle_x) / box[2] if box[2] != 0 else 0.0
    down = (center_y - middle_y) / box[3] if box[3] != 0 else 0.0
    return (across, down)


def _padding_asymmetry(inner: tuple[float, float, float, float],
                       outer: tuple[float, float, float, float]) -> float:
    """Mean left/right and top/bottom padding imbalance in holder units."""
    if outer[2] <= 0 or outer[3] <= 0:
        return 0.0
    across = abs((inner[0] - outer[0]) - (outer[0] + outer[2] - inner[0] - inner[2]))
    down = abs((inner[1] - outer[1]) - (outer[1] + outer[3] - inner[1] - inner[3]))
    return (across / outer[2] + down / outer[3]) / 2


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
                loss = 1.0 - kept_fraction
                if loss == 0.0:
                    before = _padding_asymmetry(_stored_box(kept),
                                                _stored_box(holder))
                    after = _padding_asymmetry(inner_box, outer_box)
                    loss = _clip01((after - before) * PADDING_ASYMMETRY_GAIN)
                losses.append(loss)
    if losses:
        components.append(_mean(losses))
    return components


def _implicit_containment_losses(matched: list[tuple[Element, Element]],
                                 source_canvas: Canvas) -> list[float]:
    """Containment loss for shapes that hold elements without sharing a group ID.

    A panel or card that held an element in the source should still hold it; explicit
    groups are scored by _group_components, so same-group pairs are skipped here.
    Background-like holders are layering context and skipped, as in the group term.
    """
    losses = []
    for holder, moved_holder in matched:
        if holder.type != "shape" or is_background_like(holder, source_canvas):
            continue
        for kept, moved_kept in matched:
            if (kept is holder or same_explicit_group(holder, kept)
                    or not box_contains(holder, kept)):
                continue
            inner = _stored_box(moved_kept)
            area = inner[2] * inner[3]
            if area <= 0:
                losses.append(0.0)
                continue
            losses.append(1.0 - _clip01(
                _intersection_area(inner, _stored_box(moved_holder)) / area))
    return losses


def _e_group(matched: list[tuple[Element, Element]], source_canvas: Canvas,
             source_width: float, source_height: float,
             target_width: float, target_height: float,
             target: Canvas) -> float:
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
    implicit = _implicit_containment_losses(matched, source_canvas)
    if implicit:
        penalties.append(_mean(implicit))
    relations = _relation_losses(matched, source_canvas, target)
    if not relations:
        return _mean(penalties)
    if not penalties:
        return _mean(relations)
    return ((_mean(penalties) + RELATION_GROUP_SHARE * _mean(relations))
            / (1.0 + RELATION_GROUP_SHARE))


def _relation_losses(matched: list[tuple[Element, Element]], source_canvas: Canvas,
                     target: Canvas) -> list[float]:
    """Loss for source relations that were broken: layered or aligned-and-near pairs.

    Only pairs the source itself relates count (never inferred from IDs or text): layered
    pairs (IoU above the overlap threshold, or containment between non-shape elements)
    keep their overlap; aligned pairs close together in the source keep both their
    alignment and their closeness. Explicit group members and shape-holds-element pairs
    are scored elsewhere in this term, backgrounds are context and skipped.
    """
    source_sizes = (float(source_canvas.width), float(source_canvas.height))
    target_sizes = (float(target.width), float(target.height))
    losses = []
    for left in range(len(matched)):
        for right in range(left + 1, len(matched)):
            original_a, proposed_a = matched[left]
            original_b, proposed_b = matched[right]
            if (same_explicit_group(original_a, original_b)
                    or is_background_like(original_a, source_canvas)
                    or is_background_like(original_b, source_canvas)
                    or not _positive_area(original_a) or not _positive_area(original_b)):
                continue
            holder = (original_a.type == "shape" and box_contains(original_a, original_b))                 or (original_b.type == "shape" and box_contains(original_b, original_a))
            if holder:
                continue
            if (element_iou(original_a, original_b) > SOURCE_OVERLAP_IOU_THRESHOLD
                    or contains_any(original_a, original_b)):
                before = _layer_fraction(_stored_box(original_a), _stored_box(original_b))
                after = _layer_fraction(_stored_box(proposed_a), _stored_box(proposed_b))
                losses.append(_clip01((before - after) / before) if before > 0 else 0.0)
                continue
            if normalized_center_distance(original_a, original_b,
                                          source_canvas) > RELATION_NEAR_DISTANCE:
                continue
            anchors_a, anchors_b = _anchors(original_a), _anchors(original_b)
            moved_a, moved_b = _anchors(proposed_a), _anchors(proposed_b)
            drifts = []
            for axis in (0, 1):
                for slot in range(3):
                    if abs(anchors_a[axis][slot] - anchors_b[axis][slot])                             / source_sizes[axis] <= ALIGN_SOURCE_TOLERANCE:
                        drifts.append(min(1.0, abs(moved_a[axis][slot] - moved_b[axis][slot])
                                          / target_sizes[axis] / ALIGN_DEVIATION_SCALE))
            if not drifts:
                continue
            growth = min(1.0, max(0.0, normalized_center_distance(
                proposed_a, proposed_b, target)
                - normalized_center_distance(original_a, original_b, source_canvas)
                - PAIR_SLACK) / PAIR_DISTANCE_SCALE)
            losses.append((_mean(drifts) + growth) / 2)
    return losses


def _layer_fraction(first: tuple[float, float, float, float],
                    second: tuple[float, float, float, float]) -> float:
    """Overlap area as a fraction of the smaller box."""
    smaller = min(first[2] * first[3], second[2] * second[3])
    return _intersection_area(first, second) / smaller if smaller > 0 else 0.0


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
             target: Canvas, backdrop: frozenset[str] = frozenset()) -> float:
    blocks = _space_blocks(matched, source_canvas)
    if len(blocks) < 3:
        return 0.0
    horizontal = float(target.width) >= float(target.height)
    axis_length = float(target.width) if horizontal else float(target.height)
    spans = []
    for block in blocks:
        content = [position for position in block
                   if matched[position][0].id not in backdrop] or block
        box = _union_box([_stored_box(matched[position][1])
                          for position in content])
        start = box[0] if horizontal else box[1]
        size = box[2] if horizontal else box[3]
        spans.append((start + size / 2, start, start + size, min(block)))
    spans.sort(key=lambda span: (span[0], span[3]))
    gaps = [max(0.0, spans[k + 1][1] - spans[k][2]) / axis_length
            for k in range(len(spans) - 1)]
    mean_gap = sum(gaps) / len(gaps)
    variance = sum((gap - mean_gap) ** 2 for gap in gaps) / len(gaps)
    return min(1.0, math.sqrt(variance) / (mean_gap + SPACE_GAP_SMOOTHING))


def _union_area(boxes: list[tuple[float, float, float, float]]) -> float:
    """Exact union area of axis-aligned boxes via coordinate compression."""
    spans = [(x, x + width, y, y + height)
             for x, y, width, height in boxes
             if width > 0 and height > 0]
    if not spans:
        return 0.0
    xs = sorted({edge for span in spans for edge in span[:2]})
    ys = sorted({edge for span in spans for edge in span[2:]})
    area = 0.0
    for left, right in zip(xs, xs[1:]):
        for top, bottom in zip(ys, ys[1:]):
            if any(zone_left <= left and right <= zone_right
                   and zone_top <= top and bottom <= zone_bottom
                   for zone_left, zone_right, zone_top, zone_bottom in spans):
                area += (right - left) * (bottom - top)
    return area


def _e_cover(matched: list[tuple[Element, Element]], source_canvas: Canvas,
             target_width: float, target_height: float,
             backdrop: frozenset[str] = frozenset()) -> float:
    """One minus the covered fraction of the target canvas.

    Coverage is the union area of candidate boxes for elements that are not
    background-like in the source, over the target area. Area-weighting means
    tiny elements barely register (no gaming by count); the union (not the
    sum) means overlaps never double-count. An empty eligible set covers
    nothing and scores 1.
    """
    target_area = target_width * target_height
    if target_area <= 0:
        return 1.0
    candidate = _clipped_boxes([(o, p) for o, p in matched
                                if not (is_background_like(o, source_canvas) or o.id in backdrop)],
                               1, target_width, target_height)
    raw = _clip01(1.0 - _union_area(candidate) / target_area)
    # Blank space BEYOND what the source already had (area and empty bands per axis).
    original = _clipped_boxes([(o, o) for o, _ in matched
                               if not (is_background_like(o, source_canvas) or o.id in backdrop)],
                              0, float(source_canvas.width), float(source_canvas.height))
    before = _coverage(original, float(source_canvas.width), float(source_canvas.height))
    after = _coverage(candidate, target_width, target_height)
    excess = _mean([_clip01((before[k] - after[k]) / max(before[k], COVER_EXCESS_FLOOR))
                    for k in range(3)])
    # Unchanged when the candidate is no emptier than the source; stays within [0, 1].
    return raw + (1.0 - raw) * COVER_EXCESS_SHARE * excess


def _clipped_boxes(pairs: list[tuple[Element, Element]], which: int, width: float,
                   height: float) -> list[tuple[float, float, float, float]]:
    boxes = []
    for pair in pairs:
        element = pair[which]
        left = min(max(float(element.x), 0.0), width)
        top = min(max(float(element.y), 0.0), height)
        right = min(max(float(element.x) + float(element.width), 0.0), width)
        bottom = min(max(float(element.y) + float(element.height), 0.0), height)
        boxes.append((left, top, right - left, bottom - top))
    return boxes


def _span_length(spans: list[tuple[float, float]]) -> float:
    total, end = 0.0, None
    for start, stop in sorted(spans):
        if end is None or start > end:
            total += stop - start
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
    return total


def _coverage(boxes: list[tuple[float, float, float, float]], width: float,
              height: float) -> tuple[float, float, float]:
    """Covered fractions: area, and the share of each axis crossed by some box."""
    if width <= 0 or height <= 0:
        return (0.0, 0.0, 0.0)
    solid = [b for b in boxes if b[2] > 0 and b[3] > 0]
    return (_union_area(solid) / (width * height),
            _span_length([(b[0], b[0] + b[2]) for b in solid]) / width,
            _span_length([(b[1], b[1] + b[3]) for b in solid]) / height)
