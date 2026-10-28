"""Internal layout units for S04 source analysis (no Adobe output here).

Tiered grouping over an immutable S02 Design: explicit groups default to one
cohesive unit, background-like elements stand alone, and remaining elements
form soft clusters only over strong association edges. Units preserve member
relative geometry for later slot assignment. See S04 sections 6.1 and 6.2a in
docs/planning/S04_candidate_generation.md.

Deviations from that proposal, per Developer 2 rulings:
- Explicit groups are their own tier and stay cohesive regardless of the
  association score, because same-group alone scores 0.40 against the 0.75
  threshold.
- The alignment term accepts left, center-x, or right edge agreement within
  tolerance; the proposal checks left only.
- Forced background splits separate background-like members into singleton
  units; largest-gap splits apply to ambiguous groups in split mode.

Grouping uses stored (pre-rotation) boxes, matching S03 section 3.3. Unit
extents use rotated plus stroke bounds for target-fit checks. Only geometry,
type, group_id, and fit inform grouping; underscore-prefixed fields and ID
spellings are never read. Deterministic: no RNG, ties break by source index.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from engine.candidate_validator import element_bounds
from engine.internal_schema import Canvas, Design, Element

__all__ = [
    "ALIGNMENT_TOLERANCE",
    "AMBIGUOUS_GROUP_ASSOCIATION",
    "ASSOCIATION_PROXIMITY_SCALE",
    "ASSOCIATION_THRESHOLD",
    "ASSOCIATION_WEIGHT_ALIGNMENT",
    "ASSOCIATION_WEIGHT_CONTAINS",
    "ASSOCIATION_WEIGHT_PROXIMITY",
    "ASSOCIATION_WEIGHT_SAME_GROUP",
    "BACKGROUND_AREA_FRACTION",
    "COVER_BACKGROUND_AREA_FRACTION",
    "GROUP_MODE_COHESIVE",
    "GROUP_MODE_SPLIT",
    "MAX_UNIT_SIZE",
    "SPLIT_MAX_CLUSTERS",
    "SPLIT_MIN_CLUSTERS",
    "SPLIT_SECOND_GAP_RATIO",
    "LayoutUnit",
    "association",
    "box_contains",
    "box_iou",
    "build_layout_units",
    "contains_any",
    "covers_canvas_center",
    "element_iou",
    "group_forces_split",
    "group_is_ambiguous",
    "is_aligned",
    "is_background_like",
    "min_pair_association",
    "normalized_center_distance",
    "required_group_modes",
    "same_explicit_group",
    "split_unit",
]

BACKGROUND_AREA_FRACTION = 0.75
COVER_BACKGROUND_AREA_FRACTION = 0.75
ASSOCIATION_THRESHOLD = 0.75
ASSOCIATION_WEIGHT_SAME_GROUP = 0.40
ASSOCIATION_WEIGHT_CONTAINS = 0.25
ASSOCIATION_WEIGHT_PROXIMITY = 0.20
ASSOCIATION_WEIGHT_ALIGNMENT = 0.15
ASSOCIATION_PROXIMITY_SCALE = 0.20
ALIGNMENT_TOLERANCE = 0.02
MAX_UNIT_SIZE = 6
AMBIGUOUS_GROUP_ASSOCIATION = 0.75
SPLIT_MIN_CLUSTERS = 2
SPLIT_MAX_CLUSTERS = 3
SPLIT_SECOND_GAP_RATIO = 0.5
GROUP_MODE_COHESIVE = "cohesive"
GROUP_MODE_SPLIT = "split"


@dataclass(frozen=True, slots=True)
class LayoutUnit:
    """Internal grouping of source elements in source order.

    Members reference existing S02 Elements; nothing is copied or reordered
    for Adobe output. bbox is the stored-box union as (x, y, w, h); extents
    is the rotated plus stroke union as (left, top, right, bottom).
    relative_geometry holds per-member (fx, fy, fw, fh) fractions within
    bbox for uniform scaling to slots. group_id is the shared explicit ID
    when every member carries the same string, else None.
    """

    members: tuple[Element, ...]
    source_indices: tuple[int, ...]
    bbox: tuple[float, float, float, float]
    extents: tuple[float, float, float, float]
    relative_geometry: tuple[tuple[float, float, float, float], ...]
    group_id: str | None
    is_background: bool

    @classmethod
    def of(
        cls, members: Sequence[Element], indices: Sequence[int], canvas: Canvas,
    ) -> LayoutUnit:
        items = tuple(members)
        positions = tuple(indices)
        if not items or len(items) != len(positions):
            raise ValueError("members and indices must be non-empty and aligned")
        left = min(item.x for item in items)
        top = min(item.y for item in items)
        right = max(item.x + item.width for item in items)
        bottom = max(item.y + item.height for item in items)
        box_x, box_y = float(left), float(top)
        box_w, box_h = float(right - left), float(bottom - top)
        bounds = [element_bounds(item) for item in items]
        extents = (
            min(bound[0] for bound in bounds),
            min(bound[1] for bound in bounds),
            max(bound[2] for bound in bounds),
            max(bound[3] for bound in bounds),
        )
        relative = tuple(
            (
                (float(item.x) - box_x) / box_w if box_w > 0 else 0.0,
                (float(item.y) - box_y) / box_h if box_h > 0 else 0.0,
                float(item.width) / box_w if box_w > 0 else 0.0,
                float(item.height) / box_h if box_h > 0 else 0.0,
            )
            for item in items
        )
        first_group = items[0].group_id
        if isinstance(first_group, str) and all(
            item.group_id == first_group for item in items
        ):
            label = first_group
        else:
            label = None
        background = all(is_background_like(item, canvas) for item in items)
        return cls(items, positions, (box_x, box_y, box_w, box_h), extents,
                   relative, label, background)


def covers_canvas_center(element: Element, canvas: Canvas) -> bool:
    """Check stored-box coverage of the canvas center (edges inclusive)."""
    return (
        2 * element.x <= canvas.width <= 2 * (element.x + element.width)
        and 2 * element.y <= canvas.height <= 2 * (element.y + element.height)
    )


def is_background_like(element: Element, canvas: Canvas) -> bool:
    """Heuristic search hint only; never justifies removal or reordering."""
    if element.type not in ("shape", "image"):
        return False
    area = element.width * element.height
    canvas_area = canvas.width * canvas.height
    if (
        area >= BACKGROUND_AREA_FRACTION * canvas_area
        and covers_canvas_center(element, canvas)
    ):
        return True
    return (
        element.type == "image"
        and element.effective_value("fit") == "cover"
        and area >= COVER_BACKGROUND_AREA_FRACTION * canvas_area
    )


def same_explicit_group(first: Element, second: Element) -> bool:
    """Match S03: only identical string IDs group; empty string counts."""
    return isinstance(first.group_id, str) and first.group_id == second.group_id


def box_contains(outer: Element, inner: Element) -> bool:
    """Positive-area stored-box containment (edges inclusive, match S03)."""
    if not (
        outer.width > 0
        and outer.height > 0
        and inner.width > 0
        and inner.height > 0
    ):
        return False
    return (
        outer.x <= inner.x
        and outer.y <= inner.y
        and outer.x + outer.width >= inner.x + inner.width
        and outer.y + outer.height >= inner.y + inner.height
    )


def contains_any(first: Element, second: Element) -> bool:
    """Containment in either direction."""
    return box_contains(first, second) or box_contains(second, first)


def normalized_center_distance(
    first: Element, second: Element, canvas: Canvas,
) -> float:
    """Hypotenuse of canvas-normalized center deltas, unclipped."""
    try:
        dx = (
            (second.x + second.width / 2) - (first.x + first.width / 2)
        ) / canvas.width
        dy = (
            (second.y + second.height / 2) - (first.y + first.height / 2)
        ) / canvas.height
        return math.hypot(dx, dy)
    except OverflowError:
        return math.inf


def is_aligned(first: Element, second: Element, canvas: Canvas) -> bool:
    """Left, center-x, or right edge agreement within tolerance."""
    try:
        left = abs(second.x - first.x) / canvas.width
        center = abs(
            (second.x + second.width / 2) - (first.x + first.width / 2)
        ) / canvas.width
        right = abs(
            (second.x + second.width) - (first.x + first.width)
        ) / canvas.width
    except OverflowError:
        return False
    return min(left, center, right) <= ALIGNMENT_TOLERANCE


def association(first: Element, second: Element, canvas: Canvas) -> float:
    """Soft-pairing score in [0, 1]; symmetric; explicit groups tiered out."""
    same = 1.0 if same_explicit_group(first, second) else 0.0
    contained = 1.0 if contains_any(first, second) else 0.0
    distance = normalized_center_distance(first, second, canvas)
    if math.isfinite(distance):
        proximity = math.exp(-distance / ASSOCIATION_PROXIMITY_SCALE)
    else:
        proximity = 0.0
    aligned = 1.0 if is_aligned(first, second, canvas) else 0.0
    return (
        ASSOCIATION_WEIGHT_SAME_GROUP * same
        + ASSOCIATION_WEIGHT_CONTAINS * contained
        + ASSOCIATION_WEIGHT_PROXIMITY * proximity
        + ASSOCIATION_WEIGHT_ALIGNMENT * aligned
    )


def box_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    """Intersection over union for (x, y, w, h) boxes; 0 when union is 0."""
    first_x, first_y, first_w, first_h = first
    second_x, second_y, second_w, second_h = second
    inter_w = max(
        0, min(first_x + first_w, second_x + second_w) - max(first_x, second_x),
    )
    inter_h = max(
        0, min(first_y + first_h, second_y + second_h) - max(first_y, second_y),
    )
    inter = inter_w * inter_h
    union = first_w * first_h + second_w * second_h - inter
    if union <= 0:
        return 0.0
    return inter / union


def element_iou(first: Element, second: Element) -> float:
    """Stored-box IoU for two elements."""
    return box_iou(
        (first.x, first.y, first.width, first.height),
        (second.x, second.y, second.width, second.height),
    )


def min_pair_association(
    members: Sequence[Element], canvas: Canvas,
) -> float:
    """Minimum pairwise association; 1.0 when fewer than two members."""
    items = tuple(members)
    if len(items) < 2:
        return 1.0
    best = 1.0
    for pos in range(len(items)):
        for other in range(pos + 1, len(items)):
            score = association(items[pos], items[other], canvas)
            if score < best:
                best = score
    return best


def group_forces_split(members: Sequence[Element], canvas: Canvas) -> bool:
    """A group containing a background-like element must split."""
    return any(is_background_like(item, canvas) for item in members)


def group_is_ambiguous(members: Sequence[Element], canvas: Canvas) -> bool:
    """Weak internal cohesion without a forced split: try both modes."""
    items = tuple(members)
    return (
        len(items) >= 2
        and not group_forces_split(items, canvas)
        and min_pair_association(items, canvas) < AMBIGUOUS_GROUP_ASSOCIATION
    )


def _greedy_cluster(
    count: int, edges: Sequence[tuple[int, int]], cap: int,
) -> list[tuple[int, ...]]:
    """Union-find over priority-ordered edges with a per-cluster size cap."""
    if cap < 1:
        raise ValueError("cap must be at least 1")
    parent = list(range(count))
    size = [1] * count

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for first, second in edges:
        root_first, root_second = find(first), find(second)
        if root_first == root_second:
            continue
        if size[root_first] + size[root_second] > cap:
            continue
        if root_second < root_first:
            root_first, root_second = root_second, root_first
        parent[root_second] = root_first
        size[root_first] += size[root_second]
    clusters: dict[int, list[int]] = {}
    for node in range(count):
        clusters.setdefault(find(node), []).append(node)
    return [tuple(sorted(members)) for _, members in sorted(clusters.items())]


def split_unit(unit: LayoutUnit, canvas: Canvas) -> tuple[LayoutUnit, ...]:
    """Break a unit at its largest internal gap into 2-3 sub-clusters.

    Two members always become singletons. Longer units split on the axis
    with the largest stored-box gap (x wins ties); a second gap at least
    half the largest yields three clusters, else two. Fully overlapping
    units split by position order into balanced clusters. Members keep
    source order within each cluster; clusters order by min source index.
    """
    members = unit.members
    indices = unit.source_indices
    count = len(members)
    if count < SPLIT_MIN_CLUSTERS:
        return (unit,)
    if count == SPLIT_MIN_CLUSTERS:
        first = LayoutUnit.of((members[0],), (indices[0],), canvas)
        second = LayoutUnit.of((members[1],), (indices[1],), canvas)
        return (first, second)
    starts_x = [item.x for item in members]
    starts_y = [item.y for item in members]
    ends_x = [item.x + item.width for item in members]
    ends_y = [item.y + item.height for item in members]
    order_x = sorted(range(count), key=lambda k: (starts_x[k], indices[k]))
    order_y = sorted(range(count), key=lambda k: (starts_y[k], indices[k]))
    gaps_x = [
        starts_x[order_x[k + 1]] - ends_x[order_x[k]] for k in range(count - 1)
    ]
    gaps_y = [
        starts_y[order_y[k + 1]] - ends_y[order_y[k]] for k in range(count - 1)
    ]
    best_x = max(gaps_x)
    best_y = max(gaps_y)
    if best_x >= best_y:
        order, gaps = order_x, gaps_x
    else:
        order, gaps = order_y, gaps_y
    largest = max(gaps)
    first_cut = gaps.index(largest)
    cuts = [first_cut]
    if largest > 0:
        runner_up = max(
            (gap for pos, gap in enumerate(gaps) if pos != first_cut),
            default=0,
        )
        if runner_up >= SPLIT_SECOND_GAP_RATIO * largest and runner_up > 0:
            second_cut = next(
                pos for pos, gap in enumerate(gaps)
                if gap == runner_up and pos != first_cut
            )
            cuts = sorted([first_cut, second_cut])
        slices = []
        start = 0
        for cut in cuts:
            slices.append(order[start:cut + 1])
            start = cut + 1
        slices.append(order[start:])
    else:
        parts = min(SPLIT_MAX_CLUSTERS, count)
        base, extra = divmod(count, parts)
        slices = []
        start = 0
        for part in range(parts):
            width = base + (1 if part < extra else 0)
            slices.append(order[start:start + width])
            start += width
    clusters = []
    for sliver in slices:
        ordered = sorted(sliver, key=lambda k: indices[k])
        clusters.append(
            LayoutUnit.of(
                tuple(members[k] for k in ordered),
                tuple(indices[k] for k in ordered),
                canvas,
            )
        )
    clusters.sort(key=lambda cluster: min(cluster.source_indices))
    return tuple(clusters)


def build_layout_units(
    source: Design, *, group_mode: str = GROUP_MODE_COHESIVE,
) -> tuple[LayoutUnit, ...]:
    """Partition source elements into layout units for candidate search.

    Explicit groups stay cohesive except forced background splits; in split
    mode ambiguous groups also split at their largest gap. Background-like
    elements are always singletons. Un grouped elements cluster only over
    association edges at or above threshold, capped per unit. Target-fit
    forcing happens during generator slot assignment, not here.
    """
    if group_mode not in (GROUP_MODE_COHESIVE, GROUP_MODE_SPLIT):
        raise ValueError(
            f"group_mode must be {GROUP_MODE_COHESIVE!r} or {GROUP_MODE_SPLIT!r}",
        )
    canvas = source.source_canvas
    elements = source.elements
    grouped: dict[str, list[int]] = {}
    ungrouped: list[int] = []
    for pos, item in enumerate(elements):
        if isinstance(item.group_id, str):
            grouped.setdefault(item.group_id, []).append(pos)
        elif not is_background_like(item, canvas):
            ungrouped.append(pos)
    units: list[LayoutUnit] = []
    for positions in grouped.values():
        members = tuple(elements[pos] for pos in positions)
        if len(positions) < 2:
            units.append(LayoutUnit.of(members, tuple(positions), canvas))
            continue
        if group_forces_split(members, canvas):
            background = [
                pos for pos in positions
                if is_background_like(elements[pos], canvas)
            ]
            background_set = set(background)
            rest = [pos for pos in positions if pos not in background_set]
            for pos in background:
                units.append(
                    LayoutUnit.of((elements[pos],), (pos,), canvas),
                )
            if rest:
                units.append(
                    LayoutUnit.of(
                        tuple(elements[pos] for pos in rest), tuple(rest), canvas,
                    ),
                )
            continue
        if group_mode == GROUP_MODE_SPLIT and group_is_ambiguous(
            members, canvas,
        ):
            cohesive = LayoutUnit.of(members, tuple(positions), canvas)
            units.extend(split_unit(cohesive, canvas))
            continue
        units.append(LayoutUnit.of(members, tuple(positions), canvas))
    if len(ungrouped) > 1:
        scored: list[tuple[float, int, int]] = []
        for left_pos in range(len(ungrouped)):
            for right_pos in range(left_pos + 1, len(ungrouped)):
                left = ungrouped[left_pos]
                right = ungrouped[right_pos]
                score = association(elements[left], elements[right], canvas)
                if score >= ASSOCIATION_THRESHOLD:
                    scored.append((-score, left, right))
        scored.sort()
        position_of = {pos: rank for rank, pos in enumerate(ungrouped)}
        edges = [
            (position_of[left], position_of[right])
            for _, left, right in scored
        ]
        for cluster in _greedy_cluster(len(ungrouped), edges, MAX_UNIT_SIZE):
            positions = tuple(ungrouped[rank] for rank in cluster)
            units.append(
                LayoutUnit.of(
                    tuple(elements[pos] for pos in positions), positions, canvas,
                ),
            )
    else:
        for pos in ungrouped:
            units.append(LayoutUnit.of((elements[pos],), (pos,), canvas))
    grouped_positions = {
        pos for positions in grouped.values() for pos in positions
    }
    for pos, item in enumerate(elements):
        if pos not in grouped_positions and is_background_like(item, canvas):
            units.append(LayoutUnit.of((item,), (pos,), canvas))
    units.sort(key=lambda unit: min(unit.source_indices))
    return tuple(units)


def required_group_modes(source: Design) -> tuple[str, ...]:
    """Group modes the generator should try: both when any group is ambiguous."""
    canvas = source.source_canvas
    seen: dict[str, list[Element]] = {}
    for item in source.elements:
        if isinstance(item.group_id, str):
            seen.setdefault(item.group_id, []).append(item)
    for members in seen.values():
        if len(members) >= 2 and group_is_ambiguous(members, canvas):
            return (GROUP_MODE_COHESIVE, GROUP_MODE_SPLIT)
    return (GROUP_MODE_COHESIVE,)
