"""Bounded post-stretch local layout refinement for S04 (D.1.8).

Takes an already valid candidate and tries to improve it through small, deliberate
geometric edits: unit translations, text-box expansion with font growth, shape/panel
fitting, label recentering, bar lengthening, image rescaling and gap equalization.
It is a deterministic best-improvement local search over the EXISTING deterministic
scorer and validator; it never replaces either and never widens the K6 allowlist
(x, y, width, height, text font_size; every edit goes through `layout_edit`).

Search. Step sizes shrink coarse to fine (`step_fractions` of each target axis). At
each step, proposals are generated in three priority tiers and evaluated tier by tier;
the best proposal that beats the current score by more than `min_improvement` is
accepted and the search restarts from the cheapest tier. A step ends when no tier
improves; the search ends when all steps are done or a budget is spent.
- Tier 1, defects: text below the readable-size preference or overflowing, unwanted
  overlaps or crowding of unrelated units, broken shape-content containment.
- Tier 2, polish: alignment drift, uneven gaps, off-center button labels, container
  padding, bar length.
- Tier 3, opportunistic: fill blank space by scaling whole units up about their center,
  tighten text boxes, grow text, rescale images, center content.

Readability first. Text below `min_readable_font` (a configurable quality preference,
not an Adobe rule) is a tier-1 defect. Before any font is allowed to shrink, the search
tries larger boxes, other anchors, and pushing neighboring units out of the way
(`_resolve`), growing the font along a ladder up to the source size and never above a
text that was larger in the source. Shrinking is never a hard rejection: once a text fit
shrinks a font by more than `font_shrink_onset` (a fraction of the starting size), a
penalty that grows quadratically with the excess shrink is subtracted from the score the
search compares (`font_shrink_weight` scales it). The reported scores stay unpenalized.

Protection. Units (explicit groups, button shape+label, source-overlapping/contained
foreground elements) move rigidly, so group offsets and intentional overlaps survive.
Backdrop panels and backgrounds never translate; they are resized to keep holding their
contents. A proposal is rejected, never repaired, if it breaks a containment that held
in the starting layout, enlarges an unintentional overlap, leaves the canvas, fails
`validate_candidate` before or after text fitting, or yields a non-finite score.
Intentional overlap is read from the SOURCE: same group, containment, IoU above the
D.1.3 threshold, or background-like. Randomness is not used (the config seed is
reserved); ties break by tier, operator order and proposal order. Every state is
evaluated at most once, which also rules out oscillation.

The result always carries a valid layout: the input candidate when nothing improves.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from engine.internal_schema import Canvas, Design
from engine.layout_edits import layout_edit
from engine.layout_typography import SINGLE_LINE_EM, estimated_fit
from engine.layout_units import box_iou, contains_any, is_background_like, same_explicit_group
from engine.shape_roles import ShapeRole, classify_shapes
from engine.shape_stretch import _Context, _stretch_bar

__all__ = [
    "REFINEMENT_VERSION",
    "RefinementConfig",
    "RefinementResult",
    "diagnose",
    "refine_candidate",
]

REFINEMENT_VERSION = "layout_refinement_v1"

Box = tuple[float, float, float, float]
_EPS = 1e-6
# Overlap area (px^2) at or below this is float dust, not an overlap.
_OVERLAP_DUST = 1e-6
# Intentional-overlap IoU boundary, the same value as D.1.3 / the objective.
_SOURCE_OVERLAP_IOU = 0.02
_LABEL_WIDTH_SLACK = 1.04
_LABEL_PIXEL_SLACK = 2.01
_LINE_EM = SINGLE_LINE_EM * 1.001
_FONT_LADDER = (1.1, 1.25, 1.5, 2.0)
# Whole-unit scale ratios for blank-space filling (the unit grows about its center).
_FILL_LADDER = (1.15, 1.35, 1.6, 2.0)
_PAD_X_EM = (0.4, 2.5)
_PAD_Y_EM = (0.15, 1.2)
_MAX_RESOLVE_PASSES = 3
_ALIGN_SOURCE_TOLERANCE = 0.02
_ALIGN_MAX_DRIFT_FRACTION = 0.15


@dataclass(frozen=True)
class RefinementConfig:
    """All search limits and preferences. Every field is a quality/resource choice.

    max_evaluations bounds validated+scored proposals per candidate (the work budget);
    max_rounds bounds accepted improvements per candidate; max_proposals_per_tier bounds
    proposals evaluated per tier per round. Proposal GENERATION is cheap geometry and is
    not counted, but is itself bounded by the proposal cap per operator family.
    """

    max_evaluations: int = 120
    max_rounds: int = 8
    max_proposals_per_tier: int = 24
    step_fractions: tuple[float, ...] = (0.08, 0.04, 0.02, 0.01)
    min_improvement: float = 1e-4
    min_gap_fraction: float = 0.03
    min_readable_font: float = 8.0
    # Font shrink relative to the starting layout is free up to this fraction, then penalized
    # (quadratically in the excess) rather than rejected: a quality preference, not a rule.
    font_shrink_onset: float = 0.2
    font_shrink_weight: float = 0.2
    seed: int = 0

    def __post_init__(self):
        for name in ("max_evaluations", "max_rounds", "max_proposals_per_tier"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not self.step_fractions or not all(
                math.isfinite(f) and 0 < f <= 1 for f in self.step_fractions):
            raise ValueError("step_fractions must be non-empty fractions in (0, 1]")
        for name in ("min_improvement", "min_gap_fraction", "min_readable_font"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0 <= self.font_shrink_onset < 1:
            raise ValueError("font_shrink_onset must be in [0, 1)")
        if not math.isfinite(self.font_shrink_weight) or self.font_shrink_weight < 0:
            raise ValueError("font_shrink_weight must be finite and non-negative")


@dataclass(frozen=True)
class RefinementResult:
    """Outcome of one refinement. `adobe` is always a valid layout (the input if unchanged)."""

    adobe: dict[str, Any]
    changed: bool
    score_before: float
    score_after: float
    operators: tuple[str, ...]
    evaluations: int
    rejected_invalid: int
    rejected_invariant: int
    duplicates: int
    errors: int
    accepted: int
    stopped: str


@dataclass(frozen=True)
class _Proposal:
    op: str
    boxes: Mapping[str, Box]
    fonts: Mapping[str, float]


@dataclass
class _State:
    boxes: dict[str, Box]
    fonts: dict[str, float]

    def copy(self) -> _State:
        return _State(dict(self.boxes), dict(self.fonts))

    def key(self) -> tuple:
        return (tuple(sorted((i, tuple(round(v, 6) for v in b)) for i, b in self.boxes.items())),
                tuple(sorted((i, round(f, 6)) for i, f in self.fonts.items())))


def _extract(adobe: Mapping[str, Any]) -> _State:
    return _State(
        {e["id"]: (float(e["x"]), float(e["y"]), float(e["width"]), float(e["height"]))
         for e in adobe["elements"]},
        {e["id"]: float(e["font_size"]) for e in adobe["elements"] if e["type"] == "text"})


def _overlap_area(a: Box, b: Box) -> float:
    w = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    h = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def _inside(inner: Box, outer: Box, tol: float = 1e-6) -> bool:
    return (inner[0] >= outer[0] - tol and inner[1] >= outer[1] - tol
            and inner[0] + inner[2] <= outer[0] + outer[2] + tol
            and inner[1] + inner[3] <= outer[1] + outer[3] + tol)


def _clip(value: float, bounds: tuple[float, float]) -> float:
    return min(bounds[1], max(bounds[0], value))


class _Search:
    def __init__(self, source: Design, target: Canvas, candidate: Mapping[str, Any],
                 score: Callable[[dict], float], validate: Callable[..., Any],
                 fit_text: Callable[[dict], dict] | None, measure: Callable | None,
                 config: RefinementConfig) -> None:
        self.source, self.target, self.config = source, target, config
        self.score_fn, self.validate_fn, self.fit_fn = score, validate, fit_text
        self.measure = measure or estimated_fit
        self.width, self.height = float(target.width), float(target.height)
        self.min_gap = config.min_gap_fraction * min(self.width, self.height)
        self.elements = list(source.elements)
        self.by_id = source.elements_by_id
        self.roles = classify_shapes(source)
        self.canvas = source.source_canvas
        self.start_adobe = {"canvas": dict(candidate["canvas"]),
                            "elements": [dict(e) for e in candidate["elements"]]}
        self.items = {e["id"]: e for e in self.start_adobe["elements"]}
        self.start = _extract(self.start_adobe)
        self._build_structure()
        self.seen: set[tuple] = set()
        self.counts = dict(evaluations=0, invalid=0, invariant=0, duplicates=0, errors=0)

    # ---- structure: units, intentional overlaps, invariants -----------------------
    def _backdrop(self, ident: str) -> bool:
        info = self.roles.get(ident)
        return bool(info and info.role in (ShapeRole.PANEL, ShapeRole.BACKGROUND)) \
            or is_background_like(self.by_id[ident], self.canvas)

    def _build_structure(self) -> None:
        ids = [e.id for e in self.elements]
        parent = {i: i for i in ids}

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

        for a_pos, a in enumerate(self.elements):
            for b in self.elements[a_pos + 1:]:
                if same_explicit_group(a, b):
                    union(a.id, b.id)
                    continue
                if self._backdrop(a.id) or self._backdrop(b.id):
                    continue
                if contains_any(a, b) or box_iou(
                        (a.x, a.y, a.width, a.height), (b.x, b.y, b.width, b.height)
                ) > _SOURCE_OVERLAP_IOU:
                    union(a.id, b.id)
        for ident, info in self.roles.items():
            if info.role is ShapeRole.BUTTON:
                for inner in info.contained_ids:
                    union(ident, inner)
        groups: dict[str, list[str]] = {}
        for ident in ids:
            groups.setdefault(find(ident), []).append(ident)
        self.units = {root: tuple(members) for root, members in groups.items()}
        self.unit_of = {i: root for root, members in self.units.items() for i in members}
        self.immovable = {i for i in ids if self._backdrop(i)}
        self.intentional = {
            (a.id, b.id) for a_pos, a in enumerate(self.elements) for b in self.elements[a_pos + 1:]
            if self.unit_of[a.id] == self.unit_of[b.id] or self._backdrop(a.id) or self._backdrop(b.id)
            or contains_any(a, b) or same_explicit_group(a, b)
            or box_iou((a.x, a.y, a.width, a.height), (b.x, b.y, b.width, b.height)) > _SOURCE_OVERLAP_IOU}
        self.eligible_pairs = [(a.id, b.id) for a_pos, a in enumerate(self.elements)
                               for b in self.elements[a_pos + 1:]
                               if (a.id, b.id) not in self.intentional
                               and a.width > 0 and a.height > 0 and b.width > 0 and b.height > 0]
        self.containments = [
            (a.id, b.id) for a in self.elements for b in self.elements
            if a.id != b.id and not is_background_like(a, self.canvas)
            and a.width > 0 and a.height > 0 and b.width > 0 and b.height > 0
            and _inside((b.x, b.y, b.width, b.height), (a.x, a.y, a.width, a.height), 0.0)]
        self.held = [(o, i) for o, i in self.containments
                     if _inside(self.start.boxes[i], self.start.boxes[o])]
        self.start_overlap = {p: _overlap_area(self.start.boxes[p[0]], self.start.boxes[p[1]])
                              for p in self.eligible_pairs}
        self.texts = sorted((e for e in self.elements if e.type == "text"),
                            key=lambda e: (-float(e.font_size), self.elements.index(e)))

    def invariants_ok(self, state: _State) -> bool:
        if any(not _inside(state.boxes[i], state.boxes[o]) for o, i in self.held):
            return False
        for pair in self.eligible_pairs:
            area = _overlap_area(state.boxes[pair[0]], state.boxes[pair[1]])
            if area > self.start_overlap[pair] + _OVERLAP_DUST:
                return False
        return True

    # ---- text helpers ----------------------------------------------------------------
    def _probe(self, ident: str, box: Box, font: float) -> dict:
        item = dict(self.items[ident])
        item.update(x=box[0], y=box[1], width=box[2], height=box[3], font_size=font)
        return item

    def fits(self, ident: str, box: Box, font: float) -> bool:
        return bool(self.measure(self._probe(ident, box, font))["fits"])

    def natural_width(self, ident: str, font: float) -> float:
        probe = self._probe(ident, (0, 0, 1e6, 1e6), font)
        return float(self.measure(probe)["measured_width"])

    def font_cap(self, text, state: _State) -> float:
        larger = [state.fonts[o.id] for o in self.texts
                  if o.id != text.id and float(o.font_size) > float(text.font_size)]
        return min([float(text.font_size), *larger]) if larger else float(text.font_size)

    def single_line(self, text) -> bool:
        return ("\n" not in text.content
                and float(text.height) <= 1.8 * float(text.font_size)
                * float(text.effective_value("line_height")))

    # ---- geometry helpers -----------------------------------------------------------------
    def clamp_box(self, box: Box) -> Box:
        x = min(max(box[0], _EPS), max(_EPS, self.width - box[2] - _EPS))
        y = min(max(box[1], _EPS), max(_EPS, self.height - box[3] - _EPS))
        return (x, y, box[2], box[3])

    def unit_bbox(self, state: _State, ids) -> Box:
        boxes = [state.boxes[i] for i in ids]
        left, top = min(b[0] for b in boxes), min(b[1] for b in boxes)
        right, bottom = max(b[0] + b[2] for b in boxes), max(b[1] + b[3] for b in boxes)
        return (left, top, right - left, bottom - top)

    def translate(self, state: _State, ids, dx: float, dy: float, *, clamp: bool = True
                  ) -> dict[str, Box] | None:
        if any(i in self.immovable for i in ids):
            return None
        bx, by, bw, bh = self.unit_bbox(state, ids)
        lo_x, hi_x = _EPS - bx, self.width - _EPS - (bx + bw)
        lo_y, hi_y = _EPS - by, self.height - _EPS - (by + bh)
        if not clamp and not (lo_x <= dx <= hi_x and lo_y <= dy <= hi_y):
            return None
        dx, dy = min(max(dx, lo_x), max(lo_x, hi_x)), min(max(dy, lo_y), max(lo_y, hi_y))
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return None
        return {i: (state.boxes[i][0] + dx, state.boxes[i][1] + dy, state.boxes[i][2], state.boxes[i][3])
                for i in ids}

    def _resolve(self, state: _State, changed: set[str]) -> _State | None:
        """Push units out of newly created unintentional overlaps (reflow neighbors)."""
        state = state.copy()
        changed = set(changed)
        for _ in range(_MAX_RESOLVE_PASSES):
            clash = next(((a, b) for a, b in self.eligible_pairs
                          if (a in changed) != (b in changed)
                          and _overlap_area(state.boxes[a], state.boxes[b])
                          > self.start_overlap[(a, b)] + _OVERLAP_DUST), None)
            if clash is None:
                return state
            mover, anchor = (clash[1], clash[0]) if clash[0] in changed else (clash[0], clash[1])
            ids = self.units[self.unit_of[mover]]
            if any(i in changed for i in ids):
                return None
            moved_box, fixed = state.boxes[mover], state.boxes[anchor]
            need_x = min(fixed[0] + fixed[2], moved_box[0] + moved_box[2]) - max(fixed[0], moved_box[0]) + self.min_gap
            need_y = min(fixed[1] + fixed[3], moved_box[1] + moved_box[3]) - max(fixed[1], moved_box[1]) + self.min_gap
            options = []
            for axis, need in (("x", need_x), ("y", need_y)):
                for sign in (1, -1):
                    away = ((moved_box[0] + moved_box[2] / 2) - (fixed[0] + fixed[2] / 2)) if axis == "x" \
                        else ((moved_box[1] + moved_box[3] / 2) - (fixed[1] + fixed[3] / 2))
                    if (sign > 0) != (away >= 0):
                        continue
                    options.append((need, axis, sign))
            moved = None
            for need, axis, sign in sorted(options):
                moved = self.translate(state, ids, sign * need if axis == "x" else 0.0,
                                       sign * need if axis == "y" else 0.0, clamp=False)
                if moved is not None:
                    break
            if moved is None:
                return None
            state.boxes.update(moved)
            changed.update(ids)
        return None

    # ---- proposals ---------------------------------------------------------------------------------
    def _pair_clearance(self, state: _State, a: str, b: str) -> tuple[float, float]:
        ba, bb = state.boxes[a], state.boxes[b]
        gx = max(bb[0] - (ba[0] + ba[2]), ba[0] - (bb[0] + bb[2]))
        gy = max(bb[1] - (ba[1] + ba[3]), ba[1] - (bb[1] + bb[3]))
        return gx, gy

    def crowded_pairs(self, state: _State) -> list[tuple[str, str]]:
        found = []
        for a, b in self.eligible_pairs:
            if self._backdrop(a) or self._backdrop(b):
                continue
            gx, gy = self._pair_clearance(state, a, b)
            if (gx < 0 and gy < self.min_gap) or (gy < 0 and gx < self.min_gap) \
                    or (gx >= 0 and gy >= 0 and math.hypot(gx, gy) < self.min_gap):
                found.append((a, b))
        return found

    def small_texts(self, state: _State) -> list:
        return [t for t in self.texts if t.content
                and state.fonts[t.id] < min(self.config.min_readable_font, float(t.font_size)) - 1e-9]

    def unfit_texts(self, state: _State) -> list:
        return [t for t in self.texts if t.content
                and not self.fits(t.id, state.boxes[t.id], state.fonts[t.id])]

    def broken_containers(self, state: _State) -> list[str]:
        return [i for i, info in self.roles.items() if info.contained_ids
                and info.role in (ShapeRole.PANEL, ShapeRole.BUTTON)
                and any(not _inside(state.boxes[c], state.boxes[i]) for c in info.contained_ids)]

    def off_center_labels(self, state: _State) -> list[str]:
        out = []
        for i, info in self.roles.items():
            if info.role is ShapeRole.BUTTON:
                s, lab = state.boxes[i], state.boxes[info.contained_ids[0]]
                if abs((s[0] + s[2] / 2) - (lab[0] + lab[2] / 2)) > 0.5 or abs((s[1] + s[3] / 2) - (lab[1] + lab[3] / 2)) > 0.5:
                    out.append(i)
        return out

    def _make(self, op: str, state: _State, new_boxes: Mapping[str, Box] | None,
              new_fonts: Mapping[str, float] | None = None, resolve: bool = False) -> _Proposal | None:
        trial = state.copy()
        trial.boxes.update(new_boxes or {})
        trial.fonts.update(new_fonts or {})
        if resolve:
            trial = self._resolve(trial, set(new_boxes or {}))
            if trial is None:
                return None
        boxes = {i: b for i, b in trial.boxes.items() if b != state.boxes[i]}
        fonts = {i: f for i, f in trial.fonts.items() if f != state.fonts[i]}
        return _Proposal(op, boxes, fonts) if boxes or fonts else None

    def separate_proposals(self, state: _State, st_x: float, st_y: float) -> list[_Proposal]:
        out = []
        for a, b in self.crowded_pairs(state):
            ua, ub = self.units[self.unit_of[a]], self.units[self.unit_of[b]]
            if ua == ub:
                continue
            ba, bb = state.boxes[a], state.boxes[b]
            gx, gy = self._pair_clearance(state, a, b)
            for axis in ("y", "x"):
                gap = gy if axis == "y" else gx
                need = self.min_gap - gap
                if need <= 0:
                    continue
                if axis == "y":
                    sign = 1.0 if (bb[1] + bb[3] / 2) >= (ba[1] + ba[3] / 2) else -1.0
                else:
                    sign = 1.0 if (bb[0] + bb[2] / 2) >= (ba[0] + ba[2] / 2) else -1.0
                d = (sign * need if axis == "x" else 0.0, sign * need if axis == "y" else 0.0)
                for mover, share in ((ub, 1.0), (ua, -1.0)):
                    moved = self.translate(state, mover, d[0] * share, d[1] * share)
                    if moved is not None:
                        out.append(self._make("separate", state, moved))
                half = self.translate(state, ub, d[0] / 2, d[1] / 2)
                other = self.translate(state, ua, -d[0] / 2, -d[1] / 2)
                if half is not None and other is not None:
                    out.append(self._make("separate", state, {**half, **other}))
        return [p for p in out if p]

    def text_growth_proposals(self, state: _State, text) -> list[_Proposal]:
        """Readability: larger font with a larger box, other anchors, neighbors pushed away."""
        ident = text.id
        box, font = state.boxes[ident], state.fonts[ident]
        cap = self.font_cap(text, state)
        out = []
        for ratio in _FONT_LADDER:
            target_font = font * ratio
            if target_font > cap + 1e-9:
                target_font = cap
                if target_font <= font * 1.01:
                    continue
            if self.single_line(text):
                width = self.natural_width(ident, target_font) * _LABEL_WIDTH_SLACK + _LABEL_PIXEL_SLACK
                height = target_font * _LINE_EM + _LABEL_PIXEL_SLACK
            else:
                scale = target_font / font
                width, height = box[2] * scale, box[3] * scale
            if width > self.width - 2 * _EPS or height > self.height - 2 * _EPS:
                continue
            align = text.effective_value("text_align")
            anchors = []
            cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
            anchors.append(("keep", (box[0] if align != "right" else box[0] + box[2] - width,
                                     box[1])))
            anchors.append(("center", (cx - width / 2, cy - height / 2)))
            anchors.append(("bottom", (box[0] if align != "right" else box[0] + box[2] - width,
                                       box[1] + box[3] - height)))
            for _, (x, y) in anchors:
                new_box = self.clamp_box((x, y, width, height))
                boxes = {ident: new_box}
                info = self._button_of(ident)
                if info is not None:
                    shape_id, pads = info
                    px, py = pads[0] * target_font, pads[1] * target_font
                    sw, sh_h = new_box[2] + 2 * px, new_box[3] + 2 * py
                    scx, scy = new_box[0] + new_box[2] / 2, new_box[1] + new_box[3] / 2
                    shape_box = self.clamp_box((scx - sw / 2, scy - sh_h / 2, sw, sh_h))
                    boxes[shape_id] = shape_box
                    if not _inside(new_box, shape_box):
                        continue
                proposal = self._make("grow_text", state, boxes, {ident: target_font}, resolve=True)
                if proposal:
                    out.append(proposal)
        return out

    def _button_of(self, label_id: str):
        for ident, info in self.roles.items():
            if info.role is ShapeRole.BUTTON and info.contained_ids and info.contained_ids[0] == label_id:
                shape, label = self.by_id[ident], self.by_id[label_id]
                size = float(label.font_size)
                nat = self.natural_width(label_id, size)
                px = _clip((float(shape.width) - nat) / 2 / size, _PAD_X_EM)
                py = _clip((float(shape.height) - _LINE_EM * size) / 2 / size, _PAD_Y_EM)
                return ident, (px, py)
        return None

    def text_expand_proposals(self, state: _State, text, st_x: float, st_y: float) -> list[_Proposal]:
        ident = text.id
        x, y, w, h = state.boxes[ident]
        out = []
        for name, new in (("right", (x, y, w + st_x, h)), ("left", (x - st_x, y, w + st_x, h)),
                          ("both", (x - st_x / 2, y, w + st_x, h)), ("down", (x, y, w, h + st_y)),
                          ("up", (x, y - st_y, w, h + st_y))):
            box = self.clamp_box(new)
            out.append(self._make("expand_text", state, {ident: box}, resolve=True))
        return [p for p in out if p]

    def text_tighten_proposals(self, state: _State, text) -> list[_Proposal]:
        ident = text.id
        x, y, w, h = state.boxes[ident]
        font = state.fonts[ident]
        if not text.content or "\n" in text.content:
            return []
        out = []
        nat = self.natural_width(ident, font) * _LABEL_WIDTH_SLACK + _LABEL_PIXEL_SLACK
        if w > nat * 1.15 + 4:
            align = text.effective_value("text_align")
            nx = x if align == "left" else (x + w - nat if align == "right" else x + (w - nat) / 2)
            out.append(self._make("tighten_text", state, {ident: (nx, y, nat, h)}))
        need_h = font * _LINE_EM + _LABEL_PIXEL_SLACK
        if h > need_h * 1.3 + 2 and self.fits(ident, (x, y, w, need_h), font):
            out.append(self._make("tighten_text", state, {ident: (x, y + (h - need_h) / 2, w, need_h)}))
        return [p for p in out if p]

    def container_proposals(self, state: _State) -> list[_Proposal]:
        out = []
        for ident, info in self.roles.items():
            if not info.contained_ids or info.role not in (ShapeRole.PANEL, ShapeRole.BUTTON):
                continue
            shape = self.by_id[ident]
            held = [state.boxes[c] for c in info.contained_ids]
            left, top = min(b[0] for b in held), min(b[1] for b in held)
            right, bottom = max(b[0] + b[2] for b in held), max(b[1] + b[3] for b in held)
            src = [self.by_id[c] for c in info.contained_ids]
            s_left, s_top = min(float(e.x) for e in src), min(float(e.y) for e in src)
            s_right, s_bottom = max(float(e.x + e.width) for e in src), max(float(e.y + e.height) for e in src)
            unit = 1.0
            if s_right > s_left and s_bottom > s_top:
                unit = min((right - left) / (s_right - s_left), (bottom - top) / (s_bottom - s_top))
            pad = (max(0.0, s_left - float(shape.x)) * unit, max(0.0, s_top - float(shape.y)) * unit,
                   max(0.0, float(shape.x + shape.width) - s_right) * unit,
                   max(0.0, float(shape.y + shape.height) - s_bottom) * unit)
            current = state.boxes[ident]
            if info.role is ShapeRole.BUTTON:
                px, py = (pad[0] + pad[2]) / 2, (pad[1] + pad[3]) / 2
                new = (left - px, top - py, right - left + 2 * px, bottom - top + 2 * py)
            else:
                new = (min(current[0], left - pad[0]), min(current[1], top - pad[1]),
                       max(current[0] + current[2], right + pad[2]) - min(current[0], left - pad[0]),
                       max(current[1] + current[3], bottom + pad[3]) - min(current[1], top - pad[1]))
            out.append(self._make("fit_container", state, {ident: self.clamp_box(new)}, resolve=True))
        return [p for p in out if p]

    def label_proposals(self, state: _State) -> list[_Proposal]:
        out = []
        for ident in self.off_center_labels(state):
            label_id = self.roles[ident].contained_ids[0]
            s, lab = state.boxes[ident], state.boxes[label_id]
            moved = (s[0] + (s[2] - lab[2]) / 2, s[1] + (s[3] - lab[3]) / 2, lab[2], lab[3])
            if _inside(moved, s):
                out.append(self._make("recenter_label", state, {label_id: moved}))
            recentered = (lab[0] + lab[2] / 2 - s[2] / 2, lab[1] + lab[3] / 2 - s[3] / 2, s[2], s[3])
            out.append(self._make("recenter_label", state, {ident: self.clamp_box(recentered)}))
        return [p for p in out if p]

    def alignment_proposals(self, state: _State) -> list[_Proposal]:
        out = []
        for a, b in self.eligible_pairs:
            ua, ub = self.units[self.unit_of[a]], self.units[self.unit_of[b]]
            if ua == ub or self._backdrop(a) or self._backdrop(b):
                continue
            ea, eb = self.by_id[a], self.by_id[b]
            sa = ((float(ea.x), float(ea.x + ea.width / 2), float(ea.x + ea.width)),
                  (float(ea.y), float(ea.y + ea.height / 2), float(ea.y + ea.height)))
            sb = ((float(eb.x), float(eb.x + eb.width / 2), float(eb.x + eb.width)),
                  (float(eb.y), float(eb.y + eb.height / 2), float(eb.y + eb.height)))
            ba, bb = state.boxes[a], state.boxes[b]
            ca = ((ba[0], ba[0] + ba[2] / 2, ba[0] + ba[2]), (ba[1], ba[1] + ba[3] / 2, ba[1] + ba[3]))
            cb = ((bb[0], bb[0] + bb[2] / 2, bb[0] + bb[2]), (bb[1], bb[1] + bb[3] / 2, bb[1] + bb[3]))
            for axis, size in ((0, float(self.canvas.width)), (1, float(self.canvas.height))):
                target_size = self.width if axis == 0 else self.height
                for slot in range(3):
                    if abs(sa[axis][slot] - sb[axis][slot]) / size > _ALIGN_SOURCE_TOLERANCE:
                        continue
                    drift = cb[axis][slot] - ca[axis][slot]
                    if abs(drift) < 0.5 or abs(drift) > _ALIGN_MAX_DRIFT_FRACTION * target_size:
                        continue
                    for mover, sign in ((ub, -1.0), (ua, 1.0)):
                        d = sign * drift
                        moved = self.translate(state, mover, d if axis == 0 else 0.0, d if axis == 1 else 0.0)
                        if moved is not None:
                            out.append(self._make("snap", state, moved))
        return [p for p in out if p]

    def spacing_proposals(self, state: _State) -> list[_Proposal]:
        content = [u for u in self.units.values() if not any(i in self.immovable for i in u)]
        out = []
        horizontal = self.width >= self.height
        axis = 0 if horizontal else 1
        extents = sorted(((self.unit_bbox(state, u), u) for u in content),
                         key=lambda pair: (pair[0][axis], pair[1]))
        if len(extents) >= 3:
            starts = [e[0][axis] for e in extents]
            sizes = [e[0][axis + 2] for e in extents]
            ends = [s + z for s, z in zip(starts, sizes)]
            gaps = [starts[k + 1] - ends[k] for k in range(len(extents) - 1)]
            if all(g >= 0 for g in gaps):
                span = starts[-1] - ends[0]
                equal = (span - sum(sizes[1:-1])) / (len(extents) - 1)
                if equal >= 0 and max(gaps) - min(gaps) > 0.5:
                    moves, cursor = {}, ends[0]
                    for k in range(1, len(extents) - 1):
                        d = (cursor + equal) - starts[k]
                        moved = self.translate(state, extents[k][1], d if axis == 0 else 0.0,
                                               d if axis == 1 else 0.0, clamp=False)
                        if moved is not None:
                            moves.update(moved)
                        cursor = cursor + equal + sizes[k]
                    if moves:
                        out.append(self._make("equalize_gaps", state, moves))
        return [p for p in out if p]

    def center_proposals(self, state: _State) -> list[_Proposal]:
        content = [i for u in self.units.values() if not any(m in self.immovable for m in u) for i in u]
        if not content:
            return []
        bx, by, bw, bh = self.unit_bbox(state, content)
        dx, dy = self.width / 2 - (bx + bw / 2), self.height / 2 - (by + bh / 2)
        out = []
        for ddx, ddy in ((dx, dy), (dx, 0.0), (0.0, dy)):
            moved = self.translate(state, tuple(content), ddx, ddy, clamp=False)
            if moved is not None:
                out.append(self._make("center_content", state, moved))
        return [p for p in out if p]

    def bar_proposals(self, state: _State) -> list[_Proposal]:
        out = []
        for ident, info in self.roles.items():
            if info.role not in (ShapeRole.BAR_H, ShapeRole.BAR_V):
                continue
            adobe = self._emit(state)
            ctx = _Context(self.source, self.target, adobe, self.measure)
            if _stretch_bar(ctx, self.by_id[ident], info.role is ShapeRole.BAR_H, 1.0):
                out.append(self._make("bar_extent", state, {ident: ctx.boxes[ident]}))
        return [p for p in out if p]

    def fill_proposals(self, state: _State) -> list[_Proposal]:
        """Fill blank space: scale a whole unit up about its center, fonts included.

        The unit's internal offsets scale with it, so layering and alignment inside it are
        kept; neighbors are pushed aside by `_resolve`; fonts never exceed the source size.
        """
        out = []
        for members in self.units.values():
            if any(i in self.immovable for i in members):
                continue
            bx, by, bw, bh = self.unit_bbox(state, members)
            if bw <= 0 or bh <= 0:
                continue
            cx, cy = bx + bw / 2, by + bh / 2
            limit = min((cx - _EPS) / (bw / 2), (self.width - _EPS - cx) / (bw / 2),
                        (cy - _EPS) / (bh / 2), (self.height - _EPS - cy) / (bh / 2))
            ratios = [r for r in _FILL_LADDER if r <= limit]
            if limit > 1.05 and (not ratios or limit < _FILL_LADDER[-1] and limit > ratios[-1] * 1.05):
                ratios.append(limit)
            for ratio in ratios:
                boxes = {i: (cx + (state.boxes[i][0] - cx) * ratio, cy + (state.boxes[i][1] - cy) * ratio,
                             state.boxes[i][2] * ratio, state.boxes[i][3] * ratio) for i in members}
                fonts = {}
                for i in members:
                    if i in state.fonts:
                        fonts[i] = max(state.fonts[i], min(state.fonts[i] * ratio,
                                                           self.font_cap(self.by_id[i], state)))
                proposal = self._make("fill_space", state, boxes, fonts, resolve=True)
                if proposal:
                    out.append(proposal)
        return out

    def image_proposals(self, state: _State, frac: float) -> list[_Proposal]:
        out = []
        for e in self.elements:
            if e.type != "image" or e.id in self.immovable or e.effective_value("fit") == "cover":
                continue
            x, y, w, h = state.boxes[e.id]
            for factor in (1 + 2 * frac, 1 - 2 * frac):
                nw, nh = w * factor, h * factor
                box = self.clamp_box((x + (w - nw) / 2, y + (h - nh) / 2, nw, nh))
                if abs(box[2] - nw) > 1e-9:
                    continue
                out.append(self._make("scale_image", state, {e.id: box}, resolve=factor > 1))
        return [p for p in out if p]

    def proposals(self, tier: int, state: _State, frac: float) -> list[_Proposal]:
        st_x, st_y = frac * self.width, frac * self.height
        props: list[_Proposal] = []
        if tier == 1:
            for t in self.small_texts(state):
                props += self.text_growth_proposals(state, t)
            for t in self.unfit_texts(state):
                props += self.text_expand_proposals(state, t, st_x, st_y)
            props += self.separate_proposals(state, st_x, st_y)
            if self.broken_containers(state):
                props += self.container_proposals(state)
        elif tier == 2:
            props += self.label_proposals(state)
            props += self.container_proposals(state)
            props += self.alignment_proposals(state)
            props += self.spacing_proposals(state)
            props += self.bar_proposals(state)
        else:
            props += self.fill_proposals(state)
            for t in self.texts:
                if t.content:
                    props += self.text_growth_proposals(state, t)
                    props += self.text_tighten_proposals(state, t)
            props += self.image_proposals(state, frac)
            props += self.center_proposals(state)
        unique, keys = [], set()
        for p in props:
            key = (tuple(sorted((i, tuple(round(v, 6) for v in b)) for i, b in p.boxes.items())),
                   tuple(sorted((i, round(f, 6)) for i, f in p.fonts.items())))
            if key not in keys:
                keys.add(key)
                unique.append(p)
        return unique[:self.config.max_proposals_per_tier]

    # ---- evaluation ---------------------------------------------------------------------------------------
    def _emit(self, state: _State) -> dict[str, Any]:
        elements = []
        for original in self.start_adobe["elements"]:
            ident = original["id"]
            box = state.boxes[ident]
            changes: dict[str, Any] = {"x": box[0], "y": box[1], "width": box[2], "height": box[3]}
            if original["type"] == "text":
                changes["font_size"] = state.fonts[ident]
            elements.append(layout_edit(original, changes))
        return {"canvas": dict(self.start_adobe["canvas"]), "elements": elements}

    def shrink_penalty(self, fonts: Mapping[str, float], base_fonts: Mapping[str, float]) -> float:
        """Graded cost of shrinking fonts below `font_shrink_onset`; 0 up to the onset."""
        onset, span = self.config.font_shrink_onset, 1.0 - self.config.font_shrink_onset
        total = 0.0
        for ident, base in base_fonts.items():
            if base > 0:
                excess = (1.0 - fonts[ident] / base) - onset
                if excess > 1e-9:
                    total += (excess / span) ** 2
        return self.config.font_shrink_weight * total

    def evaluate(self, state: _State, proposal: _Proposal, base_fonts: Mapping[str, float]):
        trial = state.copy()
        trial.boxes.update(proposal.boxes)
        trial.fonts.update(proposal.fonts)
        key = trial.key()
        if key in self.seen:
            self.counts["duplicates"] += 1
            return None
        self.seen.add(key)
        self.counts["evaluations"] += 1
        try:
            raw = self._emit(trial)
            if not self.validate_fn(self.source, raw, self.target).is_valid:
                self.counts["invalid"] += 1
                return None
            if self.fit_fn is not None:
                raw = self.fit_fn(raw)
            fitted = _extract(raw)
            if not self.validate_fn(self.source, raw, self.target).is_valid:
                self.counts["invalid"] += 1
                return None
            if not self.invariants_ok(fitted):
                self.counts["invariant"] += 1
                return None
            value = float(self.score_fn(raw))
            if not math.isfinite(value):
                self.counts["errors"] += 1
                return None
            penalized = value - self.shrink_penalty(fitted.fonts, base_fonts)
            return penalized, raw, fitted, value
        except Exception:
            self.counts["errors"] += 1
            return None

    def run(self) -> RefinementResult:
        cfg = self.config
        raw0 = self.start_adobe
        score0 = float(self.score_fn(raw0))
        if not math.isfinite(score0) or not self.validate_fn(self.source, raw0, self.target).is_valid:
            return self._result(raw0, score0, score0, [], 0, "start_not_improvable")
        current, state, score, compared = raw0, self.start, score0, score0
        self.seen.add(state.key())
        applied: list[str] = []
        stopped = "converged"
        for frac in cfg.step_fractions:
            while True:
                if len(applied) >= cfg.max_rounds:
                    stopped = "max_rounds"
                    break
                if self.counts["evaluations"] >= cfg.max_evaluations:
                    stopped = "max_evaluations"
                    break
                best = None
                for tier in (1, 2, 3):
                    for proposal in self.proposals(tier, state, frac):
                        if self.counts["evaluations"] >= cfg.max_evaluations:
                            break
                        outcome = self.evaluate(state, proposal, self.start.fonts)
                        if outcome and outcome[0] > compared + cfg.min_improvement \
                                and (best is None or outcome[0] > best[0][0] + 1e-12):
                            best = (outcome, proposal)
                    if best:
                        break
                if best is None:
                    break
                (penalized, raw, fitted, value), proposal = best
                current, state, score, compared = raw, fitted, value, penalized
                applied.append(proposal.op)
            if stopped in ("max_rounds", "max_evaluations"):
                break
        return self._result(current, score0, score, applied, len(applied), stopped)

    def _result(self, adobe, before, after, applied, accepted, stopped) -> RefinementResult:
        c = self.counts
        return RefinementResult(adobe, bool(applied), before, after, tuple(applied), c["evaluations"],
                                c["invalid"], c["invariant"], c["duplicates"], c["errors"], accepted, stopped)


def refine_candidate(source: Design, target: Canvas, candidate: Mapping[str, Any], *,
                     score: Callable[[dict], float], validate: Callable[..., Any],
                     fit_text: Callable[[dict], dict] | None = None,
                     measure: Callable | None = None,
                     config: RefinementConfig | None = None) -> RefinementResult:
    """Improve a valid candidate by bounded local search; return it unchanged if nothing helps.

    `score` maps an Adobe dict to the deterministic score (higher is better); `validate`
    is `validate_candidate`-shaped; `fit_text` is the pool's text refit; `measure` is an
    optional renderer measurement. Never raises for an unimproved or unscoreable input.
    """
    search = _Search(source, target, candidate, score, validate, fit_text, measure,
                     config or RefinementConfig())
    return search.run()


def diagnose(source: Design, target: Canvas, candidate: Mapping[str, Any], *,
             measure: Callable | None = None, config: RefinementConfig | None = None) -> dict[str, int]:
    """Counts of tier-1/2 layout problems the refiner would act on (0 everywhere = nothing to fix)."""
    search = _Search(source, target, candidate, lambda _: 0.0, lambda *_: type("R", (), {"is_valid": True})(),
                     None, measure, config or RefinementConfig())
    state = search.start
    return dict(crowded_pairs=len(search.crowded_pairs(state)), small_text=len(search.small_texts(state)),
                unfit_text=len(search.unfit_texts(state)),
                broken_containers=len(search.broken_containers(state)),
                off_center_labels=len(search.off_center_labels(state)))

