"""Validated S04 pool and diverse training selection; no ML or rendering imports."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import zip_longest
import hashlib
import json
import logging
import math
from typing import Callable, Iterable

from engine import candidate_generator as generator
from engine.candidate_validator import ValidationReport, validate_candidate
from engine.internal_schema import Canvas, Design
from engine.layout_objective import OBJECTIVE_VERSION, score_deterministic
from engine.layout_typography import fit_candidate_text
from engine.shape_roles import ShapeRole, classify_shapes
from engine.shape_stretch import SHAPE_STRETCH_VERSION, VARIANT_PROPORTIONAL, stretch_variants

logger = logging.getLogger(__name__)


def identity(value: dict) -> str:
    # Same canonical bytes and sha256: prefix as annotation.candidates.
    return "sha256:" + hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PoolConfig:
    max_raw_candidates: int = 64
    max_retained_candidates: int = 12
    near_duplicate_distance: float = 0.025
    seed: int = 42
    # Shape-stretch variants (D.1.7) are derived from the best valid base layouts
    # AFTER the raw round-robin and budgeted separately, so they can neither starve
    # nor be starved by the family streams. 0 disables them.
    max_stretch_bases: int = 3
    max_stretch_candidates: int = 12

    def __post_init__(self):
        for name in ("max_raw_candidates", "max_retained_candidates"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("max_stretch_bases", "max_stretch_candidates"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not math.isfinite(self.near_duplicate_distance) or not 0 <= self.near_duplicate_distance <= 1:
            raise ValueError("near_duplicate_distance must be in [0, 1]")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")


@dataclass(frozen=True)
class PoolCandidate:
    candidate_id: str
    source_design_id: str
    target_id: str
    family: str
    position: int
    seed: int
    generator_version: str
    objective_version: str
    adobe_json: str
    validation: ValidationReport
    score: float | None
    error: str | None = None
    # Shape-stretch provenance; defaults describe an ordinary family candidate.
    # Never written into the Adobe JSON.
    variant: str = VARIANT_PROPORTIONAL
    base_candidate_id: str | None = None
    shape_stretch_version: str | None = None

    @property
    def design(self) -> Design:
        return Design.from_adobe(self.to_adobe())

    def to_adobe(self) -> dict:
        # Fresh copies prevent consumers changing the snapshot that was evaluated.
        return json.loads(self.adobe_json)


@dataclass(frozen=True)
class CandidatePool:
    candidates: tuple[PoolCandidate, ...]
    retained: tuple[PoolCandidate, ...]
    winner: PoolCandidate | None
    failures: tuple[str, ...]
    exact_duplicates: int
    near_duplicates: int


def default_families(target: Canvas):
    families = [
        (generator.FAMILY_SCALE_BASELINE, lambda s, t: [generator.baseline_candidate(s, t)]),
        (generator.FAMILY_HORIZONTAL_FLOW, generator.horizontal_flow_candidates),
        (generator.FAMILY_VERTICAL_STACK, generator.vertical_stack_candidates),
        (generator.FAMILY_HERO_LEFT, generator.hero_left_candidates),
        (generator.FAMILY_HERO_RIGHT, generator.hero_right_candidates),
        (generator.FAMILY_TWO_COLUMN, generator.two_column_candidates),
        (generator.FAMILY_CONTROLLED_OVERLAY, generator.controlled_overlay_candidates),
    ]
    if target.width / target.height > generator.FAMILY_ASPECT_CUTOFF:
        families.append((generator.FAMILY_BANNER_ZONES, generator.banner_zones_candidates))
    return families


def layout_distance(a: PoolCandidate, b: PoolCandidate, target: Canvas) -> float:
    """Mean normalized center/size/font/relative-group displacement (0..1)."""
    left, right = a.design.elements, b.design.elements_by_id
    terms, groups = [], {}
    for e in left:
        f = right[e.id]
        dx = ((e.x + e.width / 2) - (f.x + f.width / 2)) / target.width
        dy = ((e.y + e.height / 2) - (f.y + f.height / 2)) / target.height
        center = min(1.0, math.hypot(dx, dy) / 0.25)
        size = min(1.0, (abs(math.log((e.width + 1) / (f.width + 1)))
                         + abs(math.log((e.height + 1) / (f.height + 1)))) / 2)
        font = min(1.0, abs(math.log(e.font_size / f.font_size))) if e.type == "text" else 0.0
        terms.append(0.55 * center + 0.30 * size + 0.15 * font)
        if isinstance(e.group_id, str):
            groups.setdefault(e.group_id, []).append((dx, dy))
    relative = [min(1.0, math.hypot(x - sum(p[0] for p in points) / len(points),
                                  y - sum(p[1] for p in points) / len(points)) / 0.25)
                for points in groups.values() for x, y in points]
    return 0.9 * sum(terms) / max(1, len(terms)) + 0.1 * sum(relative) / max(1, len(relative))


SHAPE_DISTINCT_DISTANCE = 0.1


def shape_distance(a: PoolCandidate, b: PoolCandidate, target: Canvas) -> float:
    """Largest per-shape displacement (same terms as layout_distance, 0..1).

    layout_distance averages over every element, so a variant that only resizes
    a few shapes looks like a near-duplicate of its base; this keeps it distinct.
    """
    left, right = a.design.elements, b.design.elements_by_id
    worst = 0.0
    for e in left:
        if e.type != "shape":
            continue
        f = right[e.id]
        dx = ((e.x + e.width / 2) - (f.x + f.width / 2)) / target.width
        dy = ((e.y + e.height / 2) - (f.y + f.height / 2)) / target.height
        center = min(1.0, math.hypot(dx, dy) / 0.25)
        size = min(1.0, (abs(math.log((e.width + 1) / (f.width + 1)))
                         + abs(math.log((e.height + 1) / (f.height + 1)))) / 2)
        worst = max(worst, 0.55 * center + 0.30 * size)
    return worst


def build_candidate_pool(source: Design, target: Canvas, config: PoolConfig | None = None,
                         *, families: Iterable[tuple[str, Callable]] | None = None,
                         text_measurement: Callable | None = None) -> CandidatePool:
    config = config or PoolConfig()
    source_id, target_id = identity(source.to_adobe()), identity(target.to_adobe())
    failures, records, streams = [], [], []
    version = generator.GENERATOR_VERSION + ("+measured_text" if text_measurement else "+word_fit")

    def make_record(family, position, raw, variant=VARIANT_PROPORTIONAL, base_id=None):
        """Validate, text-fit, snapshot and score one proposal; never raises."""
        stretched = variant != VARIANT_PROPORTIONAL
        extra = dict(variant=variant, base_candidate_id=base_id,
                     shape_stretch_version=SHAPE_STRETCH_VERSION if stretched else None)
        label = f"{family}[{position}]" + (f"+{variant}" if stretched else "")
        try:
            # Reject malformed/protected-field edits before text refinement.
            if validate_candidate(source, raw, target).is_valid:
                raw = fit_candidate_text(raw, text_measurement)
            encoded = json.dumps(raw, ensure_ascii=False, allow_nan=False)
            snapshot = json.loads(encoded)
            report = validate_candidate(source, snapshot, target)
            score, error = None, None
            if report.is_valid:
                try:
                    score = score_deterministic(source, [Design.from_adobe(snapshot)], target)[0]
                    if not math.isfinite(score):
                        raise ValueError("non-finite score")
                except Exception as exc:
                    score, error = None, f"{label}: scoring failed: {exc}"
                    failures.append(error)
                    logger.warning(error)
            return PoolCandidate(identity(snapshot), source_id, target_id, family, position,
                                 config.seed, version + ("+" + SHAPE_STRETCH_VERSION if stretched else ""),
                                 OBJECTIVE_VERSION, encoded, report, score, error, **extra)
        except Exception as exc:
            # Count malformed proposals against the budget too.
            message = f"{label}: invalid proposal: {type(exc).__name__}: {exc}"
            logger.warning(message)
            failures.append(message)
            return PoolCandidate("", source_id, target_id, family, position, config.seed,
                                 generator.GENERATOR_VERSION, OBJECTIVE_VERSION, "null",
                                 ValidationReport(False, False, False, (message,)), None, message, **extra)

    for family, produce in (default_families(target) if families is None else families):
        try:
            streams.append((family, iter(produce(source, target)), 0))
        except Exception as exc:
            message = f"{family}: generation failed: {type(exc).__name__}: {exc}"
            failures.append(message)
            logger.warning(message)
    # max_raw_candidates is the hard total: stretch variants spend a reserved share of
    # it (a quarter, capped by max_stretch_candidates) so tight budgets still see them.
    try:
        stretchable = bool(config.max_stretch_bases and config.max_stretch_candidates) and any(
            info.role not in (ShapeRole.FIXED, ShapeRole.PROPORTION_SENSITIVE)
            for info in classify_shapes(source).values())
    except Exception as exc:
        stretchable = False
        failures.append(f"shape_stretch: classification failed: {type(exc).__name__}: {exc}")
    reserve = min(config.max_stretch_candidates, config.max_raw_candidates // 4) if stretchable else 0
    stream_limit = config.max_raw_candidates - reserve
    # Round-robin budgets prevent an early prolific family starving later families.
    while streams and len(records) < stream_limit:
        following = []
        for family, stream, position in streams:
            if len(records) >= stream_limit:
                break
            try:
                raw = next(stream)
            except StopIteration:
                continue
            except Exception as exc:
                message = f"{family}[{position}]: generation failed: {type(exc).__name__}: {exc}"
                failures.append(message)
                logger.warning(message)
                continue
            following.append((family, stream, position + 1))
            records.append(make_record(family, position, raw))
        streams = following
    # Shape-stretch variants: derived from the best distinct valid bases, never from
    # other variants, and only when the source has a stretchable shape role.
    if stretchable:
        bases, taken = [], set()
        for record in sorted((r for r in records if r.validation.is_valid and r.score is not None),
                             key=lambda r: (-r.score, r.position)):
            if record.candidate_id not in taken and len(bases) < config.max_stretch_bases:
                taken.add(record.candidate_id)
                bases.append(record)
        proposals = []
        for record in bases:
            try:
                proposals.append([(record, v) for v in stretch_variants(
                    source, target, record.to_adobe(), measure=text_measurement)])
            except Exception as exc:
                message = f"{record.family}[{record.position}]: shape stretch failed: {type(exc).__name__}: {exc}"
                failures.append(message)
                logger.warning(message)
        # Interleave bases so the budget never starves a lower-ranked base's variants.
        order = [item for rank in zip_longest(*proposals) for item in rank if item is not None]
        for record, variant in order[:min(config.max_stretch_candidates,
                                          max(0, config.max_raw_candidates - len(records)))]:
            records.append(make_record(record.family, len(records), variant.adobe,
                                       variant.name, record.candidate_id))
    valid = [r for r in records if r.validation.is_valid]
    ranked = sorted(valid, key=lambda r: -(r.score if r.score is not None else -math.inf))
    winner = ranked[0] if ranked else None
    unique, seen = [], set()
    for record in ranked:
        if record.candidate_id not in seen:
            unique.append(record)
            seen.add(record.candidate_id)
    distinct = []
    for record in unique:
        if all(layout_distance(record, other, target) > config.near_duplicate_distance
               or shape_distance(record, other, target) > SHAPE_DISTINCT_DISTANCE for other in distinct):
            distinct.append(record)
    retained = []
    while distinct and len(retained) < config.max_retained_candidates:
        counts = {r.family: sum(x.family == r.family for x in retained) for r in distinct}
        # Family balance precedes score after the production winner; lower-score
        # alternatives stay available without changing production selection.
        chosen = min(distinct, key=lambda r: (counts[r.family],
                     -(r.score if r.score is not None else -math.inf), r.position))
        retained.append(chosen)
        distinct.remove(chosen)
    return CandidatePool(tuple(records), tuple(retained), winner, tuple(failures),
                         len(valid) - len(unique), len(unique) - len(retained) - len(distinct))
