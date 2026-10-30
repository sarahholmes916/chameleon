"""DEVELOPMENT-ONLY S05 diagnostic candidate generator. This is NOT the S04 production generator.

Purpose: give the tournament UI real, rendered, structurally valid resized layouts of one source for one
target canvas until S04 exists. It only moves/resizes elements (x, y, width, height, text font_size)
through `engine.layout_edits.layout_edit`, so immutable fields and image proportions are preserved, and
every candidate must pass `engine.candidate_validator.validate_candidate` before it is rendered or stored.

Some variants are deliberately poor (tiny text, overflow, overlap, split groups, flat hierarchy). The
strategy names are provenance only: they are hidden from annotators and must never be used as quality
labels or training targets. Deliberately invalid cases go to a separate validator-diagnostics folder.
Randomness is seeded and the number of attempts is bounded.
"""

from __future__ import annotations

import copy
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path

from engine.candidate_validator import validate_candidate
from engine.internal_schema import Canvas, Design
from engine.layout_edits import layout_edit
from engine.layout_search import baseline_proposal

GENERATOR_ID = "s05_dev_controlled_diagnostic"
GENERATOR_VERSION = "dev_controlled_v1"
GOOD_KINDS = ("tiles", "tiles", "fit")
POOR_KINDS = ("tiny_text", "text_overflow", "overlapping_tiles", "split_groups", "flat_hierarchy", "squeezed_spacing")


# ------------------------------------------------------------------ geometry helpers

def _split(source: dict) -> tuple[list[dict], list[dict]]:
    """Geometric (never ID-based) background guess: shapes/cover images spanning ~the whole canvas."""
    w, h = source["canvas"]["width"], source["canvas"]["height"]
    back, content = [], []
    for e in source["elements"]:
        spans = e["width"] >= .85 * w and e["height"] >= .85 * h
        is_bg = spans and (e["type"] == "shape" or (e["type"] == "image" and e.get("fit") == "cover"))
        (back if is_bg else content).append(e)
    return back, content


def _bbox(els: list[dict]) -> tuple[float, float, float, float]:
    return (min(e["x"] for e in els), min(e["y"] for e in els),
            max(e["x"] + e["width"] for e in els), max(e["y"] + e["height"] for e in els))


def _center(e: dict, axis: int) -> float:
    return (e["x"] + e["width"] / 2) if axis == 0 else (e["y"] + e["height"] / 2)


@dataclass
class Plan:
    boxes: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)
    fonts: dict[str, float] = field(default_factory=dict)


def _fit_plan(content, tw, th, *, margin, ax, ay, font_mult=1.0) -> Plan:
    x0, y0, x1, y1 = _bbox(content)
    bw, bh = max(x1 - x0, 1e-9), max(y1 - y0, 1e-9)
    m = margin * min(tw, th)
    s = min((tw - 2 * m) / bw, (th - 2 * m) / bh)
    ox = m + ((tw - 2 * m) - bw * s) * ax - x0 * s
    oy = m + ((th - 2 * m) - bh * s) * ay - y0 * s
    plan = Plan()
    for e in content:
        plan.boxes[e["id"]] = (ox + e["x"] * s, oy + e["y"] * s, e["width"] * s, e["height"] * s)
        if e["type"] == "text":
            plan.fonts[e["id"]] = max(1.0, e["font_size"] * s * font_mult)
    return plan


def _bands(content, k, split_axis, rng, *, keep_groups=True, scatter=False):
    n = len(content)
    if k < 2 or n < k:
        return None
    if scatter:
        order = sorted(content, key=lambda e: (_center(e, split_axis), e["id"]))
        band_of = {e["id"]: i % k for i, e in enumerate(order)}  # interleave: breaks groups on purpose
    else:
        order = sorted(content, key=lambda e: (_center(e, split_axis), e["id"]))
        centers = [_center(e, split_axis) for e in order]
        gaps = sorted(((centers[i + 1] - centers[i], i) for i in range(n - 1)), reverse=True)
        pool = gaps[:max(k, min(len(gaps), k + 1))]
        if len(pool) < k - 1:
            return None
        cuts = sorted(i + 1 for _, i in rng.sample(pool, k - 1))
        band_of, start = {}, 0
        for bi, stop in enumerate(cuts + [n]):
            for e in order[start:stop]:
                band_of[e["id"]] = bi
            start = stop
        if keep_groups:
            by_group: dict[str, list[str]] = {}
            for e in content:
                if e.get("group_id") is not None:
                    by_group.setdefault(e["group_id"], []).append(e["id"])
            for members in by_group.values():
                target = round(sum(band_of[i] for i in members) / len(members))
                for i in members:
                    band_of[i] = target
    bands = [[e for e in content if band_of[e["id"]] == bi] for bi in range(k)]
    return bands if all(bands) else None


def _tile_plan(content, sw, sh, tw, th, k, rng, *, margin, gutter, ax, ay, shared_scale, font_mult=1.0,
               keep_groups=True, scatter=False) -> Plan | None:
    d = 0 if tw / th >= sw / sh else 1  # arrange tiles along the target's longer relative axis
    o = 1 - d
    bands = _bands(content, k, split_axis=o, rng=rng, keep_groups=keep_groups, scatter=scatter)
    if bands is None:
        return None
    t = (tw, th)
    m = margin * min(tw, th)
    g = gutter * min(tw, th)
    slot_main = (t[d] - 2 * m - (k - 1) * g) / k
    slot_cross = t[o] - 2 * m
    boxes = [_bbox(b) for b in bands]

    def extent(bb, axis):
        return max(bb[axis + 2] - bb[axis], 1e-9)

    scales = [min(slot_main / extent(bb, d), slot_cross / extent(bb, o)) for bb in boxes]
    if shared_scale:
        scales = [min(scales)] * k
    plan = Plan()
    for bi, (band, bb, s) in enumerate(zip(bands, boxes, scales)):
        start_d = m + bi * (slot_main + g) + (slot_main - extent(bb, d) * s) * ax
        start_o = m + (slot_cross - extent(bb, o) * s) * ay
        for e in band:
            pos, size = [e["x"], e["y"]], [e["width"], e["height"]]
            new = [0.0, 0.0]
            new[d] = start_d + (pos[d] - bb[d]) * s
            new[o] = start_o + (pos[o] - bb[o]) * s
            plan.boxes[e["id"]] = (new[0], new[1], size[0] * s, size[1] * s)
            if e["type"] == "text":
                plan.fonts[e["id"]] = max(1.0, e["font_size"] * s * font_mult)
    return plan


# ------------------------------------------------------------------ candidate construction

def _build(source: dict, back: list[dict], plan: Plan, tw: int, th: int) -> dict:
    ids_back = {e["id"] for e in back}
    elements = []
    for e in source["elements"]:
        if e["id"] in ids_back:
            x, y, w, h = 0.0, 0.0, float(tw), float(th)
        else:
            x, y, w, h = plan.boxes[e["id"]]
        x, y, w, h = round(x, 3), round(y, 3), round(w, 3), round(h, 3)
        x, y = max(0.0, x), max(0.0, y)
        w, h = min(w, tw - x) if x + w - tw <= .01 else w, min(h, th - y) if y + h - th <= .01 else h
        changes = {"x": x, "y": y, "width": w, "height": h}
        if e["type"] == "text":
            changes["font_size"] = round(plan.fonts.get(e["id"], e["font_size"]), 2)
        elements.append(layout_edit(e, changes))
    return {"canvas": {"width": tw, "height": th}, "elements": elements}


def propose(source: dict, target: dict, kind: str, rng: random.Random) -> tuple[dict | None, dict]:
    """One candidate for a strategy with seeded parameters, or (None, params) if it cannot apply."""
    sw, sh = source["canvas"]["width"], source["canvas"]["height"]
    tw, th = target["width"], target["height"]
    back, content = _split(source)
    if not content:
        return None, {}
    params = dict(kind=kind, margin=round(rng.uniform(.02, .08), 3), ax=rng.choice((0, .5, 1)),
                  ay=rng.choice((0, .5, 1)))
    plan: Plan | None
    if kind == "fit":
        plan = _fit_plan(content, tw, th, margin=params["margin"], ax=params["ax"], ay=params["ay"])
    elif kind in ("tiles", "overlapping_tiles", "split_groups"):
        params.update(k=rng.choice((2, 3)), shared_scale=rng.random() < .6,
                      gutter=round(rng.uniform(.02, .07), 3))
        if kind == "overlapping_tiles":
            params["gutter"] = round(-rng.uniform(.12, .25), 3)
        plan = _tile_plan(content, sw, sh, tw, th, params["k"], rng, margin=params["margin"], gutter=params["gutter"],
                          ax=params["ax"], ay=params["ay"], shared_scale=params["shared_scale"],
                          keep_groups=kind != "split_groups", scatter=kind == "split_groups")
    else:  # tiny_text, text_overflow, flat_hierarchy, squeezed_spacing start from a uniform fit
        mult = {"tiny_text": rng.uniform(.35, .5), "text_overflow": rng.uniform(1.5, 1.9)}.get(kind, 1.0)
        params["font_mult"] = round(mult, 3)
        plan = _fit_plan(content, tw, th, margin=params["margin"], ax=params["ax"], ay=params["ay"], font_mult=mult)
        if plan and kind == "flat_hierarchy" and plan.fonts:
            sizes = sorted(plan.fonts.values())
            median = sizes[len(sizes) // 2]
            plan.fonts = {k: median for k in plan.fonts}
        if plan and kind == "squeezed_spacing":
            params["squeeze"] = round(rng.uniform(.45, .65), 3)
            cy = sum(b[1] + b[3] / 2 for b in plan.boxes.values()) / len(plan.boxes)
            plan.boxes = {k: (x, cy + (y - cy) * params["squeeze"], w, h) for k, (x, y, w, h) in plan.boxes.items()}
    if plan is None:
        return None, params
    return _build(source, back, plan, tw, th), params


def distinct(a: dict, b: dict) -> bool:
    """Layouts differ in some box by >1.5% of the canvas or in some text size by >12%."""
    W, H = a["canvas"]["width"], a["canvas"]["height"]
    for x, y in zip(a["elements"], b["elements"]):
        if any(abs(x[k] - y[k]) / d > .015 for k, d in (("x", W), ("y", H), ("width", W), ("height", H))):
            return True
        if x["type"] == "text" and abs(x["font_size"] - y["font_size"]) / max(x["font_size"], y["font_size"]) > .12:
            return True
    return False


def validity(source_design: Design, candidate: dict, target: Canvas) -> list[str]:
    report = validate_candidate(source_design, candidate, target)
    return [] if report.is_valid else list(report.violations)


# ------------------------------------------------------------------ invalid diagnostics

def invalid_cases(source: dict, target: dict) -> list[dict]:
    """Intentionally invalid outputs (bypassing layout_edit) with the violation each is meant to trigger."""
    design = Design.from_adobe(source)
    canvas = Canvas.from_adobe(target)
    cases = []

    def add(name, expected, candidate):
        found = validity(design, candidate, canvas)
        cases.append(dict(name=name, expected_category=expected, candidate=candidate,
                          validator_violations=found, false_accept=not found))

    stretched = baseline_proposal(design, canvas, 0)
    add("stretch_to_target", "image_aspect", stretched)
    shifted = copy.deepcopy(stretched)
    for e in shifted["elements"]:
        e["x"] = e["x"] + target["width"]
    add("shifted_off_canvas", "off_canvas", shifted)
    fam = copy.deepcopy(baseline_proposal(design, canvas, 1))
    texts = [e for e in fam["elements"] if e["type"] == "text"]
    if texts:
        texts[0]["font_family"] = "Invented Sans"
        add("changed_font_family", "immutable_field", fam)
    dropped = copy.deepcopy(baseline_proposal(design, canvas, 1))
    dropped["elements"].pop()
    add("dropped_element", "identity", dropped)
    wrong = copy.deepcopy(baseline_proposal(design, canvas, 1))
    wrong["canvas"] = {"width": target["width"] + 1, "height": target["height"]}
    add("wrong_canvas", "canvas", wrong)
    return cases


# ------------------------------------------------------------------ set builder

@dataclass
class BuildResult:
    out_dir: Path
    manifest: Path
    accepted: int
    attempts: int
    rejected: dict
    diagnostics: int


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def build_candidate_set(source_path: Path | str, target: dict, out_dir: Path | None, *, renderer, seed: int = 42,
                        count: int = 8, poor: int = 2, max_attempts: int = 240,
                        repo_root: Path | None = None) -> BuildResult:
    """Generate, validate, render and record `count` distinct candidates; raise if the budget is exhausted."""
    from scripts import source_dataset as sd
    from annotation.candidates import MANIFEST_NAME, MANIFEST_VERSION, REPO_ROOT, candidate_id_of

    repo_root = repo_root or REPO_ROOT
    source_path = Path(source_path)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_id = "sha256:" + sd.digest(source)
    design, canvas = Design.from_adobe(source), Canvas.from_adobe(target)
    set_id = f"{source_id[-12:]}_{target['width']}x{target['height']}_s{seed}"
    out_dir = Path(out_dir) if out_dir else repo_root / "data/candidates" / set_id
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError("Use a new/empty output directory; candidate sets are never overwritten")
    for folder in ("candidates", "previews", "diagnostics"):
        (out_dir / folder).mkdir(parents=True, exist_ok=True)
    source_preview, family = None, "unknown"
    dataset_root = source_path.parent.parent
    if source_path.parent.name == "sources" and (dataset_root / "manifest.jsonl").is_file():
        for record in sd.read_records(dataset_root):
            if record["source_design_id"] == source_id:
                family = record["template_family_id"]
                source_preview = _rel(dataset_root / record["preview_path"], repo_root) if record["preview_path"] else None
    renderer_version = "{package}+{commit}+{converter_sha256}".format(
        package=renderer.version.get("package"), commit=str(renderer.version.get("commit"))[:12],
        converter_sha256=str(renderer.version.get("converter_sha256"))[:12])
    good_needed, poor_needed = count - poor, poor
    accepted, pixels, rejected, records = [], set(), {}, []
    attempts = 0
    while len(accepted) < count and attempts < max_attempts:
        rng = random.Random(f"{GENERATOR_VERSION}:{seed}:{attempts}")
        want_poor = poor_needed > 0 and (attempts % 3 == 2 or good_needed == 0)
        kind = rng.choice(POOR_KINDS) if want_poor else rng.choice(GOOD_KINDS)
        attempts += 1
        candidate, params = propose(source, target, kind, rng)
        if candidate is None:
            rejected["not_applicable"] = rejected.get("not_applicable", 0) + 1
            continue
        if validity(design, candidate, canvas):
            rejected["validator"] = rejected.get("validator", 0) + 1
            continue
        if kind in GOOD_KINDS:  # poor variants intentionally skip text fitting
            fitted, _ = renderer.fit_text(candidate)
            candidate = fitted
            if validity(design, candidate, canvas):
                rejected["validator_after_fit"] = rejected.get("validator_after_fit", 0) + 1
                continue
        if any(not distinct(candidate, prior) for prior, _ in accepted):
            rejected["near_duplicate"] = rejected.get("near_duplicate", 0) + 1
            continue
        cid = candidate_id_of(candidate)
        stem = cid[-12:]
        preview = out_dir / "previews" / f"{stem}.png"
        try:
            warnings = renderer.render(candidate, preview)
            if warnings:
                raise ValueError("renderer warnings")
            pixel = sd.verify_png(preview, candidate)
        except Exception:  # noqa: BLE001 - any render failure rejects the candidate, never the run
            preview.unlink(missing_ok=True)
            rejected["render"] = rejected.get("render", 0) + 1
            continue
        if pixel in pixels:
            preview.unlink(missing_ok=True)
            rejected["pixel_duplicate"] = rejected.get("pixel_duplicate", 0) + 1
            continue
        pixels.add(pixel)
        path = out_dir / "candidates" / f"{stem}.json"
        sd.write_json(path, candidate)
        accepted.append((candidate, cid))
        if kind in POOR_KINDS:
            poor_needed -= 1
        else:
            good_needed -= 1
        records.append(dict(
            schema_version=MANIFEST_VERSION, candidate_id=cid, source_design_id=source_id, template_family_id=family,
            source_path=_rel(source_path, repo_root), source_preview_path=source_preview, target_canvas=target,
            candidate_path=_rel(path, repo_root), preview_path=_rel(preview, repo_root), generator_id=GENERATOR_ID,
            generator_version=GENERATOR_VERSION, renderer_version=renderer_version, set_id=set_id,
            provenance=dict(seed=seed, attempt=attempts - 1, strategy=kind, params=params, pixel_sha256=pixel,
                            note="strategy is provenance only; never a quality label")))
    if len(accepted) < count:
        raise RuntimeError(f"only {len(accepted)}/{count} candidates after {attempts} attempts: {rejected}")
    manifest = out_dir / MANIFEST_NAME
    sd.write_jsonl(manifest, records)
    diagnostics = invalid_cases(source, target)
    rows = []
    for case in diagnostics:
        path = out_dir / "diagnostics" / f"{case['name']}.json"
        sd.write_json(path, case.pop("candidate"))
        rows.append(dict(case, path=_rel(path, repo_root), source_design_id=source_id, target_canvas=target))
    sd.write_jsonl(out_dir / "diagnostics" / "diagnostics.jsonl", rows)
    sd.write_json(out_dir / "set.json", dict(
        set_id=set_id, generator_id=GENERATOR_ID, generator_version=GENERATOR_VERSION, seed=seed, count=count,
        poor_variants_requested=poor, attempts=attempts, rejected_attempts=rejected, source_design_id=source_id,
        target_canvas=target, renderer_version=renderer_version, development_only=True,
        note="Diagnostic S05 candidates. Not the S04 generator; strategies are not quality labels."))
    return BuildResult(out_dir, manifest, len(accepted), attempts, rejected, len(rows))


def relpath(path: str | Path, start: str | Path) -> str:
    return Path(os.path.relpath(path, start)).as_posix()
