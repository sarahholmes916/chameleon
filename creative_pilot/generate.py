"""Per-task candidate generation: S04 baselines + creative restructuring + controlled degradation.

For each (source, target) task this builds a pool, validates every candidate with the production validator,
checks renderer-measured text fit, removes structural near-duplicates, renders every retained candidate with the
official preview renderer and writes `candidate_manifest.v1` rows compatible with `annotation.candidates`.
`intended_category` is generator intent only (provenance); it is never used as a label.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections import Counter
from pathlib import Path

from annotation.candidates import MANIFEST_NAME, MANIFEST_VERSION, REPO_ROOT, candidate_id_of
from creative_pilot import degrade as dg
from creative_pilot import metrics, operators as ops
from creative_pilot.blocks import analyse
from creative_pilot.fitting import Measurer
from creative_pilot.layout import Fitter
from engine.candidate_pool import PoolConfig, build_candidate_pool
from engine.candidate_validator import validate_candidate
from engine.internal_schema import Canvas, Design
from scripts import source_dataset as sd

GENERATOR_ID = "creative_pilot"
GENERATOR_VERSION = "creative_pilot_v1"
DUP_DISTANCE = .045
QUOTAS = dict(creative=4, conventional=2, mediocre=3, poor=2)
TARGETS = ((1600, 400), (728, 90), (90, 728))


def proxy_quality(f: dict) -> float:
    """Selection aid only (keeps obviously broken proposals out of the 'creative' pool). Not a label."""
    read = min(1.0, (f.get("font_rel_min") or 0) / .09)
    fill = 1 - f["empty_cell_fraction"]
    overlap = min(1.0, f["content_overlap_fraction"] * 3)
    balance = 1 - min(1.0, f["balance_offset"] * 3)
    return (.35 * read + .25 * fill + .15 * f["order_agreement"] + .15 * balance - .2 * overlap
            - .15 * (f["occluded_pairs"] > 0) - .2 * (f.get("text_fit_failures") or 0) / max(f["n_text"], 1))


def renderer_version_string(renderer) -> str:
    v = renderer.version
    return "{}+{}+{}".format(v.get("package"), str(v.get("commit"))[:12], str(v.get("converter_sha256"))[:12])


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def pixel_metrics(path: Path) -> dict:
    """Facts measured on the rendered PNG itself (not on JSON boxes)."""
    import numpy as np
    from PIL import Image
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    gx, gy = (48, max(3, round(48 * h / w))) if w >= h else (max(3, round(48 * w / h)), 48)
    flat = 0
    ink_cells = 0
    bgc = np.median(arr.reshape(-1, 3), axis=0)
    for j in range(gy):
        for i in range(gx):
            cell = arr[j * h // gy:(j + 1) * h // gy, i * w // gx:(i + 1) * w // gx]
            if cell.size == 0:
                continue
            std = float(cell.reshape(-1, 3).std(axis=0).max())
            flat += std < 3.0
            ink_cells += float(np.abs(cell.reshape(-1, 3) - bgc).max()) > 40
    n = gx * gy
    return dict(flat_cell_fraction=round(flat / n, 4), ink_cell_fraction=round(ink_cells / n, 4))


class TaskGenerator:
    def __init__(self, renderer, out_root: Path, *, seed: int = 42, quotas: dict | None = None):
        self.renderer = renderer
        self.meas = Measurer(renderer)
        self.out_root = Path(out_root)
        self.seed = seed
        self.quotas = quotas or QUOTAS
        self.rv = renderer_version_string(renderer)

    # ----------------------------------------------------------------------- pools
    def s04_candidates(self, source: dict, design: Design, canvas: Canvas) -> list[dict]:
        cfg = PoolConfig(max_raw_candidates=32, max_retained_candidates=6, max_stretch_bases=1, max_refine_bases=1)
        pool = build_candidate_pool(design, canvas, cfg, text_measurement=self.renderer.text_measurement)
        out = []
        ranked = sorted([c for c in pool.retained if c.validation.is_valid], key=lambda c: -(c.score or 0))
        for r in ranked:
            out.append(dict(candidate=r.to_adobe(), family=r.family, variant=r.variant, refined=bool(r.parent_candidate_id),
                            s04_rank=len(out), winner=pool.winner is not None and r.candidate_id == pool.winner.candidate_id))
        return out

    def creative_proposals(self, ctx, source, design, canvas, T, budget: int, diag: Counter) -> list[dict]:
        wide = T[0] >= T[1]
        recipes = ops.WIDE_RECIPES if wide else ops.TALL_RECIPES
        fitter = Fitter(self.meas, ctx)
        out, seen = [], []
        attempts = 0
        trial = 0
        while len(out) < budget and attempts < budget * 6:
            for recipe in recipes:
                attempts += 1
                rng = random.Random(f"{self.seed}:{ctx_id(source)}:{T}:{recipe}:{trial}")
                params = ops.sample_params(recipe, wide, rng)
                try:
                    spec = ops.build_spec(ctx, T, recipe, params, rng)
                    if spec is None:
                        diag["spec_none"] += 1
                        continue
                    plan = ops.realize(ctx, T, fitter, spec, rng)
                    if plan is None:
                        diag["realize_infeasible"] += 1
                        continue
                    cand = metrics.assemble(source, plan.placements, T, self.meas)
                except Exception as exc:  # noqa: BLE001 - one bad proposal must not end the task
                    diag[f"error:{type(exc).__name__}"] += 1
                    continue
                if cand is None:
                    diag["assemble_none"] += 1
                    continue
                if not validate_candidate(design, cand, canvas).is_valid:
                    diag["invalid"] += 1
                    continue
                feats = metrics.layout_features(source, cand, self.meas)
                if feats.get("text_fit_failures"):
                    diag["text_fit_failure"] += 1
                    continue
                if feats["occluded_pairs"]:
                    diag["occluded"] += 1
                    continue
                desc = metrics.descriptor(cand, source)
                if any(metrics.structure_distance(desc, s) < DUP_DISTANCE for s in seen):
                    diag["near_duplicate_raw"] += 1
                    continue
                seen.append(desc)
                out.append(dict(candidate=cand, recipe=recipe, params={k: v for k, v in plan.params.items()
                                                                       if k in ("gamma", "floor", "emph", "wmode", "n", "hero", "valign", "amp",
                                                                                "m", "hero_scale", "n_cells", "hero_pos")},
                                features=feats, descriptor=desc, proxy=proxy_quality(feats)))
                if len(out) >= budget:
                    break
            trial += 1
        diag["creative_attempts"] += attempts
        return out

    def select_creative(self, proposals: list[dict], avoid: list[dict], count: int) -> list[dict]:
        if not proposals:
            return []
        pool = sorted(proposals, key=lambda p: -p["proxy"])
        keep_n = max(count * 2, int(len(pool) * .7))
        pool = pool[:keep_n]
        chosen: list[dict] = []
        per_recipe = Counter()
        while pool and len(chosen) < count:
            def score(p):
                ref = [c["descriptor"] for c in chosen] + [a["descriptor"] for a in avoid]
                dmin = min((metrics.structure_distance(p["descriptor"], d) for d in ref), default=1.0)
                return .55 * p["proxy"] + .45 * dmin - .08 * per_recipe[p["recipe"]]
            best = max(pool, key=score)
            pool.remove(best)
            ref = [c["descriptor"] for c in chosen] + [a["descriptor"] for a in avoid]
            if any(metrics.structure_distance(best["descriptor"], d) < DUP_DISTANCE * 1.6 for d in ref):
                continue
            chosen.append(best)
            per_recipe[best["recipe"]] += 1
        return chosen

    def degraded(self, parents: list[dict], ctx, source, design, canvas, T, count: int, severity: tuple, taken: list[dict],
                 tag: str, rng: random.Random, diag: Counter) -> list[dict]:
        out, used_ops, tries = [], Counter(), 0
        while len(out) < count and tries < count * 14 and parents:
            tries += 1
            parent = rng.choice(parents)
            op_names = sorted(dg.OPERATORS, key=lambda o: (used_ops[o], rng.random()))
            op = op_names[0] if tries % 3 else rng.choice(dg.OPERATORS)
            sev = rng.uniform(*severity)
            cand = dg.degrade(parent["candidate"], ctx, op, sev, f"{self.seed}:{tag}:{tries}", self.meas)
            ops_applied = [(op, round(sev, 2))]
            if cand is not None and tag == "poor" and rng.random() < .5:
                op2 = rng.choice([o for o in dg.OPERATORS if o != op])
                cand2 = dg.degrade(cand, ctx, op2, sev, f"{self.seed}:{tag}:{tries}b", self.meas)
                if cand2 is not None:
                    cand, ops_applied = cand2, ops_applied + [(op2, round(sev, 2))]
            if cand is None:
                diag[f"{tag}_not_applicable"] += 1
                continue
            metrics.repair_text_fit(cand, self.meas)   # a weakness is never allowed to be a text overflow
            if not validate_candidate(design, cand, canvas).is_valid:
                diag[f"{tag}_invalid"] += 1
                continue
            feats = metrics.layout_features(source, cand, self.meas)
            if feats.get("text_fit_failures"):
                diag[f"{tag}_text_unfittable"] += 1   # even 1 px text overflows: a broken layout, not a weak one
                continue
            desc = metrics.descriptor(cand, source)
            if any(metrics.structure_distance(desc, s["descriptor"]) < DUP_DISTANCE for s in taken + out):
                diag[f"{tag}_near_duplicate"] += 1
                continue
            for o_name, _ in ops_applied:
                used_ops[o_name] += 1
            out.append(dict(candidate=cand, parent=parent, ops=ops_applied, features=feats, descriptor=desc))
        return out

    # ----------------------------------------------------------------------- one task
    def run_task(self, rec: dict, source: dict, source_path: Path, T: tuple) -> dict:
        t0 = time.time()
        tw, th = T
        design, canvas = Design.from_adobe(source), Canvas.from_adobe(dict(width=tw, height=th))
        sid = rec["source_design_id"]
        task_id = f"{rec['index']:03d}_{rec['template_family_id']}_{tw}x{th}"
        out_dir = self.out_root / "candidates" / task_id
        for sub in ("candidates", "previews"):
            (out_dir / sub).mkdir(parents=True, exist_ok=True)
        diag: Counter = Counter()
        ctx = analyse(source)
        rng = random.Random(f"{self.seed}:{sid}:{T}")
        s04 = self.s04_candidates(source, design, canvas)
        s04_items = []
        for item in s04:
            feats = metrics.layout_features(source, item["candidate"], self.meas)
            item.update(features=feats, descriptor=metrics.descriptor(item["candidate"], source))
            if not any(metrics.structure_distance(item["descriptor"], x["descriptor"]) < DUP_DISTANCE for x in s04_items):
                s04_items.append(item)
        conventional = s04_items[:self.quotas["conventional"]]
        proposals = self.creative_proposals(ctx, source, design, canvas, T, 36, diag)
        creative = self.select_creative(proposals, conventional, self.quotas["creative"])
        taken = conventional + creative
        # Degrade only layouts that are not already unreadable, so "mediocre" means a good layout with one weakness.
        readable_s04 = [c for c in conventional if (c["features"].get("font_rel_min") or 0) >= .05]
        parents = creative + readable_s04 or creative + conventional
        mediocre = self.degraded(parents, ctx, source, design, canvas, T, self.quotas["mediocre"], (.25, .5), taken, "mediocre", rng, diag)
        taken = taken + mediocre
        poor = self.degraded(parents, ctx, source, design, canvas, T, self.quotas["poor"], (.7, 1.0), taken, "poor", rng, diag)
        entries = []
        for it in conventional:
            entries.append(("conventional", "s04", it, dict(s04_family=it["family"], s04_variant=it["variant"], s04_refined=it["refined"],
                                                              s04_winner=it["winner"], s04_rank=it["s04_rank"])))
        for it in creative:
            entries.append(("creative", "creative_restructure", it, dict(recipe=it["recipe"], recipe_params=it["params"])))
        for tag, group in (("mediocre", mediocre), ("poor", poor)):
            for it in group:
                entries.append((tag, "controlled_degradation", it, dict(
                    degradation_ops=[o for o, _ in it["ops"]], degradation_severity=[s for _, s in it["ops"]],
                    weakness=[dg.WEAKNESS[o] for o, _ in it["ops"]],
                    parent_recipe=it["parent"].get("recipe") or f"s04:{it['parent'].get('family')}",
                    parent_candidate_id=candidate_id_of(it["parent"]["candidate"]))))
        records, pixels, render_fail = [], set(), 0
        for intended, strategy, it, extra in entries:
            cand = it["candidate"]
            cid = candidate_id_of(cand)
            stem = cid[-12:]
            cpath, ppath = out_dir / "candidates" / f"{stem}.json", out_dir / "previews" / f"{stem}.png"
            try:
                warnings = self.renderer.render(cand, ppath)
                if warnings:
                    raise ValueError(f"renderer warnings: {warnings}")
                pixel = sd.verify_png(ppath, cand)
            except Exception as exc:  # noqa: BLE001 - any render failure drops this candidate only
                ppath.unlink(missing_ok=True)
                render_fail += 1
                diag["render_failure"] += 1
                diag[f"render_failure:{type(exc).__name__}"] += 1
                continue
            if pixel in pixels:
                ppath.unlink(missing_ok=True)
                diag["pixel_duplicate"] += 1
                continue
            pixels.add(pixel)
            sd.write_json(cpath, cand)
            feats = it["features"]
            records.append(dict(
                schema_version=MANIFEST_VERSION, candidate_id=cid, source_design_id=sid,
                template_family_id=rec["template_family_id"], source_path=rel(source_path),
                source_preview_path=rec.get("preview_abs"), target_canvas=dict(width=tw, height=th),
                candidate_path=rel(cpath), preview_path=rel(ppath), generator_id=GENERATOR_ID if strategy != "s04" else "s04",
                generator_version=GENERATOR_VERSION if strategy != "s04" else "s04_generator_v0.3+measured_text",
                renderer_version=self.rv, set_id=task_id,
                provenance=dict(pilot_version=GENERATOR_VERSION, task_id=task_id, intended_category=intended, strategy=strategy,
                                seed=self.seed, pixel_sha256=pixel, features={k: (round(v, 4) if isinstance(v, float) else v)
                                                                               for k, v in feats.items()},
                                render_metrics=pixel_metrics(ppath),
                                note="intended_category and strategy are provenance only; never a quality label", **extra)))
        sd.write_jsonl(out_dir / MANIFEST_NAME, records)
        counts = Counter(r["provenance"]["intended_category"] for r in records)
        cr = [r["provenance"]["features"].get("min_font") or 0 for r in records if r["provenance"]["intended_category"] == "creative"]
        summary = dict(task_id=task_id, creative_retained=len(cr), best_creative_min_font=max(cr) if cr else None,
                       readability_limited=(not cr) or max(cr) < 6.0, source_design_id=sid, template_family_id=rec["template_family_id"], target=dict(width=tw, height=th),
                       retained=len(records), by_intended=dict(counts), s04_valid=len(s04), creative_proposals=len(proposals),
                       render_failures=render_fail, diagnostics=dict(diag), seconds=round(time.time() - t0, 1),
                       manifest=rel(out_dir / MANIFEST_NAME))
        sd.write_json(out_dir / "task.json", summary)
        return summary


def ctx_id(source: dict) -> str:
    return sd.digest(source)[:12]
