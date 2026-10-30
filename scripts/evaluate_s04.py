"""Reproducible S04 structural/heuristic benchmark and annotation-ready previews."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import html
import json
import math
from pathlib import Path
import statistics
import time

from engine.candidate_pool import PoolConfig, build_candidate_pool, identity
from engine.internal_schema import Canvas, Design
from engine.layout_objective import OBJECTIVE_VERSION, PROVISIONAL_WEIGHTS
from engine.shape_roles import ShapeRole, classify_shapes
from engine.shape_stretch import SHAPE_STRETCH_VERSION, VARIANT_PROPORTIONAL
from scripts.source_dataset import SourceRenderer, verify_png, write_json


def render_checked(renderer, design, path):
    warnings = renderer.render(design, path)
    verify_png(path, design)
    return warnings


_STRETCHABLE = (ShapeRole.BUTTON, ShapeRole.PANEL, ShapeRole.BACKGROUND, ShapeRole.BAR_H, ShapeRole.BAR_V)


def shape_metrics(source: Design, adobe: dict) -> dict:
    """Objective shape facts for one candidate (no quality judgement).

    aspect change is ln((w'/h') / (w/h)) against the source shape. Buttons report
    whether the label box is inside the shape box and the padding balance; padding
    in ems of the label font makes sibling buttons comparable.
    """
    roles = classify_shapes(source)
    placed = {e["id"]: e for e in adobe["elements"]}
    stretch, sensitive, buttons = [], [], []
    for ident, info in roles.items():
        original, now = source.elements_by_id[ident], placed[ident]
        if min(original.width, original.height, now["width"], now["height"]) <= 0:
            continue
        change = abs(math.log((now["width"] / now["height"]) / (original.width / original.height)))
        if info.role in _STRETCHABLE:
            stretch.append(change)
        elif info.role is ShapeRole.PROPORTION_SENSITIVE:
            sensitive.append(change)
        if info.role is ShapeRole.BUTTON:
            label = placed[info.contained_ids[0]]
            left, top = label["x"] - now["x"], label["y"] - now["y"]
            right = now["x"] + now["width"] - label["x"] - label["width"]
            bottom = now["y"] + now["height"] - label["y"] - label["height"]
            em = label["font_size"]
            buttons.append(dict(
                id=ident, label_contained=min(left, top, right, bottom) >= -1e-6,
                padding_lr_asymmetry=abs(left - right) / now["width"],
                padding_tb_asymmetry=abs(top - bottom) / now["height"],
                pad_x_em=(left + right) / 2 / em, pad_y_em=(top + bottom) / 2 / em,
                aspect=now["width"] / now["height"], label_font_size=em))
    pads = [b["pad_y_em"] for b in buttons]
    return dict(
        stretchable_shapes=len(stretch), proportion_sensitive_shapes=len(sensitive), buttons=buttons,
        mean_abs_log_aspect_change=statistics.fmean(stretch) if stretch else 0.0,
        max_proportion_sensitive_log_aspect_change=max(sensitive) if sensitive else 0.0,
        labels_contained=sum(b["label_contained"] for b in buttons), label_count=len(buttons),
        button_pad_y_em_spread=(statistics.pstdev(pads) / statistics.fmean(pads)
                                if len(pads) > 1 and statistics.fmean(pads) > 0 else 0.0))


def schematic_svg(adobe: dict) -> str:
    """Flat box drawing of a layout. NOT the official renderer: no fonts, no images."""
    canvas = adobe["canvas"]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {canvas["width"]} {canvas["height"]}" '
             f'width="{canvas["width"]}" height="{canvas["height"]}">',
             f'<rect width="{canvas["width"]}" height="{canvas["height"]}" fill="#fff"/>']
    order = sorted(enumerate(adobe["elements"]), key=lambda pair: (pair[1].get("z_index", 0), pair[0]))
    for _, e in order:
        box = f'x="{e["x"]}" y="{e["y"]}" width="{e["width"]}" height="{e["height"]}"'
        if e["type"] == "shape":
            fill = html.escape(e.get("fill_color", "#888"), quote=True)
            common = f'fill="{fill}" fill-opacity="{e.get("opacity", 1)}"'
            if e.get("shape_type") == "ellipse":
                parts.append(f'<ellipse cx="{e["x"] + e["width"] / 2}" cy="{e["y"] + e["height"] / 2}" '
                             f'rx="{e["width"] / 2}" ry="{e["height"] / 2}" {common}/>')
            else:
                radius = min(e.get("corner_radius", 0), e["width"] / 2, e["height"] / 2)
                parts.append(f'<rect {box} rx="{radius}" {common}/>')
        elif e["type"] == "image":
            parts.append(f'<rect {box} fill="#ddd" stroke="#999" stroke-width="0.5"/>')
        else:
            parts.append(f'<rect {box} fill="none" stroke="#c33" stroke-width="0.4"/>'
                         f'<text x="{e["x"]}" y="{e["y"] + e["font_size"]}" font-size="{e["font_size"]}" '
                         f'font-family="sans-serif" fill="#000">{html.escape(e["content"][:60])}</text>')
    parts.append("</svg>")
    return "".join(parts)


def make_preview(renderer, design, path):
    """PNG through the official renderer, or a schematic SVG when no renderer is given."""
    if renderer is None:
        path = path.with_suffix(".svg")
        path.write_text(schematic_svg(design), encoding="utf-8")
        return path, ["schematic preview: not rendered by the official renderer"]
    return path, render_checked(renderer, design, path)


def _summarize(pool):
    winner = pool.winner
    return dict(generated=len(pool.candidates), valid=sum(r.validation.is_valid for r in pool.candidates),
                winner=winner.candidate_id if winner else None, winner_family=winner.family if winner else None,
                winner_variant=winner.variant if winner else None, winner_score=winner.score if winner else None)


def _comparison_gallery(output, renderer, source_preview, before_pool, pool, report):
    """Original, proportional-only winner, best candidate per stretch variant, selected winner."""
    items = [("original source", source_preview)]
    plan = []
    if before_pool and before_pool.winner:
        plan.append(("before: proportional-only winner", before_pool.winner))
    best = {}
    for r in pool.candidates:
        if r.variant != VARIANT_PROPORTIONAL and r.validation.is_valid and r.score is not None:
            if r.variant not in best or r.score > best[r.variant].score:
                best[r.variant] = r
    plan += [(f"stretch variant: {name}", rec) for name, rec in sorted(best.items())]
    if pool.winner:
        plan.append(("after: selected winner", pool.winner))
    for slot, (label, record) in enumerate(plan):
        path = (output / f"compare_{slot:02}.png").resolve()
        try:
            shown, _ = make_preview(renderer, record.to_adobe(), path)
            items.append((f"{label}: {record.family} / {record.variant}, score {record.score:.4f}", shown))
        except Exception as exc:
            report["render_failures"].append(f"compare {label}: {type(exc).__name__}: {exc}")
    note = "schematic boxes, NOT the official renderer" if renderer is None else "official renderer previews"
    figures = "".join(
        f'<figure><img width="520" src="{html.escape(Path(path).name)}" alt="">'
        f'<figcaption>{html.escape(caption)}</figcaption></figure>'
        for caption, path in items if path is not None)
    return (f'<meta charset="utf-8"><h1>Shape-stretch comparison</h1><p>{note}. Scores are the provisional '
            f'deterministic objective ({OBJECTIVE_VERSION}), not human judgement; a higher score is not proof '
            f'of a better design.</p><div style="display:flex;flex-wrap:wrap;gap:12px">{figures}</div>')


def evaluate_task(task, output, renderer, config, *, family_id="unknown", render_pool=False, measured_fit=False,
                  compare_baseline=True):
    """Evaluate one (source, target). renderer=None writes schematic SVG previews and no manifest."""
    output.mkdir(parents=True, exist_ok=False)
    source = Design.from_adobe(task["source"])
    target = Canvas.from_adobe(task["target_canvas"])
    measure = renderer.text_measurement if (measured_fit and renderer is not None) else None
    before_pool, before_seconds = None, None
    if compare_baseline:
        # Same generator and scorer with shape-stretch variants disabled: the proportional-only engine.
        start = time.perf_counter()
        before_pool = build_candidate_pool(source, target, replace(config, max_stretch_bases=0), text_measurement=measure)
        before_seconds = time.perf_counter() - start
    start = time.perf_counter()
    pool = build_candidate_pool(source, target, config, text_measurement=measure)
    elapsed = time.perf_counter() - start
    report = dict(generated=len(pool.candidates), valid=sum(r.validation.is_valid for r in pool.candidates),
                  invalid=sum(not r.validation.is_valid for r in pool.candidates),
                  exact_duplicates=pool.exact_duplicates, near_duplicates=pool.near_duplicates,
                  retained=len(pool.retained), generation_seconds=elapsed, config=asdict(config),
                  failures=list(pool.failures), target=target.to_adobe(), template_family_id=family_id,
                  family_counts=dict(Counter(r.family for r in pool.candidates)),
                  family_invalid=dict(Counter(r.family for r in pool.candidates if not r.validation.is_valid)),
                  renderer_version=renderer.version if renderer is not None else "schematic_svg_not_official_renderer",
                  measured_fit=measure is not None, objective_version=OBJECTIVE_VERSION,
                  shape_stretch_version=SHAPE_STRETCH_VERSION,
                  provisional_weights=asdict(PROVISIONAL_WEIGHTS), render_failures=[], candidates=[])
    report["family_failure_rates"] = {
        family: report["family_invalid"].get(family, 0) / count
        for family, count in report["family_counts"].items()
    }
    report["variant_counts"] = dict(Counter(r.variant for r in pool.candidates))
    report["variant_invalid"] = dict(Counter(r.variant for r in pool.candidates if not r.validation.is_valid))
    stretched = [r for r in pool.candidates if r.variant != VARIANT_PROPORTIONAL]
    best_by_variant = {}
    for r in pool.candidates:
        if r.validation.is_valid and r.score is not None and r.score > best_by_variant.get(r.variant, -math.inf):
            best_by_variant[r.variant] = r.score
    after_summary = _summarize(pool)
    before_summary = _summarize(before_pool) if before_pool else None
    report["after"] = dict(after_summary, seconds=elapsed,
                           metrics=shape_metrics(source, pool.winner.to_adobe()) if pool.winner else None)
    report["before"] = (dict(before_summary, seconds=before_seconds,
                             metrics=shape_metrics(source, before_pool.winner.to_adobe()) if before_pool.winner else None)
                        if before_pool else None)
    report["shape_stretch"] = dict(
        stretched_generated=len(stretched), stretched_valid=sum(r.validation.is_valid for r in stretched),
        best_score_by_variant=best_by_variant,
        stretched_wins=bool(pool.winner and pool.winner.variant != VARIANT_PROPORTIONAL),
        winner_score_delta=(after_summary["winner_score"] - before_summary["winner_score"]
                            if before_summary and after_summary["winner_score"] is not None
                            and before_summary["winner_score"] is not None else None),
        retained_variants=dict(Counter(r.variant for r in pool.retained)))
    source_path = (output / "source.json").resolve()
    source_preview = (output / "source.png").resolve()
    write_json(source_path, task["source"])
    try:
        source_preview, _ = make_preview(renderer, task["source"], source_preview)
    except Exception as exc:
        report["render_failures"].append(f"source: {exc}")
        source_preview = None
    selected = list(pool.retained) if render_pool else []
    if pool.winner and pool.winner not in selected:
        selected.insert(0, pool.winner)
    manifest, gallery = [], []
    for i, record in enumerate(pool.candidates):
        report["candidates"].append(dict(candidate_id=record.candidate_id, family=record.family,
                                         validation=asdict(record.validation), score=record.score,
                                         objective_version=record.objective_version, variant=record.variant,
                                         base_candidate_id=record.base_candidate_id,
                                         shape_metrics=(shape_metrics(source, record.to_adobe())
                                                        if record.validation.is_valid else None)))
    for i, record in enumerate(selected):
        design = record.to_adobe()
        path, preview = (output / f"candidate_{i:03}.json").resolve(), (output / f"candidate_{i:03}.png").resolve()
        write_json(path, design)
        try:
            preview, warnings = make_preview(renderer, design, preview)
            measurements = ([renderer.text_measurement(e) for e in design["elements"] if e["type"] == "text"]
                            if renderer is not None else [])
            report.setdefault("renders", []).append(dict(candidate_id=record.candidate_id, warnings=warnings,
                                                          text_measurements=measurements))
            manifest.extend([] if renderer is None else [dict(candidate_id=record.candidate_id, source_design_id=record.source_design_id,
                                 template_family_id=family_id, source_path=str(source_path),
                                 source_preview_path=str(source_preview) if source_preview else None,
                                 target_canvas=target.to_adobe(), candidate_path=str(path), preview_path=str(preview),
                                 generator_id="s04", generator_version=record.generator_version,
                                 renderer_version=json.dumps(renderer.version, sort_keys=True),
                                 set_id=output.name, provenance=dict(family=record.family, seed=config.seed,
                                 objective_version=record.objective_version, target_id=record.target_id,
                                 variant=record.variant, base_candidate_id=record.base_candidate_id,
                                 shape_stretch_version=record.shape_stretch_version))])
            gallery.append(f'<figure><img width="600" src="{preview.name}"><figcaption>'
                           f'{html.escape(record.family)} / {html.escape(record.variant)}</figcaption></figure>')
        except Exception as exc:
            report["render_failures"].append(f"{record.candidate_id}: {type(exc).__name__}: {exc}")
    report["winner"] = pool.winner.candidate_id if pool.winner else None
    (output / "comparison.html").write_text(
        _comparison_gallery(output, renderer, source_preview, before_pool, pool, report), encoding="utf-8")
    report["total_seconds"] = time.perf_counter() - start
    write_json(output / "report.json", report)
    (output / "candidate_manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in manifest), encoding="utf-8")
    (output / "index.html").write_text('<meta charset="utf-8"><h1>S04 review</h1><p>Heuristic selection; no human quality measurements.</p>' + "".join(gallery), encoding="utf-8")
    return report


def parse_target(text):
    width, _, height = text.lower().partition("x")
    return dict(width=int(width), height=int(height))


def aggregate(summaries):
    """Cross-task totals for the before/after comparison (objective facts only)."""
    rows = [row for row in summaries if row.get("before") and row.get("after")]
    deltas = [row["shape_stretch"]["winner_score_delta"] for row in rows
              if row["shape_stretch"]["winner_score_delta"] is not None]
    after_metrics = [row["after"]["metrics"] for row in rows if row["after"]["metrics"]]
    before_metrics = [row["before"]["metrics"] for row in rows if row["before"]["metrics"]]
    contained = lambda metrics: (sum(m["labels_contained"] for m in metrics), sum(m["label_count"] for m in metrics))  # noqa: E731
    return dict(
        tasks=len(summaries), compared=len(rows),
        generated=sum(r["generated"] for r in summaries), valid=sum(r["valid"] for r in summaries),
        stretched_generated=sum(r["shape_stretch"]["stretched_generated"] for r in summaries),
        stretched_valid=sum(r["shape_stretch"]["stretched_valid"] for r in summaries),
        stretched_wins=sum(r["shape_stretch"]["stretched_wins"] for r in summaries),
        stretched_losses=sum(1 for r in summaries if r["shape_stretch"]["stretched_generated"]
                             and not r["shape_stretch"]["stretched_wins"]),
        no_stretch_available=sum(1 for r in summaries if not r["shape_stretch"]["stretched_generated"]),
        mean_winner_score_delta=statistics.fmean(deltas) if deltas else None,
        min_winner_score_delta=min(deltas) if deltas else None,
        max_winner_score_delta=max(deltas) if deltas else None,
        seconds_before=sum(r["before"]["seconds"] for r in rows), seconds_after=sum(r["after"]["seconds"] for r in rows),
        labels_contained_before=contained(before_metrics), labels_contained_after=contained(after_metrics),
        max_proportion_sensitive_log_aspect_change_after=max(
            (m["max_proportion_sensitive_log_aspect_change"] for m in after_metrics), default=0.0),
        render_failures=sum(len(r["render_failures"]) for r in summaries))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources", type=Path, help="Optional generated source JSON directory")
    parser.add_argument("--source-limit", type=int, default=3)
    parser.add_argument("--source-roles", default="",
                        help="Comma-separated shape roles (button,panel,background,bar_h,bar_v); keep only sources having one")
    parser.add_argument("--targets", default="728x90",
                        help="Comma-separated WxH targets for generated sources (adobe samples also keep their own)")
    parser.add_argument("--render-pool", action="store_true")
    parser.add_argument("--measured-fit", action="store_true", help="Refine text using real renderer measurements before scoring")
    parser.add_argument("--schematic", action="store_true",
                        help="Draw flat SVG boxes instead of using the official renderer (no manifests, no measured fit)")
    parser.add_argument("--no-compare", action="store_true", help="Skip the proportional-only baseline run")
    parser.add_argument("--max-raw", type=int, default=64)
    parser.add_argument("--retain", type=int, default=12)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output exists; use a new directory to preserve previous runs")
    if args.schematic and args.measured_fit:
        parser.error("--measured-fit needs the official renderer")
    renderer = None if args.schematic else SourceRenderer()
    config = PoolConfig(max_raw_candidates=args.max_raw, max_retained_candidates=args.retain)
    targets = [parse_target(text) for text in args.targets.split(",") if text]
    wanted = {ShapeRole(name) for name in args.source_roles.split(",") if name}
    tasks = [(p.stem, json.loads(p.read_text(encoding="utf-8")), "adobe_sample") for p in sorted(Path("input").glob("*.json"))]
    if args.sources:
        manifest_path = args.sources.parent / "manifest.jsonl"
        lineage = ({row["source_design_id"]: row["template_family_id"]
                    for row in (json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip())}
                   if manifest_path.exists() else {})
        chosen = 0
        for p in sorted(args.sources.glob("*.json")):
            source = json.loads(p.read_text(encoding="utf-8"))
            if wanted and not wanted & {i.role for i in classify_shapes(Design.from_adobe(source)).values()}:
                continue
            if chosen >= args.source_limit:
                break
            chosen += 1
            family = lineage.get(identity(source), "unknown")
            for target in targets:
                tasks.append((f"{p.stem}_{target['width']}x{target['height']}",
                              dict(source=source, target_canvas=target), family))
    summaries = []
    for name, task, family in tasks:
        report = evaluate_task(task, args.output / name, renderer, config, family_id=family,
                               render_pool=args.render_pool, measured_fit=args.measured_fit,
                               compare_baseline=not args.no_compare)
        summaries.append(dict(task=name, **{k: v for k, v in report.items() if k not in ("candidates", "renders")}))
        print(name, report["valid"], "valid /", report["generated"], "generated; stretched wins:",
              report["shape_stretch"]["stretched_wins"], "; render failures", len(report["render_failures"]), flush=True)
    write_json(args.output / "summary.json", summaries)
    totals = aggregate(summaries)
    write_json(args.output / "aggregate.json", totals)
    (args.output / "index.html").write_text('<meta charset="utf-8"><h1>S04 evaluation</h1>' + "".join(
        f'<p>{html.escape(name)}: <a href="{html.escape(name)}/index.html">pool</a> | '
        f'<a href="{html.escape(name)}/comparison.html">before/after comparison</a></p>' for name, _, _ in tasks), encoding="utf-8")
    return int(any(row["render_failures"] or not row["winner"] for row in summaries))


if __name__ == "__main__":
    raise SystemExit(main())
