"""Generate, validate and render the creative-pilot source designs (reuses scripts.source_dataset gates)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from creative_pilot import families as fam
from scripts import source_dataset as sd
from scripts.source_variation import load_profile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "creative_pilot_v1" / "sources"


def default_counts(total: int = 100) -> dict:
    """Level 1: 4 families x5 = 20; level 2: 12 families (2 x5 + 10 x4) = 50; level 3: 6 families x5 = 30."""
    counts = {}
    l2 = [n for n, v in fam.FAMILIES.items() if v["difficulty"] == 2]
    for n, v in fam.FAMILIES.items():
        counts[n] = 5 if v["difficulty"] != 2 else (5 if l2.index(n) < 2 else 4)
    assert sum(counts.values()) == total, sum(counts.values())
    return counts


def generate_sources(output: Path, *, per_family: int | dict = 5, seed: int = 1000, renderer=None,
                     only: list[str] | None = None, max_attempts_factor: int = 6) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new/empty output directory; existing datasets are never overwritten")
    renderer = renderer or sd.SourceRenderer()
    profile = load_profile()
    for folder in ("sources", "previews"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    names = [n for n in fam.FAMILIES if not only or n in only]
    records, rejected, accepted, pixels = [], [], [], {}
    quota = (lambda n: per_family[n]) if isinstance(per_family, dict) else (lambda n: per_family)
    for name in names:
        findex = list(fam.FAMILIES).index(name)
        got, variant = 0, 0
        while got < quota(name) and variant < quota(name) * max_attempts_factor:
            src_seed = seed + findex * 1000 + variant
            source, params = fam.make_creative_source(name, src_seed, variant, profile=profile)
            report = sd.validate_source(source)
            measurements, reason = [], None
            if report["accepted"]:
                source, measurements = renderer.fit_text(source)
                report = sd.validate_source(source)
                if any(not m["fits"] for m in measurements):
                    report["accepted"] = False
                    report["issues"].append(dict(code="TEXT_OVERFLOW", rule_class="internal",
                                                 message="text does not fit at the 12 px source floor"))
            sid = "sha256:" + sd.digest(source)
            attempt = dict(family=name, variant=variant, seed=src_seed, source_design_id=sid,
                           issues=[i["code"] + ": " + i["message"] for i in report["issues"]])
            variant += 1
            if not report["accepted"]:
                rejected.append(dict(attempt, reason="validation"))
                continue
            dup = next((s for s, prior in accepted if s == sid or sd.near_identical(source, prior)), None)
            if dup:
                rejected.append(dict(attempt, reason="near_duplicate", duplicate_of=dup))
                continue
            stem = f"{len(records):04d}_{name}_{sid[-12:]}"
            preview = output / "previews" / f"{stem}.png"
            try:
                warnings = renderer.render(source, preview)
                if warnings:
                    raise ValueError(f"renderer warnings: {warnings}")
                pixel = sd.verify_png(preview, source)
            except Exception as exc:  # noqa: BLE001 - a render failure rejects one attempt, not the run
                preview.unlink(missing_ok=True)
                rejected.append(dict(attempt, reason="render", error=f"{type(exc).__name__}: {exc}"))
                continue
            if pixel in pixels:
                preview.unlink(missing_ok=True)
                rejected.append(dict(attempt, reason="pixel_duplicate"))
                continue
            pixels[pixel] = sid
            path = output / "sources" / f"{stem}.json"
            sd.write_json(path, source)
            accepted.append((sid, source))
            got += 1
            elements = source["elements"]
            records.append(dict(
                source_design_id=sid, template_family_id=name, source_family_id=name,
                content_category=fam.FAMILIES[name]["composition"], difficulty_level=fam.FAMILIES[name]["difficulty"],
                source_provenance="original_synthetic_creative_v1", seed=src_seed, variant=variant - 1,
                generation_parameters={k: v for k, v in params.items() if k not in ("x_knots", "y_knots")},
                generator_version=fam.GENERATOR_VERSION, canvas=source["canvas"], element_count=len(elements),
                group_count=len({e["group_id"] for e in elements if e.get("group_id")}),
                translucent_count=sum(e.get("opacity", 1) < 1 for e in elements),
                assets=sorted({e["asset_id"] for e in elements if e["type"] == "image"}),
                fonts=sorted({e["font_family"] for e in elements if e["type"] == "text"}),
                validation=report, renderer_version=renderer.version, text_measurements=measurements,
                status="rendered", review_status="pending", split="unassigned", pixel_sha256=pixel,
                source_path=path.relative_to(output).as_posix(), preview_path=preview.relative_to(output).as_posix(),
                preview_sha256=sd.file_hash(preview)))
    sd.write_jsonl(output / "manifest.jsonl", records)
    sd.write_jsonl(output / "rejected_attempts.jsonl", rejected)
    summary = dict(families=len(names), per_family=per_family, requested=sum(quota(n) for n in names), accepted=len(records), rejected_attempts=len(rejected),
                   by_family={n: sum(r["template_family_id"] == n for r in records) for n in names},
                   by_difficulty={str(k): sum(r["difficulty_level"] == k for r in records) for k in (1, 2, 3)})
    sd.write_json(output / "summary.json", summary)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--per-family", type=int, default=0, help="0 = the 100-source plan (20/50/30 by difficulty)")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args(argv)
    print(json.dumps(generate_sources(a.output, per_family=a.per_family or default_counts(), seed=a.seed, only=a.only), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
