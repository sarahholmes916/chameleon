"""Generate candidate pools for every (source, target) task of a creative-pilot source dataset (parallel)."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from creative_pilot.generate import TARGETS, TaskGenerator
from scripts import source_dataset as sd

_STATE: dict = {}


def _init(out_root: str, seed: int):
    _STATE["gen"] = TaskGenerator(sd.SourceRenderer(), Path(out_root), seed=seed)


def _run_source(args):
    rec, sources_root = args
    gen = _STATE["gen"]
    source_path = Path(sources_root) / rec["source_path"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    rec = dict(rec, preview_abs=None)
    prev = Path(sources_root) / rec["preview_path"]
    try:
        rec["preview_abs"] = prev.resolve().relative_to(Path(gen.out_root).resolve().parents[1]).as_posix()
    except ValueError:
        rec["preview_abs"] = str(prev.resolve())
    out = []
    for T in TARGETS:
        try:
            out.append(gen.run_task(rec, source, source_path, T))
        except Exception as exc:  # noqa: BLE001 - record and continue; failures are reported, never hidden
            out.append(dict(task_id=f"{rec['index']:03d}_{rec['template_family_id']}_{T[0]}x{T[1]}", error=f"{type(exc).__name__}: {exc}"))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-root", type=Path, required=True, help="directory holding sources/ (manifest.jsonl); candidates/ is written here")
    ap.add_argument("--limit", type=int, default=0, help="only the first N sources (smoke tests)")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    root = a.pilot_root
    recs = [json.loads(l) for l in (root / "sources" / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    recs = [dict(r, index=i) for i, r in enumerate(recs)][::a.stride]
    if a.limit:
        recs = recs[:a.limit]
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(str(root), a.seed)) as ex:
        futs = [ex.submit(_run_source, (r, str(root / "sources"))) for r in recs]
        for n, f in enumerate(as_completed(futs), 1):
            results.extend(f.result())
            print(f"  sources done {n}/{len(futs)}  elapsed {time.time() - t0:.0f}s", flush=True)
    results.sort(key=lambda r: r["task_id"])
    sd.write_jsonl(root / "tasks.jsonl", results)
    ok = [r for r in results if "error" not in r]
    print(json.dumps(dict(tasks=len(results), failed_tasks=len(results) - len(ok), candidates=sum(r["retained"] for r in ok),
                          seconds=round(time.time() - t0)), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
