"""CPU inference cost of a ranker checkpoint through the production scoring path (score_candidates).

`python -m training.benchmark_inference CHECKPOINT [--threads 1]`.  Reports import time, checkpoint load time and ms per candidate
(graph build + forward) over the held-out evaluation pools.  Local numbers on this machine only; they are not the M5 Pro reference.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    t0 = time.perf_counter()
    import torch
    import torch_geometric
    from engine.ranker import load_ranker_checkpoint, score_candidates
    import_seconds = time.perf_counter() - t0
    torch.set_num_threads(a.threads)

    from engine.internal_schema import Canvas
    from training import blind_eval as B

    pools = B.load_pools()
    t0 = time.perf_counter()
    ck = load_ranker_checkpoint(a.checkpoint)
    load_seconds = time.perf_counter() - t0
    rows = []
    for pool in pools:
        source, target = B._design(pool.source_path), Canvas.from_adobe({"width": pool.target[0], "height": pool.target[1]})
        designs = [B._design(c.path) for c in pool.candidates]
        t0 = time.perf_counter()
        score_candidates(source, designs, target, checkpoint=ck)
        rows.append((len(designs), len(source.elements), time.perf_counter() - t0))
    n = sum(r[0] for r in rows)
    total = sum(r[2] for r in rows)
    per_pool = sorted(r[2] for r in rows)
    result = dict(
        checkpoint=str(a.checkpoint), threads=a.threads, torch=str(torch.__version__), pyg=str(torch_geometric.__version__),
        cuda_build=bool(torch.version.cuda), python=platform.python_version(), machine=f"{platform.system()} {platform.machine()}",
        import_seconds=round(import_seconds, 2), checkpoint_load_seconds=round(load_seconds, 3), pools=len(rows), candidates=n,
        ms_per_candidate_mean=round(1000 * total / n, 1), pool_seconds_median=round(per_pool[len(per_pool) // 2], 3),
        pool_seconds_p95=round(per_pool[int(.95 * len(per_pool)) - 1], 3), pool_seconds_max=round(per_pool[-1], 3),
        max_elements=max(r[1] for r in rows), note="first pool includes one-time warm-up; numbers are from the development machine")
    try:
        import psutil
        result["peak_rss_mb"] = round(psutil.Process().memory_info().peak_wset / 2**20) if hasattr(psutil.Process().memory_info(), "peak_wset") else round(psutil.Process().memory_info().rss / 2**20)
    except Exception:
        pass
    text = json.dumps(result, indent=1)
    print(text)
    if a.out:
        a.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
