"""Combine + audit the human labels and cache their graph tensors.  `python -m training.build_dataset`"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from engine.graph_features import FEATURE_VERSION_V3
from training import data as D

OUT = D.PILOT_DIR / "training_v1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-version", default=FEATURE_VERSION_V3)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pairs = D.load_all_pairs()
    report, graphs = D.audit(pairs, a.feature_version)
    report["seconds"] = round(time.time() - t0, 1)
    report["inputs_sha256"] = {str(p.relative_to(D.REPO_ROOT)): D.sha256_file(p) for p in
                               (D.OLD_COMPARISONS, D.OLD_CANDIDATES, D.NEW_COMPARISONS, D.PROPOSALS, D.NEW_SPLITS)}
    report["old_split"] = D.old_family_split()
    (a.out / "combined_comparisons.jsonl").write_text("".join(json.dumps(D.pair_row(p), sort_keys=True) + "\n" for p in pairs), encoding="utf-8")
    (a.out / "audit_report.json").write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8")
    torch.save(graphs, a.out / f"graphs_{a.feature_version}.pt")
    print(json.dumps({k: v for k, v in report.items() if k not in ("inputs_sha256",)}, indent=1, sort_keys=True))
    return 1 if report["problem_counts"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
