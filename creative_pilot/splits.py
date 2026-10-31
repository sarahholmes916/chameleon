"""Family-disjoint train / validation / test assignment for the creative pilot.

Whole composition families go to one split, stratified by difficulty level, so no source of a held-out family has a
sibling in training. Candidates, degraded negatives and every comparison inherit the split of their source. The
assignment depends only on family names and difficulty (never on candidates, labels or scores), so tuning the
generator or the labeler on training families cannot touch test families.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from creative_pilot.families import FAMILIES

QUOTAS = {1: (2, 1, 1), 2: (8, 2, 2), 3: (4, 1, 1)}  # train, val, test families per difficulty level
SEED = 11


def assign_families(seed: int = SEED) -> dict:
    out = {}
    for level, (n_train, n_val, n_test) in QUOTAS.items():
        names = sorted((n for n, v in FAMILIES.items() if v["difficulty"] == level),
                       key=lambda n: hashlib.sha256(f"{seed}:{n}".encode()).hexdigest())
        assert len(names) == n_train + n_val + n_test, (level, len(names))
        for i, n in enumerate(names):
            out[n] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    return out


def build(pilot_root: Path, calibration: Path | None = None) -> dict:
    fam_split = assign_families()
    sources = [json.loads(l) for l in (pilot_root / "sources" / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    by_source = {r["source_design_id"]: fam_split[r["template_family_id"]] for r in sources}
    human_families = []
    if calibration and calibration.exists():
        c = json.loads(calibration.read_text(encoding="utf-8"))["counts"]
        human_families = sorted(set(c["calibration_families"]) | set(c["heldout_families"]))
    overlap = sorted(set(fam_split) & set(human_families))
    report = dict(
        policy="whole families per split, stratified by difficulty; split depends only on family name and difficulty",
        seed=SEED, family_split=fam_split, source_split=by_source,
        counts=dict(families=dict(Counter(fam_split.values())), sources=dict(Counter(by_source.values()))),
        families_by_split={s: sorted(f for f, v in fam_split.items() if v == s) for s in ("train", "val", "test")},
        existing_human_ranking_families=human_families, family_overlap_with_existing_human_data=overlap,
        notes=["No new family overlaps the existing human-ranked template families (overlap list above).",
               "All sources share the same three bundled assets, two font families and palette sampler as the existing "
               "sources, so style leakage across datasets is possible even without family overlap.",
               "Existing human comparisons were used to calibrate/measure the rendered-metrics labeler by family "
               "(see labeler_calibration.json); keep those held-out families out of any final GNN evaluation that also "
               "claims to be untouched by the labeler."])
    (pilot_root / "splits.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-root", type=Path, required=True)
    a = ap.parse_args(argv)
    r = build(a.pilot_root, a.pilot_root / "labeler_calibration.json")
    print(json.dumps(dict(counts=r["counts"], families_by_split=r["families_by_split"], overlap=r["family_overlap_with_existing_human_data"]), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
