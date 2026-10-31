"""Synthetic-label loss weighting experiment: original human pairs weight 1.0, verified synthetic pairs weight lambda.

`python -m training.weighting` (needs `python -m training.build_dataset` first).  Same GATv2 architecture, graph_v3 features, family-disjoint
splits, optimiser and seeds as the earlier run.  Earlier checkpoints are never overwritten (weighted ones go to checkpoints/weighted/).

Rules fixed BEFORE any result was looked at:
  * early stopping inside a run: weighted validation accuracy (weights as in training), weighted validation loss as tie-break;
  * choosing lambda: the highest mean accuracy over seeds on the ORIGINAL-human validation pairs (53 pairs); ties -> higher pooled
    unweighted validation accuracy; remaining ties -> smaller lambda;
  * test pairs are evaluated and reported for every lambda but are never used for any choice.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from engine.graph_features import FEATURE_VERSION_V3
from engine.ranker import LayoutRanker, RankerConfig, save_ranker_checkpoint
from training import data as D
from training import train as T

OUT = D.PILOT_DIR / "training_v1"
LAMBDAS = (0.25, 0.5, 0.75)
SEEDS = T.SEEDS


def weight_fn_for(lam: float):
    return lambda p: 1.0 if p.origin == D.ORIGIN_OLD else lam


def pick_lambda(rows: dict[float, dict]) -> float:
    """The pre-registered validation-only rule."""
    return max(sorted(rows), key=lambda lam: (rows[lam]["val_old_accuracy"]["mean"], rows[lam]["val_pooled_accuracy"]["mean"], -lam))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    ap.add_argument("--lambdas", type=float, nargs="*", default=list(LAMBDAS))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    device, out = torch.device(a.device), a.out
    ck_dir = out / "checkpoints" / "weighted"
    ck_dir.mkdir(parents=True, exist_ok=True)

    pairs = D.load_all_pairs()
    graphs = torch.load(out / f"graphs_{FEATURE_VERSION_V3}.pt", weights_only=False)
    both = (D.ORIGIN_OLD, D.ORIGIN_NEW)
    train, val = T.select(pairs, graphs, "train", both), T.select(pairs, graphs, "val", both)
    val_old, val_new = [p for p in val if p.origin == D.ORIGIN_OLD], [p for p in val if p.origin == D.ORIGIN_NEW]
    tests = {"old_test": T.select(pairs, graphs, "test", (D.ORIGIN_OLD,)), "new_test": T.select(pairs, graphs, "test", (D.ORIGIN_NEW,)),
             "all_test": T.select(pairs, graphs, "test")}
    train_old = [p for p in train if p.origin == D.ORIGIN_OLD]
    print("device", device, "| train", len(train), "val", len(val), "(old", len(val_old), "new", len(val_new), ") test", {k: len(v) for k, v in tests.items()}, flush=True)

    runs, states = [], {}
    for lam in a.lambdas:
        wf = weight_fn_for(lam)
        for seed in a.seeds:
            t0 = time.time()
            r = T.train_run(train, val, graphs, seed, device, weight_fn=wf)
            model = r.pop("model")
            rec = dict(lam=lam, seed=seed, best_epoch=r["best_epoch"], epochs_run=r["epochs_run"], stopped_early=r["stopped_early"],
                       weighted_val_accuracy=round(r["val_accuracy"], 4), weighted_val_loss=round(r["val_loss"], 4), seconds=round(time.time() - t0, 1))
            rec["val_old"] = T.metrics_from(T.logits(model, val_old, graphs, device), val_old)
            rec["val_new"] = T.metrics_from(T.logits(model, val_new, graphs, device), val_new)
            rec["val_pooled"] = T.metrics_from(T.logits(model, val, graphs, device), val)
            rec["train_all"] = T.metrics_from(T.logits(model, train, graphs, device), train)
            rec["train_old"] = T.metrics_from(T.logits(model, train_old, graphs, device), train_old)
            rec["test"] = {k: T.metrics_from(T.logits(model, v, graphs, device), v) for k, v in tests.items()}
            states[(lam, seed)] = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
            runs.append(rec)
            print(f"lambda {lam:.2f} seed {seed}: ep {rec['best_epoch']:3d}/{rec['epochs_run']:3d} val_old {rec['val_old']['accuracy']:.3f} "
                  f"val_pooled {rec['val_pooled']['accuracy']:.3f} | test old {rec['test']['old_test']['accuracy']:.3f} "
                  f"new {rec['test']['new_test']['accuracy']:.3f} all {rec['test']['all_test']['accuracy']:.3f} | {rec['seconds']}s", flush=True)

    # earlier runs (already trained, reused as-is for the comparison; lambda 1.0 = combined, 'original-only' = lambda 0)
    prev = json.loads((out / "training_results.json").read_text(encoding="utf-8"))
    summary = {}

    def pack(rs: list[dict], val_old_get, val_pooled_get, train_get, test_get) -> dict:
        return dict(runs=len(rs), val_old_accuracy=T.mean_sd([val_old_get(r) for r in rs]), val_pooled_accuracy=T.mean_sd([val_pooled_get(r) for r in rs]),
                    train_accuracy=T.mean_sd([train_get(r) for r in rs]),
                    **{k: T.mean_sd([test_get(r, k) for r in rs]) for k in tests})

    for lam in a.lambdas:
        rs = [r for r in runs if r["lam"] == lam]
        summary[lam] = pack(rs, lambda r: r["val_old"]["accuracy"], lambda r: r["val_pooled"]["accuracy"], lambda r: r["train_all"]["accuracy"],
                            lambda r, k: r["test"][k]["accuracy"])
        summary[lam]["best_epoch"] = T.mean_sd([r["best_epoch"] for r in rs])
    prev_rows = {}
    for name, label in (("old_only", "0 (original only)"), ("old_plus_new", "1.0 (combined, unweighted)")):
        rs = [r for r in prev["runs"] if r["regimen"] == name and r["seed"] in a.seeds]
        prev_rows[label] = pack(rs, lambda r: r["val_old"]["accuracy"] if "val_old" in r else r["val_accuracy"],
                                lambda r: r["val_accuracy"],  # old_only validates on original pairs only, so pooled == original
                                lambda r: r["train"]["accuracy"], lambda r, k: r["test"][k]["accuracy"])

    chosen = pick_lambda(summary)
    # paired (same seed) differences vs the earlier models
    def per_seed(rs, key):
        return {r["seed"]: r["test"][key]["accuracy"] for r in rs}
    paired = {}
    for lam in a.lambdas:
        rs = [r for r in runs if r["lam"] == lam]
        for base in ("old_only", "old_plus_new"):
            br = [r for r in prev["runs"] if r["regimen"] == base]
            for key in tests:
                d = [per_seed(rs, key)[s] - per_seed(br, key)[s] for s in a.seeds]
                paired[f"lambda_{lam}_minus_{base}:{key}"] = dict(per_seed=[round(x, 3) for x in d], **T.mean_sd(d))

    # checkpoints: best seed per lambda by weighted validation (never by test); earlier checkpoints untouched
    saved = {}
    for lam in a.lambdas:
        rs = [r for r in runs if r["lam"] == lam]
        best = max(rs, key=lambda r: (r["weighted_val_accuracy"], -r["weighted_val_loss"]))
        model = LayoutRanker(RankerConfig(dropout=T.HYPERPARAMETERS["dropout"]))
        model.load_state_dict(states[(lam, best["seed"])])
        path = ck_dir / f"ranker_synth_weight_{lam:g}_best.pt"
        assert not path.exists(), f"refusing to overwrite {path}"
        splits = dict(train=sorted({p.source_id for p in train}), validation=sorted({p.source_id for p in val}),
                      test=sorted({p.source_id for p in tests["all_test"]}))
        save_ranker_checkpoint(path, model, training_metadata=dict(
            status="trained", source_splits=splits, seed=best["seed"],
            metrics=dict(weighted_val_accuracy=best["weighted_val_accuracy"], val_old_accuracy=best["val_old"]["accuracy"],
                         val_pooled_accuracy=best["val_pooled"]["accuracy"], train_accuracy=best["train_all"]["accuracy"],
                         best_epoch=best["best_epoch"], synthetic_loss_weight=lam)))
        path.with_suffix(".training.json").write_text(json.dumps(dict(
            lam=lam, original_weight=1.0, feature_version=FEATURE_VERSION_V3, hyperparameters=T.HYPERPARAMETERS, seed=best["seed"], device=str(device),
            selection="best weighted validation accuracy among seeds", note="GPU scatter ops use atomics: reruns are statistically, not bit-wise, reproducible."), indent=1))
        saved[str(lam)] = str(path.relative_to(D.REPO_ROOT))

    result = dict(rule="lambda chosen by mean original-human validation accuracy; test never used for selection", lambdas=list(a.lambdas), seeds=a.seeds,
                  chosen_lambda=chosen, weighted_runs_summary={str(k): v for k, v in summary.items()}, earlier_models=prev_rows,
                  paired_vs_earlier=paired, checkpoints=saved, device=str(device), torch=str(torch.__version__),
                  runs=runs)
    (out / "weighting_results.json").write_text(json.dumps(result, indent=1, sort_keys=True), encoding="utf-8")
    print("chosen lambda (validation rule):", chosen)
    for label, row in {**prev_rows, **{f"{k}": v for k, v in summary.items()}}.items():
        print(f"{str(label):28s} val_old {row['val_old_accuracy']['mean']} val_pooled {row['val_pooled_accuracy']['mean']} | test old {row['old_test']['mean']}±{row['old_test']['sd']} "
              f"new {row['new_test']['mean']}±{row['new_test']['sd']} all {row['all_test']['mean']}±{row['all_test']['sd']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
