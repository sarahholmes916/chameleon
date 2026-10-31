"""Pairwise-preference training of the existing GATv2 LayoutRanker (Appendix E.3 settings, no tuning).

`python -m training.train` trains three data regimens x several seeds on family-disjoint splits:
  old_only      my original tournament labels
  old_plus_new  original labels + human-verified synthetic labels
  new_only      human-verified synthetic labels only
Validation (early stopping / model selection) never sees test families. Test numbers are reported, never used to choose.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F
from torch_geometric.data import Batch

from engine.graph_features import FEATURE_VERSION_V3
from engine.internal_schema import Canvas, Design
from engine.layout_objective import evaluate_objective
from engine.ranker import LayoutRanker, RankerConfig, load_ranker_checkpoint, save_ranker_checkpoint, score_candidates
from training import data as D

OUT = D.PILOT_DIR / "training_v1"
HYPERPARAMETERS = dict(optimizer="AdamW", lr=1e-3, weight_decay=1e-4, batch_pairs=32, grad_clip=1.0, max_epochs=100, patience=12,
                       dropout=0.10, loss="BCEWithLogits(s_a - s_b, y)", ties_and_both_bad="excluded from the loss")
SEEDS = (42, 43, 44, 45, 46)
REGIMENS = ("old_only", "old_plus_new", "new_only")


# ----------------------------------------------------------------------------- data views
def select(pairs: list[D.Pair], graphs: dict, split: str | None = None, origins: tuple[str, ...] | None = None) -> list[D.Pair]:
    return [p for p in D.usable_pairs(pairs, graphs) if (split is None or p.split == split) and (origins is None or p.origin in origins)]


def regimen_origins(name: str) -> tuple[str, ...]:
    return {"old_only": (D.ORIGIN_OLD,), "new_only": (D.ORIGIN_NEW,), "old_plus_new": (D.ORIGIN_OLD, D.ORIGIN_NEW)}[name]


def make_batch(pairs: list[D.Pair], graphs: dict, device) -> tuple[Batch, torch.Tensor]:
    data = [graphs[p.key_a()] for p in pairs] + [graphs[p.key_b()] for p in pairs]
    y = torch.tensor([p.y for p in pairs], dtype=torch.float32, device=device)
    return Batch.from_data_list(data).to(device), y


@torch.no_grad()
def logits(model: LayoutRanker, pairs: list[D.Pair], graphs: dict, device, chunk: int = 128) -> torch.Tensor:
    model.eval()
    out = []
    for i in range(0, len(pairs), chunk):
        part = pairs[i:i + chunk]
        batch, _ = make_batch(part, graphs, device)
        s = model(batch)
        out.append((s[:len(part)] - s[len(part):]).cpu())
    return torch.cat(out) if out else torch.zeros(0)


def metrics_from(d: torch.Tensor, pairs: list[D.Pair], weight_fn=None) -> dict:
    """Pairwise accuracy and BCE; with `weight_fn(pair) -> float` both are weighted means (default: every pair counts 1)."""
    if not pairs:
        return dict(n=0, accuracy=None, loss=None)
    y = torch.tensor([p.y for p in pairs])
    correct = ((d > 0).float() == y).float()
    per_loss = F.binary_cross_entropy_with_logits(d, y, reduction="none")
    w = torch.tensor([weight_fn(p) for p in pairs], dtype=torch.float32) if weight_fn else torch.ones(len(pairs))
    return dict(n=len(pairs), accuracy=float((correct * w).sum() / w.sum()), loss=float((per_loss * w).sum() / w.sum()))


# ----------------------------------------------------------------------------- GNN training
def train_run(train: list[D.Pair], val: list[D.Pair], graphs: dict, seed: int, device, verbose: bool = False, weight_fn=None) -> dict:
    h = HYPERPARAMETERS
    torch.manual_seed(seed)
    random.seed(seed)
    model = LayoutRanker(RankerConfig(dropout=h["dropout"])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=h["lr"], weight_decay=h["weight_decay"])
    gen = torch.Generator().manual_seed(seed)
    best = dict(acc=-1.0, loss=float("inf"), epoch=0, state=None)
    history, stale = [], 0
    for epoch in range(1, h["max_epochs"] + 1):
        model.train()
        order = torch.randperm(len(train), generator=gen).tolist()
        total, count = 0.0, 0
        for i in range(0, len(order), h["batch_pairs"]):
            part = [train[j] for j in order[i:i + h["batch_pairs"]]]
            batch, y = make_batch(part, graphs, device)
            s = model(batch)
            per_pair = F.binary_cross_entropy_with_logits(s[:len(part)] - s[len(part):], y, reduction="none")
            if weight_fn is None:
                loss = per_pair.mean()
            else:  # weighted mean: a down-weighted pair pulls less, the loss scale stays comparable
                w = torch.tensor([weight_fn(p) for p in part], dtype=torch.float32, device=device)
                loss = (per_pair * w).sum() / w.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), h["grad_clip"])
            opt.step()
            total += float(loss.detach()) * len(part)
            count += len(part)
        v = metrics_from(logits(model, val, graphs, device), val, weight_fn)
        history.append(dict(epoch=epoch, train_loss=total / count, val_accuracy=v["accuracy"], val_loss=v["loss"]))
        improved = v["accuracy"] > best["acc"] + 1e-12 or (abs(v["accuracy"] - best["acc"]) <= 1e-12 and v["loss"] < best["loss"] - 1e-9)
        if improved:
            best.update(acc=v["accuracy"], loss=v["loss"], epoch=epoch, state={k: t.detach().cpu().clone() for k, t in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
        if verbose:
            print(f"  ep{epoch:3d} train_loss={total / count:.4f} val_acc={v['accuracy']:.3f} val_loss={v['loss']:.4f} best_ep={best['epoch']}", flush=True)
        if stale >= h["patience"]:
            break
    model.load_state_dict(best["state"])
    return dict(model=model, best_epoch=best["epoch"], epochs_run=len(history), stopped_early=len(history) < h["max_epochs"],
                val_accuracy=best["acc"], val_loss=best["loss"], history=history)


# ----------------------------------------------------------------------------- baselines
def objective_scores(pairs: list[D.Pair]) -> dict[str, float]:
    """Deterministic production objective (a cost; higher score = better).  Per (source, candidate, target)."""
    designs, out = {}, {}
    for p in pairs:
        if p.source_id not in designs:
            designs[p.source_id] = Design.from_adobe(json.loads(D.abs_path(p.source_path).read_text(encoding="utf-8")))
        target = Canvas.from_adobe({"width": p.target[0], "height": p.target[1]})
        for key, path in ((p.key_a(), p.a_path), (p.key_b(), p.b_path)):
            if key not in out:
                cand = Design.from_adobe(json.loads(D.abs_path(path).read_text(encoding="utf-8")))
                out[key] = -evaluate_objective(designs[p.source_id], cand, target).total
    return out


def objective_correct(pairs: list[D.Pair], scores: dict) -> list[float]:
    res = []
    for p in pairs:
        d = scores[p.key_a()] - scores[p.key_b()]
        res.append(0.5 if d == 0 else float((d > 0) == (p.label == "A_BETTER")))
    return res


def pooled_features(g) -> torch.Tensor:
    x, e = g.x_numeric, g.edge_attr
    return torch.cat([x.mean(0), x.max(0).values, e.mean(0), torch.tensor([float(x.shape[0])]).log()])


def train_linear_baseline(train, val, graphs, seed: int) -> dict:
    """Pooled-feature logistic model on feature differences: no message passing, no GNN (simple learned baseline)."""
    torch.manual_seed(seed)
    cache = {k: pooled_features(g) for k, g in graphs.items()}
    def diff(pairs): return torch.stack([cache[p.key_a()] - cache[p.key_b()] for p in pairs])
    xt, xv = diff(train), diff(val)
    yt, yv = torch.tensor([p.y for p in train]), torch.tensor([p.y for p in val])
    scale = xt.abs().mean(0).clamp_min(1e-6)
    xt, xv = xt / scale, xv / scale
    w = torch.zeros(xt.shape[1], requires_grad=True)
    opt = torch.optim.AdamW([w], lr=1e-2, weight_decay=1e-2)
    best = (-1.0, float("inf"), w.detach().clone())
    stale = 0
    for _ in range(500):
        opt.zero_grad()
        F.binary_cross_entropy_with_logits(xt @ w, yt).backward()
        opt.step()
        with torch.no_grad():
            d = xv @ w
            acc = float(((d > 0).float() == yv).float().mean())
            loss = float(F.binary_cross_entropy_with_logits(d, yv))
        if acc > best[0] + 1e-12 or (abs(acc - best[0]) <= 1e-12 and loss < best[1] - 1e-9):
            best, stale = (acc, loss, w.detach().clone()), 0
        else:
            stale += 1
        if stale >= 30:
            break
    return dict(w=best[2], scale=scale, cache=cache, val_accuracy=best[0])


def linear_correct(b: dict, pairs: list[D.Pair]) -> list[float]:
    d = torch.stack([b["cache"][p.key_a()] - b["cache"][p.key_b()] for p in pairs]) / b["scale"] @ b["w"]
    return [float((x > 0) == (p.label == "A_BETTER")) for x, p in zip(d.tolist(), pairs)]


# ----------------------------------------------------------------------------- reporting helpers
def cluster_ci(correct: list[float], clusters: list[str], reps: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Bootstrap over source designs (comparisons from one source are dependent)."""
    if not correct:
        return (float("nan"), float("nan"))
    groups = defaultdict(list)
    for c, k in zip(correct, clusters):
        groups[k].append(c)
    keys, rng, stats = sorted(groups), random.Random(seed), []
    for _ in range(reps):
        draw = [rng.choice(keys) for _ in keys]
        vals = [c for k in draw for c in groups[k]]
        stats.append(sum(vals) / len(vals))
    stats.sort()
    return (stats[int(.025 * reps)], stats[int(.975 * reps) - 1])


def mean_sd(values: list[float]) -> dict:
    if not values:
        return dict(mean=None, sd=None, n=0)
    m = sum(values) / len(values)
    sd = (sum((v - m) ** 2 for v in values) / max(1, len(values) - 1)) ** .5
    return dict(mean=round(m, 4), sd=round(sd, 4), n=len(values))


def breakdown(pairs: list[D.Pair], correct: list[float], key) -> dict:
    groups = defaultdict(list)
    for p, c in zip(pairs, correct):
        groups[key(p)].append(c)
    return {str(k): dict(n=len(v), accuracy=round(sum(v) / len(v), 3)) for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--feature-version", default=FEATURE_VERSION_V3)
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    if a.max_epochs:
        HYPERPARAMETERS["max_epochs"] = a.max_epochs
    device = torch.device(a.device)
    out = a.out
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    pairs = D.load_all_pairs()
    graphs = torch.load(out / f"graphs_{a.feature_version}.pt", weights_only=False)
    usable = D.usable_pairs(pairs, graphs)
    test_sets = {
        "old_test": select(pairs, graphs, "test", (D.ORIGIN_OLD,)),
        "new_test": select(pairs, graphs, "test", (D.ORIGIN_NEW,)),
        "all_test": select(pairs, graphs, "test"),
    }
    print("device", device, "| usable", len(usable), "| test sets", {k: len(v) for k, v in test_sets.items()}, flush=True)

    # baseline 1: deterministic production objective (no fitting at all)
    obj = objective_scores(usable)
    baselines = {k: {"objective_accuracy": round(sum(objective_correct(v, obj)) / len(v), 4), "n": len(v)} for k, v in test_sets.items()}
    for split in ("train", "val"):
        v = select(pairs, graphs, split)
        baselines[f"all_{split}"] = {"objective_accuracy": round(sum(objective_correct(v, obj)) / len(v), 4), "n": len(v)}
    baselines["majority_A_rate"] = {k: round(sum(p.label == "A_BETTER" for p in v) / len(v), 3) for k, v in test_sets.items()}

    runs, linear = [], {}
    for regimen in REGIMENS:
        origins = regimen_origins(regimen)
        train, val = select(pairs, graphs, "train", origins), select(pairs, graphs, "val", origins)
        lin = train_linear_baseline(train, val, graphs, seed=42)
        linear[regimen] = {k: dict(n=len(v), accuracy=round(sum(linear_correct(lin, v)) / len(v), 4)) for k, v in test_sets.items()}
        linear[regimen]["val_accuracy"] = round(lin["val_accuracy"], 4)
        for seed in a.seeds:
            t0 = time.time()
            r = train_run(train, val, graphs, seed, device, a.verbose)
            model = r.pop("model")
            record = dict(regimen=regimen, seed=seed, n_train=len(train), n_val=len(val), best_epoch=r["best_epoch"], epochs_run=r["epochs_run"],
                          stopped_early=r["stopped_early"], val_accuracy=round(r["val_accuracy"], 4), val_loss=round(r["val_loss"], 4),
                          seconds=round(time.time() - t0, 1), history=r["history"])
            record["train"] = metrics_from(logits(model, train, graphs, device), train)
            record["test"] = {}
            record["_correct"] = {}
            for name, tp in test_sets.items():
                d = logits(model, tp, graphs, device)
                record["test"][name] = metrics_from(d, tp)
                record["_correct"][name] = [float((x > 0) == (p.label == "A_BETTER")) for x, p in zip(d.tolist(), tp)]
            for part, origin in (("val_old", (D.ORIGIN_OLD,)), ("val_new", (D.ORIGIN_NEW,))):
                if origin[0] in origins:
                    vv = select(pairs, graphs, "val", origin)
                    record[part] = metrics_from(logits(model, vv, graphs, device), vv)
            record["_state"] = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
            runs.append(record)
            t = record["test"]
            print(f"{regimen:13s} seed {seed}: ep {r['best_epoch']:3d}/{r['epochs_run']:3d} train_acc {record['train']['accuracy']:.3f} "
                  f"val_acc {r['val_accuracy']:.3f} | test old {t['old_test']['accuracy']:.3f} new {t['new_test']['accuracy']:.3f} "
                  f"all {t['all_test']['accuracy']:.3f} | {record['seconds']}s", flush=True)

    # ------------------------------------------------------------------ summary + checkpoints
    summary = {}
    for regimen in REGIMENS:
        rs = [r for r in runs if r["regimen"] == regimen]
        summary[regimen] = dict(
            n_train=rs[0]["n_train"], n_val=rs[0]["n_val"], runs=len(rs),
            train_accuracy=mean_sd([r["train"]["accuracy"] for r in rs]), val_accuracy=mean_sd([r["val_accuracy"] for r in rs]),
            best_epoch=mean_sd([r["best_epoch"] for r in rs]),
            **{name: mean_sd([r["test"][name]["accuracy"] for r in rs]) for name in test_sets})
        # model selection is by VALIDATION only
        best = max(rs, key=lambda r: (r["val_accuracy"], -r["val_loss"]))
        summary[regimen]["selected_seed"] = best["seed"]
        summary[regimen]["selected_test"] = {k: v for k, v in best["test"].items()}
        ci = {}
        for name, tp in test_sets.items():
            ci[name] = cluster_ci(best["_correct"][name], [p.source_id for p in tp])
        summary[regimen]["selected_test_ci95_source_bootstrap"] = {k: [round(v[0], 3), round(v[1], 3)] for k, v in ci.items()}
        summary[regimen]["selected_breakdown_all_test"] = dict(
            by_target=breakdown(test_sets["all_test"], best["_correct"]["all_test"], lambda p: p.target_name),
            by_origin=breakdown(test_sets["all_test"], best["_correct"]["all_test"], lambda p: p.origin),
            by_pair_type=breakdown(test_sets["all_test"], best["_correct"]["all_test"], lambda p: p.pair_type or "n/a"),
            by_family=breakdown(test_sets["all_test"], best["_correct"]["all_test"], lambda p: p.family))
        # checkpoint: only declared-disjoint source IDs; metrics are train/validation only
        train_p, val_p = select(pairs, graphs, "train", regimen_origins(regimen)), select(pairs, graphs, "val", regimen_origins(regimen))
        splits = dict(train=sorted({p.source_id for p in train_p}), validation=sorted({p.source_id for p in val_p}),
                      test=sorted({p.source_id for p in test_sets["all_test"]}))
        model = LayoutRanker(RankerConfig(dropout=HYPERPARAMETERS["dropout"]))
        model.load_state_dict(best["_state"])
        path = out / "checkpoints" / f"ranker_{regimen}_best.pt"
        save_ranker_checkpoint(path, model, training_metadata=dict(
            status="trained", source_splits=splits, seed=best["seed"],
            metrics=dict(val_accuracy=best["val_accuracy"], val_loss=best["val_loss"], train_accuracy=best["train"]["accuracy"],
                         best_epoch=best["best_epoch"], n_train_pairs=len(train_p), n_val_pairs=len(val_p))))
        (path.with_suffix(".training.json")).write_text(json.dumps(dict(
            regimen=regimen, feature_version=a.feature_version, hyperparameters=HYPERPARAMETERS, seed=best["seed"], device=str(device),
            torch=str(torch.__version__), selection="best validation accuracy then lower validation loss among seeds",
            inputs_sha256=json.loads((out / "audit_report.json").read_text())["inputs_sha256"], history=best["history"],
            note="GPU scatter ops use atomics: reruns are statistically, not bit-wise, reproducible."), indent=1))
        summary[regimen]["checkpoint"] = str(path.relative_to(D.REPO_ROOT)) if path.is_relative_to(D.REPO_ROOT) else str(path)

    # checkpoint round-trip: reload on CPU through the production inference path and re-score the selected model's test pairs
    roundtrip = {}
    for regimen in REGIMENS:
        ck = load_ranker_checkpoint(out / "checkpoints" / f"ranker_{regimen}_best.pt", expected_config=RankerConfig(dropout=HYPERPARAMETERS["dropout"]))
        tp = test_sets["old_test"]
        designs, ok = {}, 0
        for p in tp:
            designs.setdefault(p.source_id, Design.from_adobe(json.loads(D.abs_path(p.source_path).read_text(encoding="utf-8"))))
            cands = [Design.from_adobe(json.loads(D.abs_path(x).read_text(encoding="utf-8"))) for x in (p.a_path, p.b_path)]
            sa, sb = score_candidates(designs[p.source_id], cands, Canvas.from_adobe({"width": p.target[0], "height": p.target[1]}), checkpoint=ck)
            ok += float((sa > sb) == (p.label == "A_BETTER"))
        roundtrip[regimen] = dict(old_test_accuracy_via_score_candidates_cpu=round(ok / len(tp), 4), n=len(tp),
                                  training_status=ck.training_metadata["status"])

    result = dict(hyperparameters=HYPERPARAMETERS, seeds=a.seeds, feature_version=a.feature_version, device=str(device),
                  torch=str(torch.__version__), counts=dict(usable_pairs=len(usable), test={k: len(v) for k, v in test_sets.items()}),
                  baselines=baselines, linear_pooled_baseline=linear, summary=summary, checkpoint_roundtrip=roundtrip,
                  runs=[{k: v for k, v in r.items() if not k.startswith("_")} for r in runs], wall_seconds=round(time.time() - t_start, 1))
    (out / "training_results.json").write_text(json.dumps(result, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("baselines", "linear_pooled_baseline", "checkpoint_roundtrip")}, indent=1))
    print(json.dumps({r: {k: v for k, v in s.items() if k not in ("selected_breakdown_all_test",)} for r, s in summary.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
