"""Calibrate the rendered-metrics labeler against existing human comparisons (family-disjoint).

The labeler scores a candidate from objective measurements of the candidate JSON and its rendered PNG
(smallest/median relative text size, renderer-measured fit, empty space, overlap, hidden content, group spread,
balance, reading-order agreement, headline rank, flat rendered area). A pairwise logistic model on feature
differences predicts which of two layouts a person preferred.

Leakage control: human comparisons are split BY TEMPLATE FAMILY. The model used to propose labels is fitted on the
calibration families only; agreement is reported on held-out families that were never used for fitting or for any
threshold. Leave-one-family-out CV is reported as a second, more stable estimate. With ~370 comparisons from 11
families this is small-sample evidence, not a guarantee, and it is tested on generator outputs (S04 and a
development generator) that differ from the new creative layouts - a distribution shift documented in the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPORTS = ROOT / "data" / "rankings" / "sarah" / "exports"
FEATURES = ("log_min_font", "rel_min_font", "rel_median_font", "frac_fonts_lt6", "fit_failure_rate", "empty_cells",
            "longest_empty", "overlap", "hidden_content", "group_spread", "balance_offset", "order_agreement",
            "headline_first", "flat_render", "content_area")
VERSION = "rendered_metrics_labeler_v1"


def phi(features: dict, render: dict) -> list[float]:
    """Feature vector of one candidate (all inputs are objective measurements)."""
    n = max(features.get("n_text") or 1, 1)
    mn = features.get("min_font") or 1.0
    spread = features.get("group_spread_max")
    return [math.log(max(mn, 1.0)), min(features.get("font_rel_min") or 0.0, .15), min(features.get("font_rel_median") or 0.0, .25),
            (features.get("fonts_lt6") or 0) / n, (features.get("text_fit_failures") or 0) / n, features["empty_cell_fraction"],
            features["longest_empty_stretch"], min(features["content_overlap_fraction"], 1.0), float(features["occluded_pairs"] > 0),
            (min(spread, 6.0) / 6.0) if spread else 0.0, features["balance_offset"], features["order_agreement"],
            float(features.get("headline_rank") == 0), render.get("flat_cell_fraction", 0.0), min(features["content_area_fraction"], 1.5)]


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 60, intercept: bool = False) -> np.ndarray:
    """Ridge logistic regression by Newton steps (numpy only)."""
    if intercept:
        X = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(X.shape[1])
    reg = np.eye(X.shape[1]) * l2
    if intercept:
        reg[-1, -1] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ (p - y) + reg @ w
        h = (X * (p * (1 - p))[:, None]).T @ X + reg + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(h, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def human_candidate_features(exports: Path = EXPORTS, cache: Path | None = None) -> dict:
    """phi() for every candidate in the human export, from its JSON and rendered preview (no labels used)."""
    if cache and cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    from creative_pilot import metrics
    from creative_pilot.fitting import Measurer
    from creative_pilot.generate import pixel_metrics
    from scripts.source_dataset import SourceRenderer
    meas = Measurer(SourceRenderer())
    cands = read_jsonl(exports / "comparisons_all_candidates.jsonl")
    out = {}
    for c in cands:
        source = json.loads(Path(c["source_path"]).read_text(encoding="utf-8"))
        cand = json.loads(Path(c["candidate_path"]).read_text(encoding="utf-8"))
        feats = metrics.layout_features(source, cand, meas)
        render = pixel_metrics(Path(c["preview_path"]))
        out[c["candidate_id"]] = dict(features=feats, render=render, phi=phi(feats, render),
                                      target=f"{c['target_canvas']['width']}x{c['target_canvas']['height']}",
                                      family=c["template_family_id"], generator_version=c["generator_version"])
    if cache:
        cache.write_text(json.dumps(out), encoding="utf-8")
    return out


def split_families(comps: list[dict], seed_search=range(1, 200)) -> tuple[list[str], list[str], int]:
    """Deterministic family-disjoint split. The seed is chosen only for coverage (>=8 decisive held-out comparisons at each
    extreme target and ~25-33% of comparisons held out) - never from agreement results."""
    fams = sorted({c["template_family_id"] for c in comps})
    for seed in seed_search:
        order = sorted(fams, key=lambda f: hashlib.sha256(f"{seed}:{f}".encode()).hexdigest())
        held = order[:4]
        sel = [c for c in comps if c["template_family_id"] in held and c["label"] in ("A_BETTER", "B_BETTER")]
        share = sum(c["template_family_id"] in held for c in comps) / len(comps)
        ok = all(sum(f"{c['target_width']}x{c['target_height']}" == t for c in sel) >= 8 for t in ("1600x400", "728x90", "90x728"))
        if ok and .22 <= share <= .34:
            return sorted(f for f in fams if f not in held), sorted(held), seed
    raise RuntimeError("no family split satisfies the coverage constraints")


def pair_matrix(comps, feats, standardize=None):
    X, y, meta = [], [], []
    for c in comps:
        if c["label"] not in ("A_BETTER", "B_BETTER"):
            continue
        a, b = feats.get(c["candidate_a_id"]), feats.get(c["candidate_b_id"])
        if not a or not b:
            continue
        d = np.array(a["phi"]) - np.array(b["phi"])
        X.append(d)
        y.append(1.0 if c["label"] == "A_BETTER" else 0.0)
        meta.append(c)
    return np.array(X), np.array(y), meta


def fit_pairwise(comps, feats, l2=2.0):
    X, y, _ = pair_matrix(comps, feats)
    scale = X.std(axis=0) + 1e-6
    Xs = X / scale
    Xa, ya = np.vstack([Xs, -Xs]), np.concatenate([y, 1 - y])  # symmetric: swapping A/B flips the label
    return fit_logistic(Xa, ya, l2=l2), scale


def predict_p(w, scale, fa, fb):
    d = (np.array(fa["phi"]) - np.array(fb["phi"])) / scale
    return float(1 / (1 + math.exp(-float(np.clip(d @ w, -30, 30)))))


def absolute_quality(comps, feats, l2=2.0):
    """Quality logit trained on: 'bad' = candidates in BOTH_BAD comparisons; 'good' = winners of decisive comparisons
    that never appeared in a BOTH_BAD one. Used only to propose BOTH_BAD, with a threshold chosen on calibration data."""
    bad, good = set(), set()
    for c in comps:
        if c["label"] == "BOTH_BAD":
            bad.update((c["candidate_a_id"], c["candidate_b_id"]))
        elif c["label"] in ("A_BETTER", "B_BETTER"):
            good.add(c["preferred_candidate_id"])
    good -= bad
    ids = [i for i in sorted(bad | good) if i in feats]
    X = np.array([feats[i]["phi"] for i in ids])
    y = np.array([1.0 if i in good else 0.0 for i in ids])
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-6
    w = fit_logistic((X - mu) / sd, y, l2=l2, intercept=True)
    return w, mu, sd, len(bad), len(good)


def quality(w, mu, sd, f):
    x = (np.array(f["phi"]) - mu) / sd
    return float(x @ w[:-1] + w[-1])


def evaluate(comps, feats, model, quality_model, tau, delta):
    w, scale = model
    qw, qmu, qsd = quality_model
    rows = []
    for c in comps:
        a, b = feats.get(c["candidate_a_id"]), feats.get(c["candidate_b_id"])
        if not a or not b:
            continue
        p = predict_p(w, scale, a, b)
        qa, qb = quality(qw, qmu, qsd, a), quality(qw, qmu, qsd, b)
        if max(qa, qb) < tau:
            pred = "BOTH_BAD"
        elif abs(p - .5) < delta:
            pred = "TIE"
        else:
            pred = "A_BETTER" if p > .5 else "B_BETTER"
        rows.append(dict(c=c, p=p, pred=pred, target=f"{c['target_width']}x{c['target_height']}", margin=abs(p - .5)))
    return rows


def summarize(rows):
    dec = [r for r in rows if r["c"]["label"] in ("A_BETTER", "B_BETTER")]
    direction = [r for r in dec if r["pred"] in ("A_BETTER", "B_BETTER")]
    acc = lambda rs: round(sum(r["pred"] == r["c"]["label"] for r in rs) / len(rs), 3) if rs else None
    out = dict(comparisons=len(rows), decisive=len(dec), decisive_with_direction=len(direction),
               directional_agreement=acc(direction), directional_agreement_all_decisive=acc(dec),
               by_target={t: dict(n=sum(r["target"] == t for r in direction), agreement=acc([r for r in direction if r["target"] == t]))
                          for t in sorted({r["target"] for r in rows})})
    conf = Counter((r["c"]["label"], r["pred"]) for r in rows)
    out["confusion"] = {f"{k[0]}->{k[1]}": v for k, v in sorted(conf.items())}
    bb = [r for r in rows if r["c"]["label"] == "BOTH_BAD"]
    pb = [r for r in rows if r["pred"] == "BOTH_BAD"]
    tp = sum(r["c"]["label"] == "BOTH_BAD" for r in pb)
    out["both_bad"] = dict(human=len(bb), predicted=len(pb), true_positive=tp,
                           precision=round(tp / len(pb), 3) if pb else None, recall=round(tp / len(bb), 3) if bb else None)
    ties = [r for r in rows if r["c"]["label"] == "TIE"]
    out["tie"] = dict(human=len(ties), predicted=sum(r["pred"] == "TIE" for r in rows),
                      human_ties_predicted_tie=sum(r["pred"] == "TIE" for r in ties))
    bins = []
    for lo, hi in ((0, .1), (.1, .2), (.2, .3), (.3, .4), (.4, .51)):
        rs = [r for r in direction if lo <= r["margin"] < hi]
        bins.append(dict(margin_lo=lo, margin_hi=hi, n=len(rs), agreement=acc(rs)))
    out["agreement_by_margin_bin"] = bins
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exports", type=Path, default=EXPORTS)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "creative_pilot_v1")
    a = ap.parse_args(argv)
    out = a.out
    comps = [c for c in read_jsonl(a.exports / "comparisons_all.jsonl") if not c.get("voided")]
    feats = human_candidate_features(a.exports, out / "human_candidate_features.json")
    cal_f, held_f, seed = split_families(comps)
    cal = [c for c in comps if c["template_family_id"] in cal_f]
    held = [c for c in comps if c["template_family_id"] in held_f]
    model = fit_pairwise(cal, feats)
    qw, qmu, qsd, nbad, ngood = absolute_quality(cal, feats)
    # thresholds are chosen on CALIBRATION data only
    cal_scores = {i: quality(qw, qmu, qsd, f) for i, f in feats.items()}
    best_tau, best_f1 = -9.0, -1.0
    bb_cal = [c for c in cal if c["label"] == "BOTH_BAD"]
    for tau in np.linspace(-3, 3, 61):
        tp = sum(max(cal_scores[c["candidate_a_id"]], cal_scores[c["candidate_b_id"]]) < tau for c in bb_cal)
        pred = sum(max(cal_scores[c["candidate_a_id"]], cal_scores[c["candidate_b_id"]]) < tau for c in cal)
        prec, rec = (tp / pred if pred else 0), (tp / len(bb_cal) if bb_cal else 0)
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        if f1 > best_f1:
            best_tau, best_f1 = float(tau), f1
    delta = .06
    q = (qw, qmu, qsd)
    held_rows = evaluate(held, feats, model, q, best_tau, delta)
    cal_rows = evaluate(cal, feats, model, q, best_tau, delta)
    # leave-one-family-out CV (directional agreement only)
    lofo, pooled, lofo_margins = {}, [], []
    for f in sorted({c["template_family_id"] for c in comps}):
        tr = [c for c in comps if c["template_family_id"] != f]
        te = [c for c in comps if c["template_family_id"] == f and c["label"] in ("A_BETTER", "B_BETTER")]
        if len(te) < 3:
            continue
        m = fit_pairwise(tr, feats)
        hits = []
        for c in te:
            p = predict_p(m[0], m[1], feats[c["candidate_a_id"]], feats[c["candidate_b_id"]])
            hits.append((p > .5) == (c["label"] == "A_BETTER"))
        lofo[f] = dict(n=len(hits), agreement=round(sum(hits) / len(hits), 3))
        pooled += hits
        lofo_margins += [(abs(predict_p(m[0], m[1], feats[c["candidate_a_id"]], feats[c["candidate_b_id"]]) - .5), h)
                         for c, h in zip(te, hits)]
    lofo_bins = []
    for lo, hi in ((0, .1), (.1, .2), (.2, .3), (.3, .4), (.4, .51)):
        hs = [h for mg, h in lofo_margins if lo <= mg < hi]
        lofo_bins.append(dict(margin_lo=lo, margin_hi=hi, n=len(hs), agreement=round(sum(hs) / len(hs), 3) if hs else None))
    # transparent BOTH_BAD rule: both layouts have a tiny smallest font. Threshold chosen on CALIBRATION comparisons only.
    def bb_stats(rows, t):
        pred = [c for c in rows if max(feats[c["candidate_a_id"]]["features"]["min_font"], feats[c["candidate_b_id"]]["features"]["min_font"]) < t]
        tp = sum(c["label"] == "BOTH_BAD" for c in pred)
        n_bb = sum(c["label"] == "BOTH_BAD" for c in rows)
        prec, rec = (tp / len(pred) if pred else 0.0), (tp / n_bb if n_bb else 0.0)
        return dict(threshold_px=t, predicted=len(pred), true_positive=tp, human_both_bad=n_bb, precision=round(prec, 3),
                    recall=round(rec, 3), f1=round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0)
    rule_t = max((1, 2, 3, 4, 5, 6, 7, 8), key=lambda t: bb_stats(cal, t)["f1"])
    both_bad_rule = dict(rule="both candidates have smallest font below threshold", chosen_on="calibration families only",
                         calibration=bb_stats(cal, rule_t), heldout=bb_stats(held, rule_t))
    # simple baselines on the same held-out decisive comparisons
    def baseline(rows, idx, sign=1.0):
        """Pick the candidate with the larger (sign=+1) / smaller (sign=-1) value of one feature; ties count 0.5."""
        dec = [r for r in rows if r["c"]["label"] in ("A_BETTER", "B_BETTER")]
        ok = 0.0
        for r in dec:
            fa, fb = (sign * feats[r["c"][k]]["phi"][idx] for k in ("candidate_a_id", "candidate_b_id"))
            ok += ((fa > fb) == (r["c"]["label"] == "A_BETTER")) if fa != fb else .5
        return round(ok / len(dec), 3) if dec else None
    # deterministic S04 objective as a comparison (never used for fitting)
    objective = objective_agreement(comps, feats, held_f)
    counts = dict(human_comparisons=len(comps), calibration_families=cal_f, heldout_families=held_f, split_seed=seed,
                  calibration_comparisons=len(cal), heldout_comparisons=len(held),
                  calibration_decisive=sum(c["label"] in ("A_BETTER", "B_BETTER") for c in cal),
                  heldout_decisive=sum(c["label"] in ("A_BETTER", "B_BETTER") for c in held),
                  both_bad_quality_training=dict(bad_candidates=nbad, good_candidates=ngood))
    result = dict(
        version=VERSION, features=list(FEATURES), counts=counts,
        heldout=summarize(held_rows), calibration_in_sample=summarize(cal_rows),
        leave_one_family_out=dict(per_family=lofo, pooled_directional_agreement=round(sum(pooled) / len(pooled), 3), pooled_n=len(pooled)),
        baselines_heldout_directional=dict(larger_min_font=baseline(held_rows, 0), larger_content_area=baseline(held_rows, 14),
                                           less_empty_cells=baseline(held_rows, 5, -1.0), less_flat_render=baseline(held_rows, 13, -1.0)),
        s04_objective_vs_human=objective,
        thresholds=dict(both_bad_tau=best_tau, tie_delta=delta, both_bad_f1_calibration=round(best_f1, 3), both_bad_rule_px=rule_t),
        both_bad_rule=both_bad_rule, out_of_family_agreement_by_margin_bin=lofo_bins,
        model=dict(pairwise_weights=dict(zip(FEATURES, [round(float(x), 4) for x in model[0]])), pairwise_scale=[float(x) for x in model[1]],
                   quality_weights=dict(zip(list(FEATURES) + ["intercept"], [round(float(x), 4) for x in qw])),
                   quality_mean=[float(x) for x in qmu], quality_std=[float(x) for x in qsd]),
        notes=["Held-out directional agreement is measured on families never used to fit weights or choose thresholds.",
               "In-sample numbers are reported only to show fit; they are NOT generalisation estimates.",
               "Human data comes from S04 and a development generator, not from the new creative layouts: transfer is unverified.",
               "Tie and BOTH_BAD detection rest on very few human examples; treat them as weak."])
    (out / "labeler_calibration.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("counts", "heldout", "leave_one_family_out", "thresholds", "s04_objective_vs_human")}, indent=1)[:5000])
    return 0


def objective_agreement(comps, feats, held_families):
    """How often the production deterministic objective picks the human-preferred layout (descriptive, no fitting)."""
    try:
        from engine.internal_schema import Canvas, Design
        from engine.layout_objective import evaluate_objective
        cands = {c["candidate_id"]: c for c in read_jsonl(EXPORTS / "comparisons_all_candidates.jsonl")}
        cache, hits, rows = {}, defaultdict(list), 0
        for c in comps:
            if c["label"] not in ("A_BETTER", "B_BETTER"):
                continue
            scores = []
            for cid in (c["candidate_a_id"], c["candidate_b_id"]):
                if cid not in cache:
                    m = cands[cid]
                    src = Design.from_adobe(json.loads(Path(m["source_path"]).read_text(encoding="utf-8")))
                    cand = Design.from_adobe(json.loads(Path(m["candidate_path"]).read_text(encoding="utf-8")))
                    cache[cid] = 1.0 - evaluate_objective(src, cand, Canvas.from_adobe(m["target_canvas"])).total  # total is a cost; score = 1 - J
                scores.append(cache[cid])
            if scores[0] == scores[1]:
                continue
            ok = (scores[0] > scores[1]) == (c["label"] == "A_BETTER")
            hits["all"].append(ok)
            hits["heldout_families" if c["template_family_id"] in held_families else "calibration_families"].append(ok)
            hits[f"{c['target_width']}x{c['target_height']}"].append(ok)
        return {k: dict(n=len(v), agreement=round(sum(v) / len(v), 3)) for k, v in hits.items()}
    except Exception as exc:  # noqa: BLE001 - descriptive extra; report why it is missing instead of failing calibration
        return dict(unavailable=f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
