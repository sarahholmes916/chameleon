"""Select informative candidate pairs and PROPOSE preference labels (never human labels).

Evaluation methods, in priority order (each proposal records which one produced it):
  ai_visual_assessment            tiers given by an image-capable model reading blind contact sheets of the rendered PNGs
                                  (only for the fixed-stride sample of tasks; see visual_review.py / visual_tiers.py)
  controlled_degradation_checked  the pair is (parent, degraded child), the named weakness is known by construction, AND the
                                  rendered-metrics model independently prefers the parent by a clear margin
  rendered_metrics_model          pairwise logistic model on measured features of the JSON and rendered PNG, calibrated on
                                  existing human comparisons by family; asserted only when the margin is large
  rule_both_unreadable            weak rule: both layouts have a tiny smallest font (low confidence, flagged)
  none -> SKIP                    ambiguous; needs a human
Generator family, intended category and the deterministic S04 objective are NOT used to decide labels. Intended
category is used only to make sure the sampled pairs cover the requested comparison types.
Every proposal has review_status `synthetic_proposed`; none is ever `human_verified` here.
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

from creative_pilot import calibrate
from creative_pilot.visual_tiers import ASSESSOR, TIERS
from scripts import source_dataset as sd

METHOD_VERSION = "creative_pilot_labels_v1"
PAIR_TYPES = (("creative", "conventional", 2), ("creative", "creative", 2), ("creative", "degraded", 1),
              ("degraded", "degraded", 1), ("conventional", "degraded", 1), ("parent_child", None, 1))
MARGIN_MIN = .15
KNOWN_WORSENING = set()  # filled below


def _h(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest(), 16)


def utility(model: dict, row: dict) -> float:
    phi = np.array(calibrate.phi(row["provenance"]["features"], row["provenance"]["render_metrics"]))
    w = np.array([model["pairwise_weights"][f] for f in calibrate.FEATURES])
    return float(phi @ (w / np.array(model["pairwise_scale"])))


def bin_agreement(model_json: dict, margin: float) -> tuple[float | None, int]:
    for b in model_json["out_of_family_agreement_by_margin_bin"]:
        if b["margin_lo"] <= margin < b["margin_hi"]:
            return b["agreement"], b["n"]
    return None, 0


def category(row: dict) -> str:
    c = row["provenance"]["intended_category"]
    return "degraded" if c in ("mediocre", "poor") else c


def select_pairs(rows: list[dict], utils: dict, n_pairs: int, rng: random.Random) -> list[tuple]:
    """Up to n_pairs distinct (row_a, row_b, pair_type) with the requested type mix; informative (closer) pairs favoured."""
    ids = {r["candidate_id"]: r for r in rows}
    chosen, seen = [], set()

    def eligible(kind_a, kind_b):
        out = []
        if kind_a == "parent_child":
            for r in rows:
                pid = r["provenance"].get("parent_candidate_id")
                if pid in ids and category(r) == "degraded":
                    out.append((ids[pid], r))
            return out
        for i, a in enumerate(rows):
            for b in rows[i + 1:]:
                ca, cb = category(a), category(b)
                if {ca, cb} == {kind_a, kind_b} or (kind_a == kind_b and ca == cb == kind_a):
                    out.append((a, b))
        return out

    for ka, kb, count in PAIR_TYPES:
        for _ in range(count):
            pool = [p for p in eligible(ka, kb) if frozenset((p[0]["candidate_id"], p[1]["candidate_id"])) not in seen]
            if not pool:
                continue
            weights = [math.exp(-abs(utils[a["candidate_id"]] - utils[b["candidate_id"]]) / .6) + .15 for a, b in pool]
            a, b = rng.choices(pool, weights=weights, k=1)[0]
            seen.add(frozenset((a["candidate_id"], b["candidate_id"])))
            chosen.append((a, b, ka if kb is None else f"{ka}_vs_{kb}"))
    # top up with any remaining pairs if some type had no eligible candidates
    if len(chosen) < n_pairs:
        rest = [(a, b) for i, a in enumerate(rows) for b in rows[i + 1:] if frozenset((a["candidate_id"], b["candidate_id"])) not in seen]
        rng.shuffle(rest)
        for a, b in rest[:n_pairs - len(chosen)]:
            seen.add(frozenset((a["candidate_id"], b["candidate_id"])))
            chosen.append((a, b, "other"))
    return chosen


def describe(ra: dict, rb: dict, label: str, model: dict) -> dict:
    """Short, measured explanation facets (typography, hierarchy, balance/spacing, composition, space use, trade-offs)."""
    fa, fb = ra["provenance"]["features"], rb["provenance"]["features"]
    pa, pb = ra["provenance"]["render_metrics"], rb["provenance"]["render_metrics"]
    f = lambda v, d=1: "n/a" if v is None else (f"{v:.{d}f}")
    better, worse = ("A", "B") if label == "A_BETTER" else ("B", "A") if label == "B_BETTER" else (None, None)
    typo = f"smallest text A {f(fa['min_font'])}px vs B {f(fb['min_font'])}px; median A {f(fa['median_font'])}px vs B {f(fb['median_font'])}px"
    hier = f"headline is the largest text: A {'yes' if fa.get('headline_rank') == 0 else 'no'}, B {'yes' if fb.get('headline_rank') == 0 else 'no'}; size spread A {f(fa.get('font_ratio_max_min'))}x vs B {f(fb.get('font_ratio_max_min'))}x"
    bal = f"balance offset A {f(fa['balance_offset'], 2)} vs B {f(fb['balance_offset'], 2)}; longest empty stretch A {f(fa['longest_empty_stretch'], 2)} vs B {f(fb['longest_empty_stretch'], 2)} of the long axis"
    comp = f"overlap A {f(fa['content_overlap_fraction'], 2)} vs B {f(fb['content_overlap_fraction'], 2)}; hidden-content pairs A {fa['occluded_pairs']} vs B {fb['occluded_pairs']}; reading-order agreement A {f(fa['order_agreement'], 2)} vs B {f(fb['order_agreement'], 2)}"
    space = f"empty grid cells A {f(fa['empty_cell_fraction'], 2)} vs B {f(fb['empty_cell_fraction'], 2)}; flat rendered area A {f(pa['flat_cell_fraction'], 2)} vs B {f(pb['flat_cell_fraction'], 2)}"
    ta = ", ".join(f"{k}" for k in ("larger smallest text" if (fa["min_font"] or 0) > (fb["min_font"] or 0) else "smaller smallest text",))
    trade = ("no major trade-off measured" if not better else
             f"{better} wins on the weighted measures; {worse} is {'larger in' if ((fa if worse == 'A' else fb)['content_area_fraction'] > (fa if better == 'A' else fb)['content_area_fraction']) else 'smaller in'} content area")
    return dict(typography=typo, hierarchy=hier, balance_spacing=bal, composition=comp, space_use=space, tradeoffs=trade)


def propose(rows_by_task: dict, model_json: dict, key: dict, split_of: dict, seed: int = 42, n_pairs: int = 8) -> list[dict]:
    model = model_json["model"]
    num_to_id = {t: {int(k): v for k, v in d["numbering"].items()} for t, d in key.items()}
    tier_of = {}
    for t, tiers in TIERS.items():
        for n, tier in tiers.items():
            tier_of[num_to_id[t][n]] = tier
    out = []
    for task_id, rows in sorted(rows_by_task.items()):
        utils = {r["candidate_id"]: utility(model, r) for r in rows}
        rng = random.Random(f"{seed}:{task_id}")
        for ra, rb, ptype in select_pairs(rows, utils, n_pairs, rng):
            if _h(task_id, ra["candidate_id"], rb["candidate_id"], seed) % 2:
                ra, rb = rb, ra
            a, b = ra["candidate_id"], rb["candidate_id"]
            ua, ub = utils[a], utils[b]
            p = 1 / (1 + math.exp(-(ua - ub)))
            margin = abs(p - .5)
            label, conf, basis, method, reason = "SKIP", 0.0, "none", "none", "ambiguous: measured differences are small or mixed; needs a human"
            expl_label = "A_BETTER" if p > .5 else "B_BETTER"
            ta, tb = tier_of.get(a), tier_of.get(b)
            fa, fb = ra["provenance"]["features"], rb["provenance"]["features"]
            parent_child = rb["provenance"].get("parent_candidate_id") == a or ra["provenance"].get("parent_candidate_id") == b
            if ta is not None and tb is not None:
                method = ASSESSOR
                if min(ta, tb) >= 3:
                    label, conf, reason = "BOTH_BAD", .5, f"visual assessment: A tier {ta}, B tier {tb} (neither acceptable)"
                elif ta == tb:
                    label, conf, reason = "TIE", .5, f"visual assessment: both tier {ta}"
                else:
                    label = "A_BETTER" if ta < tb else "B_BETTER"
                    gap = abs(ta - tb)
                    conf = {1: .6, 2: .75, 3: .85}[gap]
                    reason = f"visual assessment: A tier {ta}, B tier {tb} (1 strong .. 4 poor)"
                basis = "uncalibrated_visual_tier_gap"
                expl_label = label
            elif parent_child and margin >= .1 and ((p > .5) == (ra["candidate_id"] == (rb["provenance"].get("parent_candidate_id") or ""))
                                                    or (p < .5) == (rb["candidate_id"] == (ra["provenance"].get("parent_candidate_id") or ""))):
                parent_is_a = rb["provenance"].get("parent_candidate_id") == a
                label = "A_BETTER" if parent_is_a else "B_BETTER"
                child = rb if parent_is_a else ra
                ops = child["provenance"].get("degradation_ops", [])
                method = "controlled_degradation_checked"
                conf, basis = (bin_agreement(model_json, margin)[0] or .6), "empirical_out_of_family_bin_agreement"
                reason = f"B/A is a degraded copy of the other ({', '.join(child['provenance'].get('weakness', ops))}); the rendered-metrics model agrees"
                expl_label = label
            elif max(fa["min_font"] or 0, fb["min_font"] or 0) < model_json["thresholds"]["both_bad_rule_px"]:
                label, conf, basis, method = "BOTH_BAD", .25, "weak_rule_heldout_precision_0.17_to_0.38", "rule_both_unreadable"
                reason = f"both layouts have smallest text under {model_json['thresholds']['both_bad_rule_px']}px; weak rule, human check needed"
                expl_label = label
            elif margin >= MARGIN_MIN:
                label = "A_BETTER" if p > .5 else "B_BETTER"
                agree, n = bin_agreement(model_json, margin)
                conf, basis, method = agree, f"empirical_out_of_family_bin_agreement(n={n})", "rendered_metrics_model"
                reason = f"rendered-metrics model prefers {'A' if p > .5 else 'B'} (margin {margin:.2f})"
            wcid = _h("cid", task_id, *sorted((a, b)))
            comparison_id = format(wcid, "x")[:32]
            tw, th = ra["target_canvas"]["width"], ra["target_canvas"]["height"]
            out.append(dict(
                comparison_id=comparison_id, proposal_id="prop_" + comparison_id[:16], schema_version="ai_proposal.v1",
                task_id=task_id, candidate_a_id=a, candidate_b_id=b, source_design_id=ra["source_design_id"],
                template_family_id=ra["template_family_id"], split_group_id=ra["template_family_id"],
                split=split_of[ra["source_design_id"]], target_width=tw, target_height=th, pair_type=ptype,
                proposed_label=label, confidence=None if conf is None else round(float(conf), 3), confidence_kind=basis,
                confidence_is_calibrated=False,  # never claimed: bins come from a small human sample and a different generator distribution
                explanation=dict(summary=reason, **describe(ra, rb, expl_label if expl_label in ("A_BETTER", "B_BETTER") else "", model_json)),
                evaluation_method=method, method_version=METHOD_VERSION, visually_assessed=method == ASSESSOR,
                review_status="synthetic_proposed", metric_margin=round(margin, 4), metric_p_a_better=round(p, 4),
                candidate_a_generator_version=ra["generator_version"], candidate_b_generator_version=rb["generator_version"],
                candidate_a_path=ra["candidate_path"], candidate_b_path=rb["candidate_path"],
                candidate_a_preview=ra["preview_path"], candidate_b_preview=rb["preview_path"],
                source_path=ra["source_path"], source_preview_path=ra.get("source_preview_path"),
                screen_left_candidate_id=a if _h("side", comparison_id) % 2 == 0 else b,
                blind_eval_subset=_h("blind", comparison_id) % 100 < 15))
    return out


def load_inputs(pilot_root: Path):
    rows = [json.loads(l) for l in (pilot_root / "candidate_manifest_all.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["set_id"]].append(r)
    model = json.loads((pilot_root / "labeler_calibration.json").read_text(encoding="utf-8"))
    key = json.loads((pilot_root / "visual_review" / "key.json").read_text(encoding="utf-8"))
    split_of = json.loads((pilot_root / "splits.json").read_text(encoding="utf-8"))["source_split"]
    return by_task, model, key, split_of


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-root", type=Path, required=True)
    ap.add_argument("--pairs-per-task", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    by_task, model, key, split_of = load_inputs(a.pilot_root)
    proposals = propose(by_task, model, key, split_of, a.seed, a.pairs_per_task)
    out = a.pilot_root / "labels"
    out.mkdir(parents=True, exist_ok=True)
    sd.write_jsonl(out / "proposals.jsonl", proposals)
    summary = dict(
        proposals=len(proposals), tasks=len(by_task), by_label=dict(Counter(p["proposed_label"] for p in proposals)),
        by_method=dict(Counter(p["evaluation_method"] for p in proposals)),
        visually_assessed=sum(p["visually_assessed"] for p in proposals), by_split=dict(Counter(p["split"] for p in proposals)),
        by_target=dict(Counter(f"{p['target_width']}x{p['target_height']}" for p in proposals)),
        by_pair_type=dict(Counter(p["pair_type"] for p in proposals)),
        blind_eval_subset=sum(p["blind_eval_subset"] for p in proposals),
        confidence_summary={k: dict(n=len(v), mean=round(sum(v) / len(v), 3)) for k, v in
                            {m: [p["confidence"] for p in proposals if p["evaluation_method"] == m and p["confidence"] is not None]
                             for m in {p["evaluation_method"] for p in proposals}}.items() if v},
        method_version=METHOD_VERSION,
        caveats=["All proposals are synthetic_proposed. None is a human judgment.",
                 "Confidence is an uncalibrated estimate: metric-model values reuse binned out-of-family agreement from a small human sample; visual/rule values are not calibrated.",
                 "Labels never use generator family, intended category or the deterministic objective as the deciding signal."])
    (out / "proposals_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
