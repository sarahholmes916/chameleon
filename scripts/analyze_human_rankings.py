"""Phase A: analyze existing human ranking exports (read-only on the exports)."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPORTS = ROOT / "data" / "rankings" / "sarah" / "exports"
OUT = ROOT / "data" / "creative_pilot_v1"


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def family_of(cand):
    prov = cand.get("provenance", {})
    orig = prov.get("original_provenance", prov)
    return orig.get("family") or cand.get("generator_id") or "unknown", orig.get("variant")


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - m) / d, 3), round((c + m) / d, 3)]


def median(x):
    return sorted(x)[len(x) // 2] if x else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exports", default=str(EXPORTS))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    exp, out = Path(a.exports), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    comps = read_jsonl(exp / "comparisons_all.jsonl")
    cands = {c["candidate_id"]: c for c in read_jsonl(exp / "comparisons_all_candidates.jsonl")}

    def target(c):
        return f"{c['target_width']}x{c['target_height']}"

    stats = {}
    stats["counts"] = dict(
        comparisons=len(comps), candidates=len(cands),
        tournaments=len({c["tournament_id"] for c in comps}),
        sources=len({c["source_design_id"] for c in comps}),
        template_families=len({c["template_family_id"] for c in comps}),
        labels=dict(Counter(c["label"] for c in comps)),
        voided=sum(bool(c.get("voided")) for c in comps),
        usable_for_pairwise_loss=sum(bool(c.get("usable_for_pairwise_loss")) for c in comps),
        match_kinds=dict(Counter(c["match_kind"] for c in comps)),
        with_reason_tags=sum(bool(c["reason_tags"]) for c in comps),
        with_notes=sum(bool(c["note"]) for c in comps),
        repeats=sum(bool(c.get("repeat_of")) for c in comps),
    )
    res = Counter()
    missing = []
    for cid, c in cands.items():
        for k in ("candidate_path", "preview_path", "source_path", "source_preview_path"):
            ok = bool(c.get(k)) and Path(c[k]).exists()
            res[(k, ok)] += 1
            if not ok:
                missing.append((cid, k, c.get(k)))
    stats["reference_resolution"] = {f"{k}:{'ok' if ok else 'missing'}": n for (k, ok), n in sorted(res.items())}
    stats["missing_examples"] = missing[:10]
    stats["comparisons_with_unknown_candidate"] = sum(
        c["candidate_a_id"] not in cands or c["candidate_b_id"] not in cands for c in comps)
    stats["generator_ids"] = dict(Counter(c.get("generator_id") for c in cands.values()))
    stats["generator_versions"] = dict(Counter(c.get("generator_version") for c in cands.values()))
    stats["reason_tag_counts"] = dict(Counter(t for c in comps for t in c["reason_tags"]))

    fam, var, gen, tgt, srcfam, same_fam, tgt_same = (defaultdict(Counter) for _ in range(7))
    tourn_fam = defaultdict(lambda: defaultdict(Counter))
    for c in comps:
        if c.get("voided"):
            continue
        tgt[target(c)][c["label"]] += 1
        srcfam[c["template_family_id"]][c["label"]] += 1
        ca, cb = cands.get(c["candidate_a_id"]), cands.get(c["candidate_b_id"])
        if not ca or not cb:
            continue
        same_family = family_of(ca)[0] == family_of(cb)[0]
        if same_family:
            same_fam[family_of(ca)[0]][c["label"]] += 1
            tgt_same[target(c)][c["label"]] += 1
        for cand in (ca, cb):
            if same_family:
                break  # a family beating itself carries no family-level signal
            f, v = family_of(cand)
            g = cand.get("generator_version")
            if c["label"] in ("A_BETTER", "B_BETTER"):
                key = "win" if c["preferred_candidate_id"] == cand["candidate_id"] else "loss"
            elif c["label"] == "BOTH_BAD":
                key = "both_bad"
            else:
                key = "tie"
            fam[f][key] += 1
            var[f"{f}/{v}"][key] += 1
            gen[g][key] += 1
            tourn_fam[c["tournament_id"]][f][key] += 1

    def rate(d):
        o = {}
        for k, v in sorted(d.items(), key=lambda kv: str(kv[0])):
            n = v["win"] + v["loss"]
            o[str(k)] = dict(v, decisive=n, win_rate=round(v["win"] / n, 3) if n else None,
                             wilson95_comparison_level=wilson(v["win"], n))
        return o

    stats["same_family_comparisons"] = {k: dict(v) for k, v in same_fam.items()}
    stats["by_generator_family_cross_family_only"] = rate(fam)
    stats["by_generator_family"] = stats["by_generator_family_cross_family_only"]
    stats["by_family_variant"] = rate(var)
    stats["by_generator_version"] = rate(gen)
    stats["by_target"] = {k: dict(v, n=sum(v.values())) for k, v in sorted(tgt.items())}
    stats["by_template_family"] = {k: dict(v, n=sum(v.values())) for k, v in sorted(srcfam.items())}

    tl = defaultdict(Counter)
    for t, fams in tourn_fam.items():
        for f, cn in fams.items():
            net = cn["win"] - cn["loss"]
            tl[f]["pos" if net > 0 else "neg" if net < 0 else "zero"] += 1
    stats["tournament_level_family_net"] = {k: dict(v) for k, v in sorted(tl.items())}

    tour_tgt = defaultdict(set)
    for c in comps:
        tour_tgt[c["tournament_id"]].add((c["source_design_id"], target(c)))
    stats["source_target_tasks"] = len({x for s in tour_tgt.values() for x in s})
    stats["tournaments_per_target"] = dict(Counter(t for s in tour_tgt.values() for _, t in s))
    stats["BOTH_BAD_rate_by_target"] = {k: round(v["BOTH_BAD"] / sum(v.values()), 3) for k, v in tgt.items()}
    stats["BOTH_BAD_rate_by_template_family"] = {
        k: dict(n=sum(v.values()), rate=round(v["BOTH_BAD"] / sum(v.values()), 3)) for k, v in srcfam.items()}

    geo = {}
    for cid, c in cands.items():
        p = c.get("candidate_path")
        if not p or not Path(p).exists():
            continue
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        cv = d["canvas"]
        area = cv["width"] * cv["height"]
        els = d["elements"]
        sizes = [e["font_size"] for e in els if e.get("type") == "text" and "font_size" in e]
        covered = sum(max(0, e["width"]) * max(0, e["height"]) for e in els if e.get("type") != "shape")
        geo[cid] = dict(min_font=min(sizes) if sizes else None, med_font=median(sizes), n=len(els),
                        content_area_frac=round(min(covered / area, 3), 3))
    stats["geometry_resolved"] = len(geo)

    diffs = []
    for c in comps:
        if c["label"] in ("A_BETTER", "B_BETTER") and c["preferred_candidate_id"]:
            w = c["preferred_candidate_id"]
            loser = c["candidate_b_id"] if w == c["candidate_a_id"] else c["candidate_a_id"]
            gw, gl = geo.get(w), geo.get(loser)
            if gw and gl and gw["min_font"] and gl["min_font"]:
                diffs.append((gw["min_font"] - gl["min_font"], gw["content_area_frac"] - gl["content_area_frac"]))
    if diffs:
        stats["winner_vs_loser"] = dict(
            n=len(diffs),
            winner_larger_min_font=sum(d[0] > 0 for d in diffs),
            winner_smaller_min_font=sum(d[0] < 0 for d in diffs),
            equal_min_font=sum(d[0] == 0 for d in diffs),
            winner_larger_content_area=sum(d[1] > 0 for d in diffs),
            winner_smaller_content_area=sum(d[1] < 0 for d in diffs),
            note="descriptive, comparison-level; comparisons within a tournament are not independent")
    bb = [geo[x]["min_font"] for c in comps if c["label"] == "BOTH_BAD"
          for x in (c["candidate_a_id"], c["candidate_b_id"]) if x in geo and geo[x]["min_font"]]
    allf = [g["min_font"] for g in geo.values() if g["min_font"]]
    stats["min_font_median"] = dict(all_candidates=median(allf), in_both_bad_comparisons=median(bb))

    # Reason tags are attached to a comparison, not to a candidate: report them by label and target only.
    tag_label, tag_target = defaultdict(Counter), defaultdict(Counter)
    for c in comps:
        for t in c["reason_tags"]:
            tag_label[c["label"]][t] += 1
            tag_target[target(c)][t] += 1
    stats["reason_tags_by_label"] = {k: dict(v) for k, v in tag_label.items()}
    stats["reason_tags_by_target"] = {k: dict(v) for k, v in tag_target.items()}
    stats["comparisons_without_tags_by_label"] = dict(Counter(c["label"] for c in comps if not c["reason_tags"]))

    # Cluster (tournament) bootstrap for S04-vs-S04 cross-family win rates: resample whole tournaments.
    import random
    rng = random.Random(0)
    per_t = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # tournament -> family -> [win, decisive]
    for c in comps:
        ca, cb = cands.get(c["candidate_a_id"]), cands.get(c["candidate_b_id"])
        if not ca or not cb or c["label"] not in ("A_BETTER", "B_BETTER"):
            continue
        fa, fb = family_of(ca)[0], family_of(cb)[0]
        if fa == fb:
            continue
        for f, cid in ((fa, ca["candidate_id"]), (fb, cb["candidate_id"])):
            per_t[c["tournament_id"]][f][1] += 1
            per_t[c["tournament_id"]][f][0] += c["preferred_candidate_id"] == cid
    tids = sorted(per_t)
    families = sorted({f for t in per_t.values() for f in t})
    boot = {}
    for f in families:
        point = [sum(per_t[t][f][0] for t in tids if f in per_t[t]), sum(per_t[t][f][1] for t in tids if f in per_t[t])]
        samples = []
        for _ in range(2000):
            draw = [rng.choice(tids) for _ in tids]
            w = sum(per_t[t][f][0] for t in draw if f in per_t[t])
            n = sum(per_t[t][f][1] for t in draw if f in per_t[t])
            if n:
                samples.append(w / n)
        samples.sort()
        boot[f] = dict(wins=point[0], decisive=point[1], tournaments_with_family=sum(f in per_t[t] for t in tids),
                       win_rate=round(point[0] / point[1], 3) if point[1] else None,
                       cluster_bootstrap95=[round(samples[int(.025 * len(samples))], 3),
                                            round(samples[int(.975 * len(samples))], 3)] if len(samples) > 50 else None)
    stats["family_cluster_bootstrap_cross_family"] = boot
    sizes = Counter(c["tournament_id"] for c in comps)
    stats["comparisons_per_tournament"] = dict(min=min(sizes.values()), max=max(sizes.values()),
                                               mean=round(sum(sizes.values()) / len(sizes), 2))
    stats["tournaments_per_source"] = dict(Counter(
        len({c["tournament_id"] for c in comps if c["source_design_id"] == s})
        for s in {c["source_design_id"] for c in comps}))
    stats["sources_by_generator_cohort"] = {
        k: len({c["source_design_id"] for c in comps
                if (cands[c["candidate_a_id"]]["generator_version"].startswith("dev") == (k == "dev_controlled"))})
        for k in ("dev_controlled", "s04")}

    (out / "human_ranking_stats.json").write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    for k in ("counts", "reference_resolution", "by_target", "BOTH_BAD_rate_by_target", "generator_versions",
              "by_generator_family", "by_family_variant", "tournament_level_family_net", "winner_vs_loser",
              "reason_tag_counts", "min_font_median", "source_target_tasks", "tournaments_per_target",
              "BOTH_BAD_rate_by_template_family"):
        print(k, json.dumps(stats.get(k)))


if __name__ == "__main__":
    main()
