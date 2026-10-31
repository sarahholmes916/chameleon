"""Blind visual comparison of the layouts selected by three ranking methods from one shared candidate pool.

  python -m training.blind_eval build    # score pools, measure agreement, sample the gallery (private key kept apart)
  python -m training.blind_eval serve    # Gradio gallery: pick the best anonymized layout (or tie / all bad); results saved
  python -m training.blind_eval report   # reveal identities and tally what you chose

Methods: `objective` (deterministic S04 objective, lower cost is better), `gnn_original` (checkpoint trained on original human
labels only), `gnn_combined` (original + verified synthetic labels). Nothing here touches the production engine. Pools exclude the
deliberately degraded (mediocre/poor) candidates; only sources from held-out test families are used, so neither GNN trained on them.
The gallery conditions on disagreement, so its win rates describe disagreement cases only.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from training import data as D

OUT = D.PILOT_DIR / "blind_eval"
PRIVATE = OUT / "private"
CHECKPOINTS = D.PILOT_DIR / "training_v1" / "checkpoints"
METHODS = {"objective": None, "gnn_original": CHECKPOINTS / "ranker_old_only_best.pt", "gnn_combined": CHECKPOINTS / "ranker_old_plus_new_best.pt"}
EXTREME = ("1600x400", "728x90", "90x728")
POOL_CATEGORIES = ("creative", "conventional")  # provenance used only to leave out intentionally degraded candidates
DB_NAME = "results.db"
TIE, ALL_BAD = "TIE", "ALL_BAD"


@dataclass(frozen=True)
class Cand:
    candidate_id: str
    path: str
    preview: str
    generator: str
    category: str


@dataclass(frozen=True)
class TaskPool:
    task_id: str
    origin: str  # "new" (creative pilot) or "old" (original tournament)
    family: str
    source_id: str
    source_path: str
    source_preview: str
    target: tuple[int, int]
    candidates: tuple[Cand, ...]

    @property
    def target_name(self) -> str:
        return f"{self.target[0]}x{self.target[1]}"


# ----------------------------------------------------------------------------- pools
def load_pools() -> list[TaskPool]:
    """Held-out-family tasks only: sources none of the evaluated checkpoints were trained or early-stopped on."""
    pools: list[TaskPool] = []
    splits = json.loads(D.NEW_SPLITS.read_text(encoding="utf-8"))["source_split"]
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in D.read_jsonl(D.PILOT_DIR / "candidate_manifest_all.jsonl"):
        if splits.get(r["source_design_id"]) == "test" and r["provenance"].get("intended_category") in POOL_CATEGORIES:
            groups[r["set_id"]].append(r)
    for set_id, rows in sorted(groups.items()):
        r0 = rows[0]
        pools.append(TaskPool(
            task_id=set_id, origin="new", family=r0["template_family_id"], source_id=r0["source_design_id"], source_path=r0["source_path"],
            source_preview=r0["source_preview_path"], target=(r0["target_canvas"]["width"], r0["target_canvas"]["height"]),
            candidates=tuple(sorted((Cand(r["candidate_id"], r["candidate_path"], r["preview_path"], r["generator_id"],
                                          r["provenance"].get("intended_category", "")) for r in rows), key=lambda c: c.candidate_id))))
    old: dict[str, list[dict]] = defaultdict(list)
    for r in D.read_jsonl(D.OLD_CANDIDATES):
        if r["template_family_id"] in D.OLD_TEST_FAMILIES:
            old[r["set_id"]].append(r)
    for set_id, rows in sorted(old.items()):
        r0 = rows[0]
        pools.append(TaskPool(
            task_id="orig_" + set_id, origin="old", family=r0["template_family_id"], source_id=r0["source_design_id"], source_path=r0["source_path"],
            source_preview=r0["source_preview_path"], target=(r0["target_canvas"]["width"], r0["target_canvas"]["height"]),
            candidates=tuple(sorted((Cand(r["candidate_id"], r["candidate_path"], r["preview_path"], r.get("generator_id", ""), "tournament")
                                     for r in rows), key=lambda c: c.candidate_id))))
    return [p for p in pools if len(p.candidates) >= 2]


# ----------------------------------------------------------------------------- scoring
def _design(path: str):
    from engine.internal_schema import Design
    return Design.from_adobe(json.loads(D.abs_path(path).read_text(encoding="utf-8")))


def objective_scores(pool: TaskPool) -> list[float]:
    from engine.internal_schema import Canvas
    from engine.layout_objective import evaluate_objective
    source, target = _design(pool.source_path), Canvas.from_adobe({"width": pool.target[0], "height": pool.target[1]})
    return [-evaluate_objective(source, _design(c.path), target).total for c in pool.candidates]  # higher = better


def gnn_scores(pool: TaskPool, checkpoint) -> list[float]:
    from engine.internal_schema import Canvas
    from engine.ranker import score_candidates
    source, target = _design(pool.source_path), Canvas.from_adobe({"width": pool.target[0], "height": pool.target[1]})
    return score_candidates(source, [_design(c.path) for c in pool.candidates], target, checkpoint=checkpoint)


def argmax_first(scores: list[float]) -> int:
    best = max(scores)
    return next(i for i, s in enumerate(scores) if s == best)


def select_all(pools: list[TaskPool], methods: dict[str, Path | None]) -> dict[str, dict[str, str]]:
    """{task_id: {method: selected candidate_id}} using one shared pool per task."""
    from engine.ranker import load_ranker_checkpoint
    checkpoints = {m: (load_ranker_checkpoint(p) if p else None) for m, p in methods.items()}
    out: dict[str, dict[str, str]] = {}
    for pool in pools:
        picks = {}
        for method, ck in checkpoints.items():
            scores = objective_scores(pool) if ck is None else gnn_scores(pool, ck)
            picks[method] = pool.candidates[argmax_first(scores)].candidate_id
        out[pool.task_id] = picks
    return out


# ----------------------------------------------------------------------------- agreement + sampling
def agreement_report(pools: list[TaskPool], picks: dict[str, dict[str, str]], methods: list[str]) -> dict:
    def rate(rows):
        return dict(tasks=len(rows), all_methods_same=sum(len(set(picks[p.task_id].values())) == 1 for p in rows))
    report = dict(overall=rate(pools))
    report["distinct_selected_layouts"] = dict(sorted(Counter(len(set(picks[p.task_id].values())) for p in pools).items()))
    report["pairwise_same_pick"] = {f"{a}~{b}": sum(picks[p.task_id][a] == picks[p.task_id][b] for p in pools)
                                    for i, a in enumerate(methods) for b in methods[i + 1:]}
    report["by_target"] = {t: rate([p for p in pools if p.target_name == t]) for t in sorted({p.target_name for p in pools})}
    report["by_origin"] = {o: rate([p for p in pools if p.origin == o]) for o in sorted({p.origin for p in pools})}
    report["pool_size"] = dict(min=min(len(p.candidates) for p in pools), max=max(len(p.candidates) for p in pools),
                               mean=round(sum(len(p.candidates) for p in pools) / len(pools), 2))
    gen = {}
    for m in methods:
        c = Counter()
        for p in pools:
            sel = next(x for x in p.candidates if x.candidate_id == picks[p.task_id][m])
            c[f"{sel.generator}:{sel.category}"] += 1
        gen[m] = dict(c)
    report["selected_candidate_source"] = gen
    return report


def sample_tasks(pools: list[TaskPool], picks: dict[str, dict[str, str]], n: int = 30, seed: int = 7, other_cap: int = 5) -> list[TaskPool]:
    """Disagreement tasks only (nothing to judge otherwise), stratified by target then round-robin over families (seeded)."""
    disagree = [p for p in pools if len(set(picks[p.task_id].values())) >= 2]
    rng = random.Random(seed)
    by_target: dict[str, list[TaskPool]] = defaultdict(list)
    for p in disagree:
        by_target[p.target_name].append(p)

    def spread(items: list[TaskPool]) -> list[TaskPool]:
        by_family: dict[str, list[TaskPool]] = defaultdict(list)
        for p in sorted(items, key=lambda x: x.task_id):
            by_family[p.family].append(p)
        for v in by_family.values():
            rng.shuffle(v)
        order = sorted(by_family)
        rng.shuffle(order)
        out = []
        while any(by_family.values()):
            for f in order:
                if by_family[f]:
                    out.append(by_family[f].pop())
        return out

    other = spread([p for t, v in by_target.items() if t not in EXTREME for p in v])[:min(other_cap, n // 6)]
    quota = {t: 0 for t in EXTREME}
    room = n - len(other)
    pools_t = {t: spread(by_target.get(t, [])) for t in EXTREME}
    while room > 0 and any(len(pools_t[t]) > quota[t] for t in EXTREME):
        for t in EXTREME:
            if room > 0 and len(pools_t[t]) > quota[t]:
                quota[t] += 1
                room -= 1
    chosen = [p for t in EXTREME for p in pools_t[t][:quota[t]]] + other
    rng.shuffle(chosen)
    return chosen


def build_manifests(chosen: list[TaskPool], picks: dict[str, dict[str, str]], seed: int = 7) -> tuple[dict, dict]:
    """Public manifest (what the gallery may show) and private key (which methods chose each option)."""
    public, private = [], []
    for p in chosen:
        rng = random.Random(f"{seed}:{p.task_id}")
        by_id = {c.candidate_id: c for c in p.candidates}
        method_by_cand: dict[str, list[str]] = defaultdict(list)
        for method, cid in picks[p.task_id].items():
            method_by_cand[cid].append(method)
        cids = sorted(method_by_cand)
        rng.shuffle(cids)
        public.append(dict(task_id=p.task_id, target=list(p.target), source_preview=p.source_preview,
                           options=[dict(position=i + 1, candidate_id=c, preview=by_id[c].preview) for i, c in enumerate(cids)]))
        private.append(dict(task_id=p.task_id, origin=p.origin, family=p.family, target=p.target_name,
                            options=[dict(position=i + 1, candidate_id=c, methods=sorted(method_by_cand[c]),
                                          generator=by_id[c].generator, category=by_id[c].category) for i, c in enumerate(cids)]))
    return dict(version="blind_eval.v1", seed=seed, tasks=public), dict(version="blind_eval.v1", seed=seed, tasks=private)


# ----------------------------------------------------------------------------- results store (torch/gradio free)
def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE IF NOT EXISTS judgments (
        judgment_id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL, annotator TEXT NOT NULL, choice TEXT NOT NULL,
        chosen_candidate_id TEXT, displayed_positions TEXT NOT NULL, created_at_utc TEXT NOT NULL, voided_at_utc TEXT)""")
    conn.commit()
    return conn


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def judged_ids(conn, annotator: str) -> set[str]:
    return {r["task_id"] for r in conn.execute("SELECT task_id FROM judgments WHERE annotator=? AND voided_at_utc IS NULL", (annotator,))}


def next_task(public: dict, conn, annotator: str) -> dict | None:
    done = judged_ids(conn, annotator)
    return next((t for t in public["tasks"] if t["task_id"] not in done), None)


def record(conn, public: dict, task_id: str, annotator: str, choice: str) -> None:
    task = next(t for t in public["tasks"] if t["task_id"] == task_id)
    cid = None
    if choice not in (TIE, ALL_BAD):
        cid = next((o["candidate_id"] for o in task["options"] if o["position"] == int(choice)), None)
        if cid is None:
            raise ValueError(f"task {task_id} has no option {choice}")
    if task_id in judged_ids(conn, annotator):
        raise ValueError("already judged; undo first")
    conn.execute("INSERT INTO judgments (task_id, annotator, choice, chosen_candidate_id, displayed_positions, created_at_utc) VALUES (?,?,?,?,?,?)",
                 (task_id, annotator, choice if cid is None else "OPTION", cid, json.dumps([o["position"] for o in task["options"]]), _now()))
    conn.commit()


def undo_last(conn, annotator: str) -> str | None:
    row = conn.execute("SELECT judgment_id, task_id FROM judgments WHERE annotator=? AND voided_at_utc IS NULL ORDER BY judgment_id DESC LIMIT 1", (annotator,)).fetchone()
    if row:
        conn.execute("UPDATE judgments SET voided_at_utc=? WHERE judgment_id=?", (_now(), row["judgment_id"]))
        conn.commit()
    return row["task_id"] if row else None


# ----------------------------------------------------------------------------- tally
def sign_test_p(wins: int, losses: int) -> float | None:
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def tally(private: dict, conn, annotator: str | None = None, origin: str | None = None) -> dict:
    methods = sorted({m for t in private["tasks"] for o in t["options"] for m in o["methods"]})
    tasks = {t["task_id"]: t for t in private["tasks"]}
    rows = conn.execute("SELECT * FROM judgments WHERE voided_at_utc IS NULL ORDER BY judgment_id").fetchall()
    latest: dict[tuple, sqlite3.Row] = {}
    for r in rows:
        if annotator is None or r["annotator"] == annotator:
            latest[(r["task_id"], r["annotator"])] = r
    res = {m: Counter() for m in methods}
    h2h = {(a, b): Counter() for i, a in enumerate(methods) for b in methods[i + 1:]}
    outcomes = Counter()
    detail = []
    judged = 0
    for (task_id, who), r in sorted(latest.items()):
        t = tasks[task_id]
        if origin is not None and t["origin"] != origin:
            continue
        judged += 1
        pick = {m: o["candidate_id"] for o in t["options"] for m in o["methods"]}
        if r["choice"] in (TIE, ALL_BAD):
            outcomes[r["choice"]] += 1
            winners: set[str] = set()
        else:
            outcomes["OPTION"] += 1
            opt = next(o for o in t["options"] if o["candidate_id"] == r["chosen_candidate_id"])
            winners = set(opt["methods"])
        for m in methods:
            key = "best_shared" if (m in winners and len(winners) > 1) else "best_unique" if m in winners else r["choice"].lower() if r["choice"] in (TIE, ALL_BAD) else "not_chosen"
            res[m][key] += 1
            res[m]["best_any"] += int(m in winners)
        for (a, b), c in h2h.items():
            if pick[a] == pick[b]:
                c["same_pick"] += 1
            elif a in winners:
                c["a_wins"] += 1
            elif b in winners:
                c["b_wins"] += 1
            elif r["choice"] in (TIE, ALL_BAD):
                c["tie_or_all_bad"] += 1
            else:
                c["third_option_won"] += 1
        detail.append(dict(task_id=task_id, annotator=who, target=t["target"], origin=t["origin"], choice=r["choice"],
                           chosen_methods=sorted(winners), options={str(o["position"]): o["methods"] for o in t["options"]}))
    return dict(judged_task_judgments=judged, outcomes=dict(outcomes),
                per_method={m: dict(c, share_best_any=round(c["best_any"] / judged, 3) if judged else None) for m, c in res.items()},
                head_to_head={f"{a} vs {b}": dict(c, sign_test_p=sign_test_p(c["a_wins"], c["b_wins"])) for (a, b), c in h2h.items()},
                detail=detail)


# ----------------------------------------------------------------------------- build / serve / report
def cmd_build(a) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    pools = load_pools()
    print(f"{len(pools)} held-out tasks ({Counter(p.origin for p in pools)}), targets {dict(Counter(p.target_name for p in pools))}", flush=True)
    picks = select_all(pools, METHODS)
    methods = list(METHODS)
    agree = agreement_report(pools, picks, methods)
    chosen = sample_tasks(pools, picks, n=a.n, seed=a.seed)
    public, private = build_manifests(chosen, picks, a.seed)
    (OUT / "gallery_manifest.json").write_text(json.dumps(public, indent=1), encoding="utf-8")
    (PRIVATE / "answer_key.json").write_text(json.dumps(private, indent=1), encoding="utf-8")
    (PRIVATE / "selections_all_eligible.json").write_text(json.dumps(
        {p.task_id: dict(origin=p.origin, family=p.family, target=p.target_name, pool=len(p.candidates), picks=picks[p.task_id]) for p in pools},
        indent=1), encoding="utf-8")
    agree["gallery"] = dict(tasks=len(chosen), by_target=dict(Counter(p.target_name for p in chosen)), by_origin=dict(Counter(p.origin for p in chosen)),
                            by_family=dict(Counter(p.family for p in chosen)),
                            options_per_task=dict(Counter(len(t["options"]) for t in private["tasks"])), seed=a.seed,
                            eligible_disagreement_tasks=sum(len(set(picks[p.task_id].values())) >= 2 for p in pools))
    (PRIVATE / "agreement_report.json").write_text(json.dumps(agree, indent=1), encoding="utf-8")
    print(json.dumps(agree, indent=1))
    return 0


def cmd_report(a) -> int:
    private = json.loads((PRIVATE / "answer_key.json").read_text(encoding="utf-8"))
    conn = connect(OUT / DB_NAME)
    result = tally(private, conn, a.annotator)
    result["by_origin"] = {o: {k: v for k, v in tally(private, conn, a.annotator, o).items() if k != "detail"} for o in ("old", "new")}
    (OUT / "blind_eval_results.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "detail"}, indent=1))
    return 0


def cmd_serve(a) -> int:
    from training.blind_eval_ui import launch
    return launch(OUT, a.port, a.annotator)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--n", type=int, default=30)
    b.add_argument("--seed", type=int, default=7)
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=7880)
    s.add_argument("--annotator", default="sarah")
    r = sub.add_parser("report")
    r.add_argument("--annotator", default=None)
    a = ap.parse_args()
    return dict(build=cmd_build, serve=cmd_serve, report=cmd_report)[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
