"""Combine human preference labels, audit them, and turn them into graph pairs for the GATv2 ranker.

Inputs are read-only. Only ``A_BETTER`` / ``B_BETTER`` rows feed the pairwise loss (Appendix E.3); ``TIE`` and
``BOTH_BAD`` rows are kept in the combined file but flagged unusable for that loss. Splits are by whole template
family, so every source design, sibling candidate and degraded copy stays on one side.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from engine.candidate_validator import validate_candidate
from engine.graph_features import FEATURE_VERSION_V3, build_candidate_graph
from engine.internal_schema import Canvas, Design
from scripts.source_dataset import digest

REPO_ROOT = Path(__file__).resolve().parents[1]
HUMAN_DIR = REPO_ROOT / "data" / "rankings" / "sarah" / "exports"
PILOT_DIR = REPO_ROOT / "data" / "creative_pilot_v1"
OLD_COMPARISONS = HUMAN_DIR / "comparisons_all.jsonl"
OLD_CANDIDATES = HUMAN_DIR / "comparisons_all_candidates.jsonl"
NEW_COMPARISONS = PILOT_DIR / "ai_review_exports" / "synthetic_human_verified.jsonl"
PROPOSALS = PILOT_DIR / "labels" / "proposals.jsonl"
NEW_SPLITS = PILOT_DIR / "splits.json"

ORIGIN_OLD, ORIGIN_NEW = "human_original", "synthetic_verified"
# The four held-out families of the rendered-metrics labeler (creative_pilot/calibrate.py): the proposal model never saw them.
OLD_TEST_FAMILIES = ("adobe_sample", "centered_editorial", "image_panel", "offer_coupon")
OLD_VAL_FAMILY_COUNT = 2
SPLIT_SEED = 42
DIRECTIONAL = ("A_BETTER", "B_BETTER")


@dataclass(frozen=True)
class Pair:
    comparison_id: str
    origin: str
    family: str
    split: str
    source_id: str
    source_path: str
    target: tuple[int, int]
    a_id: str
    b_id: str
    a_path: str
    b_path: str
    label: str
    pair_type: str = ""
    method: str = ""
    agreed_with_proposal: bool | None = None
    generator_a: str = ""
    generator_b: str = ""

    @property
    def y(self) -> float:
        return 1.0 if self.label == "A_BETTER" else 0.0

    @property
    def usable(self) -> bool:
        return self.label in DIRECTIONAL

    @property
    def target_name(self) -> str:
        return f"{self.target[0]}x{self.target[1]}"

    def key_a(self):
        return (self.source_id, self.a_id, self.target)

    def key_b(self):
        return (self.source_id, self.b_id, self.target)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def abs_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def old_val_families(test_families: Iterable[str] = OLD_TEST_FAMILIES, seed: int = SPLIT_SEED) -> tuple[str, ...]:
    """Deterministic validation families chosen from the original families that are not test families."""
    rows = [r for r in read_jsonl(OLD_COMPARISONS)]
    families = sorted({r["template_family_id"] for r in rows} - set(test_families))
    return tuple(sorted(random.Random(seed).sample(families, OLD_VAL_FAMILY_COUNT)))


def old_family_split() -> dict[str, str]:
    rows = read_jsonl(OLD_COMPARISONS)
    families = sorted({r["template_family_id"] for r in rows})
    val = set(old_val_families())
    return {f: ("test" if f in OLD_TEST_FAMILIES else "val" if f in val else "train") for f in families}


def load_old_pairs() -> list[Pair]:
    """My original tournament comparisons (voided rows excluded)."""
    cands = {c["candidate_id"]: c for c in read_jsonl(OLD_CANDIDATES)}
    split = old_family_split()
    out = []
    for r in read_jsonl(OLD_COMPARISONS):
        if r.get("voided"):
            continue
        a, b = cands[r["candidate_a_id"]], cands[r["candidate_b_id"]]
        out.append(Pair(
            comparison_id=r["comparison_id"], origin=ORIGIN_OLD, family=r["template_family_id"], split=split[r["template_family_id"]],
            source_id=r["source_design_id"], source_path=a["source_path"], target=(r["target_width"], r["target_height"]),
            a_id=a["candidate_id"], b_id=b["candidate_id"], a_path=a["candidate_path"], b_path=b["candidate_path"],
            label=r["label"], pair_type="tournament", method="human", generator_a=a.get("generator_id", ""),
            generator_b=b.get("generator_id", "")))
    return out


def load_new_pairs() -> list[Pair]:
    """Synthetic proposals that I explicitly accepted or corrected (rejected and unverified rows are never read)."""
    family_split = json.loads(NEW_SPLITS.read_text(encoding="utf-8"))["family_split"]
    proposals = {p["comparison_id"]: p for p in read_jsonl(PROPOSALS)}
    out = []
    for r in read_jsonl(NEW_COMPARISONS):
        if r.get("verification_status") != "human_verified" or r.get("voided"):
            continue
        p = proposals.get(r["comparison_id"], {})
        out.append(Pair(
            comparison_id=r["comparison_id"], origin=ORIGIN_NEW, family=r["template_family_id"], split=family_split[r["template_family_id"]],
            source_id=r["source_design_id"], source_path=r["source_path"], target=(r["target_width"], r["target_height"]),
            a_id=r["candidate_a_id"], b_id=r["candidate_b_id"], a_path=r["candidate_a_path"], b_path=r["candidate_b_path"],
            label=r["label"], pair_type=p.get("pair_type", ""), method=r.get("proposal_method", ""),
            agreed_with_proposal=r.get("human_agreed_with_proposal"), generator_a=r.get("candidate_a_generator_version", ""),
            generator_b=r.get("candidate_b_generator_version", "")))
    return out


def load_all_pairs() -> list[Pair]:
    return load_old_pairs() + load_new_pairs()


def pair_row(p: Pair) -> dict:
    return dict(comparison_id=p.comparison_id, origin=p.origin, label=p.label, usable_for_pairwise_loss=p.usable, split=p.split,
                template_family_id=p.family, source_design_id=p.source_id, source_path=p.source_path,
                target_width=p.target[0], target_height=p.target[1], candidate_a_id=p.a_id, candidate_b_id=p.b_id,
                candidate_a_path=p.a_path, candidate_b_path=p.b_path, pair_type=p.pair_type, proposal_method=p.method,
                human_agreed_with_proposal=p.agreed_with_proposal)


# ----------------------------------------------------------------------------- audits
def audit(pairs: list[Pair], feature_version: str = FEATURE_VERSION_V3) -> tuple[dict, dict]:
    """Return (report, graph cache). Resolution, hashes, Stage 1 validity, graph features, duplicates, conflicts, leakage."""
    report: dict = dict(feature_version=feature_version)
    problems: dict[str, list] = defaultdict(list)
    sources: dict[str, Design] = {}
    source_json: dict[str, dict] = {}
    cand_json: dict[str, dict] = {}
    graphs: dict[tuple, object] = {}
    validity: dict[tuple, bool] = {}

    for p in pairs:
        if p.source_id not in sources:
            sp = abs_path(p.source_path)
            if not sp.exists():
                problems["missing_source_file"].append(p.comparison_id)
                continue
            raw = json.loads(sp.read_text(encoding="utf-8"))
            if "sha256:" + digest(raw) != p.source_id:
                problems["source_hash_mismatch"].append(p.comparison_id)
            source_json[p.source_id] = raw
            sources[p.source_id] = Design.from_adobe(raw)

    for p in pairs:
        if p.source_id not in sources:
            continue
        for cid, path, key in ((p.a_id, p.a_path, p.key_a()), (p.b_id, p.b_path, p.key_b())):
            if key in graphs or key in validity:
                continue
            cp = abs_path(path)
            if not cp.exists():
                problems["missing_candidate_file"].append(p.comparison_id)
                validity[key] = False
                continue
            raw = json.loads(cp.read_text(encoding="utf-8"))
            if "sha256:" + digest(raw) != cid:
                problems["candidate_hash_mismatch"].append(p.comparison_id)
            cand_json[cid] = raw
            target = Canvas.from_adobe({"width": p.target[0], "height": p.target[1]})
            rep = validate_candidate(sources[p.source_id], raw, target)
            validity[key] = rep.is_valid
            if not rep.is_valid:
                problems["stage1_invalid_candidate"].append((p.comparison_id, cid, list(rep.violations)[:2]))
            try:
                graphs[key] = build_candidate_graph(sources[p.source_id], Design.from_adobe(raw), target, feature_version=feature_version)
            except Exception as exc:  # report, do not hide
                problems["graph_build_failed"].append((p.comparison_id, cid, str(exc)[:200]))

    resolved = [p for p in pairs if p.key_a() in graphs and p.key_b() in graphs]
    report["comparisons_total"] = len(pairs)
    report["comparisons_by_origin_label"] = {f"{o}:{lab}": n for (o, lab), n in sorted(Counter((p.origin, p.label) for p in pairs).items())}
    report["comparisons_with_both_graphs"] = len(resolved)
    report["unique_candidate_graphs"] = len(graphs)
    report["candidates_stage1_valid"] = sum(validity.values())
    report["candidates_stage1_invalid"] = len(validity) - sum(validity.values())

    shapes = Counter()
    for g in graphs.values():
        n, e = g.x_numeric.shape[0], g.edge_index.shape[1]
        shapes[(n <= 64) and (e == n * (n - 1))] += 1
    report["graphs_dense_edge_count_ok"] = shapes[True]
    report["graphs_dense_edge_count_bad"] = shapes[False]

    # duplicates / conflicts, on unordered candidate pairs within one source and target
    groups: dict[tuple, list[Pair]] = defaultdict(list)
    for p in pairs:
        groups[(p.source_id, p.target, frozenset((p.a_id, p.b_id)))].append(p)
    dup, conflicts = [], []
    for key, rows in groups.items():
        if len(rows) < 2:
            continue
        dup.append([r.comparison_id for r in rows])
        winners = set()
        for r in rows:
            if r.label == "A_BETTER":
                winners.add(r.a_id)
            elif r.label == "B_BETTER":
                winners.add(r.b_id)
        if len(winners) > 1:
            conflicts.append([r.comparison_id for r in rows])
    report["duplicate_unordered_pairs"] = len(dup)
    report["duplicate_pairs_examples"] = dup[:5]
    report["conflicting_directional_labels"] = len(conflicts)
    report["identical_candidate_pairs"] = sum(1 for p in pairs if p.a_id == p.b_id)
    report["duplicate_comparison_ids"] = len(pairs) - len({p.comparison_id for p in pairs})

    # preference cycles A>B>C>A (within one source/target) among directional labels
    edges: dict[tuple, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    for p in pairs:
        if p.label == "A_BETTER":
            edges[(p.source_id, p.target)][p.a_id].add(p.b_id)
        elif p.label == "B_BETTER":
            edges[(p.source_id, p.target)][p.b_id].add(p.a_id)
    cycles = 0
    for graph in edges.values():
        state: dict[str, int] = {}

        def visit(node: str) -> bool:
            state[node] = 1
            for nxt in graph.get(node, ()):
                if state.get(nxt) == 1 or (nxt not in state and visit(nxt)):
                    return True
            state[node] = 2
            return False

        if any(n not in state and visit(n) for n in list(graph)):
            cycles += 1
    report["source_target_groups_with_preference_cycle"] = cycles

    # leakage
    by_split = defaultdict(lambda: defaultdict(set))
    for p in pairs:
        by_split[p.split]["families"].add(p.family)
        by_split[p.split]["sources"].add(p.source_id)
        by_split[p.split]["candidates"].update((p.a_id, p.b_id))
    leaks = {}
    names = sorted(by_split)
    for i, s in enumerate(names):
        for t in names[i + 1:]:
            for kind in ("families", "sources", "candidates"):
                inter = by_split[s][kind] & by_split[t][kind]
                if inter:
                    leaks[f"{s}&{t}:{kind}"] = sorted(inter)[:5] + ([f"... {len(inter)} total"] if len(inter) > 5 else [])
    report["leakage_between_splits"] = leaks
    report["split_summary"] = {s: dict(families=len(by_split[s]["families"]), sources=len(by_split[s]["sources"]),
                                       candidates=len(by_split[s]["candidates"]),
                                       comparisons=sum(1 for p in pairs if p.split == s),
                                       usable=sum(1 for p in pairs if p.split == s and p.usable)) for s in names}
    report["split_by_origin"] = {f"{o}:{s}": n for (o, s), n in sorted(Counter((p.origin, p.split) for p in pairs if p.usable).items())}
    report["problems"] = {k: v[:10] for k, v in problems.items()}
    report["problem_counts"] = {k: len(v) for k, v in problems.items()}
    return report, graphs


def usable_pairs(pairs: Iterable[Pair], graphs: dict) -> list[Pair]:
    return [p for p in pairs if p.usable and p.key_a() in graphs and p.key_b() in graphs]
