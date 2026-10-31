"""Blind-evaluation mechanics: pooling, sampling, anonymised manifests, saved judgments, tally.  Fixtures only; no checkpoints or real data."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from training import blind_eval as B  # noqa: E402

METHODS = ["objective", "gnn_original", "gnn_combined"]


def pool(task_id, target=(1600, 400), origin="new", family="f", n=4):
    cands = tuple(B.Cand(f"{task_id}-c{i}", f"{task_id}-c{i}.json", f"{task_id}-c{i}.png", "gen", "creative") for i in range(n))
    return B.TaskPool(task_id, origin, family, "s-" + task_id, "s.json", "s.png", target, cands)


def picks_for(pools, pattern):
    """pattern: {task_id: (objective_idx, original_idx, combined_idx)}"""
    return {p.task_id: dict(zip(METHODS, (p.candidates[i].candidate_id for i in pattern[p.task_id]))) for p in pools}


def test_agreement_counts_all_same_and_distinct_layouts():
    pools = [pool("a"), pool("b"), pool("c", target=(90, 728))]
    picks = picks_for(pools, {"a": (0, 0, 0), "b": (0, 1, 1), "c": (0, 1, 2)})
    report = B.agreement_report(pools, picks, METHODS)
    assert report["overall"] == dict(tasks=3, all_methods_same=1)
    assert report["distinct_selected_layouts"] == {1: 1, 2: 1, 3: 1}
    assert report["pairwise_same_pick"]["gnn_original~gnn_combined"] == 2
    assert report["by_target"]["90x728"] == dict(tasks=1, all_methods_same=0)


def test_sampling_only_uses_disagreement_is_stratified_and_deterministic():
    targets = [(1600, 400), (728, 90), (90, 728), (600, 1000)]
    pools = [pool(f"t{i}", target=targets[i % 4], family=f"fam{i % 3}", origin="old" if i % 4 == 3 else "new") for i in range(40)]
    pattern = {p.task_id: ((0, 0, 0) if i % 5 == 0 else (0, 1, 2)) for i, p in enumerate(pools)}
    picks = picks_for(pools, pattern)
    a, b = B.sample_tasks(pools, picks, n=12, seed=3), B.sample_tasks(pools, picks, n=12, seed=3)
    assert [p.task_id for p in a] == [p.task_id for p in b] and len(a) == 12
    assert all(len(set(picks[p.task_id].values())) >= 2 for p in a)               # unanimous tasks are never shown
    counts = {t: sum(p.target_name == t for p in a) for t in B.EXTREME}
    assert min(counts.values()) >= 3 and sum(p.target_name == "600x1000" for p in a) <= 2
    assert [p.task_id for p in B.sample_tasks(pools, picks, n=12, seed=4)] != [p.task_id for p in a]


def test_manifests_hide_identity_dedupe_options_and_shuffle_positions():
    pools = [pool("a"), pool("b")]
    picks = picks_for(pools, {"a": (0, 1, 1), "b": (0, 1, 2)})
    public, private = B.build_manifests(pools, picks, seed=5)
    text = str(public)
    assert not any(m in text for m in METHODS) and "methods" not in text
    assert [len(t["options"]) for t in public["tasks"]] == [2, 3]
    shared = next(o for o in private["tasks"][0]["options"] if len(o["methods"]) == 2)
    assert shared["methods"] == ["gnn_combined", "gnn_original"]
    assert [o["position"] for o in public["tasks"][1]["options"]] == [1, 2, 3]
    assert B.build_manifests(pools, picks, seed=5)[0] == public                        # same seed -> same display order


def test_judgments_persist_resume_undo_and_reject_double_votes(tmp_path):
    pools = [pool("a"), pool("b")]
    picks = picks_for(pools, {"a": (0, 1, 2), "b": (0, 1, 2)})
    public, _ = B.build_manifests(pools, picks, seed=1)
    conn = B.connect(tmp_path / "r.db")
    first = B.next_task(public, conn, "me")["task_id"]
    B.record(conn, public, first, "me", "2")
    with pytest.raises(ValueError):
        B.record(conn, public, first, "me", "1")
    with pytest.raises(ValueError):
        B.record(conn, public, public["tasks"][1]["task_id"], "me", "9")                 # no such option
    conn.close()
    again = B.connect(tmp_path / "r.db")                                                   # resume after restart
    assert B.next_task(public, again, "me")["task_id"] != first
    assert B.next_task(public, again, "someone_else")["task_id"] == public["tasks"][0]["task_id"]
    assert B.undo_last(again, "me") == first and B.next_task(public, again, "me")["task_id"] == public["tasks"][0]["task_id"]
    assert again.execute("SELECT COUNT(*) FROM judgments").fetchone()[0] == 1               # voided, not deleted (audit trail)


def test_tally_credits_methods_and_head_to_head(tmp_path):
    pools = [pool("a"), pool("b"), pool("c"), pool("d")]
    picks = picks_for(pools, {"a": (0, 1, 1), "b": (0, 1, 2), "c": (0, 0, 1), "d": (0, 1, 2)})
    public, private = B.build_manifests(pools, picks, seed=2)
    conn = B.connect(tmp_path / "r.db")

    def option_of(task_id, method):
        return str(next(o["position"] for t in private["tasks"] if t["task_id"] == task_id for o in t["options"] if method in o["methods"]))
    B.record(conn, public, "a", "me", option_of("a", "gnn_combined"))      # shared by both GNNs
    B.record(conn, public, "b", "me", option_of("b", "objective"))
    B.record(conn, public, "c", "me", B.TIE)
    B.record(conn, public, "d", "me", B.ALL_BAD)
    t = B.tally(private, conn, "me")
    assert t["outcomes"] == {"OPTION": 2, B.TIE: 1, B.ALL_BAD: 1}
    assert t["per_method"]["gnn_combined"]["best_shared"] == 1 and t["per_method"]["gnn_original"]["best_any"] == 1
    assert t["per_method"]["objective"]["best_unique"] == 1 and t["per_method"]["objective"]["tie"] == 1
    h = t["head_to_head"]["gnn_combined vs objective"]
    assert h["a_wins"] == 1 and h["b_wins"] == 1 and h["tie_or_all_bad"] == 2
    assert t["head_to_head"]["gnn_combined vs gnn_original"]["same_pick"] == 1          # task a: both picked one layout
    assert B.tally(private, conn, "me", origin="old")["judged_task_judgments"] == 0


def test_sign_test():
    assert B.sign_test_p(0, 0) is None
    assert B.sign_test_p(5, 5) == 1.0
    assert B.sign_test_p(10, 0) == pytest.approx(2 / 1024)
