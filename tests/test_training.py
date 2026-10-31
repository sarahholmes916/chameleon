"""Training-pipeline mechanics (label audit, pairwise loop, early stopping) on tiny deterministic CPU fixtures.

These tests establish software contracts only; they say nothing about layout quality or about the real labels.
"""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from engine.graph_features import build_candidate_graph
from engine.internal_schema import Canvas, Design
from scripts.source_dataset import digest
from training import data as D
from training import train as T


TARGET = (100, 80)


def source_json():
    elements = []
    for i in range(3):
        item = dict(id=f"e{i}", type="text" if i < 2 else "image", x=5 + 10 * i, y=5 + 8 * i, width=30, height=12, group_id=None)
        if item["type"] == "text":
            item.update(content=f"Text {i}", font_family="Family", font_size=10, color="#000000")
        else:
            item.update(asset_id="a.png")
        elements.append(item)
    return dict(canvas=dict(width=TARGET[0], height=TARGET[1]), elements=elements)


def candidate_json(shift, font=10):
    value = source_json()
    for i, e in enumerate(value["elements"]):
        e["x"] += shift * (i + 1)
        if e["type"] == "text":
            e["font_size"] = font
    return value


def write(tmp_path, name, value):
    path = tmp_path / name
    path.write_text(json.dumps(value))
    return str(path), "sha256:" + digest(value)


def make_pair(tmp_path, n, a, b, label, split="train", family="fam", source=None):
    sp, sid = write(tmp_path, f"source_{n if source else 0}.json", source or source_json())
    ap, aid = write(tmp_path, f"a{a}.json", candidate_json(a))
    bp, bid = write(tmp_path, f"b{b}.json", candidate_json(b))
    return D.Pair(comparison_id=f"c{n}", origin=D.ORIGIN_OLD, family=family, split=split, source_id=sid, source_path=sp, target=TARGET,
                  a_id=aid, b_id=bid, a_path=ap, b_path=bp, label=label)


def test_only_directional_labels_are_usable(tmp_path):
    rows = [make_pair(tmp_path, i, 1, 2, label) for i, label in enumerate(("A_BETTER", "B_BETTER", "TIE", "BOTH_BAD"))]
    assert [p.usable for p in rows] == [True, True, False, False]
    assert [p.y for p in rows[:2]] == [1.0, 0.0]


def test_audit_reports_clean_data_and_graph_shapes(tmp_path):
    other = source_json()
    other["elements"][0]["x"] += 1  # a distinct source design, so the val pair shares nothing with the train pair
    pairs = [make_pair(tmp_path, 1, 1, 2, "A_BETTER"), make_pair(tmp_path, 2, 3, 4, "B_BETTER", split="val", family="other", source=other)]
    report, graphs = D.audit(pairs)
    assert not report["problem_counts"] and report["comparisons_with_both_graphs"] == 2
    assert report["candidates_stage1_invalid"] == 0 and report["graphs_dense_edge_count_bad"] == 0
    assert report["conflicting_directional_labels"] == 0 and not report["leakage_between_splits"]
    assert all(g.edge_index.shape[1] == g.x_numeric.shape[0] * (g.x_numeric.shape[0] - 1) for g in graphs.values())
    assert all(torch.isfinite(g.x_numeric).all() and torch.isfinite(g.edge_attr).all() for g in graphs.values())


def test_audit_flags_duplicates_conflicts_missing_files_and_leakage(tmp_path):
    first = make_pair(tmp_path, 1, 1, 2, "A_BETTER")
    swapped = D.Pair(**{**first.__dict__, "comparison_id": "c2", "a_id": first.b_id, "b_id": first.a_id,
                        "a_path": first.b_path, "b_path": first.a_path})  # same pair shown in the other order, same winner
    flipped = D.Pair(**{**swapped.__dict__, "comparison_id": "c3", "label": "A_BETTER"})  # now the opposite winner
    leak = D.Pair(**{**make_pair(tmp_path, 4, 5, 6, "A_BETTER", split="test").__dict__})  # same source + family in train and test
    missing = D.Pair(**{**first.__dict__, "comparison_id": "c5", "b_id": "sha256:none", "b_path": str(tmp_path / "gone.json")})
    report, _ = D.audit([first, swapped, flipped, leak, missing])
    assert report["duplicate_unordered_pairs"] == 1 and report["conflicting_directional_labels"] == 1
    assert report["problem_counts"].get("missing_candidate_file") == 1
    assert report["source_target_groups_with_preference_cycle"] == 1
    assert any(key.endswith(":sources") for key in report["leakage_between_splits"])
    assert any(key.endswith(":families") for key in report["leakage_between_splits"])


def test_audit_detects_candidate_that_breaks_stage_one(tmp_path):
    bad = candidate_json(1)
    bad["elements"][0]["content"] = "changed text"  # immutable field
    sp, sid = write(tmp_path, "source.json", source_json())
    ap, aid = write(tmp_path, "bad.json", bad)
    bp, bid = write(tmp_path, "ok.json", candidate_json(2))
    pair = D.Pair("c1", D.ORIGIN_OLD, "f", "train", sid, sp, TARGET, aid, bid, ap, bp, "A_BETTER")
    report, _ = D.audit([pair])
    assert report["candidates_stage1_invalid"] == 1 and report["problem_counts"]["stage1_invalid_candidate"] == 1


def graph_for(tmp_path, shift):
    source = Design.from_adobe(source_json())
    return build_candidate_graph(source, Design.from_adobe(candidate_json(shift)), Canvas.from_adobe(dict(width=TARGET[0], height=TARGET[1])))


def tiny_problem(tmp_path, count=12):
    pairs, graphs = [], {}
    for n in range(count):
        a, b = 1 + n % 4, 5 + n % 3
        pair = make_pair(tmp_path, n, a, b, "A_BETTER" if n % 2 else "B_BETTER")
        pairs.append(pair)
        graphs[pair.key_a()], graphs[pair.key_b()] = graph_for(tmp_path, a), graph_for(tmp_path, b)
    return pairs, graphs


def test_pairwise_loop_trains_selects_best_epoch_and_is_deterministic_on_cpu(tmp_path, monkeypatch):
    monkeypatch.setitem(T.HYPERPARAMETERS, "max_epochs", 4)
    pairs, graphs = tiny_problem(tmp_path)
    cpu = torch.device("cpu")
    first = T.train_run(pairs[:8], pairs[8:], graphs, seed=7, device=cpu)
    second = T.train_run(pairs[:8], pairs[8:], graphs, seed=7, device=cpu)
    assert first["epochs_run"] == 4 and 1 <= first["best_epoch"] <= 4
    assert first["history"] == second["history"]
    scores = T.logits(first["model"], pairs[8:], graphs, cpu)
    assert scores.shape == (4,) and torch.isfinite(scores).all()
    assert T.metrics_from(scores, pairs[8:])["accuracy"] == pytest.approx(first["val_accuracy"])


def test_early_stopping_waits_for_patience(tmp_path, monkeypatch):
    monkeypatch.setitem(T.HYPERPARAMETERS, "max_epochs", 30)
    monkeypatch.setitem(T.HYPERPARAMETERS, "patience", 2)
    monkeypatch.setitem(T.HYPERPARAMETERS, "lr", 0.0)  # frozen weights: nothing improves after epoch 1
    pairs, graphs = tiny_problem(tmp_path)
    run = T.train_run(pairs[:8], pairs[8:], graphs, seed=1, device=torch.device("cpu"))
    assert run["best_epoch"] == 1 and run["epochs_run"] == 3 and run["stopped_early"]


def test_metrics_cluster_ci_and_breakdown(tmp_path):
    pairs = [make_pair(tmp_path, i, 1, 2, "A_BETTER") for i in range(4)]
    m = T.metrics_from(torch.tensor([2.0, 1.0, -1.0, -3.0]), pairs)
    assert m["n"] == 4 and m["accuracy"] == 0.5 and m["loss"] > 0
    lo, hi = T.cluster_ci([1.0, 1.0, 0.0, 0.0], ["s1", "s1", "s2", "s2"], reps=200)
    assert 0.0 <= lo <= 0.5 <= hi <= 1.0
    assert T.breakdown(pairs, [1.0, 0.0, 1.0, 1.0], lambda p: p.family) == {"fam": dict(n=4, accuracy=0.75)}


def test_pair_weights_scale_metrics_and_loss_deterministically(tmp_path, monkeypatch):
    pairs = [make_pair(tmp_path, i, 1, 2, "A_BETTER") for i in range(4)]
    d = torch.tensor([2.0, 1.0, -1.0, -3.0])                        # first two correct, last two wrong
    heavy_on_correct = T.metrics_from(d, pairs, lambda p: 3.0 if p.comparison_id in ("c0", "c1") else 1.0)
    assert heavy_on_correct["accuracy"] == pytest.approx(0.75) and heavy_on_correct["accuracy"] > T.metrics_from(d, pairs)["accuracy"]
    assert T.metrics_from(d, pairs, lambda p: 1.0) == T.metrics_from(d, pairs)           # unit weights == unweighted
    monkeypatch.setitem(T.HYPERPARAMETERS, "max_epochs", 3)
    problem, graphs = tiny_problem(tmp_path)
    cpu = torch.device("cpu")
    plain = T.train_run(problem[:8], problem[8:], graphs, seed=3, device=cpu)
    unit = T.train_run(problem[:8], problem[8:], graphs, seed=3, device=cpu, weight_fn=lambda p: 1.0)
    assert plain["history"] == unit["history"]                                           # weight 1 is exactly the old behaviour
    half = T.train_run(problem[:8], problem[8:], graphs, seed=3, device=cpu, weight_fn=lambda p: 0.5 if p.comparison_id in ("c0", "c2") else 1.0)
    assert half["history"] != plain["history"] and all(torch.isfinite(torch.tensor(h["train_loss"])) for h in half["history"])


def test_weighting_rule_uses_validation_only():
    from training import weighting as W
    rows = {0.25: dict(val_old_accuracy=dict(mean=0.90), val_pooled_accuracy=dict(mean=0.85)),
            0.5: dict(val_old_accuracy=dict(mean=0.90), val_pooled_accuracy=dict(mean=0.87)),
            0.75: dict(val_old_accuracy=dict(mean=0.80), val_pooled_accuracy=dict(mean=0.99))}
    assert W.pick_lambda(rows) == 0.5                      # tie on original-human validation -> higher pooled validation
    assert W.weight_fn_for(0.25)(D.Pair("c", D.ORIGIN_NEW, "f", "train", "s", "p", (1, 1), "a", "b", "x", "y", "A_BETTER")) == 0.25
    assert W.weight_fn_for(0.25)(D.Pair("c", D.ORIGIN_OLD, "f", "train", "s", "p", (1, 1), "a", "b", "x", "y", "A_BETTER")) == 1.0
