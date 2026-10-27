"""Neural mechanics and checkpoint contracts, not aesthetic-quality evidence."""

import ast
from copy import deepcopy
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
from torch import nn
from torch_geometric.data import Batch

from engine.graph_features import build_candidate_graph
from engine.internal_schema import Canvas, Design
from engine.ranker import (
    CHECKPOINT_VERSION, RANKER_VERSION, CheckpointUnavailable, LayoutRanker,
    ModelInferenceError, RankerConfig, load_ranker_checkpoint,
    save_ranker_checkpoint, score_candidates,
)
from engine.text_embeddings import TextEmbeddingCache


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def seeded_cpu():
    # Tiny graphs are faster with one CPU thread; restore caller settings/RNG.
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        yield
    torch.set_num_threads(previous)


def source_design(n=4):
    elements = []
    for i in range(n):
        item = dict(id=f"opaque-{i}", type="text" if i % 2 == 0 else "image",
                    x=i * 7, y=i * 3, width=20 + i, height=10 + i, group_id=None,
                    extension={"keep": [None, 1, "literal"]})
        if item["type"] == "text":
            item.update(content=f"Literal text {i}", font_family="Family", font_size=12, color="#fff")
        else:
            item.update(asset_id="unavailable.png")
        elements.append(item)
    return Design.from_adobe(dict(canvas=dict(width=100, height=80), elements=elements))


def candidate_design(source, shift=0, target=None):
    value = source.to_adobe()
    value["canvas"] = (target or source.canvas).to_adobe()
    for i, element in enumerate(value["elements"]):
        element["x"] += shift * (i + 1)
        element["height"] += abs(shift)
    return Design.from_adobe(value)


class TestEncoder:
    __test__ = False
    encoder_version = "test-encoder@fixed-revision"

    def encode(self, content):
        result = torch.linspace(-1, 1, 384)
        result[0] = len(content)
        return result


def graph_for(n=4, shift=0, cache=None):
    source = source_design(n)
    return build_candidate_graph(source, candidate_design(source, shift), source.canvas,
                                 text_embeddings=cache)


def provenance(trained=False):
    # Synthetic metadata tests schema/availability only; these are not human
    # labels, an actual training run, or evidence of preference accuracy.
    return dict(status="trained" if trained else "untrained", seed=42,
                source_splits=dict(train=["fixture-train"] if trained else [],
                                   validation=["fixture-validation"] if trained else [], test=[]),
                metrics={"fixture_metric": 0.5} if trained else {})


def checkpoint_for(tmp_path, model=None, trained=True):
    model = model or LayoutRanker()
    path = tmp_path / "test-ranker.pt"
    save_ranker_checkpoint(path, model, training_metadata=provenance(trained))
    return load_ranker_checkpoint(path, expected_config=model.config)


def rewrite_manifest(path, change):
    payload = torch.load(path, weights_only=True)
    change(payload["manifest"])
    canonical = json.dumps(payload["manifest"], sort_keys=True, separators=(",", ":"), allow_nan=False)
    payload["manifest_sha256"] = sha256(canonical.encode()).hexdigest()
    torch.save(payload, path)


def test_ranker_receives_candidate_typography_and_can_learn_from_it():
    source = source_design(1)
    candidate = source.to_adobe()
    candidate["elements"][0]["font_size"] = 60
    small = build_candidate_graph(source, source, source.canvas)
    large = build_candidate_graph(source, Design.from_adobe(candidate), source.canvas)
    batch = Batch.from_data_list([small, large])
    batch.x_numeric.requires_grad_(True)
    model = LayoutRanker().eval()
    scores = model(batch)
    # Fixed seed verifies sensitivity/mechanics, not which typography is better.
    assert scores[0].item() != scores[1].item()
    scores.sum().backward()
    assert torch.isfinite(batch.x_numeric.grad).all()
    assert torch.all(batch.x_numeric.grad[:, 53:55].abs().sum(dim=0) > 0)


@pytest.mark.parametrize("width", [53, 55])
def test_old_graph_width_is_rejected_even_if_version_is_relabelled(width):
    graph = graph_for()
    graph.x_numeric = graph.x_numeric[:, :width]
    with pytest.raises(ValueError):
        LayoutRanker()(graph)


@pytest.mark.parametrize("version,width", [("gatv2_ranker_v1", 53), ("gatv2_ranker_v2", 55)])
def test_old_ranker_architecture_checkpoint_is_rejected(tmp_path, version, width):
    path = tmp_path / "old.pt"
    save_ranker_checkpoint(path, LayoutRanker(), training_metadata=provenance())
    rewrite_manifest(path, lambda manifest: manifest["architecture"].update(
        version=version, numeric_dim=width))
    with pytest.raises(CheckpointUnavailable, match="incompatible"):
        load_ranker_checkpoint(path)


def test_exact_s03_architecture_and_versions():
    model = LayoutRanker()
    assert RANKER_VERSION == "gatv2_ranker_v3"
    assert CHECKPOINT_VERSION == "ranker_checkpoint_v1"
    assert (model.text_projection.in_features, model.text_projection.out_features) == (384, 32)
    assert len(model.convs) == len(model.skips) == len(model.norms) == 3
    for conv, width in zip(model.convs, (90, 128, 128)):
        assert (conv.in_channels, conv.out_channels, conv.heads, conv.edge_dim) == (width, 32, 4, 31)
        assert conv.concat and conv.add_self_loops
        assert conv.fill_value == 0.0
        assert conv.dropout == 0.0
        assert not conv.share_weights and conv.res is None
    assert isinstance(model.skips[0], nn.Linear)
    assert (model.skips[0].in_features, model.skips[0].out_features) == (90, 128)
    assert all(isinstance(skip, nn.Identity) for skip in model.skips[1:])
    assert all(isinstance(norm, nn.LayerNorm) and norm.normalized_shape == (128,) for norm in model.norms)
    assert model.feature_dropout.p == model.head[2].p == 0.1
    assert (model.head[0].in_features, model.head[0].out_features) == (256, 128)
    assert isinstance(model.head[1], nn.ReLU)
    assert (model.head[3].in_features, model.head[3].out_features) == (128, 1)


@pytest.mark.parametrize("filename,n", [
    ("chameleon_sample_level1_warmup.json", 4),
    ("chameleon_sample_level2_core.json", 11),
    ("chameleon_sample_level3_advanced.json", 17),
])
def test_real_sample_scores_are_finite_and_designs_unchanged(filename, n):
    task = json.loads((ROOT / "input" / filename).read_text(encoding="utf-8"))
    source = Design.from_adobe(task["source"])
    target = Canvas.from_adobe(task["target_canvas"])
    candidate = candidate_design(source, 2, target)
    before = source.to_adobe(), candidate.to_adobe()
    graph = build_candidate_graph(source, candidate, target)
    original_graph = graph.clone()
    model = LayoutRanker().eval()
    with torch.no_grad():
        score = model(graph)
    assert score.shape == (1,) and score.dtype == torch.float32
    assert torch.isfinite(score).all()
    assert graph.num_nodes == n
    assert graph.edge_index.shape[1] == n * (n - 1)
    assert before == (source.to_adobe(), candidate.to_adobe())
    for name in ("x_numeric", "text_vec", "text_mask", "edge_index", "edge_attr"):
        assert torch.equal(graph[name], original_graph[name])


def test_singleton_empty_edges_and_deterministic_eval():
    graph = graph_for(1)
    assert graph.edge_index.shape == (2, 0)
    model = LayoutRanker().eval()
    with torch.no_grad():
        a, b = model(graph), model(graph)
    assert a.shape == (1,)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert graph.edge_attr.shape == (0, 31)  # Self loops stay inside GATv2.


def test_batch_scores_match_separate_runs_and_remain_isolated():
    model = LayoutRanker().eval()
    graphs = [graph_for(n, shift=i * 2) for i, n in enumerate((1, 4, 17))]
    with torch.no_grad():
        individual = torch.cat([model(g) for g in graphs])
        batched = model(Batch.from_data_list(graphs))
        changed = model(Batch.from_data_list([graphs[0], graph_for(4, shift=45), graphs[2]]))
    assert batched.shape == (3,)
    torch.testing.assert_close(batched, individual, rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(changed[[0, 2]], batched[[0, 2]], rtol=1e-5, atol=2e-6)
    assert abs(changed[1].item() - batched[1].item()) > 1e-5


def test_permuting_nodes_and_edges_preserves_whole_graph_score():
    graph = graph_for(4, shift=5)
    permuted = graph.clone()
    permutation = torch.tensor([2, 0, 3, 1])
    inverse = permutation.argsort()
    for name in ("x_numeric", "text_vec", "text_mask"):
        permuted[name] = graph[name][permutation]
    permuted.edge_index = inverse[graph.edge_index].flip(1)
    permuted.edge_attr = graph.edge_attr.flip(0)
    permuted.element_ids = tuple(graph.element_ids[i] for i in permutation)
    model = LayoutRanker().eval()
    with torch.no_grad():
        torch.testing.assert_close(model(graph), model(permuted), rtol=1e-5, atol=2e-6)


def test_pooling_concatenates_each_graphs_mean_then_max_after_final_elu():
    model = LayoutRanker().train()
    norm_outputs, pooled_inputs, dropouts = [], [], []
    handles = [
        model.norms[-1].register_forward_hook(lambda _m, _a, output: norm_outputs.append(output.detach())),
        model.head.register_forward_pre_hook(lambda _m, args: pooled_inputs.append(args[0].detach())),
        model.feature_dropout.register_forward_hook(lambda *_args: dropouts.append(True)),
    ]
    model(Batch.from_data_list([graph_for(1), graph_for(4)]))
    for handle in handles:
        handle.remove()
    nodes = torch.nn.functional.elu(norm_outputs[0])
    expected = torch.stack([
        torch.cat((part.mean(0), part.max(0).values)) for part in (nodes[:1], nodes[1:])
    ])
    torch.testing.assert_close(pooled_inputs[0], expected, rtol=0, atol=0)
    assert len(dropouts) == 2  # Feature dropout only after the first two stages.


def test_nonfinite_model_output_is_rejected():
    model = LayoutRanker().eval()
    with torch.no_grad():
        model.head[-1].bias.fill_(float("nan"))
    with pytest.raises(ValueError, match="non-finite scores"):
        model(graph_for())


def test_backprop_uses_candidate_geometry_edges_and_trainable_text_projection():
    cache = TextEmbeddingCache(TestEncoder())
    graph = graph_for(4, shift=3, cache=cache)
    graph.x_numeric.requires_grad_(True)
    graph.edge_attr.requires_grad_(True)
    graph.text_vec.requires_grad_(True)
    model = LayoutRanker(RankerConfig(text_encoder_version=cache.encoder_version)).eval()
    captured = []
    handle = model.convs[0].register_forward_pre_hook(lambda _module, args: captured.append(args[0].detach().clone()))
    model(graph).square().mean().backward()
    handle.remove()
    assert captured[0].shape == (4, 90)
    assert torch.equal(captured[0][:, :57], graph.x_numeric)
    assert torch.equal(captured[0][:, -1:], graph.text_mask)
    assert torch.count_nonzero(captured[0][graph.text_mask[:, 0] == 0, 57:89]) == 0
    assert captured[0][:, 57:89].min() >= 0  # ReLU projection.
    assert graph.text_vec.grad is None  # Frozen external embeddings.
    assert graph.x_numeric.grad[:, 42:50].abs().sum() > 0
    assert graph.edge_attr.grad.abs().sum() > 0
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert model.text_projection.weight.grad.abs().sum() > 0
    assert all(conv.lin_edge.weight.grad.abs().sum() > 0 for conv in model.convs)
    assert model.skips[0].weight.grad.abs().sum() > 0


def test_training_step_changes_weights_and_eval_disables_dropout():
    model = LayoutRanker()
    batch = Batch.from_data_list([graph_for(4), graph_for(4, shift=5)])
    before = model.convs[0].lin_l.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    values = model(batch)
    # Arbitrary target checks optimization mechanics, not preference labels.
    loss = (values - torch.tensor([0.2, -0.1])).square().mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert not torch.equal(before, model.convs[0].lin_l.weight)
    assert not torch.equal(model(batch), model(batch))
    model.eval()
    assert torch.equal(model(batch), model(batch))


def test_sparse_graph_requires_declared_topology_and_runs_when_declared():
    graph = graph_for(65)
    with pytest.raises(ValueError, match="topology"):
        LayoutRanker()(graph)
    model = LayoutRanker(RankerConfig(graph_topologies=("dense_v1", "sparse_v1"))).eval()
    scores = model(Batch.from_data_list([graph_for(1), graph]))
    assert scores.shape == (2,) and torch.isfinite(scores).all()


@pytest.mark.parametrize("field,value,message", [
    ("context_version", "graph_v1", "feature version"),
    ("context_version", "graph_v2", "feature version"),
    ("graph_topology", "unknown", "topology"),
    ("text_encoder_version", "different", "encoder version"),
    ("text_embedding_status", "complete", "status"),
    ("context_version", None, "one string"),
    ("text_embedding_status", ["none", "none"], "one string"),
])
def test_graph_metadata_mismatch_rejected(field, value, message):
    graph = graph_for()
    graph[field] = value
    with pytest.raises(ValueError, match=message):
        LayoutRanker()(graph)


@pytest.mark.parametrize("field", ["x_numeric", "text_vec", "text_mask", "edge_attr"])
@pytest.mark.parametrize("problem", ["shape", "dtype", "nan"])
def test_tensor_contract_rejected_before_message_passing(field, problem):
    graph = graph_for()
    if problem == "shape":
        graph[field] = graph[field][:, :-1]
    elif problem == "dtype":
        graph[field] = graph[field].double()
    else:
        graph[field][0, 0] = float("nan")
    with pytest.raises(ValueError, match=field):
        LayoutRanker()(graph)


@pytest.mark.parametrize("problem", ["dtype", "shape", "negative", "range", "cross_batch", "empty_graph"])
def test_invalid_edges_and_batch_assignment_rejected(problem):
    graph = Batch.from_data_list([graph_for(4), graph_for(1)])
    if problem == "dtype":
        graph.edge_index = graph.edge_index.int()
    elif problem == "shape":
        graph.edge_index = graph.edge_index.t()
    elif problem == "negative":
        graph.edge_index[0, 0] = -1
    elif problem == "range":
        graph.edge_index[0, 0] = graph.num_nodes
    elif problem == "cross_batch":
        graph.edge_index[0, 0] = 4
    else:
        graph.batch[:] = 0
    with pytest.raises(ValueError):
        LayoutRanker()(graph)


def test_mixed_batch_checks_every_graphs_metadata():
    graph = graph_for()
    graph.context_version = "wrong"
    with pytest.raises(ValueError, match="feature version"):
        LayoutRanker()(Batch.from_data_list([graph_for(), graph]))


def test_text_checkpoint_requires_exact_encoder_and_complete_embeddings():
    cache = TextEmbeddingCache(TestEncoder())
    model = LayoutRanker(RankerConfig(text_encoder_version=cache.encoder_version))
    with pytest.raises(ValueError, match="encoder version"):
        model(graph_for())
    graph = graph_for(cache=cache)
    graph.text_vec[0] = 0
    graph.text_mask[0] = 0
    graph.text_embedding_status = "partial"
    with pytest.raises(ValueError, match="complete"):
        model(graph)
    with pytest.raises(ValueError, match="encoder version"):
        LayoutRanker()(graph_for(cache=cache))


@pytest.mark.parametrize("problem", ["fractional_mask", "nontext_mask", "hidden_vector", "semantic_mask", "semantic_cos"])
def test_missingness_cannot_silently_leak_text_information(problem):
    graph = graph_for()
    if problem == "fractional_mask":
        graph.text_mask[0] = 0.5
    elif problem == "nontext_mask":
        graph.text_mask[1] = 1
    elif problem == "hidden_vector":
        graph.text_vec[0, 0] = 1
    elif problem == "semantic_mask":
        graph.edge_attr[0, 18] = 1
    else:
        graph.edge_attr[0, 17] = 0.9
    with pytest.raises(ValueError):
        LayoutRanker()(graph)


def test_text_enabled_no_text_design_and_zero_available_vector():
    source = source_design(2)
    source = replace(source, elements=(source.elements[1],))
    cache = TextEmbeddingCache(TestEncoder())
    model = LayoutRanker(RankerConfig(text_encoder_version=cache.encoder_version))
    graph = build_candidate_graph(source, source, source.canvas, text_embeddings=cache)
    assert graph.text_embedding_status == "no_text"
    assert torch.isfinite(model(graph)).all()
    graph = graph_for(cache=cache)
    graph.text_vec[0] = 0
    with pytest.raises(ValueError, match="nonzero"):
        model(graph)


@pytest.mark.parametrize("changes", [
    dict(dropout=-0.1), dict(dropout=1), dict(dropout=True), dict(dropout=float("nan")),
    dict(graph_topologies=()), dict(graph_topologies=("dense_v1", "dense_v1")),
    dict(graph_topologies="dense_v1"), dict(graph_topologies=("unknown",)),
    dict(text_encoder_version=""),
])
def test_invalid_config_is_rejected(changes):
    with pytest.raises(ValueError):
        RankerConfig(**changes)


def test_checkpoint_round_trip_manifest_hashes_and_rng_preservation(tmp_path, monkeypatch):
    model = LayoutRanker().eval()
    graph = graph_for()
    expected = model(graph).detach()
    metadata = provenance()
    path = tmp_path / "model.pt"
    save_ranker_checkpoint(path, model, training_metadata=metadata)
    payload = torch.load(path, weights_only=True)
    manifest = payload["manifest"]
    assert manifest["checkpoint_version"] == CHECKPOINT_VERSION
    assert manifest["numeric_feature_version"] == manifest["edge_feature_version"] == "graph_v3"
    assert manifest["architecture"]["pyg_version"] == "2.8.1"
    assert manifest["runtime_versions"]["pyg"] == "2.8.1"
    assert len(manifest["weights_sha256"]) == len(payload["manifest_sha256"]) == 64
    calls = []
    original_load = torch.load

    def checked_load(*args, **kwargs):
        calls.append(kwargs)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", checked_load)
    rng_before = torch.get_rng_state().clone()
    restored = load_ranker_checkpoint(path, expected_config=model.config)
    assert torch.equal(rng_before, torch.get_rng_state())
    assert calls == [dict(map_location="cpu", weights_only=True)]
    assert not restored.model.training
    assert restored.training_metadata == metadata
    restored.training_metadata["status"] = "trained"
    assert restored.training_metadata["status"] == "untrained"
    torch.testing.assert_close(restored.model(graph), expected, rtol=0, atol=0)


@pytest.mark.parametrize("field,value", [
    ("checkpoint_version", "future"), ("numeric_feature_version", "graph_v1"),
    ("edge_feature_version", "graph_v1"),
    ("numeric_feature_version", "graph_v2"), ("edge_feature_version", "graph_v2"),
    ("architecture", {"version": "different", "numeric_dim": 54}),
])
def test_incompatible_checkpoint_versions_and_dimensions_rejected(tmp_path, field, value):
    path = tmp_path / "model.pt"
    save_ranker_checkpoint(path, LayoutRanker(), training_metadata=provenance())
    rewrite_manifest(path, lambda manifest: manifest.update({field: value}))
    with pytest.raises(CheckpointUnavailable, match="incompatible"):
        load_ranker_checkpoint(path)


@pytest.mark.parametrize("problem", ["missing", "corrupt", "manifest_hash", "weights_hash", "nan_weights", "wrong_dtype", "missing_weight"])
def test_unavailable_or_damaged_checkpoint_never_returns_random_model(tmp_path, problem, monkeypatch):
    path = tmp_path / "model.pt"
    model = LayoutRanker()
    if problem == "missing_weight":
        state = model.state_dict()
        state.pop("head.3.bias")
        monkeypatch.setattr(model, "state_dict", lambda: state)
    save_ranker_checkpoint(path, model, training_metadata=provenance())
    if problem == "missing":
        path = tmp_path / "absent.pt"
    elif problem == "corrupt":
        path.write_bytes(b"not a torch checkpoint")
    elif problem != "missing_weight":
        payload = torch.load(path, weights_only=True)
        if problem == "manifest_hash":
            payload["manifest"]["config"]["dropout"] = 0.2
        elif problem == "weights_hash":
            payload["state_dict"]["head.3.bias"] += 1
        elif problem == "nan_weights":
            payload["state_dict"]["head.3.bias"][:] = float("nan")
        else:
            payload["state_dict"]["head.3.bias"] = payload["state_dict"]["head.3.bias"].double()
        torch.save(payload, path)
    with pytest.raises(CheckpointUnavailable):
        load_ranker_checkpoint(path)


def test_expected_checkpoint_encoder_and_topology_are_enforced(tmp_path):
    path = tmp_path / "model.pt"
    save_ranker_checkpoint(path, LayoutRanker(), training_metadata=provenance())
    for config in (RankerConfig(text_encoder_version="other"), RankerConfig(graph_topologies=("sparse_v1",))):
        with pytest.raises(CheckpointUnavailable, match="expected"):
            load_ranker_checkpoint(path, expected_config=config)


@pytest.mark.parametrize("problem", ["missing_config", "missing_runtime", "different_pyg"])
def test_checkpoint_never_fills_missing_contract_settings(tmp_path, problem):
    path = tmp_path / "model.pt"
    save_ranker_checkpoint(path, LayoutRanker(), training_metadata=provenance())

    def change(manifest):
        if problem == "missing_config":
            manifest["config"].pop("text_encoder_version")
        elif problem == "missing_runtime":
            manifest["runtime_versions"].pop("torch")
        else:
            manifest["runtime_versions"]["pyg"] = "unknown-version"

    rewrite_manifest(path, change)
    with pytest.raises(CheckpointUnavailable):
        load_ranker_checkpoint(path)


@pytest.mark.parametrize("problem", ["overlap", "duplicate", "empty_train", "empty_metrics", "nan_metric", "missing_seed", "bad_status"])
def test_checkpoint_training_metadata_must_be_explicit_and_source_disjoint(tmp_path, problem):
    metadata = provenance(True)
    if problem == "overlap":
        metadata["source_splits"]["test"] = metadata["source_splits"]["train"]
    elif problem == "duplicate":
        metadata["source_splits"]["train"] *= 2
    elif problem == "empty_train":
        metadata["source_splits"]["train"] = []
    elif problem == "empty_metrics":
        metadata["metrics"] = {}
    elif problem == "nan_metric":
        metadata["metrics"]["bad"] = float("nan")
    elif problem == "missing_seed":
        metadata.pop("seed")
    else:
        metadata["status"] = "maybe"
    with pytest.raises(ValueError):
        save_ranker_checkpoint(tmp_path / "bad.pt", LayoutRanker(), training_metadata=metadata)
    assert not (tmp_path / "bad.pt").exists()


def test_checkpoint_loader_also_rejects_source_leakage(tmp_path):
    path = tmp_path / "model.pt"
    save_ranker_checkpoint(path, LayoutRanker(), training_metadata=provenance(True))
    rewrite_manifest(path, lambda m: m["training"]["source_splits"].update(test=["fixture-train"]))
    with pytest.raises(CheckpointUnavailable, match="disjoint"):
        load_ranker_checkpoint(path)


def test_score_candidates_needs_trained_checkpoint_and_empty_list_is_empty(tmp_path):
    source = source_design()
    assert score_candidates(source, [], source.canvas) == []
    for checkpoint in (None, checkpoint_for(tmp_path, trained=False)):
        with pytest.raises(CheckpointUnavailable, match="trained"):
            score_candidates(source, [source], source.canvas, checkpoint=checkpoint)


def test_scoring_preserves_order_designs_chunking_and_model_mode(tmp_path):
    source = source_design()
    candidates = [candidate_design(source, s) for s in (9, 0, 3)]
    original = source.to_adobe(), [c.to_adobe() for c in candidates]
    checkpoint = checkpoint_for(tmp_path)
    expected = [checkpoint.model(build_candidate_graph(source, c, source.canvas)).item() for c in candidates]
    checkpoint.model.train()
    one_at_a_time = score_candidates(source, candidates, source.canvas, checkpoint=checkpoint, batch_size=1)
    batched = score_candidates(source, candidates, source.canvas, checkpoint=checkpoint, batch_size=2)
    assert checkpoint.model.training
    assert batched == pytest.approx(expected, abs=2e-6)
    assert one_at_a_time == pytest.approx(expected, abs=2e-6)
    assert original == (source.to_adobe(), [c.to_adobe() for c in candidates])
    assert all(isinstance(score, float) for score in batched)
    assert all(p.grad is None for p in checkpoint.model.parameters())

    @dataclass
    class TeammateCandidate:
        design: Design
        candidate_id: str

    wrapped = [TeammateCandidate(c, str(i)) for i, c in enumerate(candidates)]
    assert score_candidates(source, wrapped, source.canvas, checkpoint=checkpoint) == pytest.approx(expected, abs=2e-6)


def test_scoring_encoder_compatibility_failure_is_explicit(tmp_path):
    source = source_design()
    cache = TextEmbeddingCache(TestEncoder())
    checkpoint = checkpoint_for(tmp_path, LayoutRanker(RankerConfig(text_encoder_version=cache.encoder_version)))
    with pytest.raises(ModelInferenceError, match="encoder"):
        score_candidates(source, [source], source.canvas, checkpoint=checkpoint)
    values = score_candidates(source, [source], source.canvas, checkpoint=checkpoint, text_embeddings=cache)
    assert len(values) == 1

    class UnavailableEncoder(TestEncoder):
        def encode(self, content):
            return None

    with pytest.raises(ModelInferenceError, match="complete"):
        score_candidates(source, [source], source.canvas, checkpoint=checkpoint,
                         text_embeddings=TextEmbeddingCache(UnavailableEncoder()))


@pytest.mark.parametrize("problem", ["candidate_ids", "exception", "nan", "wrong_shape"])
def test_scoring_failures_are_explicit_and_restore_model_mode(tmp_path, monkeypatch, problem):
    source = source_design()
    checkpoint = checkpoint_for(tmp_path)
    checkpoint.model.train()
    candidate = source
    if problem == "candidate_ids":
        candidate = source_design(2)
    elif problem == "exception":
        def fail(_graph):
            raise RuntimeError("test inference failure")
        monkeypatch.setattr(checkpoint.model, "forward", fail)
    elif problem == "nan":
        monkeypatch.setattr(checkpoint.model, "forward", lambda _g: torch.tensor([float("nan")]))
    else:
        monkeypatch.setattr(checkpoint.model, "forward", lambda _g: torch.tensor(0.1))
    with pytest.raises(ModelInferenceError):
        score_candidates(source, [candidate], source.canvas, checkpoint=checkpoint)
    assert checkpoint.model.training


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_forward_backward_and_cpu_eval_agree():
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        cpu_model = LayoutRanker().eval()
        batch = Batch.from_data_list([graph_for(1), graph_for(4), graph_for(17)])
        cpu_scores = cpu_model(batch).detach()
        gpu_model = deepcopy(cpu_model).cuda()
        gpu_scores = gpu_model(batch.clone().to("cuda"))
        assert gpu_scores.shape == (3,) and torch.isfinite(gpu_scores).all()
        torch.testing.assert_close(gpu_scores.cpu(), cpu_scores, rtol=1e-4, atol=2e-5)
        gpu_scores.square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in gpu_model.parameters())


def test_python_310_syntax():
    for path in (ROOT / "engine" / "ranker.py", Path(__file__)):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 10))
