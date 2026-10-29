"""S03 whole-candidate GATv2 scores and explicit checkpoint/inference boundaries.

Raw LayoutRanker instances are untrained neural networks. Only checkpoints
declared trained may enter score_candidates; quality still needs human evaluation.
This module never generates, validates, renders, or changes Adobe layouts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
from typing import Any, Protocol

import torch
from torch import nn
from torch.nn import functional as F
import torch_geometric
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool

from engine.graph_features import (
    FEATURE_VERSION_V3, FEATURE_VERSION_V4, SUPPORTED_FEATURE_VERSIONS, build_candidate_graph,
    feature_dimensions,
)
from engine.internal_schema import Canvas, Design
from engine.text_embeddings import TEXT_DIMENSION, TextEmbeddingCache
from engine.text_fit import GLYPH_ADVANCE_EM, TEXT_FIT_ESTIMATOR_VERSION


RANKER_VERSION = "gatv2_ranker_v3"  # the default (graph_v3) architecture
RANKER_VERSIONS = {FEATURE_VERSION_V3: "gatv2_ranker_v3", FEATURE_VERSION_V4: "gatv2_ranker_v4"}
CHECKPOINT_VERSION = "ranker_checkpoint_v1"


def _architecture(feature_version: str = FEATURE_VERSION_V3) -> dict[str, Any]:
    numeric_dim, edge_dim = feature_dimensions(feature_version)
    architecture = dict(
        version=RANKER_VERSIONS[feature_version], numeric_dim=numeric_dim, edge_dim=edge_dim,
        text_dim=TEXT_DIMENSION,
        text_projection_dim=32, hidden_dim=128, heads=4, layers=3,
        pooling=["mean", "max"], mlp_hidden_dim=128,
        residual="linear_first_identity_rest", normalization="node_layer_norm",
        masked_text_projection=True, attention_dropout=0.0,
        add_self_loops=True, fill_value=0.0, pyg_version="2.8.1",
    )
    if feature_version == FEATURE_VERSION_V4:
        # graph_v4 feature values depend on the text-fit estimate; record it so a changed
        # estimator cannot silently reuse a checkpoint trained on the old values.
        architecture.update(text_fit_estimator=TEXT_FIT_ESTIMATOR_VERSION,
                            text_fit_glyph_advance_em=float(GLYPH_ADVANCE_EM))
    return architecture


@dataclass(frozen=True, slots=True)
class RankerConfig:
    """Versioned architecture's configurable dropout and input compatibility.

    Default: numeric-only, dense topology. Declare sparse topology explicitly
    when preparing experiments that train/evaluate on it as well.
    """

    dropout: float = 0.10
    text_encoder_version: str = "none"
    graph_topologies: tuple[str, ...] = ("dense_v1",)
    feature_version: str = FEATURE_VERSION_V3

    def __post_init__(self) -> None:
        if (isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float))
                or not math.isfinite(self.dropout) or not 0 <= self.dropout < 1):
            raise ValueError("dropout must be finite and in [0, 1)")
        if not isinstance(self.text_encoder_version, str) or not self.text_encoder_version:
            raise ValueError("text_encoder_version must be a nonempty identity or 'none'")
        if not isinstance(self.graph_topologies, (tuple, list)):
            raise ValueError("graph_topologies must be a sequence of topology identifiers")
        topologies = tuple(self.graph_topologies)
        if (not topologies or any(t not in ("dense_v1", "sparse_v1") for t in topologies)
                or len(set(topologies)) != len(topologies)):
            raise ValueError("graph_topologies must contain distinct supported identifiers")
        object.__setattr__(self, "graph_topologies", topologies)
        if self.feature_version not in SUPPORTED_FEATURE_VERSIONS:
            raise ValueError(f"feature_version must be one of {SUPPORTED_FEATURE_VERSIONS}")


def _metadata_values(graph: Data, name: str, count: int) -> list[str]:
    value = getattr(graph, name, None)
    values = [value] if isinstance(value, str) else value
    if (not isinstance(values, (tuple, list)) or len(values) != count
            or any(not isinstance(item, str) for item in values)):
        raise ValueError(f"{name} must contain one string per graph")
    return list(values)


class LayoutRanker(nn.Module):
    """The network of docs/reference/C_ranker_model.md; forward(Data | Batch) returns scores[B].

    Scores are uncalibrated and comparable only within a source/target context.
    No encoder is owned here; embedding inputs are detached, projection trains.
    """

    def __init__(self, config: RankerConfig | None = None) -> None:
        super().__init__()
        self.config = config or RankerConfig()
        self.node_dimension, self.edge_dimension = feature_dimensions(self.config.feature_version)
        self.text_projection = nn.Linear(TEXT_DIMENSION, 32)
        self.convs = nn.ModuleList([
            GATv2Conv(width, 32, heads=4, concat=True, edge_dim=self.edge_dimension,
                      dropout=0.0, add_self_loops=True, fill_value=0.0,
                      residual=False, share_weights=False)
            for width in (self.node_dimension + 32 + 1, 128, 128)
        ])
        self.skips = nn.ModuleList([nn.Linear(self.node_dimension + 32 + 1, 128), nn.Identity(), nn.Identity()])
        self.norms = nn.ModuleList([nn.LayerNorm(128) for _ in range(3)])
        self.feature_dropout = nn.Dropout(self.config.dropout)
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(self.config.dropout), nn.Linear(128, 1),
        )

    def _validate_graph(self, graph: Data) -> tuple[torch.Tensor, int]:
        if not isinstance(graph, Data):
            raise ValueError("expected a PyG Data or Batch")
        x = getattr(graph, "x_numeric", None)
        if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[0] < 1:
            raise ValueError("x_numeric must describe at least one node")
        n = x.shape[0]
        edges = getattr(graph, "edge_index", None)
        if (not isinstance(edges, torch.Tensor) or edges.dtype != torch.int64
                or edges.ndim != 2 or edges.shape[0] != 2):
            raise ValueError("edge_index must be int64 with shape [2, E]")
        device = self.text_projection.weight.device
        if edges.device != device:
            raise ValueError("graph tensors and model must be on the same device")
        for name, shape in (
            ("x_numeric", (n, self.node_dimension)), ("text_vec", (n, TEXT_DIMENSION)),
            ("text_mask", (n, 1)), ("edge_attr", (edges.shape[1], self.edge_dimension)),
        ):
            value = getattr(graph, name, None)
            if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                    or value.dtype != torch.float32):
                raise ValueError(f"{name} must be float32 with shape {shape}")
            if value.device != device:
                raise ValueError("graph tensors and model must be on the same device")
            if not torch.isfinite(value).all().item():
                raise ValueError(f"{name} must be finite")
        if edges.numel() and (edges.min().item() < 0 or edges.max().item() >= n):
            raise ValueError("edge_index refers to a missing node")
        if isinstance(graph, Batch):
            batch = graph.batch
            count = graph.num_graphs
            if (not isinstance(batch, torch.Tensor) or batch.shape != (n,)
                    or batch.dtype != torch.int64 or batch.device != device
                    or count < 1 or not torch.equal(
                        batch.unique(sorted=True), torch.arange(count, device=device))):
                raise ValueError("batch must assign nodes to nonempty contiguous graph IDs")
            if not torch.equal(batch[edges[0]], batch[edges[1]]):
                raise ValueError("edges must not cross graphs in a batch")
        else:
            batch = torch.zeros(n, dtype=torch.int64, device=device)
            count = 1
        if any(v != self.config.feature_version for v in _metadata_values(graph, "context_version", count)):
            raise ValueError("incompatible graph feature version")
        if any(v not in self.config.graph_topologies for v in _metadata_values(graph, "graph_topology", count)):
            raise ValueError("graph topology is not declared by this ranker")
        if any(v != self.config.text_encoder_version for v in _metadata_values(graph, "text_encoder_version", count)):
            raise ValueError("incompatible text encoder version")

        mask = graph.text_mask[:, 0]
        is_text = x[:, 0] == 1
        if not ((mask == 0) | (mask == 1)).all().item() or (mask[~is_text] != 0).any().item():
            raise ValueError("text masks must be binary and available only for text nodes")
        if (graph.text_vec[mask == 0] != 0).any().item():
            raise ValueError("unavailable text vectors must be zero")
        if (graph.text_vec[mask == 1].abs().sum(dim=1) == 0).any().item():
            raise ValueError("available text vectors must be nonzero")
        if self.config.text_encoder_version == "none":
            if mask.any().item():
                raise ValueError("numeric-only ranker requires absent text embeddings")
        elif not (mask[is_text] == 1).all().item():
            raise ValueError("text-enabled ranker requires complete text embeddings")
        expected_semantic = mask[edges[0]] * mask[edges[1]]
        if (not torch.equal(graph.edge_attr[:, 18], expected_semantic)
                or (graph.edge_attr[expected_semantic == 0, 17] != 0).any().item()):
            raise ValueError("semantic edge availability must agree with text masks")
        statuses = _metadata_values(graph, "text_embedding_status", count)
        for i, status in enumerate(statuses):
            text_count = int(is_text[batch == i].sum().item())
            available = int(mask[batch == i].sum().item())
            expected = ("no_text" if text_count == 0 else "none" if available == 0
                        else "complete" if available == text_count else "partial")
            if status != expected:
                raise ValueError("text_embedding_status disagrees with node masks")
        return batch, count

    def forward(self, graph: Data) -> torch.Tensor:
        batch, count = self._validate_graph(graph)
        mask = graph.text_mask.detach()
        projected = F.relu(self.text_projection(graph.text_vec.detach())) * mask
        h = torch.cat((graph.x_numeric, projected, mask), dim=1)
        for i, (conv, skip, norm) in enumerate(zip(self.convs, self.skips, self.norms)):
            h = F.elu(norm(conv(h, graph.edge_index, graph.edge_attr) + skip(h)))
            if i < 2:
                h = self.feature_dropout(h)
        pooled = torch.cat((global_mean_pool(h, batch, size=count),
                            global_max_pool(h, batch, size=count)), dim=1)
        scores = self.head(pooled).squeeze(-1)
        if not torch.isfinite(scores).all().item():
            raise ValueError("ranker produced non-finite scores")
        return scores


class CheckpointUnavailable(RuntimeError):
    """Missing, untrained, corrupt, or incompatible ranker; caller uses fallback."""


class ModelInferenceError(RuntimeError):
    """Candidate scoring failed; caller must use its deterministic fallback."""


def _json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _training_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    # Copy through JSON; no arbitrary Python objects are saved in checkpoints.
    data = json.loads(_json_text(dict(value)))
    if set(data) != {"status", "source_splits", "seed", "metrics"}:
        raise ValueError("training metadata requires status, source_splits, seed, metrics")
    if data["status"] not in ("untrained", "trained"):
        raise ValueError("training status must be untrained or trained")
    if type(data["seed"]) is not int:
        raise ValueError("training seed must be an integer")
    splits = data["source_splits"]
    if not isinstance(splits, dict) or set(splits) != {"train", "validation", "test"}:
        raise ValueError("source_splits must declare train, validation, and test source IDs")
    seen: set[str] = set()
    for ids in splits.values():
        if (not isinstance(ids, list) or any(not isinstance(v, str) or not v for v in ids)
                or len(set(ids)) != len(ids) or seen.intersection(ids)):
            raise ValueError("source split IDs must be unique and disjoint")
        seen.update(ids)
    metrics = data["metrics"]
    if (not isinstance(metrics, dict) or any(
            not isinstance(k, str) or not k or isinstance(v, bool)
            or not isinstance(v, (int, float)) or not math.isfinite(v)
            for k, v in metrics.items())):
        raise ValueError("metrics must map names to finite numbers")
    if data["status"] == "trained" and (not splits["train"] or not splits["validation"] or not metrics):
        raise ValueError("trained checkpoints require train/validation sources and recorded metrics")
    return data


def _state_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = sha256()
    for name, tensor in sorted(state.items()):
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32:
            raise ValueError("ranker state must contain float32 tensors")
        if not torch.isfinite(tensor).all().item():
            raise ValueError("ranker state must be finite")
        digest.update(_json_text([name, str(tensor.dtype), list(tensor.shape)]).encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class RankerCheckpoint:
    model: LayoutRanker
    training_metadata_json: str
    weights_sha256: str

    @property
    def training_metadata(self) -> dict[str, Any]:
        """Return a detached metadata copy; the model remains a normal nn.Module."""
        return json.loads(self.training_metadata_json)


def save_ranker_checkpoint(
    path: str | Path, model: LayoutRanker, *, training_metadata: Mapping[str, Any],
) -> None:
    """Save explicit provenance and CPU state; never infer that a model is trained."""
    training = _training_metadata(training_metadata)
    state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    # feature_version is recorded as numeric_/edge_feature_version, not duplicated inside config,
    # so a graph_v3 manifest keeps exactly its original layout.
    config = {k: v for k, v in asdict(model.config).items() if k != "feature_version"}
    manifest = dict(
        checkpoint_version=CHECKPOINT_VERSION, architecture=_architecture(model.config.feature_version),
        numeric_feature_version=model.config.feature_version,
        edge_feature_version=model.config.feature_version,
        config=config, training=training,
        runtime_versions=dict(python=platform.python_version(), torch=str(torch.__version__),
                              pyg=str(torch_geometric.__version__)),
        weights_sha256=_state_hash(state),
    )
    # JSON canonicalization also removes tuples and TorchVersion subclasses.
    manifest = json.loads(_json_text(manifest))
    torch.save(dict(manifest=manifest, manifest_sha256=sha256(_json_text(manifest).encode()).hexdigest(),
                    state_dict=state), path)


def load_ranker_checkpoint(
    path: str | Path, *, expected_config: RankerConfig | None = None,
    device: str | torch.device = "cpu",
) -> RankerCheckpoint:
    """Load tensors with weights_only=True and reject incompatible manifests/state.

    Untrained artifacts may be restored for experiments, but score_candidates
    rejects them. Loading does not consume the caller's CPU initialization RNG.
    """
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or set(payload) != {"manifest", "manifest_sha256", "state_dict"}:
            raise ValueError("invalid checkpoint envelope")
        manifest = payload["manifest"]
        if sha256(_json_text(manifest).encode()).hexdigest() != payload["manifest_sha256"]:
            raise ValueError("checkpoint manifest hash mismatch")
        if set(manifest) != {
            "checkpoint_version", "architecture", "numeric_feature_version", "edge_feature_version",
            "config", "training", "runtime_versions", "weights_sha256",
        }:
            raise ValueError("invalid checkpoint manifest fields")
        if manifest["checkpoint_version"] != CHECKPOINT_VERSION:
            raise ValueError("incompatible checkpoint version")
        feature_version = manifest["numeric_feature_version"]
        if (feature_version != manifest["edge_feature_version"]
                or feature_version not in SUPPORTED_FEATURE_VERSIONS):
            raise ValueError("incompatible numeric/edge feature version")
        if manifest["architecture"] != _architecture(feature_version):
            raise ValueError("incompatible ranker architecture/dimensions")
        if (not isinstance(manifest["config"], dict)
                or set(manifest["config"]) != {"dropout", "text_encoder_version", "graph_topologies"}):
            raise ValueError("checkpoint config must explicitly declare all settings")
        runtime = manifest["runtime_versions"]
        if (not isinstance(runtime, dict) or set(runtime) != {"python", "torch", "pyg"}
                or any(not isinstance(v, str) or not v for v in runtime.values())):
            raise ValueError("checkpoint must record Python, Torch, and PyG versions")
        if runtime["pyg"] != str(torch_geometric.__version__):
            raise ValueError("incompatible PyG runtime version")
        config = RankerConfig(**manifest["config"], feature_version=feature_version)
        if expected_config is not None and config != expected_config:
            raise ValueError("checkpoint config does not match expected encoder/topology/dropout")
        training = _training_metadata(manifest["training"])
        state = payload["state_dict"]
        if _state_hash(state) != manifest["weights_sha256"]:
            raise ValueError("checkpoint weight hash mismatch")
        with torch.random.fork_rng(devices=[]):
            model = LayoutRanker(config)
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        return RankerCheckpoint(model, _json_text(training), manifest["weights_sha256"])
    except Exception as exc:
        # This is the external artifact boundary. Never replace a failed load
        # with a randomly initialized ranker; retain the diagnostic as the cause.
        raise CheckpointUnavailable(f"cannot load ranker checkpoint {path}: {exc}") from exc


class CandidateWithDesign(Protocol):
    @property
    def design(self) -> Design: ...


def score_candidates(
    source: Design, candidates: Sequence[Design | CandidateWithDesign], target: Canvas,
    *, checkpoint: RankerCheckpoint | None = None,
    text_embeddings: TextEmbeddingCache | None = None, batch_size: int = 32,
) -> list[float]:
    """Score in input order, using a declared trained checkpoint and complete text.

    The caller supplies candidates already checked by its S04 validator. Accepts
    Design fixtures or future wrappers with .design, without owning that model.
    No sorting, geometry repair, fallback scorer, or Adobe serialization occurs.
    """
    if not candidates:
        return []
    if checkpoint is None or checkpoint.training_metadata["status"] != "trained":
        raise CheckpointUnavailable("candidate ranking requires a trained checkpoint")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    model = checkpoint.model
    was_training = model.training
    try:
        expected_encoder = model.config.text_encoder_version
        actual_encoder = text_embeddings.encoder_version if text_embeddings is not None else "none"
        if expected_encoder != actual_encoder:
            raise ValueError("scoring encoder does not match the checkpoint")
        model.eval()
        scores: list[float] = []
        device = model.text_projection.weight.device
        with torch.inference_mode():
            for start in range(0, len(candidates), batch_size):
                graphs = []
                for candidate in candidates[start:start + batch_size]:
                    design = candidate if isinstance(candidate, Design) else candidate.design
                    graphs.append(build_candidate_graph(
                        source, design, target, text_embeddings=text_embeddings,
                        feature_version=model.config.feature_version))
                batch = Batch.from_data_list(graphs).to(device)
                values = model(batch)
                if values.shape != (len(graphs),) or not torch.isfinite(values).all().item():
                    raise ValueError("ranker must return one finite scalar per candidate")
                scores.extend(values.cpu().tolist())
        return scores
    except Exception as exc:
        raise ModelInferenceError(f"candidate ranking failed: {exc}") from exc
    finally:
        model.train(was_training)
