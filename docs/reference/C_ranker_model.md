# Appendix C — S03 GATv2 ranker model, checkpoints and scoring behavior

> **Appendix C of [TECHNICAL_SPEC.md](../TECHNICAL_SPEC.md).** Authoritative detail for the
> `gatv2_ranker_v3` (default, `graph_v3`) and `gatv2_ranker_v4` (`graph_v4`) architectures, the
> `ranker_checkpoint_v1` format and scoring failure behavior. The
> `score_candidates` signature and its place in the pipeline are defined in the main specification
> (§2, §6). Input tensors are defined in [Appendix B](B_graph_features.md).

Code: [engine/ranker.py](../../engine/ranker.py). Trained checkpoints now exist (`data/creative_pilot_v1/training_v1/`; the shipped one is
`engine/checkpoints/ranker_synth_weight_0.25_best.pt`, graph_v3, numeric-only), trained with `training/`. There is still no measured
aesthetic improvement: a blind comparison showed no difference from the deterministic objective (`MODEL_SELECTION.md`).

## C.1 Model and versions

```python
from torch_geometric.data import Batch
from engine.ranker import LayoutRanker, RankerConfig

model = LayoutRanker(RankerConfig()).eval()
scores = model(Batch.from_data_list(graphs))  # float32 [number_of_graphs]
```

Constructing `LayoutRanker` initializes random weights for experiments and does not load a checkpoint.
A single `Data`
returns shape `[1]`; a `Batch` returns `[B]` in graph order, without squeezing
away the singleton batch dimension.

The architecture identifier is `gatv2_ranker_v3`; numeric and edge features use `graph_v3`
(numeric width 57, edge width 31, first GAT layer and residual input width 90). Earlier
`graph_v1`/`graph_v2` tensors and `gatv2_ranker_v1`/`v2` checkpoints are incompatible and rejected, not
padded. The `ranker_checkpoint_v1` container format is unchanged by those migrations; its architecture and
feature manifests enforce them.

`RankerConfig(feature_version="graph_v4")` selects `gatv2_ranker_v4`: numeric width 58, edge width 32, first GAT
layer and residual input width 91 (`58 + 32 + 1`). Everything else is identical. The default is `graph_v3`, and the
diagram and numbers below describe it. A model accepts only graphs whose `context_version` equals its
`feature_version`.

Architecture (PyG `GATv2Conv`, pinned to `2.8.1` in `requirements-graph.txt`):

```text
numeric fields: [N,57] -----┐
text [N,384] -> Linear 32 ---+--> concat [N,90]
mask [N,1] ------------------┘
       GATv2Conv(90 -> 128, 4 heads x 32, edge_dim=31)
       LayerNorm(128) + ELU + dropout(0.10)
       GATv2Conv(128 -> 128, 4 heads x 32, edge_dim=31)
       LayerNorm(128) + ELU + dropout(0.10)
       GATv2Conv(128 -> 128, 4 heads x 32, edge_dim=31)
       LayerNorm(128) + ELU
       concatenate global_mean_pool and global_max_pool -> [graph,256]
       MLP Linear(256,128) -> ReLU -> Dropout(.10) -> Linear(128,1)
       -> one uncalibrated scalar S(source,candidate,target)
```

Details fixed by the implementation: the text projection is `Linear(384,32)` + ReLU and is multiplied by
the presence mask, so missing or non-text projections are exactly zero even if the bias is nonzero; the
pretrained encoder is external and frozen, so gradients train the projection and GNN only. Each stage
adds an explicit skip (`Linear(90,128)` on the first stage, identity on the others), then per-node
`LayerNorm(128)` and ELU; dropout follows stages 1 and 2 only. Each `GATv2Conv` uses `heads=4`,
`concat=True`, `edge_dim=31`, attention dropout 0.0, `add_self_loops=True`, `fill_value=0.0` (self-loop
edge attributes are zero), PyG's built-in residual disabled, and separate source/target projections.

Feature/head dropout defaults to 0.10. The graph's edge arrays are unchanged by the model, including a
singleton's empty edge list.

The score is meaningful for **ranking candidates under the same source and target**, not as a universal design-quality percentage or calibrated probability. A score of 0.9 does not mean "90% good." Attention is not an importance explanation.

## C.2 Configuration and compatibility

`RankerConfig` is a frozen dataclass with:

- `dropout=0.10`, finite and in `[0,1)`.
- `text_encoder_version="none"` for numeric-only experiments, or an exact
  encoder/revision/preprocessing identity matching the embedding cache.
- `feature_version="graph_v3"` by default; `"graph_v4"` selects the v4 contract and architecture. Any other value raises `ValueError`.
- `graph_topologies=("dense_v1",)` by default. Include `"sparse_v1"` explicitly
  for experiments covering larger designs. Declaring support does not itself
  demonstrate that training data covered that topology.

The model checks shapes, float32/int64 dtypes, device agreement, finite values,
edge bounds, batch isolation, graph versions, topology, and text availability
before message passing. Numeric-only inputs require zero text vectors/masks
and zero semantic edge features. Text-enabled inputs require the exact encoder
and complete embeddings for all text nodes; no-text designs remain supported.
Partial/missing embeddings are rejected rather than converted into a numeric-only
model. Graph metadata must agree with masks, including within a mixed batch.

LayerNorm acts per node, and pooling acts per graph. Evaluation mode disables
dropout. CUDA reductions may differ slightly from CPU due to floating-point
reduction order; tests compare them with explicit tolerances, not bit identity.
No attention explanations or node-importance predictions are exposed.

## C.3 Checkpoints

```python
from engine.ranker import save_ranker_checkpoint, load_ranker_checkpoint

# Supply actual experiment provenance; this example records an untrained model.
save_ranker_checkpoint("experiment.pt", model, training_metadata={
    "status": "untrained",
    "source_splits": {"train": [], "validation": [], "test": []},
    "seed": 42,
    "metrics": {},
})
checkpoint = load_ranker_checkpoint(
    "experiment.pt", expected_config=model.config, device="cpu",
)
```

`ranker_checkpoint_v1` stores an exact architecture manifest, config, separate
numeric/edge versions, text-encoder identity, topology declarations, CPU state
tensors, training metadata, and Python/Torch/PyG versions. SHA-256 hashes cover
the manifest (including source splits and metrics) and sorted weight names,
shapes, dtypes, and bytes. These hashes detect corruption, not authenticity or
whether a claimed training run occurred.

Metadata must explicitly contain `status`, `source_splits`, `seed`, and `metrics`.
Split source IDs must be unique and disjoint across train/validation/test. A
`trained` declaration requires nonempty train and validation source lists and
finite recorded metrics; the test split can be empty during pilot work. This
validates provenance structure, not dataset contents or quality. A future training
loader must verify comparisons against those actual source manifests.

Loading uses `torch.load(..., weights_only=True, map_location="cpu")`, validates
hashes and all required config fields, strictly loads state keys/shapes/dtypes,
then moves the restored model to the requested device in evaluation mode. PyG
runtime version must match the saved runtime; Torch/Python versions are recorded
for reproducibility. An optional `expected_config` also enforces the caller's
expected encoder, topology, and dropout. Loading preserves the caller's CPU RNG.
Untrained artifacts may be restored for experiments; the scoring API rejects them.

Missing/corrupt/incompatible artifacts raise `CheckpointUnavailable` with their
underlying diagnostic. There is no replacement with random weights on failure.

The feature version is recorded as `numeric_feature_version` and `edge_feature_version`, not inside `config`, so
`graph_v3` manifests keep exactly their original layout; loading reads the version from those fields and rejects
unsupported or mismatched values and any architecture that does not match that version. A `graph_v4` architecture
manifest additionally records `text_fit_estimator` and `text_fit_glyph_advance_em`, so a checkpoint is rejected if the
text-fit estimate changes. `expected_config` also enforces `feature_version`. `score_candidates` builds each candidate
graph in the loaded model's own feature version.

## C.4 Candidate-scoring behavior

```python
from engine.ranker import score_candidates

# Requires a compatible checkpoint with actual trained provenance.
scores = score_candidates(
    source, valid_candidates, target,
    checkpoint=checkpoint,
    text_embeddings=None,  # Or the exact TextEmbeddingCache for a text model.
    batch_size=32,
)
```

The checkpoint and cache are explicit keyword-only dependencies instead of global state. Candidates
may be S02 `Design` objects or objects exposing `.design` (the `CandidateWithDesign` protocol, which a
future `LayoutCandidate` satisfies). Scores preserve candidate input order
across batches. Source/candidate designs and their JSON are unchanged.

An empty candidate sequence returns `[]`. Nonempty scoring requires a checkpoint
declared trained, or raises `CheckpointUnavailable`. Encoder mismatch, unsupported
topology, malformed candidates, runtime errors, and non-finite outputs raise
`ModelInferenceError`; no partial score list is returned. Inference runs with
gradients disabled and dropout off, restoring the model's previous training mode.
The model is a mutable PyTorch object; do not concurrently train it while scoring.

The caller owns validity filtering, deterministic fallback, sorting/selection and integration (spec §6, §7,
§8). This module checks graph identity compatibility but does not certify candidate legality.

## C.5 Coverage

Tests exercise all three Adobe samples, singleton and sparse graphs, batch
isolation, permutation invariance, candidate/edge gradients, trainable projection
and frozen input vectors, exact pooling order, dropout, checkpoint round trips,
corruption/version rejection, missing embeddings, candidate wrappers, and explicit
failure behavior. Tests use generated fixtures and synthetic metadata solely to
check mechanics; none represent human preference training or validation results.
