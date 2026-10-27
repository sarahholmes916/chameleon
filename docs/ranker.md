# S03 GATv2 ranker foundation

`engine/ranker.py` implements the scoring network in S03 section 5, consuming
the [graph_v1 tensors](graph_features.md) and the existing S02 models. It also
provides checkpoint persistence and an explicit candidate-scoring boundary.
This is an implemented, untrained architecture. No human preference dataset,
trained quality checkpoint, or measured aesthetic improvement is supplied.

## Model and versions

```python
from torch_geometric.data import Batch
from engine.ranker import LayoutRanker, RankerConfig

model = LayoutRanker(RankerConfig()).eval()
scores = model(Batch.from_data_list(graphs))  # float32 [number_of_graphs]
```

Constructing `LayoutRanker` initializes random weights for experiments. It does
not load a checkpoint. Its raw output is useful for tensor/gradient tests until
training and evaluation establish useful ranking behavior. A single `Data`
returns shape `[1]`; a `Batch` returns `[B]` in graph order, without squeezing
away the singleton batch dimension. Each scalar compares candidates under the
same source and target; it is not a quality percentage or calibrated probability.

The architecture identifier is `gatv2_ranker_v1`. Numeric and edge features
retain `graph_v1`; no feature dimensions or order changed.

| Stage | Contract |
| --- | --- |
| Text | Detached `[N,384]` input, trainable `Linear(384,32)` + ReLU, multiplied by availability mask |
| Concatenation | Numeric 53 + projected text 32 + mask 1 = 86 |
| GAT stage 1 | `GATv2Conv(86,32,heads=4,concat=True,edge_dim=31)` plus `Linear(86,128)` skip |
| GAT stages 2–3 | `GATv2Conv(128,32,heads=4,concat=True,edge_dim=31)` plus identity skips |
| Each stage | Add skip, per-node `LayerNorm(128)`, ELU; dropout after stages 1 and 2 only |
| Pooling | Per-graph mean then max concatenated to 256 |
| Score head | `Linear(256,128)`, ReLU, dropout, `Linear(128,1)` |

Feature/head dropout defaults to 0.10. Attention dropout is explicitly 0.0;
source/target projections are separate. Each GAT layer adds self loops with
zero edge attributes (`add_self_loops=True`, `fill_value=0.0`). PyG's built-in
residual is disabled because the skip connections above are explicit. The
graph's edge arrays remain unchanged, including a singleton's empty edge list.

Multiplying the text projection by its mask makes missing/non-text projections
exactly zero even if the projection learns a bias. This specifies the previously
unspecified mask application in S03 sections 4.2/5; it is recorded in the ranker
architecture version and does not change graph_v1. The pretrained encoder is
external and frozen; gradients train the projection and GNN, not its vectors.

## Configuration and compatibility

`RankerConfig` is a frozen dataclass with:

- `dropout=0.10`, finite and in `[0,1)`.
- `text_encoder_version="none"` for numeric-only experiments, or an exact
  encoder/revision/preprocessing identity matching the embedding cache.
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

## Checkpoints

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
validates provenance structure, not dataset contents or quality. A future S05
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

## Candidate-scoring handoff

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

The three positional arguments follow the proposed S03/S06 interface; the
checkpoint/cache are explicit keyword-only dependencies instead of global state.
Candidates may be S02 `Design` objects for tests or future `LayoutCandidate`
objects exposing `.design` through the `CandidateWithDesign` protocol. No second
candidate model or parser is introduced. Scores preserve candidate input order
across batches. Source/candidate designs and their JSON are unchanged.

An empty candidate sequence returns `[]`. Nonempty scoring requires a checkpoint
declared trained, or raises `CheckpointUnavailable`. Encoder mismatch, unsupported
topology, malformed candidates, runtime errors, and non-finite outputs raise
`ModelInferenceError`; no partial score list is returned. Inference runs with
gradients disabled and dropout off, restoring the model's previous training mode.
The model is a mutable PyTorch object; do not concurrently train it while scoring.

The caller owns S04 validity checks, deterministic fallback, sorting/selection,
and S06 integration. This module checks graph identity compatibility but does
not certify candidate legality. `engine/resize.py`, the baseline dependencies,
S02, and rendering remain unchanged and do not import the ML stack.

## Verification and remaining work

Use the same optional dependencies as M2A, with PyG pinned to `2.8.1` in
`requirements-graph.txt`; no additional model packages or weights are required.

```text
.venv\Scripts\python.exe -m pytest -q tests/test_ranker.py
.venv\Scripts\python.exe -m pytest -q
```

Tests exercise all three Adobe samples, singleton and sparse graphs, batch
isolation, permutation invariance, candidate/edge gradients, trainable projection
and frozen input vectors, exact pooling order, dropout, checkpoint round trips,
corruption/version rejection, missing embeddings, candidate wrappers, and explicit
failure behavior. Tests use generated fixtures and synthetic metadata solely to
check mechanics; none represent human preference training or validation results.

CPU and CUDA forward/backward checks execute with PyTorch `2.11.0+cu128` and
PyG `2.8.1` on Python `3.11.17`. CUDA tests skip explicitly where CUDA is absent.
The optional `torch-scatter` acceleration package is not installed; the tested
PyG built-in scatter path works. Python 3.10 syntax is checked, but a Python 3.10
runtime and the competition's Apple hardware have not been tested.

Next work is S05's source-disjoint preference dataset and training/evaluation
pipeline, including the pooled MLP baseline. Reliable labels still depend on
the teammate's validator and faithful renderer. No training pipeline, human
labels, evaluated checkpoint, fallback scorer, or resize integration is included.

Official API references:
[GATv2Conv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATv2Conv.html),
[per-graph pooling](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.pool.global_mean_pool.html),
and [PyTorch 2.11 checkpoint loading](https://docs.pytorch.org/docs/2.11/generated/torch.load.html).
