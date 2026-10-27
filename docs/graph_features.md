# S03 M2A: graph construction and feature extraction

The graph foundation is implemented in `engine/graph_features.py` and
`engine/text_embeddings.py`. It consumes the existing S02 models directly.
It does not generate, validate, rank, render, or train candidate layouts.

## Public graph interface

```python
from engine.graph_features import build_candidate_graph

graph = build_candidate_graph(source, candidate, target)
```

`source` and `candidate` are S02 `Design` instances; `target` is a S02 `Canvas`.
The required three-argument interface from S03 is unchanged. A compatible
keyword-only extension, `text_embeddings=TextEmbeddingCache(...)`, supplies an
optional encoder/cache without global state or model downloads.

Node `i` always represents `source.elements[i]`. Candidate elements are matched
by exact ID, even when the candidate array is reordered. Missing, unexpected,
or duplicate IDs and changed element types raise `ValueError`. Normally S02
already rejects duplicate IDs during parsing. The graph's identity checks are
not a replacement for the future S04 policy validator.

`source.canvas` must equal `source.source_canvas`; `candidate.canvas` must equal
the explicit `target`. Source geometry uses `source.source_canvas`; candidate
geometry uses `target`. The target argument is authoritative, and optional
`Design.target_canvas` context does not override it. Parsing a candidate snapshot
through `Design.from_adobe()` is supported even though that snapshot's own
`source_canvas` describes the target layout. No source data is changed.

## Feature contract and metadata

The feature order and formulas are those of
[S03 sections 3.3 and 4.1](S03_graph_ranker_spec.md#33-exact-31-dimensional-edge-feature-contract).
`NODE_FEATURE_NAMES` and `EDGE_FEATURE_NAMES` list every feature in order;
the read-only `NODE_FEATURE_INDEX` and `EDGE_FEATURE_INDEX` map names to columns.
`FEATURE_CONTRACT_VERSION` is `"graph_v3"`.

| Field | Shape / type |
| --- | --- |
| `x_numeric` | `[N, 57]`, CPU `torch.float32` |
| `text_vec` | `[N, 384]`, CPU `torch.float32` |
| `text_mask` | `[N, 1]`, CPU `torch.float32` |
| `edge_index` | `[2, E]`, CPU `torch.int64` |
| `edge_attr` | `[E, 31]`, CPU `torch.float32` |
| `element_ids` | Tuple of original IDs in source order |
| `num_nodes` | Explicit `N`, including a one-node graph with no edges |
| `context_version` | `"graph_v3"` |
| `graph_topology` | `"dense_v1"` or `"sparse_v1"` |
| `text_encoder_version` | Provider's version identity, or `"none"` |
| `text_embedding_status` | `"complete"`, `"partial"`, `"none"`, or `"no_text"` |

All returned numeric tensors are finite. Raw source and candidate boxes remain
unchanged; only derived features are clipped. Geometry calculations use rational
intermediates to avoid overflow before clipping, including large finite integers
accepted by S02. Log ratios use differences of logs to avoid overflow/underflow
of intermediate ratios. This changes numerical evaluation, not the formulas.

The model receives no learned priorities, semantic role labels, or inferred
groups. Text `color` and shape `fill_color` support the samples' numeric
comma-separated `rgb(r,g,b)` and `rgba(r,g,b,a)`, plus `#RGB`, `#RGBA`,
`#RRGGBB`, and `#RRGGBBAA`. RGB channels must be finite numbers in [0,255];
alpha must be finite in [0,1]. Decimal/exponent notation, case-insensitive
function names, and surrounding/channel whitespace are supported. Percentages,
named colors, space/slash CSS syntax, malformed literals, and out-of-range
channels are unsupported and produce zero RGB, alpha, and parse mask.
This bounded feature parser is not a full CSS renderer; it never rewrites JSON.

RGB and availability remain at columns 32:36 for shape fills and 36:40 for
text. graph_v3 appends `fill_alpha` at column 55 and `text_alpha` at column 56.
Absent hex/rgb alpha defaults to 1 for successfully parsed colors. A transparent
color has alpha 0 **and parse mask 1**, unlike an unsupported color. Alpha is
separate from element opacity (column 13); RGB is not premultiplied. Fields
irrelevant to the element type are zero. These are source style features:
candidate color edits remain forbidden by the output policy. Style-run colors
and strokes are preserved in JSON but are not separately pooled into node colors.

## Explicit conventions within the S03 contract

- For `N <= 64`, edges are `(i, j)` in lexicographic source-index order for every
  `i != j`. One node has `edge_index[2, 0]` and `edge_attr[0, 31]`.
- For `N > 64`, select eight nearest normalized source centers per node. Select
  up to sixteen nearest members of its explicit group as well. Add reverse edges
  and deduplicate; the cap applies to each node's selections before adding
  reverse edges, not to final degree. Null/absent groups do not participate;
  an empty string is a real group ID.
- Equal-distance neighbors are resolved by source index. Bridge disconnected
  components using the closest pair, breaking ties by `(i, j)` with `i < j`,
  until connected. The final directed edges are sorted lexicographically.
  Candidate geometry never affects topology. These tie rules make the
  unspecified ties in S03 section 3.2 deterministic.
- Containment requires positive area for both boxes; equal nonzero boxes contain
  one another. Touching projections have zero signed gap. Zero union has IoU 0.
- Node center deltas are computed from unclipped normalized centers, then
  clipped; clipping each center before subtraction would change the contract.
- Source normalized widths/heights/area and candidate normalized dimensions
  have range `[0, 4]`; signed normalized geometry has range `[-4, 4]`.
- Existing text/style/group/stacking/semantic features describe source elements.
  Candidate geometry blocks and the new typography columns use candidate values.

## Candidate typography (graph_v2)

The typography revision retained columns 0:53 (color parsing was subsequently
extended in graph_v3). Column 53, `candidate_font_size_log`, is
`log1p(candidate.font_size) / log1p(max(target.width, target.height))` for text.
Column 54, `font_size_log_change`, is
`clip(log(candidate.font_size) - log(source.font_size), -4, 4) / 4`.
Both are zero on non-text nodes, even if those nodes carry font-size extensions.
Source font size remains at column 16. Candidate elements are matched by exact
ID, so reordering cannot attach one element's font size to another node.

A 60 px versus 12 px candidate now produces different graph tensors even when
its box is identical. These features describe size, not measured glyph fit.
Text embeddings, original feature order, and the 31 edge calculations are
unchanged by the typography revision. That revision introduced graph_v2 with
55 numeric columns; graph_v3 adds alpha and supports the fixture color syntax,
giving a current ranker input of 57 + 32 + 1 = 90. Old graph_v1/v2 graphs and
checkpoints must be rebuilt/retrained; there is no silent padding or conversion.

## Optional text embeddings

```python
from engine.text_embeddings import TextEmbeddingCache

# provider implements encoder_version: str and encode(text) -> vector | None
cache = TextEmbeddingCache(provider)
graph = build_candidate_graph(source, candidate, target, text_embeddings=cache)
```

`EmbeddingProvider` is a typed protocol. `TextEmbeddingCache.get(text)` uses
`(encoder_version, sha256(literal_content))` as its key. Reuse the same cache
across candidates. It preserves whitespace and Unicode, including unpaired
surrogates in the hash input; it performs no text normalization. Returned vectors
are independent detached CPU copies. Changing encoder version creates new keys.
Changing it during one graph build is an error.

Without a cache, all text vectors and masks are zero. Non-text nodes always
have zero vectors/masks and never call the provider. Text-text cosine is computed
only when both masks are one, using float64 norm/dot intermediates. Other edges
have semantic cosine and mask zero.

Providers may return `None` or raise `EmbeddingUnavailable`. Zero-norm vectors
also count as unavailable because they have no cosine direction. These results
are cached; call `cache.clear()` to retry transient failures. Wrong dimensions
or non-finite float32 values raise `ValueError` rather than hiding a broken
provider. Other unexpected provider errors propagate.

`FrozenSentenceTransformerProvider(model, encoder_version=...)` adapts an
already-loaded 384-dimensional SentenceTransformer. It disables parameter
gradients, selects evaluation mode, and encodes under inference mode. Model
installation/loading and local weights remain the caller's responsibility.
For the proposed baseline use `sentence-transformers/all-MiniLM-L6-v2` with
a pinned revision, and include that revision/preprocessing identity in the
version string. No model weights were downloaded or used for this milestone's
tests; deterministic providers and a small fake encoder exercise the interface.

Missing embeddings are suitable for structural tests and a separately trained
numeric-only checkpoint. They are not equivalent inputs to a checkpoint trained
with text embeddings. Scoring must check this metadata.
The implemented [ranker scoring API](ranker.md) now enforces the exact encoder
identity and complete availability for text-enabled checkpoints, or zero
embedding/semantic inputs for numeric-only checkpoints.

## Dependencies and verification

The existing baseline requirements and S02 modules have no ML imports. Install
the graph stack separately, selecting the desired CPU/CUDA PyTorch distribution:

```text
python -m pip install -r requirements-dev.txt -r requirements-graph.txt
python -m pytest -q tests/test_internal_schema.py
python -m pytest -q tests/test_graph_features.py
python -m pytest -q
```

PyG is pinned to `2.8.1`. The local environment uses PyTorch `2.11.0+cu128`,
but all graph tests and the GATv2 smoke test execute on CPU. The project's local
`.venv` inherits the existing PyTorch installation and contains PyG; use
`.venv/Scripts/python.exe -m pytest ...` on Windows for that environment.
The graph test module skips explicitly if optional torch/PyG dependencies are
missing; a skipped module is not evidence that graph tests passed.

Tests cover all three Adobe fixtures, complete numeric feature vectors,
clipping and masks, dense/sparse topology, isolated nodes, exact-ID matching,
immutable input, optional encoder behavior, PyG batching, and a small edge-aware
layer forward/backward pass. The smoke test is not a production ranker or
preference-training pipeline. Python 3.10 syntax is checked; actual execution
was on Python 3.11.17.

Official API references used to verify the integration:
[PyG Data](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html),
[PyG GATv2Conv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATv2Conv.html),
and [SentenceTransformer](https://sbert.net/docs/package_reference/sentence_transformer/model.html).
The installed PyG 2.8.1 API was also inspected and exercised locally.

Color regression tests load all three actual Adobe sample files and check each
text color and shape fill against explicit expected RGB/alpha values. The core
sample's style-run-only orange is also exercised as a text-color literal, without
claiming that the graph pools style-run colors. Tests cover transparency versus
missingness, opacity independence, hex compatibility, malformed/unsupported
inputs, and unchanged source/candidate JSON.
