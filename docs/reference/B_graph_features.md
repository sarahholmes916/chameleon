# Appendix B — S03 graph construction and feature contract (`graph_v3`; `graph_v4` selectable)

> **Appendix B of [TECHNICAL_SPEC.md](../TECHNICAL_SPEC.md).** Authoritative detail for graph topology,
> node/edge feature tables, formulas, color parsing and text embeddings. Status, the
> `build_candidate_graph` signature and the model that consumes these tensors are defined in the main
> specification (§2, §4) and [Appendix C](C_ranker_model.md).

Code: [engine/graph_features.py](../../engine/graph_features.py), [engine/text_embeddings.py](../../engine/text_embeddings.py).
`FEATURE_CONTRACT_VERSION = "graph_v3"` is the default. `build_candidate_graph(..., feature_version="graph_v4")`
builds `graph_v4`, whose first 57 node columns and first 31 edge columns are exactly `graph_v3` and which appends one
node and one edge feature (B.4.3). B.1–B.6 describe `graph_v3` unless they say `graph_v4`. Feature order is locked:
any change requires a new version string and tests (earlier-version history is in [HISTORY.md](HISTORY.md)).

## B.1 Interface behavior

Node `i` always represents `source.elements[i]`. Candidate elements are matched
by exact ID, even when the candidate array is reordered. Missing, unexpected,
or duplicate IDs and changed element types raise `ValueError`. Normally S02
already rejects duplicate IDs during parsing. The graph's identity checks are
not an output-policy check (spec §8.1).

`source.canvas` must equal `source.source_canvas`; `candidate.canvas` must equal
the explicit `target`. Source geometry uses `source.source_canvas`; candidate
geometry uses `target`. The target argument is authoritative, and optional
`Design.target_canvas` context does not override it. Parsing a candidate snapshot
through `Design.from_adobe()` is supported even though that snapshot's own
`source_canvas` describes the target layout. No source data is changed.

### Graph object fields

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

For `graph_v4`, `x_numeric` is `[N, 58]`, `edge_attr` is `[E, 32]` and `context_version` is `"graph_v4"`; every
other field is identical. An unsupported `feature_version` raises `ValueError`.

All returned numeric tensors are finite. Raw source and candidate boxes remain
unchanged; only derived features are clipped. Geometry calculations use rational
intermediates to avoid overflow before clipping, including large finite integers
accepted by S02. Log ratios use differences of logs to avoid overflow/underflow
of intermediate ratios. This changes numerical evaluation, not the formulas.

`float32` for feature tensors; `int64` for `edge_index`; finite values only. Every candidate of the same
source has the same element index mapping and graph topology. The graph builder must not modify either
`Design`. For dense N=4 the graph has 12 directed edges; for N=17, 272. Text encoder caching must not
change the representation of repeated strings.

## B.2 Identity and graph semantics

**Node `i`:** the element at index `i` in `source.elements`, preserving original array order. `element_ids[i]` stores its exact opaque ID. Candidate elements are joined to source elements by exact ID, and a mismatch is an error; never pair by sorted coordinates, inferred role, or ID spelling.

**Single graph vs subgraphs:** the source may contain several apparent components (e.g., logo group, text group, CTA group). **They all remain nodes of one graph.** The existing `group_id` is encoded as edge evidence, not as a graph partition or a new Adobe group. This enables the model to learn interactions *between* components, such as a headline–CTA or image–logo relationship. An absent or explicit-null group value is ungrouped. An empty-string group ID is a valid real group ID. Only identical string IDs yield `same_explicit_group=1`.

**Directed edge `(i,j)`:** information flows from node `i` to node `j`; edge features describe element `j` relative to element `i`. Include both directions even though their geometry values may have opposite signs. Do **not** create semantic relation types based on names like `headline_1`, `sun_logo`, or `bg`.

**No explicit canvas node:** target dimensions and the transformation ratio are global context repeated on each node. Global graph pooling then produces a representation for scoring.

## B.3 Edge construction and the 31-dimensional edge features

### B.3.1 Dense graph (default)

```python
ids = [element.id for element in source.elements]
edge_pairs = [(i, j) for i in range(N) for j in range(N) if i != j]
edge_index = torch.tensor(edge_pairs, dtype=torch.long).T.contiguous()
# shape [2, N*(N-1)]
```

Edges are **not deleted** because source elements appear far apart or have different group IDs. That would prevent the GNN from evaluating cross-group relationships. `N=1` produces an empty explicit edge list shaped `[2, 0]`; GATv2 handles self information via its own added self-loop. Do not add duplicate self-edges in graph construction.

### B.3.2 Larger graphs (`N>64`)

If tests ever contain more than 64 elements, make a sparse graph by: (a) connecting each node bidirectionally to its `k=8` nearest centers in normalized source coordinates; (b) adding all bidirectional edges between elements sharing an explicit group, capped at 16 nearest same-group neighbors per node; (c) computing weakly connected components and repeatedly adding bidirectional edges between the closest pair of nodes belonging to different components until connected. Deduplicate and sort edges lexicographically. The same source-derived topology is used for every candidate of that source. Record `graph_topology="dense_v1"` or `"sparse_v1"` in model metadata. Do not mix topologies within a model evaluation without including both in its training distribution.

### B.3.3 Exact 31-dimensional edge feature contract

All boxes are axis-aligned, pre-rotation element bounds. Let canvas `C=(W,H)`. For pair `(i,j)`, define:

- `dx=(center_x[j]-center_x[i])/W`, `dy` analogously; negative values allowed.
- `gap_x`: zero if horizontal box projections overlap; otherwise signed separation normalized by `W` (positive when `j` is to the right of `i`), `gap_y` analogously.
- `IoU`: intersection area divided by union area; return 0 if union area is 0.
- `contains_ij`, `contains_ji`: 1 when one nonzero-area axis-aligned box contains the other, else 0.
- `align6`: the **absolute** differences in left, horizontal-center, right positions divided by `W`, then top, vertical-center, bottom divided by `H`. Zero means perfect respective alignment. Values can exceed 1 for off-canvas source input; clip to `[0,2]` as a feature-preprocessing step only.
- `log_area_ratio = clip(log((area_j+1)/(area_i+1)), -6, 6)/6` with areas measured in square pixels of the relevant canvas coordinate system.
- `z_known`: 1 if both elements explicitly provide `z_index`; else 0. `z_sign`: `sign(z_j-z_i)` if known; else 0. Do not invent a missing `z_index`.
- `semantic_cos`: cosine similarity of frozen text embeddings if both nodes contain text and embeddings are available; else 0. `semantic_mask=1` only in the former case. A cross-modality pairing (text–shape or text–image) is not represented by text-text similarity.

Feature order and ranges:

| Index range | Dimensions | Feature block |
|---|---:|---|
| 0:2 | 2 | Source relative center `(dx,dy)` |
| 2:4 | 2 | Source signed gaps `(gap_x,gap_y)` |
| 4:5 | 1 | Source IoU |
| 5:7 | 2 | Source containment flags `(i_contains_j,j_contains_i)` |
| 7:13 | 6 | Source `align6` |
| 13:14 | 1 | Source log area ratio |
| 14:15 | 1 | Same explicit group |
| 15:17 | 2 | Source z-known and z-sign |
| 17:19 | 2 | Text semantic cosine and its availability mask |
| 19:21 | 2 | Candidate relative center `(dx,dy)` |
| 21:23 | 2 | Candidate signed gaps `(gap_x,gap_y)` |
| 23:24 | 1 | Candidate IoU |
| 24:30 | 6 | Candidate `align6` |
| 30:31 | 1 | Candidate log area ratio |

Source geometry uses `source.source_canvas`; candidate geometry uses `target`. To avoid scale explosions, clip normalized displacement/gaps to `[-4,4]`, alignment differences to `[0,2]`, and ratios as specified. This clipping affects **model features only**, never the stored element geometry or serialized JSON.

### B.3.4 Worked two-node edge example

Suppose a 1000×500 source has headline `H=(x=100,y=80,w=300,h=60)` and subtitle `S=(x=105,y=150,w=290,h=40)`, both with the same explicit string `group_id`. The center of each is `x=250`, so for `H -> S`, the source relative center is `dx=0`, `dy=(170-110)/500=0.12`; their vertical bounding-box gap is `(150-140)/500=0.02`; the left-edge alignment difference is `abs(105-100)/1000=0.005`; and `same_explicit_group=1`. Reverse edge `S -> H` reverses signed center displacement and gap, but the same-group feature remains 1. Those are **measured** inputs. Neither direction receives a manually asserted 'importance weight'.

### B.3.5 What the 'edge weights' mean

The 31-vector `edge_attr[i,j]` is **not** a scalar weight. It contains independent facts and derived measurements. GATv2 internally computes learned attention for each direction, layer, and head:

\[
\alpha_{i\to j}^{(\ell,h)}=\operatorname{softmax}_{i\in\mathcal N(j)}\left(a_{\ell,h}(h_i,h_j,e_{ij})\right).
\]

That coefficient determines aggregation of messages arriving at `j`, **not** object ranking, pairwise grouping probability, causal importance, or a preservation priority. High attention can indicate that some neighbor contributes useful information, including background information. It is not a reliable standalone explanation of what a human values.

If explicit relation prediction becomes necessary in v0.2, add a supervised edge head with named outputs such as `keep_near`, `align`, and `same_visual_unit`, plus actual labels. Do not relabel attention coefficients as relation predictions.

### B.3.6 Explicit conventions

- For `N <= 64`, edges are `(i, j)` in lexicographic source-index order for every
  `i != j`. One node has `edge_index[2, 0]` and `edge_attr[0, 31]`.
- For `N > 64` (B.3.2), the cap of sixteen same-group neighbors applies to each node's selections before
  reverse edges are added, not to final degree. Null/absent groups do not participate; an empty string is a
  real group ID. Equal-distance neighbors are resolved by source index; bridge ties use `(i, j)` with
  `i < j`; the final directed edges are sorted lexicographically. Candidate geometry never affects topology.
- Containment requires positive area for both boxes; equal nonzero boxes contain
  one another. Touching projections have zero signed gap. Zero union has IoU 0.
- Node center deltas are computed from unclipped normalized centers, then
  clipped; clipping each center before subtraction would change the contract.
- Source normalized widths/heights/area and candidate normalized dimensions
  have range `[0, 4]`; signed normalized geometry has range `[-4, 4]`.
- Existing text/style/group/stacking/semantic features describe source elements.
  Candidate geometry blocks and the typography columns 53:55 use candidate values.

## B.4 Node features (57 numeric fields)

For the ranker, each element is one node containing **57 numeric fields**, a **384-dimensional frozen text vector** as a separate input, and a one-dimensional presence mask. The model projects the 384-dimensional vector to 32 and concatenates it, yielding **90 features per node** at the first GNN layer: `57 + 32 + 1 = 90`.

### B.4.1 The 57 numeric fields, in exact order

| Index range | Dim | Feature / definition |
|---|---:|---|
| 0:3 | 3 | Type one-hot `[text,image,shape]` |
| 3:7 | 4 | Original `(x/W,y/H,w/W,h/H)` |
| 7:9 | 2 | Original normalized center |
| 9:10 | 1 | Original area divided by `W*H` |
| 10:11 | 1 | Original log aspect `clip(log((w+1)/(h+1)),-4,4)/4` |
| 11:13 | 2 | `sin(rotation)`, `cos(rotation)` (degrees converted to radians) |
| 13:14 | 1 | Effective opacity; absent means 1 |
| 14:16 | 2 | `has_z_index`, `clip(z_index/32,-1,1)` (0 when absent) |
| 16:17 | 1 | Text `log1p(font_size)/log1p(max(W,H))`; zero on other types |
| 17:19 | 2 | Text bold, italic booleans; zero otherwise |
| 19:22 | 3 | Text align one-hot `[left,center,right]`; zero otherwise |
| 22:24 | 2 | Text log char count / `log1p(512)`, log word count / `log1p(256)` |
| 24:26 | 2 | Text `has_max_lines`, `clip(max_lines/20,0,1)`; null or absent -> mask/value 0 |
| 26:29 | 3 | Image effective `fit` one-hot `[contain,cover,fill]`; zero otherwise |
| 29:32 | 3 | Shape type one-hot `[rectangle,ellipse,line]`; zero otherwise |
| 32:36 | 4 | Shape fill color `(r,g,b,parse_mask)`; zero when not parseable or irrelevant |
| 36:40 | 4 | Text color `(r,g,b,parse_mask)`; zero when not parseable or irrelevant |
| 40:42 | 2 | Explicit group present as string, group size divided by `N` |
| 42:46 | 4 | Candidate `(x/TW,y/TH,w/TW,h/TH)` |
| 46:48 | 2 | Candidate center minus source center, both in normalized canvas coordinates |
| 48:50 | 2 | `clip(log((w_c+1)/(w_s+1)),-4,4)/4`, corresponding height scale |
| 50:53 | 3 | Global `clip(log(TW/TH),-4,4)/4`, `clip(log(TW/SW),-4,4)/4`, `clip(log(TH/SH),-4,4)/4` |
| 53:54 | 1 | Candidate text `log1p(font_size_c)/log1p(max(TW,TH))`; zero on other types |
| 54:55 | 1 | Text font-size change `clip(log(font_size_c/font_size_s),-4,4)/4`; zero on other types |
| 55:56 | 1 | Source shape fill alpha in [0,1]; zero for unsupported colors or other types |
| 56:57 | 1 | Source text color alpha in [0,1]; zero for unsupported colors or other types |

Here `W,H` refer to the source canvas; `TW,TH` refer to the target; `SW,SH` are aliases of source width/height in columns 50:53. Char count means `len(content)` in Python Unicode code points; word count means `len(content.split())`. There is no need to normalize or rewrite stored content. Clip all source normalized geometric numeric inputs in indices 3:10 and candidate geometry in 42:48 to `[-4,4]` **for features only**, except dimension/area terms which are lower-bounded at zero.

Color features follow the parser defined in B.6 (the only definition of color parsing).

### B.4.2 Candidate typography

Columns 53 (`candidate_font_size_log`) and 54 (`font_size_log_change`) are defined in the B.4.1 table;
both are zero on non-text nodes, even if those nodes carry font-size extensions. Source font size stays at
column 16. Candidate elements are matched by exact ID, so reordering cannot attach one element's font size
to another node. A 60 px versus 12 px candidate produces different graph tensors even when its box is
identical. These features describe size, not measured glyph fit.

### B.4.3 `graph_v4` additions (node column 57, edge column 31)

| Field | Index | Definition |
| --- | ---: | --- |
| `text_fit_height_ratio` (node) | 57 | `clip(est_height / h_c, 0, 4) / 4` for a text node whose candidate box has `w_c > 0` and `h_c > 0`, where `est_height` is the text-fit estimate defined in D.1.5 and implemented once in [engine/text_fit.py](../../engine/text_fit.py). Zero on non-text nodes (even if they carry font or content extensions) and for a candidate box with nonpositive width or height. For a valid text box the estimate is positive, so a zero on a text node identifies a degenerate box |
| `size_order_preserved` (edge) | 31 | Text–text pairs: `sign(f_s[i] − f_s[j]) · sign(f_c[i] − f_c[j])`, that is `+1` when the candidate keeps the source's font-size order, `-1` when it reverses it, `0` when either pair is tied. Zero when either node is not text. Symmetric in `(i, j)` |

Consequences: node width 58, edge width 32, ranker `gatv2_ranker_v4` with first-layer width `58 + 32 + 1 = 91`
([Appendix C](C_ranker_model.md)). The feature value depends on the estimator, so the checkpoint architecture
manifest records its identity (`text_fit_v1`, glyph advance 0.5 em). The estimate is not measured glyph layout,
its glyph advance is uncalibrated (D.1.6), and the cap means overflow beyond four box heights is not
distinguished. Why these two fields and not the other two proposed: B.7.2.

## B.5 Text vector and availability

- **Optional.** The text vector is not required for the graph, the ranker or `resize`. The default ranker configuration is numeric-only (`text_encoder_version="none"`, Appendix C.2) and the graph builder runs without a cache (all text vectors and masks zero).
- Proposed encoder, if one is used: frozen pretrained `sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions, applied to the literal `content` string; cache by `(encoder_version, content_hash)`. It has **not** been approved by Adobe: whether pretrained embedding models are permitted, and any size limit, are unanswered (spec U8, [Appendix F](F_submission_and_adobe_questions.md)). It has never been loaded or evaluated in this repository.
- Non-text elements: 384 zeros, `text_mask=0`.
- Text elements with a successfully computed embedding: 384-vector and `text_mask=1`.
- For deployment, bundle the exact compatible encoder or use a separately trained **numeric-only** checkpoint; do not silently supply zeros to a checkpoint trained to depend on embeddings and assume identical predictions.
- No image pixels, OCR, web lookup, asset-ID semantics, or LLM-generated descriptions are required by v0.1. Optional future image representations need their own permission/rule review and versioned feature contract.
- The model-side projection of the 384-vector is specified in [Appendix C](C_ranker_model.md).

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
version string. Tests use deterministic providers and a small fake encoder; no pretrained
weights have been downloaded or evaluated.

Missing embeddings are suitable for structural tests and a separately trained
numeric-only checkpoint. They are not equivalent inputs to a checkpoint trained
with text embeddings. Scoring must check this metadata.
The ranker scoring API ([Appendix C](C_ranker_model.md)) enforces the exact encoder
identity and complete availability for text-enabled checkpoints, or zero
embedding/semantic inputs for numeric-only checkpoints.

## B.6 Color parsing (single authoritative definition)

The graph receives no learned priorities, semantic role labels, or inferred
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
candidate color edits are forbidden by the output policy (spec §8.1).

Color literals are parsed for features only; stored JSON literals stay exact. Style-run colors and
strokes are preserved in JSON but not pooled into node colors, and a shape's `color` key (distinct from
`fill_color`) is not featurized.

## B.7 Typography audit and the `graph_v4` decision

> **Status (spec U13):** `graph_v4` is implemented as an **opt-in** feature version (B.4.3) after the
> information-content evaluation of B.7.2. `graph_v3` is the default and is unchanged. Whether `graph_v4`
> ranks candidates better than `graph_v3` is **untested** (B.7.3); do not claim an improvement.

### B.7.1 Audit of `graph_v3` (2026-10-28; code, tests and one experiment)

**Question:** can the GNN distinguish two candidate layouts that are identical except for text font size?

**Result: yes at the feature level; the model path is connected; usefulness after training is untested.**

- Columns 53 (`candidate_font_size_log`) and 54 (`font_size_log_change`) carry the candidate's font size (B.4.1). `tests/test_graph_features.py::test_font_only_candidate_edits_are_visible_and_joined_by_id` asserts that two candidates with identical boxes and different font sizes differ only in these columns.
- Reproduced on the Level 2 sample (baseline repair candidate versus the same candidate with every text size multiplied by 1.5, boxes unchanged): exactly columns 53 and 54 differ; `edge_index`, `edge_attr` and the candidate-box columns 42:46 are identical. An untrained `LayoutRanker` in evaluation mode returned different scalar scores for the two graphs (about -0.0315 and -0.0356); this shows the input reaches the output, not that a trained model values it correctly.
- Column 53 is not clipped. Column 54 is clipped at a size ratio of `e^4` (about 54.6x), which does not matter for realistic edits.
- Non-text nodes carry zero in both columns by design.

**Gaps (what `graph_v3` cannot express explicitly):**

1. **No font-size/box relation.** Whether a size fits its candidate box depends on the box, the text length, the line height and the size together. `graph_v3` has the pieces (columns 22:24 text length, 42:46 candidate box, 53 size) but no relation between them and no line height, so a model must learn text fit from interactions of raw columns, with little data.
2. **No typographic relation on edges.** All 31 edge features are geometric, grouping or semantic (B.3.3). Preserving a size hierarchy (headline larger than body) must be derived by attention from node columns; there is no explicit candidate size ratio or order-preserved signal between two text nodes.
3. **Frozen typography is unseen.** `line_height`, `letter_spacing` and `font_family` are constant across candidates of one source (K6), so they cannot distinguish candidates; they matter only for interpreting fit (gap 1).

### B.7.2 Information-content evaluation of the proposed features (2026-10-28)

**Question:** does each proposed `graph_v4` field carry information that the `graph_v3` columns do not already
provide, or does it duplicate them? **Rule, fixed before running:** a field duplicates `graph_v3` if a ridge-linear
model on the `graph_v3` inputs reproduces it with held-out R² of at least 0.90.

**Method.** Leave-one-sample-out over the three sample designs; 500 randomized candidates per sample (target drawn
from 12 common formats, each text size multiplied by 0.25–4, each box scaled by 0.5–2 and shifted by up to 10 % of the
canvas), giving 7,000 text nodes and 35,000 text–text edges. A ridge-linear model and a two-hidden-layer MLP map the
`graph_v3` inputs (the node row; for edges, the 31 edge features plus both endpoint rows) to each proposed field.
Only text nodes and text–text edges are scored, so rows that are trivially zero cannot inflate the result.

| Proposed field | Linear held-out R² (3 folds) | MLP held-out R² | Decision |
| --- | --- | --- | --- |
| Node `text_fit_height_ratio` | 0.66 / 0.76 / 0.75 (mean 0.72) | 0.86 / 0.85 / −3.16 (unstable) | **Kept** (57). `graph_v3` cannot reproduce it linearly: it needs line height, newline structure and absolute pixel scale, none of which `graph_v3` carries |
| Edge `candidate_size_log_ratio` | 0.998 / 0.974 / 0.985 (mean 0.986); 0.990 from `col53_i − col53_j` alone | mean 0.94 | **Dropped.** It is the difference of two `graph_v3` columns (53), so it duplicates them |
| Edge `size_order_preserved` | −0.19 / −0.62 / −0.11 | 0.84 / −0.20 / −0.52 | **Kept** (edge 31). It is a sign-agreement between two columns, not a linear function of them |
| Node `text_fit_valid` | not run | not run | **Dropped by construction.** The ratio is exactly zero on a text node only when the box is degenerate, and the type one-hot already separates text from other nodes |

**Reading the result.** This measures whether `graph_v3` makes a quantity available, not whether a ranker benefits from it.
The candidates are synthetic, there are only three source designs, and the estimator's glyph advance is an
uncalibrated placeholder. The MLP's failure on the third sample (advanced, small canvases) reflects extrapolation, not
a property of the feature. The analysis script is a one-off and is not in the repository.

**As implemented:** `graph_v4` = `graph_v3` + node column 57 + edge column 31 (B.4.3). Node width 58, edge width 32,
first GAT input width 91, ranker `gatv2_ranker_v4`, `ranker_checkpoint_v1` unchanged. `graph_v3` graphs, the v3
architecture and v3 checkpoints are untouched and not interchangeable with v4 (a model accepts only its own version).

### B.7.3 Still required before the default changes or any improvement is claimed

Nothing below has been run; there are no human labels yet.

1. **Pair set.** Candidate pairs identical except for text `font_size`, from the rendered calibration set of D.1.6, for the three sample designs and several target canvases. Each text element is covered in three cases: *overflow* (too large for its box, so text spills or is cut off), *unreadably small*, and *appropriately sized*. Include pairs where both sizes are acceptable, to see whether a model invents a preference.
2. **Labels are human only** (Appendix E tool: A better, B better, tie, both bad). The deterministic `E_read` is never the label.
3. **Models compared.** A `graph_v3` ranker, a `graph_v4` ranker (`RankerConfig(feature_version="graph_v4")`), the pooled-feature baseline (E.4) and the deterministic scorer, all trained or evaluated on identical source-design-disjoint splits (Appendix E.3).
4. **Metrics and rule.** Pairwise accuracy overall and per case, with ties and both-bad reported separately and chance (50 %) shown. The team writes down, before the run, what counts as a material improvement of `graph_v4` over `graph_v3` and over the pooled-feature baseline; it is not set here. `graph_v4` becomes the default only if it meets that rule.
5. **Before the estimator constant is calibrated** (D.1.6), treat `text_fit_height_ratio` as an uncalibrated input; if the estimator changes, bump the feature and ranker versions.
