# Adobe Chameleon — Technical Specification S03 (graph and ranker)

**Version:** 0.1 baseline with graph_v3 typography and color revisions
**Updated:** 2026-10-27
**Status:** S03 graph construction/feature extraction and the untrained GATv2
ranker/checkpoint/scoring foundation are implemented. A deterministic output
validation gate, bounded repair search, and resize integration are also implemented
(see [output validation](output_validation.md)). Diverse generation and preference
training/evaluation remain proposals in [planning/](planning/).
See [M2A interfaces and verification](graph_features.md) and [ranker interfaces](ranker.md).
**Dependencies:** S02 *Internal design representation*; Adobe v0.9 JSON schema; challenge RULES.md/FAQ.md; `engine/resize.py`  
**Primary objective:** Given a structured Adobe source design and target canvas, produce a legally structured, constraint-checked resized design and, internally, several materially different alternatives. Learn candidate preferences from human side-by-side comparisons rather than treating heuristic scores as ground-truth labels.

> **Scope caveat:** This spec is a concrete architecture and default configuration, NOT independent verification of every contest rule. The original rules and authoritative JSON Schema win if any condition here conflicts with them. Do not implement invented restrictions as if they were official rules.

---

## 0. Baseline decisions with graph_v3 feature revisions

1. **One graph per `(source design, candidate layout, target canvas)`**, with exactly one node per original Adobe element. No output nodes for clusters, canvases, or brands. A source design with 17 elements yields 17 nodes, even if it contains several explicit groups.
2. **Dense directed element graph for `1 <= N <= 64`:** for every `i != j`, include edge `i -> j`. Thus `E=N(N-1)`; 17 elements yield 272 directed edges. PyG layers add their own self-loop messages. For `N>64`, use the deterministic sparse procedure in §3.2.
3. **Edges encode relationships, not object importance or a ranking.** Each edge is a 31-dimensional numeric vector containing source and candidate geometry, group evidence, optional text similarity, and relative stacking evidence. A learned GNN attention coefficient is a per-layer/per-head message aggregation coefficient; it is **not** a human-interpretable object-importance score.
4. **No explicit object-importance head in v0.1.** The trainable output is a single **candidate-layout score** conditional on the original design and target. A separately supervised role/importance/relationship head may be added in v0.2 but must not be inferred from attention alone.
5. **One edge-aware GNN ranker** with GATv2 layers; do not require a second GNN before candidate generation. Deterministic source analysis and computational layout generation work without any checkpoint.
6. **Candidate generation is computational first**, using explicit layout families, integer/discrete decisions where helpful, local geometry refinement, and hard-rule validation. The GNN chooses among candidates but never determines whether a rule has been satisfied.
7. **Human preference comparisons** are the primary labels for ranking. Heuristic validity checks may filter candidates but may not be used as synthetic preference ground truth.
8. **Stage-one output remains Adobe JSON with only `canvas` and `elements` at root.** Internal role hypotheses, graph data, quality scores, source context, and UI metadata must never leak into returned Adobe JSON.
9. **Internal alternatives are allowed; the competition-facing resize boundary returns one design** unless the actual challenge interface explicitly changes.
10. **Keep S02 as the lossless source-of-truth layer.** Neither embedding generation nor candidate construction mutates the source design or relies on semantic interpretations of ID spellings.

## 1. Modules and stable boundaries

```text
engine/
  internal_schema.py       # S02; immutable Canvas/Element/Design + round-trip
  graph_features.py        # S03; deterministic numeric graph construction
  text_embeddings.py       # S03; optional frozen text encoder/cache
  ranker.py                # S03/S05; GATv2 scoring model
  layout_units.py          # S04; explicit and inferred layout units
  candidate_generator.py   # S04; layout-family proposals + search
  layout_objective.py      # S04; exact soft costs
  candidate_validator.py   # S04; schema/policy/geometric validation
  diversify.py             # S04; near-duplicate removal + selection
  resize.py                # existing external boundary, thin adapter
annotation/
  app.py                   # S05; local side-by-side UI
  records.py               # S05; append-only human labels
training/
  dataset.py               # S05; grouped-by-source splits and batching
  train_ranker.py          # S05; training, evaluation, checkpoints
  evaluate.py              # S05/S06; ablation and end-to-end evaluation
configs/
  graph_v3.yaml
  generator_v1.yaml
  ranker_v1.yaml
```

Key function signatures (names are proposed, but function boundaries should be preserved):

```python
@dataclass(frozen=True)
class LayoutCandidate:
    design: Design                 # target canvas; IDs/order preserved
    candidate_id: str             # internal stable identity
    family: str                   # enum name, internal only
    seed: int
    generator_version: str

@dataclass(frozen=True)
class ValidationReport:
    schema_ok: bool
    policy_ok: bool
    geometry_ok: bool
    violations: tuple[str, ...]

@dataclass(frozen=True)
class ScoredCandidate:
    candidate: LayoutCandidate
    report: ValidationReport
    score: float | None           # None if checkpoint unavailable

def build_candidate_graph(source: Design, candidate: Design, target: Canvas) -> Data: ...
def generate_candidates(source: Design, target: Canvas, config: LayoutConfig) -> list[LayoutCandidate]: ...
def validate_candidate(source: Design, candidate: Design, target: Canvas) -> ValidationReport: ...
def score_candidates(source: Design, candidates: list[LayoutCandidate], target: Canvas) -> list[float]: ...
def select_options(candidates: list[ScoredCandidate], k: int = 3) -> list[ScoredCandidate]: ...
```

`Data` means a `torch_geometric.data.Data` graph with `x_numeric`, `text_vec`, `text_mask`, `edge_index`, `edge_attr`, `element_ids`, and a `context_version`. The model's public numeric result is a scalar score; no Adobe fields are changed by inference.

**Failure policy:** if ranking or text-model loading fails, use the deterministic scorer on validated candidates only. If no candidate passes, continue generation/repair using validation diagnostics (section 6.6). Never return, rank as eligible, or serialize a known-invalid candidate as the selected output. Exhausting an explicit search budget raises `NoFeasibleLayout`; it must not return a least-violating draft. This is search exhaustion, not proof that no solution exists.

---

# S03 — Relational Graph Representation and GNN

## 2. Identity and graph semantics

**Node `i`:** the element at index `i` in `source.elements`, preserving original array order. `element_ids[i]` stores its exact opaque ID. Candidate elements are joined to source elements by exact ID, and a mismatch is an error; never pair by sorted coordinates, inferred role, or ID spelling.

**Single graph vs subgraphs:** the source may contain several apparent components (e.g., logo group, text group, CTA group). **They all remain nodes of one graph.** The existing `group_id` is encoded as edge evidence, not as a graph partition or a new Adobe group. This enables the model to learn interactions *between* components, such as a headline–CTA or image–logo relationship. An absent or explicit-null group value is ungrouped. An empty-string group ID is a valid real group ID. Only identical string IDs yield `same_explicit_group=1`.

**Directed edge `(i,j)`:** information flows from node `i` to node `j`; edge features describe element `j` relative to element `i`. Include both directions even though their geometry values may have opposite signs. Do **not** create semantic relation types based on names like `headline_1`, `sun_logo`, or `bg`.

**No explicit canvas node:** target dimensions and the transformation ratio are global context repeated on each node. Global graph pooling then produces a representation for scoring.

## 3. Edge construction

### 3.1 Dense graph (default)

```python
ids = [element.id for element in source.elements]
edge_pairs = [(i, j) for i in range(N) for j in range(N) if i != j]
edge_index = torch.tensor(edge_pairs, dtype=torch.long).T.contiguous()
# shape [2, N*(N-1)]
```

Edges are **not deleted** because source elements appear far apart or have different group IDs. That would prevent the GNN from evaluating cross-group relationships. `N=1` produces an empty explicit edge list shaped `[2, 0]`; GATv2 handles self information via its own added self-loop. Do not add duplicate self-edges in graph construction.

### 3.2 Larger graphs (`N>64`)

If tests ever contain more than 64 elements, make a sparse graph by: (a) connecting each node bidirectionally to its `k=8` nearest centers in normalized source coordinates; (b) adding all bidirectional edges between elements sharing an explicit group, capped at 16 nearest same-group neighbors per node; (c) computing weakly connected components and repeatedly adding bidirectional edges between the closest pair of nodes belonging to different components until connected. Deduplicate and sort edges lexicographically. The same source-derived topology is used for every candidate of that source. Record `graph_topology="dense_v1"` or `"sparse_v1"` in model metadata. Do not mix topologies within a model evaluation without including both in its training distribution.

M2A makes distance ties deterministic by source index. Same-group selection is
capped at 16 neighbors per node before adding reverse edges; resulting degrees
may be higher. Closest-component bridge ties use `(i, j)` with `i < j`. These
conventions refine section 3.2 without changing the feature dimensions or order.

### 3.3 Exact 31-dimensional edge feature contract

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

### 3.3a Worked two-node edge example

Suppose a 1000×500 source has headline `H=(x=100,y=80,w=300,h=60)` and subtitle `S=(x=105,y=150,w=290,h=40)`, both with the same explicit string `group_id`. The center of each is `x=250`, so for `H -> S`, the source relative center is `dx=0`, `dy=(170-110)/500=0.12`; their vertical bounding-box gap is `(150-140)/500=0.02`; the left-edge alignment difference is `abs(105-100)/1000=0.005`; and `same_explicit_group=1`. Reverse edge `S -> H` reverses signed center displacement and gap, but the same-group feature remains 1. Those are **measured** inputs. Neither direction receives a manually asserted 'importance weight'.

### 3.4 What the 'edge weights' mean

The 31-vector `edge_attr[i,j]` is **not** a scalar weight. It contains independent facts and derived measurements. GATv2 internally computes learned attention for each direction, layer, and head:

\[
\alpha_{i\to j}^{(\ell,h)}=\operatorname{softmax}_{i\in\mathcal N(j)}\left(a_{\ell,h}(h_i,h_j,e_{ij})\right).
\]

That coefficient determines aggregation of messages arriving at `j`, **not** object ranking, pairwise grouping probability, causal importance, or a preservation priority. High attention can indicate that some neighbor contributes useful information, including background information. It is not a reliable standalone explanation of what a human values.

If explicit relation prediction becomes necessary in v0.2, add a supervised edge head with named outputs such as `keep_near`, `align`, and `same_visual_unit`, plus actual labels. Do not relabel attention coefficients as relation predictions.

## 4. Exact node feature contract

For the ranker, each element is one node containing **57 numeric fields**, a **384-dimensional frozen text vector** as a separate input, and a one-dimensional presence mask. The model projects the 384-dimensional vector to 32 and concatenates it, yielding **90 features per node** at the first GNN layer: `57 + 32 + 1 = 90`.

### 4.1 The 57 numeric fields, in exact order

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

**graph_v2 migration:** append these two typography features; original columns
0:53 retain their meanings and order. Candidate font size is joined by exact
element ID, not candidate array position. The candidate size uses target-canvas
normalization; source font size at column 16 keeps source-canvas normalization.
The ratio is evaluated as a difference of logs to avoid overflow. Font sizes
are schema-valid numbers >=1; no availability mask is needed beyond type.
The normalized candidate log size is not clipped, so large sizes remain
distinguishable; the change feature saturates at +/-1. These are features,
not edits or rendered-text fit guarantees. Text vectors and the 31 edge features
are unchanged. The shared feature version (including checkpoint edge metadata)
advanced to graph_v2 for typography. The current graph_v3 adds the two alpha
columns and rgba parsing. Both graph_v1 and graph_v2 tensors/checkpoints are
rejected, not padded. The architecture is now gatv2_ranker_v3 with 90 inputs.

Here `W,H` refer to the source canvas; `TW,TH` refer to the target; `SW,SH` are aliases of source width/height in columns 50:53. Char count means `len(content)` in Python Unicode code points; word count means `len(content.split())`. There is no need to normalize or rewrite stored content. Clip all source normalized geometric numeric inputs in indices 3:10 and candidate geometry in 42:48 to `[-4,4]` **for features only**, except dimension/area terms which are lower-bounded at zero. Color parsing supports numeric comma-separated rgb()/rgba() and #RGB/#RGBA/#RRGGBB/#RRGGBBAA. Channels must be finite and in [0,255], alpha in [0,1]. RGB is normalized by 255, not premultiplied; alpha is separate from element opacity. Missing alpha defaults to 1 for supported colors. Transparent colors have mask 1 and alpha 0. Unknown, malformed, out-of-range, percentage, named, or space/slash literals get zero color features and mask 0. Whitespace, case-insensitive function names, and decimal/exponent numbers are supported. Original JSON literals stay exact. These source color features do not authorize candidate recoloring. Stroke and per-run colors remain unpooled; see graph_features.md for the full parser boundary.

### 4.2 Text vector and availability

- Text encoder baseline: frozen pretrained `sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions, applied to the literal `content` string; cache by `(encoder_version, content_hash)`.
- Non-text elements: 384 zeros, `text_mask=0`.
- Text elements with a successfully computed embedding: 384-vector and `text_mask=1`.
- For deployment, bundle the exact compatible encoder or use a separately trained **numeric-only** checkpoint; do not silently supply zeros to a checkpoint trained to depend on embeddings and assume identical predictions.
- No image pixels, OCR, web lookup, asset-ID semantics, or LLM-generated descriptions are required by v0.1. Optional future image representations need their own permission/rule review and versioned feature contract.
- The 384-vector is projected through `Linear(384,32)` + ReLU within the model. Train only the projection, not the frozen pretrained encoder at first.

### 4.3 Tensor outputs and determinism

```python
Data(
    x_numeric = FloatTensor[N, 57],
    text_vec  = FloatTensor[N, 384],
    text_mask = FloatTensor[N, 1],
    edge_index= LongTensor[2, E],
    edge_attr = FloatTensor[E, 31],
    element_ids = tuple(original_ids),
)
```

`float32` for feature tensors; `int64` for `edge_index`; finite values only. Every candidate of the same source has the same element index mapping and graph topology. The graph builder must not modify either `Design`. For dense N=4, assert 12 directed edges. For N=17, assert 272. Text encoder caching must not change the representation of repeated strings.

## 5. Scoring network: gatv2_ranker_v3

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

Set `add_self_loops=True`, `fill_value=0.0` for the internally generated self-loop edge attributes. Use an explicit residual/skip connection at each GNN stage if tensor shapes allow (`h_out = norm(conv(h,...) + projected_h)`), with its exact implementation versioned. For the initial three layers the input width differs only on the first layer, so use a `Linear(90,128)` skip projection there. The implementation **must specify the chosen PyG version** in lockfile and verify that these exact options are supported.

Implemented as `gatv2_ranker_v3` in `engine/ranker.py`, using PyG `2.8.1`
pinned in `requirements-graph.txt`. The first skip is linear and the other two
are identity; normalization is per-node LayerNorm. Text projection uses ReLU
and is multiplied by the presence mask, so missing vectors remain zero after
projection bias. Attention dropout is 0.0; feature/head dropout defaults to 0.10.
The [ranker handoff](ranker.md) specifies checkpoint version/encoder/topology
checks and explicit inference failures. This is untrained model infrastructure,
not evidence of useful preference ranking or an integrated fallback.

The score is meaningful for **ranking candidates under the same source and target**, not as a universal design-quality percentage or calibrated probability. A score of 0.9 does not mean "90% good." For human explainability do not claim attention is an importance explanation.

**Baselines to train and evaluate alongside the GNN:** (1) deterministic geometric scorer; (2) MLP on pooled element features plus aggregate layout statistics. Adopt the GNN only if it improves held-out preference decisions and end-to-end design choice sufficiently to justify complexity.

---

---

S04 (candidate generation), S05 (annotation/training) and S06 (integration, risks) are
proposals and live in [planning/](planning/):
[S04](planning/S04_candidate_generation.md), [S05](planning/S05_annotation_training.md),
[S06](planning/S06_integration.md).
