# Adobe Chameleon — Technical Specification S03–S06

**Version:** 0.1 proposed implementation baseline  
**Date:** 2026-10-26  
**Status:** S03 graph construction/feature extraction and the untrained GATv2
ranker/checkpoint/scoring foundation are implemented. A deterministic output
validation gate, bounded repair search, and resize integration are also implemented
(see [output validation](output_validation.md)). Diverse generation and preference
training/evaluation remain proposals.
See [M2A interfaces and verification](graph_features.md) and [ranker interfaces](ranker.md).
**Dependencies:** S02 *Internal design representation*; Adobe v0.9 JSON schema; challenge RULES.md/FAQ.md; `engine/resize.py`  
**Primary objective:** Given a structured Adobe source design and target canvas, produce a legally structured, constraint-checked resized design and, internally, several materially different alternatives. Learn candidate preferences from human side-by-side comparisons rather than treating heuristic scores as ground-truth labels.

> **Scope caveat:** The full Adobe schema, RULES.md, FAQ.md, renderer, and fixture files were not attached in this conversation. S02 summarizes their obligations but is itself a proposal. This spec is a concrete architecture and default configuration, NOT independent verification of every contest rule. The original rules and authoritative JSON Schema win if any condition here conflicts with them. Do not implement invented restrictions as if they were official rules.

---

## 0. Decisions frozen for v0.1

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
  graph_v1.yaml
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

For the ranker, each element is one node containing **53 numeric fields**, a **384-dimensional frozen text vector** as a separate input, and a one-dimensional presence mask. The model projects the 384-dimensional vector to 32 and concatenates it, yielding **86 features per node** at the first GNN layer: `53 + 32 + 1 = 86`.

### 4.1 The 53 numeric fields, in exact order

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

Here `W,H` refer to the source canvas; `TW,TH` refer to the target; `SW,SH` are aliases of source width/height in the final three entries. Char count means `len(content)` in Python Unicode code points; word count means `len(content.split())`. There is no need to normalize or rewrite stored content. Clip all source normalized geometric numeric inputs in indices 3:10 and candidate geometry in 42:48 to `[-4,4]` **for features only**, except dimension/area terms which are lower-bounded at zero. Unknown non-hex color literals must not be altered; only the features get `parse_mask=0`. Parse `#RGB`, `#RRGGBB`, and `#RRGGBBAA` by expanding to RGB in `[0,1]` (the final AA portion is intentionally not encoded in this v0.1 RGB feature). The Adobe literals remain exact.

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
    x_numeric = FloatTensor[N, 53],
    text_vec  = FloatTensor[N, 384],
    text_mask = FloatTensor[N, 1],
    edge_index= LongTensor[2, E],
    edge_attr = FloatTensor[E, 31],
    element_ids = tuple(original_ids),
)
```

`float32` for feature tensors; `int64` for `edge_index`; finite values only. Every candidate of the same source has the same element index mapping and graph topology. The graph builder must not modify either `Design`. For dense N=4, assert 12 directed edges. For N=17, assert 272. Text encoder caching must not change the representation of repeated strings.

## 5. Scoring network: concrete v0.1 model

```text
numeric fields: [N,53] -----┐
text [N,384] -> Linear 32 ---+--> concat [N,86]
mask [N,1] ------------------┘
       GATv2Conv(86 -> 128, 4 heads x 32, edge_dim=31)
       LayerNorm(128) + ELU + dropout(0.10)
       GATv2Conv(128 -> 128, 4 heads x 32, edge_dim=31)
       LayerNorm(128) + ELU + dropout(0.10)
       GATv2Conv(128 -> 128, 4 heads x 32, edge_dim=31)
       LayerNorm(128) + ELU
       concatenate global_mean_pool and global_max_pool -> [graph,256]
       MLP Linear(256,128) -> ReLU -> Dropout(.10) -> Linear(128,1)
       -> one uncalibrated scalar S(source,candidate,target)
```

Set `add_self_loops=True`, `fill_value=0.0` for the internally generated self-loop edge attributes. Use an explicit residual/skip connection at each GNN stage if tensor shapes allow (`h_out = norm(conv(h,...) + projected_h)`), with its exact implementation versioned. For the initial three layers the input width differs only on the first layer, so use a `Linear(86,128)` skip projection there. The implementation **must specify the chosen PyG version** in lockfile and verify that these exact options are supported.

Implemented as `gatv2_ranker_v1` in `engine/ranker.py`, using PyG `2.8.1`
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

# S04 — Candidate Layout Generation and Computational Fallback

## 6. Candidate generation policy

**Input:** immutable S02 `Design`, `Canvas target`. **Output:** 1–K candidate `Design` instances with exactly the original element IDs in the original order. No mutations of source IDs/types, text content, font families, asset IDs, group-field presence/values, or extension fields. Only contest-permitted element geometry and style fields may change; whitelist the exact allowed fields after checking authoritative RULES.md/FAQ.md.

### 6.1 Layout units

`LayoutUnit` is an internal optimization construct, **not** a new Adobe JSON group. First initialize one unit per source element. For identical explicit, non-null `group_id` strings, record a hard **membership constraint** (the group ID cannot change), but do not assume all group members form one rigid rectangle; some source groups include backgrounds and text overlay. Derive optional **soft clustering** evidence from: existing group membership, geometric containment, horizontal/vertical proximity, repeated anchor alignments. Do not use literal IDs to infer logos or CTAs. v0.1 may score these relations by fixed deterministic formulas; a learned relation head is optional v0.2.

Concrete geometric evidence: for source boxes `i,j`, define `normalized_center_distance = hypot(dx,dy)`, `same_group` as §3.3, `contains` as §3.3, and `aligned_left` when `|x_i-x_j|/SW <= .02`. For an inferred *soft* pairing, propose `association(i,j) = 0.40*same_group + 0.25*contains_any + 0.20*exp(-distance/0.20) + 0.15*aligned_left`; these coefficients are **starting heuristics**, not learned labels or Adobe rules. Exclude background-like containment relations from rigid units; only use scores to create candidate search preferences. Audit this formula on the three provided fixtures before trusting it.

### 6.2 Candidate families and generation budget

Initial composition-family enum and expected trial count:

| Family | Policy | Trials |
|---|---|---:|
| `SCALE_BASELINE` | Proportional affine scale + translation, then repair | 2 |
| `VERTICAL_STACK` | Order high-level soft units vertically with alignment variants | 4 |
| `HORIZONTAL_FLOW` | Put units into one or two horizontal bands | 4 |
| `TWO_COLUMN` | Allocate left/right regions when target ratio permits | 4 |
| `HERO_LEFT` | Largest source image region on left, other units to right | 3 |
| `HERO_RIGHT` | Mirrored hero arrangement | 3 |
| `CONTROLLED_OVERLAY` | Allow text/shape/image intentional layering inferred from source containment and z evidence | 4 |

These 24 trials are *proposal seeds*, not a guarantee of 24 valid or distinct outputs. Applicability checks suppress unhelpful families (e.g., two-column arrangement for very tall narrow targets). "Largest image" uses bounding-box area, **not** an inferred semantic product role. Every proposal must pass the same validator before learned ranking. The baseline can be used even with no NLP encoder/model checkpoint.

Defaults: `max_raw_candidates=32`, `max_valid_candidates=12`, deterministic RNG seed `42` for repeatable development, generator time budget initially `1.0s` on the reference CPU (must measure and adapt against submission time limits), and best 3 diverse internal options. Do not promise latency until benchmarked.

### 6.2a Deterministic template seeding and eligibility

For each family, first classify **background-like candidates for search only**: a shape/image with `area >= 0.75 * source_canvas_area` and source box covering the canvas center, or an image with effective `fit="cover"` and box area at least 75% of canvas. This classification is a heuristic hint and cannot justify removal, changing source z-order, or automatic full-bleed exceptions. All original elements still receive candidate boxes. Existing source containment/overlap and explicit group evidence have precedence over this hint when proposing overlay pairs.

Define candidate **layout units** as connected components of strong soft-pairing edges with association >= 0.75, but exclude background-like nodes from that grouping and cap each inferred unit at 6 nodes. Start with explicit same-group associations (where geometrically appropriate), then consider proximity edges in descending score with stable source-index tie-breaks. Unit membership exists only during candidate search; explicit Adobe `group_id` is never changed. Maintain each unit's source bounding rectangle and member-relative fractional positions. A shape that contains text can remain a separate overlay layer rather than be forced into the same rigid packing box.

For `HORIZONTAL_FLOW`, compute usable width `TW*(1-2m)` and height `TH*(1-2m)`, with initial margin fraction `m=0.04`; order units by source top coordinate then left coordinate (ties by source index), assign consecutive units left-to-right to evenly spaced region slots, and uniformly scale each unit to fit its assigned slot while preserving member-relative geometry. For `VERTICAL_STACK`, swap axes; for `TWO_COLUMN`, place alternating units by descending source unit area in the currently less-occupied column, breaking ties by source order. `HERO_LEFT/RIGHT` allocates an initial 45% width to the largest image-containing unit and 55% to the rest, then runs vertical sublayout on the remainder. `CONTROLLED_OVERLAY` starts with horizontal/vertical flow and separately reinstates candidate overlaps for source-overlapping pairs; it must retain original stacking semantics. All families run the same subsequent local geometry optimizer. These percentages/order policies are deterministic **seeding defaults** and must be tuned or rejected on evaluation; they are not claimed as inherently attractive designs.

Do not encode reading order as Adobe element-array order. The source-position sort here is only a repeatable computational starting guess; later semantic role predictions may replace it if evaluated. Retain candidate designs in original Adobe array order regardless of slot assignment.

**Overlap eligibility for v0.1's computed penalty:** eligible foreground pairs are two non-background-like, nonzero-area elements whose source-box IoU is <=0.02, neither contains the other in source, and which do not share an explicit non-null group. Ineligible does **not** mean 'should overlap'; it means 'no unconditional overlap penalty'. Source pairs already overlapping are exempt from this generic penalty, but visual diagnostics may still flag extreme target occlusion. These thresholds are heuristic and must not be used as official validity criteria.

### 6.3 Decision variables and geometric refinement

Each element has candidate variables `(x_i,y_i,w_i,h_i)`. Rotation, image fit, opacity, and text styling are fixed by default unless explicitly authorized by rules. Candidate dimension minimums are not invented as hard contest rules: a preferred minimum font size or touch target is a **soft quality preference** only when those fields can legally change. Preserve zero-area lines and allowed zero-area elements under the S02 schema; don't impose invalid positive-area requirements.

Two-stage solver:

1. **Discrete layout-family and region assignment** chooses a composition pattern and ordering of internal units. Use enumeration/beam search initially; optionally OR-Tools CP-SAT for discrete assignments and selective mutually exclusive foreground-rectangle packing.
2. **Continuous/local geometry refinement** adjusts positions, scale, and gaps, measuring constraint violations after each move; a deterministic coordinate-descent/random-restart implementation is sufficient for v0.1. CP-SAT is not required for rotations, arbitrary render-aware text, or all soft metrics.

Never add `NoOverlap2D` over every element. Background and decorative layers often intentionally overlap foreground content; only include confirmed conflicting foreground pairs in optional packing constraints.

### 6.4 Soft objective, fully defined enough to implement

First satisfy hard validation constraints. Among valid layouts minimize:

\[
J(L)=w_m E_{move}+w_a E_{align}+w_p E_{pair}+w_s E_{scale}+w_o E_{unwanted-overlap}+w_b E_{balance}
\]

Use starting weights `[w_m,w_a,w_p,w_s,w_o,w_b] = [0.10,0.15,0.25,0.15,0.25,0.10]` (sum 1.0); tune on held-out human preferences. Each term is averaged over its eligible set and bounded to `[0,1]`, so weights are comparable:

- `E_move`: average `min(1, ||center_c_norm - center_s_norm||_2 / sqrt(2))`; **ignore intentional family-wide directional reflow in this penalty** by computing it with low weight, not forcing source positions.
- `E_align`: average `min(1, |anchor_i-anchor_j|/0.1)` for source pairs with aligned corresponding anchors (`<=0.02` in source), excluding confirmed decorative/overlay pairs. Anchors are left, center-x, right, top, center-y, bottom, using appropriate axis normalization.
- `E_pair`: for pairs selected as soft associations, `min(1, max(0,d_c-d_s-0.05)/0.25)` where `d` is normalized center distance. This penalizes large increases in proximity but does not require rigid grouping.
- `E_scale`: mean `min(1, |log((area_c+1)/(area_s+1)) - log((target_area+1)/(source_area+1))|/3)` for scalable boxes. Exclude appropriate full-bleed backgrounds if corroborated by fit/geometry.
- `E_unwanted-overlap`: average candidate IoU for **eligible foreground pairs** that did not overlap in source, capped at 1; exclude allowed intentional overlays, zero-area elements, and background-like containment pairs. Eligibility must be explicitly logged for debugging.
- `E_balance`: Euclidean offset between *area-weighted* element center-of-mass and target center, normalized by half-diagonal, capped at 1; cap each object's contribution to avoid a single full-bleed shape dominating; ignore full-bleed backgrounds.

These are practical starting definitions, not a validated aesthetic metric. Add a preview-based text-legibility check if a dependable renderer is available; without a renderer, text-box area is an unreliable approximation of glyph fit. Avoid optimizing an unverified "readability" proxy as a hard constraint.

### 6.5 Hard validation and policy boundary

Validator checks in this order:

1. **Adobe JSON Schema:** correct root keys and allowed field types; conditional type-specific fields; values and nullability.
2. **Structural identity:** exact original ID multiset in original array order; original type, required policy, group presence/value and all extension data preserved; unchanged content, font families, asset references, and other fields frozen by authoritative challenge rules.
3. **Canvas:** `candidate.canvas == target`, dimensions positive integral under S02.
4. **Geometry:** finite numbers; permitted element size/position ranges and in-bounds/visible-extent rules as actually specified by Adobe; check rotated extents separately when required.
5. **Image policy:** exact permitted box-ratio distortion checks and full-bleed/`fit:'cover'` exceptions from current rules; do not infer from file name or ID.
6. **Rendering diagnostics:** warn on likely unreadable text, over-cropping, unexpected overlap, and missing assets; distinguish warnings from official disqualifying errors.

If the challenge provides a hard ~5% source-box proportion tolerance, implement it **after confirming the exact official definition and exception logic**, not by assuming every source bounding box is an asset's intrinsic ratio. S02 says the sample advanced design includes an image asset reference that may be missing: keep the element and asset_id regardless.

### 6.6 Repair/fallback and infeasibility

Reject each failed proposal and feed its violations into the next generation/repair attempt. Continue until at least one candidate passes every implemented hard check. All baseline and repaired candidates pass through the same gate; no model score, timeout, or requirement to return JSON grants an exception. Revalidate the selected output before serialization. Never return a least-violating or structurally-valid-but-policy-invalid draft.

The implemented `find_valid_layout(source, target, propose, max_attempts=8)` returns the first validated JSON snapshot and gives the prior `ValidationReport` to the proposer. The current deterministic proposer first tries scaling, then proportionally fits and translates the entire source composition, with progressively smaller retries. It is a basic geometric repair, not the full template optimizer. `resize()` uses this path without ML.

A finite search budget prevents hangs on impossible constraints or a broken generator. On exhaustion raise `NoFeasibleLayout` with all attempt reports; do not return a design. Unexpected generator failures also propagate. The unchanged CLI writes outputs only after `resize()` returns, so this failure produces no new JSON/PNG for the failed task. Existing files from earlier runs are not deleted. A successful return certifies the documented gate, not render legibility or official aesthetic acceptance. See [implemented checks and limitations](output_validation.md).

## 7. Candidate option diversification

Among candidates passing all checked hard constraints:

1. Remove exact coordinate duplicates (`x,y,w,h` rounded to 0.5 px for **dedupe comparison only**, not output storage).
2. Compute layout distance between candidates A,B with matched IDs:

\[
D(A,B)=0.6\cdot\frac{1}{N}\sum_i\min\left(1,\frac{\lVert c_i^A-c_i^B\rVert_2}{0.25}\right)+0.4\cdot\frac{1}{N}\sum_i\min\left(1,\left|\log\frac{a_i^A+\epsilon}{a_i^B+\epsilon}\right|\right)
\]

Here centers are target-normalized, areas are target-area-normalized, and `epsilon=1e-6`. These choices are proposed heuristic defaults.
3. Sort by ranker score (or deterministic score if ranker unavailable), select best first, then greedily maximize `score_normalized + 0.25 * min_distance_to_selected`. Normalize scores by min-max among the current valid candidate set (all equal -> all 0.5).
4. Prefer different composition families as a tie-breaker. Return up to `k=3`; never fabricate three options if fewer distinct valid candidates exist.
5. The contest-facing `resize()` returns exactly the first chosen design in Adobe JSON; a separate `resize_with_options()` may expose a private list for the demo or annotation app.

---

# S05 — Preference Annotation and Ranker Training

## 8. Side-by-side labeling application

**Stack:** a local Gradio Blocks app backed by SQLite or append-only JSONL plus immutable candidate files. Store Adobe JSON, renderer preview, and provenance separately. The UI never edits original submissions.

Display layout A and B **for the same source design and the same target canvas**, with matched preview dimensions and access to a toggleable source preview. Randomize A/B screen position; retain canonical candidate identities in the database. Only compare schema/policy valid candidates for *aesthetic* ranking; review invalid examples in a separate diagnostics mode, not mixed with quality labels.

Allowed responses: `A_BETTER`, `B_BETTER`, `TIE`, `BOTH_BAD`, `SKIP`. Optional single reason: `balance`, `hierarchy`, `spacing`, `readability`, `grouping`, `image_treatment`, `other`; optional freeform note. Enable keyboard shortcuts and a progress counter; show no model scores to annotators.

Canonical record (one judgment per row):

```json
{
  "schema_version": "preference.v1",
  "comparison_id": "uuid",
  "source_design_id": "sha256_of_canonical_source_json",
  "target_canvas": {"width": 728, "height": 90},
  "candidate_a_id": "sha256_of_canonical_candidate_a_json",
  "candidate_b_id": "sha256_of_canonical_candidate_b_json",
  "screen_left_candidate_id": "candidate_a_or_b_id",
  "label": "A_BETTER",
  "reason": "hierarchy",
  "annotator_id": "local_pseudonym",
  "generator_version": "generator_v1",
  "renderer_version": "renderer_v1",
  "feature_version": "graph_v1",
  "created_at_utc": "ISO_8601_timestamp"
}
```

Candidate IDs and source IDs are derived from a canonical JSON serializer used **for hashing only**; this does not change S02's round-trip contract. Store comparison records even if repeated and do not silently overwrite disagreements. Avoid label leakage: the chosen candidate identifier may not be passed as a model feature. Labelers should periodically see ~10% hidden repeats to gauge consistency and ~10% cross-annotator overlaps if two people label.

**Pair-selection policy:** initially sample pairs with the same source/target and different layout families. Once a model exists, prioritize uncertain pairs, but reserve at least 25% random pairs to reduce active-learning selection bias. Allow `BOTH_BAD` to identify generator failure; do not directly treat it as `A=B` quality. It does not supply a winner for pairwise training.

## 9. Dataset and leakage prevention

- Group all candidates, views, and pairwise comparisons from the **same source design** into the same split, regardless of target, layout family, or generator version.
- Starting split: source-design-grouped train/validation/test = `70/15/15` by source count, stratified approximately by source orientation and element-count buckets if enough sources exist. Keep final test sources untouched during hyperparameter tuning.
- Pilot: 100–200 manually judged comparisons to test UI/rendering and label consistency. Then aim for an initial 500–1,500 comparisons across many independent source designs; these are planning ranges, not data sufficiency guarantees.
- In evaluation, report number of unique source designs, unique target canvases, preference pairs, valid layouts, and annotator agreement, not just pair counts.
- Do not create training winners by running the deterministic scorer over synthetic candidates; heuristics can **generate options and rule-filter them**, but quality labels must come from people or a separately justified source of ground truth.

## 10. Training objective

For A/B non-tie comparisons under the same source and target, compute model scores `s_a`, `s_b`, difference `d=s_a-s_b`, and target `y=1` for A better else `0`. Primary loss:

\[
\mathcal{L}_{pair}=\mathrm{BCEWithLogits}(d,y).
\]

For `TIE`, optionally use a weaker squared score-difference penalty `0.25 * d^2`, separately monitored; do not force ties into a winner. `BOTH_BAD` and `SKIP` do not contribute to the pairwise preference loss. A future separately supervised absolute-quality head may use `BOTH_BAD` only with a defined quality-labeling protocol.

**Initial training hyperparameters:** AdamW, learning rate `1e-3`, weight decay `1e-4`, 32 pairs per batch, gradient norm cap `1.0`, maximum 100 epochs, patience 12 measured on validation pairwise accuracy (use validation loss as tie-break), seed 42. Log Torch/PyG/transformer versions, feature versions, checkpoint hash, splits, and all hyperparameters. Use mixed precision only after confirming stable results; the 5090 is helpful but not a substitute for better labels.

**Checkpoint contract:** save architecture config, model weights, numeric-feature version, edge-feature version, text-encoder ID, training-source split manifest, random seed, metrics, and hashes. Reject incompatible checkpoints rather than silently making shape conversions.

## 11. Evaluation / ablations

Always compare on source-disjoint held-out designs:

1. `B0`: computed objective only (no learned ranking).
2. `B1`: pooled-feature MLP, no graph message passing.
3. `B2`: GNN without text embeddings.
4. `B3`: GNN with text embeddings.
5. `B4`: GNN with text embeddings + diverse candidate selection.

Metrics: held-out binary preference accuracy excluding ties, tie behavior separately, calibration of pairwise preference probabilities (not raw scores), human end-to-end preference win-rate against B0 on a fresh holdout, proportion of outputs meeting all checked contest constraints, number of distinct alternatives, p50/p95 runtime, and failures per family.

A GNN whose offline preference accuracy improves but whose end-to-end outputs do not improve should not be claimed to have solved the design problem. Do a blinded A/B evaluation of final outputs.

---

# S06 — Integration, Acceptance Tests, and Implementation Order

## 12. End-to-end control flow

The implemented output boundary uses the shared gate:

```python
from engine.internal_schema import Canvas, Design
from engine.layout_search import baseline_proposal, find_valid_layout

def resize(source: dict, target_canvas: dict) -> dict:
    design = Design.from_adobe(source, target_canvas=target_canvas)
    target = Canvas.from_adobe(target_canvas)
    return find_valid_layout(
        design, target,
        lambda attempt, previous_report: baseline_proposal(design, target, attempt),
    )
```

Future multi-candidate/GNN integration must filter invalid candidates before scoring, keep generating if the valid set is empty, and revalidate the selected JSON before returning it. Pass an explicit compatible trained checkpoint and encoder cache to `score_candidates`; on ranker failure use deterministic scores **only over the valid set**. Neither ranking path may bypass the output gate. A bare graph/ranker call is not a legality check.

## 13. Required acceptance tests before merging

**Representation/invariance:** use all three Adobe fixtures if accessible. Round-trip source JSON exactly as decoded; preserve null vs absent, numeric types, array order, extension objects, literal text, asset IDs, group IDs, and source canvas. Assert candidate source is unchanged after generation and serialization. Preserve advanced missing-asset reference.

**Graph:** N=1 -> one node, zero explicit edges; N=4 -> 12 directed edges; N=17 -> 272 directed edges. Duplicate IDs must be rejected upstream. For `group_id=None`, `group_id=ABSENT`, and `group_id=''`, assert correct `same_explicit_group` behavior. For negative `x/y`, zero width/height, and rotated boxes, assert finite graph values and no stored-data mutations. Reverse edge relative-displacement signs must be opposites. Every candidate for the same source has the exact same ordered IDs and edge-index topology. Check `x_numeric.shape==(N,53)`, `text_vec==(N,384)`, `text_mask==(N,1)`, `edge_attr==(E,31)`, output scalar finite.

**Model:** shape tests on CPU and GPU; deterministic eval with dropout disabled; no NaN on all-zero text embeddings; batched graphs retain separate pooling outputs; reject checkpoint dimension/version mismatch. Confirm text availability behavior using a numeric-only checkpoint where needed. Attention returned for debugging must not be exposed as object-importance labels.

**Generator:** every returned candidate preserves exact ID count/order, original types/immutable fields, and target canvas. Run each family on warm-up/core/advanced designs; assert no candidate is marked valid if it violates a known rule. Check intentional background overlaps are not globally disallowed. Check invalid attempts are retried with diagnostics, a later valid attempt is returned, and exhaustion raises without emitting an invalid draft. Check infeasible layouts are not falsely passed. Diversity returns <=3 options and no exact duplicates.

**Preference UI:** reversed display order does not invert canonical labels; comparisons have same source and target; no score leakage; tie/both-bad/skip recorded correctly; repeats retained; labels survive app restart; append-only data stored with renderer/generator versions.

**Evaluation:** no source ID leaks across train/val/test; report dataset sizes by source; compare MLP/GNN to computed baseline; holdout final-output comparison uses designs not seen during annotation/training.

## 14. Implementation sequence and ownership

| Milestone | Geometry / integration owner | GPU / ML owner | Exit criterion |
|---|---|---|---|
| M1 | Finish S02 and validator | Draft exact graph tensor tests | All source fixtures round-trip |
| M2 | Implement families and computed objective | Frozen text encoder + graph builder | >1 valid candidate in at least one fixture |
| M3 | Renderer QA and option preview | Human pairwise annotation UI | Reproducible preview + persisted preferences |
| M4 | Solver repairs and diversity selection | Train MLP baseline, then GNN ranker | Validity and held-out accuracy reports |
| M5 | Integrate `resize` adapter | Pairwise/ranker ablation | One valid selected JSON, fallback works |
| M6 | Runtime / advanced case stress test | Holdout end-to-end A/B comparison | Benchmarked speed, failure modes documented |

## 15. Known risks requiring explicit review

- Adobe source designs may have no semantic group labels, so group inference is uncertain by design.
- A fully connected graph learns *potential* cross-element influence; it does not discover visually meaningful groups simply because attention exists.
- Text embeddings do not reveal image content; an image file's `asset_id` must not be interpreted as a trusted semantic category.
- Box geometry is not necessarily rendered glyph or intrinsic image geometry. Renderer fidelity is essential for human preference labels.
- The 17-element 728×90 target may impose incompatible aesthetic preferences; a formal feasible solution depends on actual contest rules.
- Model quality may be limited by source-design diversity even with thousands of pair labels. A strong MLP baseline is mandatory.
- An absent model must not cause the core resize path to fail. The deterministic pipeline is a first-class product, not an emergency afterthought.

## 16. References and authority order

1. Authoritative project files, when available: `chameleon_design_schema_v0.9.json`, `RULES.md`, `FAQ.md`, and `engine/resize.py` (highest priority for permitted outputs and constraints).
2. S02 *Internal design representation* (provided in this conversation): the intended lossless Python representation, data-preserving semantics, and current stage-one assumptions.
3. PyTorch Geometric `GATv2Conv` API (supports `edge_dim` and returned per-head attention): https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATv2Conv.html
4. Google OR-Tools CP-SAT 2D nonoverlap model, relevant only to selected packing pairs: https://github.com/google/or-tools/blob/stable/ortools/sat/cp_model.proto
5. This S03–S06 document: a **proposed, testable** implementation design. Its feature counts, weights, candidate families, and training parameters are engineering defaults to verify experimentally, not claims from Adobe or experimentally optimal settings.
