# Adobe Chameleon — Technical Specification

**Spec version:** 0.3 (consolidated; internal decisions recorded) · **Last updated:** 2026-10-28
**Contract versions:** feature contract `graph_v3` (default) and `graph_v4` (opt-in) · ranker `gatv2_ranker_v3` / `gatv2_ranker_v4` · checkpoint container `ranker_checkpoint_v1`

This is the single source of truth for the Chameleon engine's architecture and component contracts. A
developer working on any one of S02–S06 should be able to start from this document and the appendix it
points to. Planning (who does what, milestone progress) is in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md);
CI is in [CI.md](CI.md).

## 0. How to read this document

### 0.1 Status labels

| Label | Meaning |
| --- | --- |
| **Implemented** | Code exists in the repository and a module under `tests/` exercises it (verified by inspection; this document does not record test results — run the suites per [CI.md](CI.md)). |
| **Partial** | Some of the documented contract exists; the missing part is listed. |
| **Planned** | Proposed design only. No code. Never treat as working. |
| **Unresolved** | An architectural decision that cannot be established from code, official documents, or settled architecture. Listed, with a status each, in the decision register (§10.2). |

### 0.2 Precedence (highest first)

1. **Official Adobe documents:** [RULES.md](../RULES.md), [FAQ.md](../FAQ.md), [chameleon_design_schema_v0.9.json](../chameleon_design_schema_v0.9.json) (provisional, v0.9), and the `engine/resize.py` docstring supplied with the starter kit. Never edited by us. The main challenge brief (a PDF outside this repository) is also official; where it and the repository files differ, see [Appendix F](reference/F_submission_and_adobe_questions.md).
2. **Current code and tests,** for what is implemented. If this document disagrees with the code about implemented behavior, the document is wrong and must be corrected.
3. **This document (§1–§10).**
4. **Appendices A–F.** They are parts of this specification that hold detail too long for the main text. Each requirement lives in exactly one place; the main text says which. An appendix may not contradict §1–§10.
5. [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md), [CI.md](CI.md): planning and CI procedure only. They do not define contracts.
6. [AGENTS.md](../AGENTS.md): workflow for coding agents. It does not define contracts.
7. [reference/HISTORY.md](reference/HISTORY.md): non-normative.

When sources conflict, name the conflict with file and line references before changing behavior. Do not
reconcile an uncertainty by inventing a constraint.

### 0.3 Document map

| Content | Authoritative location |
| --- | --- |
| Pipeline, architectural decisions, component status | §1 |
| Shared types and every cross-component signature | §2 |
| S02 summary; full field tables, presence semantics, serialization | §3 → [Appendix A](reference/A_internal_representation.md) |
| S03 summary; topology, feature tables, color parsing, text embeddings | §4 → [Appendix B](reference/B_graph_features.md) |
| Ranker architecture, checkpoints, scoring failures | §4, §6 → [Appendix C](reference/C_ranker_model.md) |
| S04 implemented retry loop and repair; planned generator | §5 → [Appendix D](reference/D_candidate_generation.md) |
| Ranking, fallback, selection; annotation/training | §6 → [Appendix E](reference/E_annotation_training.md) |
| Integration (`resize`, CLI, deployment) | §7 |
| **All output constraints** and failure handling | **§8** |
| Tests and acceptance criteria | §9 |
| Limitations and unresolved decisions | §10 |
| Stage-one submission requirements; questions for Adobe (U1, U8, U9, U11) | [Appendix F](reference/F_submission_and_adobe_questions.md) |

---

## 1. System architecture and pipeline

### 1.1 Goal

Input: an Adobe v0.9 design (`source`: `canvas` + `elements`) and a `target_canvas`. Output: one design
with exactly `canvas` (the target) and `elements` at its root, every original element retained once
(§8.1). Internally the engine may build several candidate layouts and rank them; the competition-facing
boundary returns one.

### 1.2 Pipeline

```text
source JSON + target_canvas
        │  S02: Design.from_adobe
        ▼
   Design (source)
        │
        ├──────────────► S04 candidate production ──► LayoutCandidate(s)          [planned batch generator]
        │                  (find_valid_layout retry loop + baseline_proposal)      [implemented]
        │                              │
        │                              ▼
        │                    validate_candidate  (the only output-policy check, §8.1)
        │                              │ valid candidates only
        │                              ▼
        ├──► S03 build_candidate_graph(source, candidate.design, target)
        │                              │
        │                              ▼
        │              S05 ranking: learned score (GNN)  ─ fallback → deterministic score [planned]
        │                              │
        │                              ▼
        │                    selection / diversification                           [planned]
        │                              │
        │                              ▼
        └──────────────► revalidate selected candidate → candidate.design.to_adobe() → {canvas, elements}
```

The **currently implemented** path is only: `Design.from_adobe` → `find_valid_layout` (with
`baseline_proposal`) → validated JSON. There is no batch generator, no deterministic scorer, no learned
selection and no diversification in `resize()` today (§5, §6, §7).

### 1.3 Architectural decisions

1. **Computational path first, permanent.** Candidate generation, validation and repair work without any ML package or checkpoint. The GNN only ranks already-valid candidates and never decides validity.
2. **S02 is the lossless source-of-truth layer.** No other module defines Adobe element/canvas classes or parses Adobe JSON. Nothing mutates the source `Design`.
3. **One graph per `(source design, candidate design, target canvas)`**, exactly one node per original element; explicit Adobe groups are edge evidence, not graph partitions (Appendix B.2).
4. **Edges encode measurements, not importance.** GATv2 attention coefficients are message-aggregation weights, not object importance, grouping truth or a preference explanation.
5. **No object-importance head in v0.1.** The trainable output is one scalar score for a whole candidate conditional on source and target. A node/relation head would need its own supervised labels and is not part of this contract.
6. **One edge-aware GNN ranker** (GATv2); do not require a second GNN before candidate generation.
7. **Human pairwise preferences are the training labels.** Heuristic scores and validity filters may generate/filter candidates but are never ground-truth preference labels (Appendix E).
8. **Output is Adobe JSON with only `canvas` and `elements`.** Role hypotheses, graph data, scores, source context and UI data never appear in returned JSON.
9. **Internal alternatives are allowed; the official boundary returns one design.**
10. **Roles are never inferred from IDs or notes** (`bg`, `logo`, `_note` text, asset filenames carry no authority).

### 1.4 Component status

| Component | Code | Status |
| --- | --- | --- |
| S02 parser/serializer/models | `engine/internal_schema.py` | **Implemented**; editing APIs `with_updates`, `with_layout` **Planned** (A.8) |
| S03 graph builder, text-embedding cache, text-fit estimate | `engine/graph_features.py`, `engine/text_embeddings.py`, `engine/text_fit.py` (stdlib only) | **Implemented** (real pretrained encoder never loaded or evaluated) |
| S03/S05 GATv2 ranker, checkpoints, scoring | `engine/ranker.py` | **Implemented**, untrained; no trained checkpoint exists |
| S04 output validator | `engine/candidate_validator.py` | **Implemented** (limitations in §8.1.4) |
| S04 iterative retry + proportional repair | `engine/layout_search.py` | **Implemented** |
| S04 batch candidate generator, layout units, objective, diversification | — | **Planned** (Appendix D) |
| S05 deterministic scorer | — | **Planned**; requirements settled, interface recommended in §6.2 (U3) |
| S05 annotation UI, dataset, training, evaluation | — | **Planned** (Appendix E) |
| S06 `resize` adapter | `engine/resize.py` | **Partial**: validated proportional fit only |
| Preview renderer | `renderer/converter/` (vendored, used by `engine/main.py`) | **Implemented**; fidelity to the official scorer **Unresolved** (U9) |
| CI | `.github/workflows/ci.yml`, `scripts/ci_tests.py` | **Implemented** ([CI.md](CI.md)) |

---

## 2. Shared data structures and interfaces

This section is the only place cross-component types and signatures are defined. Appendices define
internal behavior of the functions listed here, not their signatures.

### 2.1 S02 types — **Implemented**

`Canvas`, `StyleRun`, `Element`, `Theme`, `Design`, the `ABSENT` sentinel and `SchemaError(ValueError)`
live in `engine/internal_schema.py`; their fields, presence semantics and methods are defined in
[Appendix A](reference/A_internal_representation.md) (A.1 lists the public API). A `Design` carries three
canvas references: `canvas` (coordinate system of the current layout), `source_canvas` (original) and
`target_canvas` (requested, or `None`).

### 2.2 `LayoutCandidate` — **Planned**

```python
@dataclass(frozen=True)
class LayoutCandidate:
    design: Design              # target canvas; IDs and order preserved
    candidate_id: str           # internal identity; recommended: content hash of the Adobe JSON (§10.3, U5)
    family: str                 # composition-family name, internal only
    seed: int
    generator_version: str
```

A `LayoutCandidate` wraps a `Design` with candidate-level metadata. It is **not** Adobe output: only
`candidate.design.to_adobe()` is serialized. No `LayoutCandidate` class exists in code; consumers that
exist today accept a plain `Design` (S03) or an object exposing `.design` (`CandidateWithDesign`, S05).

### 2.3 Functions and result types

| Name | Signature | Status | Code |
| --- | --- | --- | --- |
| `Design.from_adobe` | `(source: Mapping, *, target_canvas: Mapping \| None = None) -> Design` | Implemented | `internal_schema.py:421` |
| `Design.to_adobe` | `() -> dict` | Implemented | `internal_schema.py:441` |
| `generate_candidates` | `(source: Design, target: Canvas, config: LayoutConfig) -> list[LayoutCandidate]` | **Planned**; return type is settled architecture; contract in §5.4 | — |
| `LayoutConfig` | frozen dataclass; proposed fields in §5.4 | **Planned**; fields are a recommendation (U2) | — |
| `find_valid_layout` | `(source: Design, target: Canvas, propose: Proposal, *, max_attempts: int = 8) -> dict` | Implemented | `layout_search.py:23` |
| `Proposal` | `Callable[[int, ValidationReport \| None], Design \| Mapping[str, Any]]` | Implemented | `layout_search.py:20` |
| `baseline_proposal` | `(source: Design, target: Canvas, attempt: int) -> dict` | Implemented | `layout_search.py:50` |
| `validate_candidate` | `(source: Design, candidate: Design \| Mapping[str, Any], target: Canvas) -> ValidationReport` | Implemented; **the authoritative output-policy validator** | `candidate_validator.py:57` |
| `ValidationReport` | frozen dataclass `(schema_ok: bool, policy_ok: bool, geometry_ok: bool, violations: tuple[str, ...])` with property `is_valid` | Implemented | `candidate_validator.py:14` |
| `build_candidate_graph` | `(source: Design, candidate: Design, target: Canvas, *, text_embeddings: TextEmbeddingCache \| None = None, feature_version: str = "graph_v3") -> torch_geometric.data.Data` | Implemented | `graph_features.py:304` |
| `score_candidates` | `(source: Design, candidates: Sequence[Design \| CandidateWithDesign], target: Canvas, *, checkpoint: RankerCheckpoint \| None = None, text_embeddings: TextEmbeddingCache \| None = None, batch_size: int = 32) -> list[float]` | Implemented | `ranker.py:345` |
| `score_deterministic` | `(source: Design, candidates: Sequence[Design \| CandidateWithDesign], target: Canvas, *, weights: ObjectiveWeights \| None = None) -> list[float]` | **Planned**; proposed signature, mirrors `score_candidates` (§6.2, U3) | — |
| `rank_candidates` | `(source: Design, candidates: Sequence[LayoutCandidate], target: Canvas, *, checkpoint: RankerCheckpoint \| None = None, text_embeddings: TextEmbeddingCache \| None = None) -> RankedCandidates` | **Planned**; proposed learned-then-deterministic entry point (§6.4, U3) | — |
| `RankedCandidates` | `(scores: tuple[float, ...], scorer: Literal["learned", "deterministic"], fallback_reason: str \| None)` | **Planned**; proposed | — |
| `ScoredCandidate` | `(candidate: LayoutCandidate, report: ValidationReport, score: float, scorer: Literal["learned", "deterministic"])` | **Planned**; proposed: `score` is never `None` (§6.4) | — |
| `select_options` | `(candidates: list[ScoredCandidate], k: int = 3) -> list[ScoredCandidate]` | **Planned**; provisional name (U4) | — |
| `resize` | `(source: dict, target_canvas: dict) -> dict` | Implemented (**Partial**, §7) | `resize.py` |
| `resize_with_options` | internal multi-option entry point for demo/annotation | **Planned**; provisional name (U4) | — |

Checkpoint functions `save_ranker_checkpoint`, `load_ranker_checkpoint`, `LayoutRanker`, `RankerConfig`,
`RankerCheckpoint` are specified in [Appendix C](reference/C_ranker_model.md). `TextEmbeddingCache`,
`EmbeddingProvider`, `FrozenSentenceTransformerProvider` are specified in Appendix B.5.

**Interface rules.**

- `Design` is the internal representation of one complete design. `LayoutCandidate` wraps a `Design`.
- S03 graph construction receives a candidate's `Design`, never a `LayoutCandidate`.
- S05 scores and ranks **validated** candidates; S05 callers pass `candidate.design` (or the candidate object, via `.design`).
- The final adapter serializes the selected candidate's `Design` with `to_adobe()`.
- `validate_candidate` is the only function that decides whether a design satisfies output policy. S02 parses, represents and serializes; it does not check output policy (§3).
- `generate_candidates` (planned batch generator) and `find_valid_layout` (implemented iterative retry loop) are different mechanisms with separate responsibilities: generation proposes, `validate_candidate` decides, `find_valid_layout` repairs by iterating proposal and validation, ranking orders valid candidates. How they compose is specified in §5.4.
- Matching between source and candidate elements is by exact ID, never position, geometry or guessed role.

### 2.4 Exceptions

| Exception | Defined in | Meaning |
| --- | --- | --- |
| `SchemaError(ValueError)` (has `.path`) | `internal_schema.py` | Invalid Adobe JSON or invalid model construction |
| `NoFeasibleLayout(RuntimeError)` (has `.reports`) | `layout_search.py:11` | Retry budget exhausted without a valid layout; not a proof of infeasibility |
| `CheckpointUnavailable(RuntimeError)` | `ranker.py` | Missing, corrupt, incompatible or untrained checkpoint |
| `ModelInferenceError(RuntimeError)` | `ranker.py` | Any failure during scoring (encoder mismatch, unsupported topology, non-finite output, …) |
| `EmbeddingUnavailable(RuntimeError)` | `text_embeddings.py` | Provider cannot embed a string; handled by the cache (B.5) |

---

## 3. S02 — Internal representation and Adobe adapters

**Status: Implemented** (`engine/internal_schema.py`, `tests/test_internal_schema.py`). **Planned:**
`Element.with_updates`, `Design.with_layout` (A.8).

Contract (details in [Appendix A](reference/A_internal_representation.md)):

- Lossless round trip: `Design.from_adobe(x).to_adobe() == x` as a decoded JSON value, including absent vs explicit `null`, numeric `int`/`float` types, nested extension data, style runs and array order (A.2, A.3, A.9).
- All optional Adobe fields default to `ABSENT`; schema defaults are exposed only through `effective_value` (A.3).
- Models are frozen, keyword-only dataclasses; `extras` are read-only views; exported JSON is always a fresh tree (A.1).
- Every parsed original element has the internal flag `required=True`; this means "retain", not "placement is valid" (A.4).
- Parsing accepts schema-valid input that violates target constraints (for example negative coordinates, a missing asset reference, out-of-range style runs). It performs no feasibility, bounds or image-proportion checks.
- Root and canvas extension keys are rejected; element and style-run extension keys are preserved.
- S02 contains no output-policy validation; that is `validate_candidate` only (§8.1).

---

## 4. S03 — Graph construction and feature representation

**Status: Implemented** (`engine/graph_features.py`, `engine/text_embeddings.py`, `engine/ranker.py`;
tests `tests/test_graph_features.py`, `tests/test_ranker.py`; optional ML dependencies in
`requirements-graph.txt`).

Contract:

- Versions `graph_v3` (default, unchanged) and `graph_v4` (opt-in; the same columns plus a text-fit node feature and a size-order edge feature, B.4.3). Tensor shapes, node/edge feature tables, formulas, clipping, topology (`dense_v1`, `sparse_v1`), color parsing and text-embedding behavior: [Appendix B](reference/B_graph_features.md). Feature order is locked; any change requires a new version string, tests, and a ranker/checkpoint version bump together (§8.2).
- Models `gatv2_ranker_v3` (for `graph_v3`) and `gatv2_ranker_v4` (for `graph_v4`), checkpoints, and scoring failures: [Appendix C](reference/C_ranker_model.md). The model is untrained; its outputs are not evidence of design quality.
- Text-vector input and the text encoder are optional; the numeric-only path is the default (B.5, C.2). `graph_v3` distinguishes candidates that differ only in text font size (columns 53:55). `graph_v4` adds an explicit text-fit feature and a size-order edge feature after an information-content evaluation (B.7.2); whether it ranks better than `graph_v3` is untested and `graph_v3` stays the default until it is (B.7.3, U13).
- `build_candidate_graph` does not check output policy. It raises `ValueError` for missing, unexpected or duplicate IDs and changed element types (B.1).
- The baseline (S02, `resize`, validator, retry loop) imports no ML package; CI enforces this (§8.2, [CI.md](CI.md)).

---

## 5. S04 — Candidate generation and layout optimization

**Status: Partial.** The retry loop, proportional repair and validator are implemented. The batch
generator, layout units, objective and diversification are **Planned** ([Appendix D](reference/D_candidate_generation.md)).

### 5.1 Implemented: iterative retry loop (`find_valid_layout`)

- `max_attempts` must be a positive `int`, else `ValueError`.
- For each attempt index (from 0), call `propose(attempt, previous_report)`; the first previous report is `None`.
- The proposal is snapshotted (`Design.from_adobe(...).to_adobe()` for a `Design`/mapping) and validated against `validate_candidate`; the returned object is exactly the validated snapshot.
- A snapshot that fails schema parsing is recorded as `ValidationReport(False, False, False, ("schema: …",))` and the loop continues.
- The first valid snapshot returns immediately, including on the final allowed attempt.
- If all attempts fail, `NoFeasibleLayout(reports)` is raised. No last-attempt or least-violating draft is ever returned. Unexpected exceptions from `propose` propagate.
- The attempt budget is an internal reliability limit, not an Adobe time limit.

### 5.2 Implemented: proportional repair (`baseline_proposal`)

- Attempt 0: independent-axis scaling by `target.width/source.width` and `target.height/source.height` (the starter kit's uniform-stretch idea).
- Attempts ≥ 1: uniform scale `min(tw/sw, th/sh) · 0.8^(attempt−1)`, adjusted (up to four passes) so the measured extents of the whole composition — including rotation and fixed-width shape strokes (`element_bounds`) — fit inside the target; then the composition is translated to center. Relative element positions, groups, image ratios, order and immutable fields are preserved.
- Text `font_size` becomes `max(1.0, font_size · min(sx, sy))`.
- This is a geometric fallback, not a claim of attractive or legible reflow. It does not search font sizes or compositions.

### 5.3 Planned: batch generator

`generate_candidates(source, target, config) -> list[LayoutCandidate]` (§2.3) with layout units, composition
families, two-stage solver, soft objective and diversification: Appendix D. Every produced candidate must
pass `validate_candidate` before ranking (§8.1.3). `find_valid_layout` is kept as the iterative
validation/repair mechanism; §5.4 defines how the two compose.

### 5.4 Planned: generator / validator / repair integration contract (U2)

**Status:** the architecture (`generate_candidates` returns `list[LayoutCandidate]`; `find_valid_layout`
stays the iterative validation/repair loop; generation, validation/repair and ranking stay separate) is
settled direction. The signatures, fields and flow below are the **recommended contract**; implementing
`generate_candidates` requires team sign-off on them. Nothing here exists in code.

**Responsibilities.**

| Step | Owner | Rule |
| --- | --- | --- |
| Propose layouts | `generate_candidates` | Pure and deterministic for a given `(source, target, config)`. Imports no ML package. Never calls `validate_candidate`, never scores, never raises on an infeasible proposal. Returns candidates in a deterministic order. Every returned `Design` has the target canvas and the original IDs in the original order (built with the planned `Design.with_layout`, A.8). The decision variables of every text element are its box **and** its `font_size` (D.1.4); a generator that only moves or resizes boxes does not satisfy this contract. |
| Decide validity | `validate_candidate` | Only authority (§8.1). Applied to every candidate, including the baseline. |
| Repair | `find_valid_layout` | Unchanged signature and semantics (§5.1). Used per invalid candidate through a *candidate proposal*: attempt 0 proposes `candidate.design` unchanged; later attempts delegate to a repair strategy that receives the previous `ValidationReport` and ends at `baseline_proposal`. Its result is a validated dict; the pipeline wraps it in a new `LayoutCandidate` that keeps `family`, `seed` and `generator_version` and has its own `candidate_id` (U5). |
| Rank | `rank_candidates` (§6.4) | Sees only validated candidates. |

**Flow** (replaces the planned flow in §7.3 once implemented):

1. `generate_candidates(source, target, config)`. With `include_baseline=True` (default) the list always contains the proportional-baseline candidate (family `SCALE_BASELINE`), so the batch path is never less feasible than today's `resize`.
2. For each candidate in order: `validate_candidate`; if invalid, repair with at most `config.repair_attempts` attempts via `find_valid_layout`; a candidate whose repair raises `NoFeasibleLayout` is dropped and its reports are kept for diagnostics.
3. Remove duplicates by `candidate_id` (content-addressed if U5 is accepted).
4. If the valid set is empty, run today's `find_valid_layout(baseline_proposal)` once (skipped when the baseline candidate was already repaired and failed, since it is deterministic); if that also raises, `NoFeasibleLayout` propagates (§7.5, U10).
5. Rank, select, revalidate the selected design, serialize (§6.4, §7.3).

**`LayoutConfig` (recommended fields).** Frozen, hashable, JSON-serializable, so a config can be recorded in
annotation and training manifests (E.1). Every default is a proposal.

| Field | Type / default | Purpose |
| --- | --- | --- |
| `seed` | `int = 42` | Single source of randomness; same config gives the same candidates (matches D.1.2). |
| `families` | `tuple[str, ...]` = all applicable of D.1.2 | Which composition families to try. |
| `include_baseline` | `bool = True` | Always emit the `SCALE_BASELINE` candidate. |
| `max_raw_candidates` | `int = 32` | Cap on proposals before validation (D.1.2). |
| `max_valid_candidates` | `int = 12` | Cap on validated candidates passed to ranking (D.1.2). |
| `repair_attempts` | `int = 8` | `max_attempts` per candidate for `find_valid_layout` (matches today's default). |
| `group_mode` | `Literal["unit", "reflow"] = "unit"` | Start by moving explicit groups as units; `"reflow"` permits controlled reflow of a group's members (§8.1.5). |
| `objective_weights` | `ObjectiveWeights \| None = None` | Soft-objective weights for the deterministic scorer (D.1.5); `None` means the provisional defaults of D.1.5 (untested; §6.2). |

Budgets are counts, not wall-clock seconds: a wall-clock budget would make output depend on machine speed
and break reproducibility. D.1.2's "1.0 s" generator budget remains a measurement target, not a config
field. `generator_version` is stamped by the code, not configured.


---

## 6. S05 — Ranking and candidate selection

**Status: Partial.** The learned scorer is implemented but untrained; deterministic scoring, selection and
training are **Planned**.

### 6.1 Learned ranking — Implemented (untrained)

`score_candidates` (§2.3) returns one finite float per candidate in input order. Scores are comparable only
across candidates of the same source and target; they are not calibrated probabilities. It requires a
checkpoint whose training metadata declares `status="trained"`; otherwise it raises `CheckpointUnavailable`.
Details and failure cases: Appendix C.2–C.4. **No trained checkpoint exists.**

### 6.2 Deterministic scorer — Planned (U3)

**Status:** the requirements below are settled direction. The signature, return types and exception
behavior are the **recommended contract**; implementation needs team sign-off.

**Role.** The deterministic scorer is the primary scorer until a trained checkpoint exists, and the
fallback when the learned scorer fails (§6.4). It is the permanent baseline that the GNN is compared against
(Appendix E.4).

**Requirements.**

- Operates on **validated** candidates only. It does not call `validate_candidate`; validity and preference stay separate. Passing an unvalidated design is a caller error, not something the scorer detects.
- Imports no PyTorch, PyG, text encoder or checkpoint, and runs in the baseline environment (§8.2 item 7).
- Evaluates, as soft terms only, visual quality: unwanted overlap, alignment, spacing, text-readability *estimates*, and group cohesion, together with the movement, pairing, scale and balance terms already in D.1.5. The term definitions live in D.1.5; this section does not repeat them. Readability uses only stored fields (font size, box, content length, declared colors) and never inspects assets or pixels; it is an estimate, and text fit remains unverified (K15).
- Same source and target for every candidate in one call; scores are comparable only within a call.

**Proposed signature:** `score_deterministic` in the §2.3 table. Higher is better. It mirrors
`score_candidates` so callers can swap one for the other, and it returns one finite float per candidate in
input order. `ObjectiveWeights` is a frozen dataclass with one float per term of D.1.5 (nine terms).
`weights=None` means the **provisional** defaults stated in D.1.5: untested starting values that are not final,
with the calibration procedure in D.1.6. Provisional weights do not block implementing the scorer; they block
calling its output validated. Any change to a weight or constant bumps `objective_version`.

**Behavior.**

- Pure and deterministic: no randomness, no I/O, no clock. Ties are broken by input order (first wins).
- Returns `[]` for an empty candidate list (as `score_candidates`).
- Raises `ValueError` for malformed input (a candidate whose IDs do not match the source, or a changed element type), as `build_candidate_graph` does. It never returns a partial list and never substitutes a default score.
- The scorer carries an `objective_version` string. Weights and term definitions are versioned with it.

### 6.3 Selection and diversification — Planned (U4)

Among validated candidates, rank by learned score (or deterministic score) and select up to `k` meaningfully
different alternatives (D.2). The official boundary uses only the first choice.

### 6.4 Fallback policy

1. Only candidates that pass `validate_candidate` are scored or selectable. A score never overrides a failed check.
2. If the learned scorer raises `CheckpointUnavailable` or `ModelInferenceError`, the caller must rank the **valid** set with the deterministic scorer. No random-weight or zero-embedding substitution. (No caller exists yet; the deterministic scorer is Planned.)
3. If no candidate is valid, continue generation/repair within a finite budget, then raise `NoFeasibleLayout`.
4. The selected design is revalidated before serialization.

**Recommended fallback contract (U3; needs sign-off).** The planned `rank_candidates` (§2.3) is the single
place that implements items 2 and 4's ranking half:

| Situation | Behavior |
| --- | --- |
| `checkpoint` is `None`, or no trained checkpoint is configured | Use `score_deterministic`; `scorer="deterministic"`, `fallback_reason="no_checkpoint"`. This is the normal state until a model is trained, not an error. |
| `score_candidates` raises `CheckpointUnavailable` | Rank *all* candidates with `score_deterministic`; `fallback_reason` carries the exception message. |
| `score_candidates` raises `ModelInferenceError` | Same: all candidates, deterministic scorer, reason recorded. No partial learned scores are kept. |
| Any other exception from `score_candidates` | Propagates. In the implemented `score_candidates` only an invalid `batch_size` (`ValueError`, raised before the guarded block) does. |
| `score_deterministic` raises (for example `ValueError` for a candidate that does not match the source) | Propagates; there is no further fallback. |

Implemented behavior to account for: `score_candidates` wraps **every** exception raised while building graphs
and scoring (`ranker.py`, the `except Exception` block) into `ModelInferenceError`, including a graph-building
`ValueError` for a candidate whose IDs do not match the source. A broken candidate therefore triggers the
fallback, and it is the deterministic scorer's own input check (previous table row) that then surfaces it.

All candidates in one `RankedCandidates` are scored by the same scorer; learned and deterministic scores are
never mixed or compared. `ScoredCandidate.score` is therefore always a finite `float`, and the earlier
idea of `score=None` is dropped. The fallback is silent to the caller's output (the same `resize` result
type) but never silent to developers: `scorer` and `fallback_reason` are returned and must be logged. The
names `model_missing` / `model_incompatible` are not used (HISTORY.md).

### 6.5 Annotation, dataset, training, evaluation — Planned

[Appendix E](reference/E_annotation_training.md). Requirements that apply to any such work: split by source
design, persist ties/both-bad and display order, never use heuristic scores as labels, compare the GNN with
a deterministic scorer and a pooled-feature MLP.

---

## 7. S06 — Integration and output validation

### 7.1 `resize` — Partial

`engine/resize.py`: parse with `Design.from_adobe(source, target_canvas=target_canvas)`, parse the target with
`Canvas.from_adobe(target_canvas)`, then return
`find_valid_layout(design, target, lambda attempt, previous: baseline_proposal(design, target, attempt))`.
The return value is a JSON dict containing only `canvas` and `elements`. On exhaustion `NoFeasibleLayout`
propagates; no design is returned. A successful return certifies the §8.1 gate only — not text fit,
legibility, asset availability or aesthetic quality.

### 7.2 CLI (`engine/main.py`, starter-kit orchestrator)

Reads every `input/*.json` task, calls `resize`, writes `<task>_output.json` and `<task>_output.png` (the
latter via `renderer.converter.convert.render_design_to_png`, falling back to a labeled-box placeholder only
on `ImportError`), and prints a non-authoritative `self_check` (§8.1.2). `resize` is called without exception
handling: a `NoFeasibleLayout` for one task aborts the process with a nonzero exit before that task's output
is written, and later tasks in the folder are not processed. Outputs of earlier runs are not deleted.
Behavior analysis and the recommended policy are in §7.5 (U10).

### 7.3 Planned multi-candidate flow

The flow of §5.4 (generate → validate → repair → deduplicate → baseline floor), then `rank_candidates`
(learned score, deterministic fallback per §6.4) → select/diversify → revalidate the selected candidate →
`candidate.design.to_adobe()`. This is recommended architecture, not code.

### 7.4 Deployment constraints

Scoring runs `docker run --network none` on a `linux/amd64` image built from `Dockerfile` (README, FAQ Q5);
RULES.md names an Apple M5 Pro reference machine with unspecified limits. Everything needed at run time must
be installed at build time. `requirements.txt` currently contains only Pillow and `cr-renderer`; torch, PyG
and any text encoder or checkpoint are not installed in the image. How ML assets would ship (and RULES.md's
public-weights rule for fine-tuned models) is open (U8). What the official documents establish about
submission, the viewer, Docker and models, and the questions still to put to Adobe, are in
[Appendix F](reference/F_submission_and_adobe_questions.md).

### 7.5 Batch failure policy (U10) — analysis and recommendation

**Observed behavior** (`engine/main.py:116-133`, `engine/resize.py`, `tests/test_layout_search.py::test_cli_does_not_write_or_render_a_failed_task`):
`main()` loops over `sorted(INPUT_DIR.glob("*.json"))` and calls `resize` with no exception handling.
`NoFeasibleLayout` (or `SchemaError`, or a `json` decode error, or a renderer exception) in task *k* ends the
process with a traceback and a nonzero status. Tasks before *k* already have `_output.json` and `_output.png`;
task *k* has neither JSON nor PNG (the JSON is written before the PNG, so a renderer failure leaves a JSON
without a PNG); tasks after *k* are never run. The local variable `exit_code` is never set to a nonzero value.
Because the Docker `ENTRYPOINT` is `python -m engine.main` and scoring runs against many hidden tasks, one
infeasible design could cost every task sorted after it. How the official harness treats a missing output
file, a nonzero exit, or a partial folder is not documented (Appendix F, question Q-batch).

| Policy | Never emits invalid JSON? | Effect on the rest of the batch | Assessment |
| --- | --- | --- | --- |
| **A. Abort the batch** (today) | Yes | Loses all later tasks | Safe but the worst blast radius |
| **B. Record the failure and continue** | Yes, if the failed task writes no output | Every other task is scored | Recommended |
| **C. Retry with another valid generation strategy** | Yes, if every retry is validated | None while retrying | Belongs inside `resize`, not in the CLI: it is the retry loop of §5.1 and the candidate flow of §5.4. It reduces failures; it cannot eliminate them |
| D. Emit a best-effort draft | **No** | — | Rejected: conflicts with the single-valid-boundary invariant |

**Recommendation (requires team approval; nothing is implemented):**

1. Keep `resize` raising `NoFeasibleLayout`. It is a library contract: it never returns an invalid design.
2. In the CLI, handle each task independently (policy B): catch `NoFeasibleLayout` and `SchemaError` per task, write **no** `_output.json` or `_output.png` for that task, remove any stale output of the same task from an earlier run so an old file is never mistaken for a new result, print the task name and a violation summary from `NoFeasibleLayout.reports` to stderr, and continue.
3. Render failures after a valid JSON should not discard the valid JSON; record them the same way.
4. After the last task, print a summary and exit nonzero if any task failed. Whether a nonzero exit is itself penalized is an Adobe question; if it is, exit 0 with the summary is the alternative.
5. Pursue C inside `resize` (§5.4) to make failures rare.
6. Add tests: a failing task between two good ones leaves exactly the good outputs; exit status per item 4; stale outputs removed.

This also needs Adobe's answer on what the scorer does with a task that has no output (Appendix F).

---

## 8. Cross-component invariants and failure handling

### 8.1 Constraint specification (single authoritative section)

#### 8.1.1 Constraint table

Class column: **Official** = stated by Adobe documents; **Internal** = a safety policy chosen by this
project, stricter than or not stated in the official documents (must not be presented as an Adobe rule);
**Objective** = soft optimization term, never a rejection criterion; **Unresolved** = the interpretation is
not established. "Enforced" = checked by `validate_candidate` today.

| ID | Constraint | Class | Enforced | Source / notes |
| --- | --- | --- | --- | --- |
| K1 | Output root has exactly `canvas` and `elements`; canvas width/height are integers ≥ 1; `elements` non-empty | Official | Yes (schema parse) | schema `additionalProperties:false`, `minItems:1` |
| K2 | `canvas` equals the requested target | Official | Yes (geometry) | `resize.py` docstring; FAQ Q1 |
| K3 | Every source ID appears exactly once; nothing added or removed; no duplicate IDs | Official | Yes (policy) | schema `id` description; RULES.md; FAQ Q1 |
| K4 | Output element order equals the source's flat array order | Internal | Yes (policy) | not stated by Adobe; whether it is required is not established |
| K5 | Per-type required fields and enum/type/nullability validity (text: content, font_family, font_size, color; image: asset_id; shape: shape_type, fill_color) | Official | Yes (schema parse) | schema `allOf` |
| K6 | Allowed changes: `x`, `y`, `width`, `height` on every element and `font_size` on text elements. **Everything else is frozen** — `type`, `content`, `font_family`, `font_weight`/`style`, `text_align`, `line_height`, `letter_spacing`, `max_lines`, `style_runs`, `color`, `fill_color`, stroke fields, `corner_radius`, `asset_id`, `fit`, `rotation`, `opacity`, `z_index`, `group_id`, and all extension properties, including optional-field presence/`null` and numeric types | Official for `content`, `font_family` (stage one, RULES.md "Adaptive typography") and `asset_id` (no regeneration, RULES.md); **Internal** for all other frozen fields | Yes (policy) | Whether Adobe permits changing e.g. `rotation` or `opacity` is **Unresolved** (U1); the validator forbids it |
| K7 | `font_size` ≥ 1 | Official | Yes (schema parse) | schema minimum |
| K8 | Every element lies inside the target canvas, measured as the union of the stored box and its rotated bounding box, expanded by half the stroke width (`stroke_width/2·(|cos|+|sin|)`) for shapes with a non-null `stroke_color`; no tolerance (strict `<0`, `>W`, `>H`) | "Nothing off-canvas" is Official (RULES.md; `resize.py`); **the measurement above is Internal**; the official definition (rotation, stroke, tolerance, intentional bleed) is **Unresolved** (U1) | Yes (geometry) | `element_bounds`, `candidate_validator.py:25` |
| K9 | Non-`cover` images with positive-area source and candidate boxes keep their box ratio: `|(w_c·h_s)/(h_c·w_s) − 1| ≤ 1/20`. Source `fit` is read with its effective default (`contain`). No full-bleed exemption is inferred from geometry, appearance or ID | "Kept in proportion" is Official (RULES.md); the **5 % figure** appears only in the supplied `resize.py` docstring ("within ~5 %", full-bleed/`fit:"cover"` exempt); the exact official definition is **Unresolved** (U1) | Yes (geometry) | |
| K10 | A positive width/height must not become zero, and a zero dimension stays zero (zero-area lines allowed) | Internal | Yes (geometry) | anti-collapse safeguard |
| K11 | `group_id` value and presence are preserved | Internal (covered by K6) | Yes (policy) | |
| K12 | Group members keep their spatial relationship ("designer-intended unit … meant to travel together") | Objective (strong soft term). The schema states the intent but no check, so it is not a rejection criterion; §8.1.5 | **No** (scored by the planned deterministic scorer, not by the validator) | The proportional repair preserves relative geometry because it scales the whole composition; a future generator may not. Whether Adobe's scorer checks cohesion is not known (Appendix F) |
| K13 | Missing-asset references are retained like any other element | Official (same-ID contract) | Yes (via K3, K6) | Level 3 sample |
| K14 | Never regenerate, redraw or replace asset pixel content | Official, disqualifying (RULES.md) | Not machine-checked; the engine never touches pixels | |
| K15 | Text fits its box and is legible; rendered PNG quality | Official rubric (RULES.md: PNG judged by an AI evaluator) | **No**; text fit is **unverified** | Objective only (Appendix D) |
| K16 | Overlap, alignment, balance, proximity, scale fidelity | Objective | No | Appendix D.1.5 |

#### 8.1.2 Reports and tolerances

- `validate_candidate` checks, in this implemented order: (1) parse the candidate with `Design.from_adobe` (failure → `ValidationReport(False, False, False, ("schema: …",))`); (2) `canvas == target` (geometry); (3) exact ID sequence (policy); (4) per element: frozen fields via a strict JSON comparison that distinguishes `bool`/`int`/`float` (policy), degeneracy, image ratio, bounds (geometry). `violations` lists policy messages then geometry messages; `is_valid` requires all three flags and no violations.
- When parsing fails, `policy_ok` and `geometry_ok` are `False` even though those checks did not run.
- `engine/main.py:self_check` is not the validator: it checks ID sets, required fields, and axis-aligned off-canvas on the **stored** box with a 0.5 px tolerance, ignoring rotation and stroke. A self-check warning does not fail the run. The validator is stricter; do not use the self-check as a compliance statement.
- Consequence of K8: output containing intentional bleed (for example the Level 2 sample's corner circle at `(-100,-100)`, which the starter README calls an expected warning) cannot pass the gate. Whether bleed is acceptable to the official scorer is U1.

#### 8.1.3 Hard vs soft, rejection and retry

- **Hard** = a failed `validate_candidate`. A hard failure rejects the candidate; nothing outranks it (not a model score, a timeout, or a wish to return JSON).
- **Soft** = Objective terms (K16). They order valid candidates only.
- Rejected proposals are retried by `find_valid_layout` with the previous `ValidationReport` passed to the proposer (§5.1). Exhaustion raises `NoFeasibleLayout`; unexpected generator errors propagate; a least-violating draft is never returned.
- Rotated extents and strokes are conservative internal measures; they are not a claim about the official scorer's formula.

#### 8.1.4 Planned validator extensions (recommendation, U6)

Structured violation codes, an error-vs-warning severity, an "unevaluated" state instead of `False`, and
rendering diagnostics (likely unreadable text, over-cropping, unexpected overlap) have been proposed. None
exists. The recommended shape is in §10.3 (U6).

#### 8.1.5 Group handling (U7 — direction set)

Two different things are called "group preservation"; only the first is a validity constraint.

| Aspect | Class | Enforced by |
| --- | --- | --- |
| The `group_id` value and presence of every element are unchanged (K11) | **Hard** (Internal, via K6) | `validate_candidate` (implemented) |
| Spatial relationships among members of an explicit group (relative offsets, order along each axis, compactness) | **Strong soft objective** (K12) | Deterministic scorer term `E_group` (D.1.5, planned); the GNN sees `same_explicit_group` edge evidence (B.3.3) and learns from labels |
| Containment relationships (for example a button shape behind its label) | Soft objective, only where evidence is reliable | `E_group` containment part (D.1.5) |

**Generation (S04, planned).**

- Initially move the members of an explicit group together as one layout unit: one placement and scale for the group's source bounding box, with member-relative geometry preserved (D.1.1, D.1.3).
- Controlled reflow is allowed where the unit cannot be placed validly or readably (`LayoutConfig.group_mode="reflow"`, §5.4): members may be re-arranged, but the group stays contiguous and the members' order and any reliable containment are kept as far as validity allows. Text `font_size` may change per member.
- Do **not** impose a rigid transform on every group. Source groups can include a background plus overlaid text (D.1.1); a rigid transform of such a group may be impossible or undesirable.
- Groups are never inferred from IDs, notes or asset names and never become Adobe `group_id` values (§8.2 item 3).

**Scoring (S05, planned).** `E_group` penalizes loss of cohesion among valid candidates; a candidate with
poor cohesion remains valid and selectable if nothing better exists. It never overrides a failed
`validate_candidate`.

**Containment evidence.** Treat a pair as a container/contained pair only when the source shows it
geometrically (the inner box lies inside the outer box and the outer is not a canvas-sized background-like
element, D.1.3); sharing an explicit group strengthens the evidence. Numeric thresholds are unresolved and
must be tuned on the three fixtures and human labels before use. Where the evidence is ambiguous the pair is
not scored.

### 8.2 Cross-component invariants

1. Source designs and caller-owned dicts are never mutated; exported JSON is a fresh tree.
2. Nothing but `canvas` and `elements` leaves `resize`. Required flags, graph data, scores, group hypotheses and UI data stay internal.
3. Original groups are preserved in JSON; inferred layout clusters are internal views and never become Adobe groups.
4. Stored pixel values are never normalized, clipped or rounded in the output representation; clipping applies to model features only (Appendix B).
5. Original inputs may violate target constraints (negative coordinates, missing assets); constraints apply to outputs only.
6. Node/candidate pairing is by exact ID. Elements are never ordered by sorted position or inferred role.
7. The baseline path (S02, validator, retry loop, `resize`) must not import or require PyTorch, PyG, pretrained models or GPU packages.
8. Randomness uses explicit seeds. Learned experiments record configurations, data manifests, checkpoint versions and metrics.
9. Graph features, text encoder, checkpoint, normalization and (when it exists) the generator are versioned together. An incompatible or missing ML asset triggers an explicit error and the deterministic fallback (§6.4), never bogus confidence or silent padding.
10. Attention coefficients are never presented as importance, grouping or explanation.

### 8.3 Failure handling

| Failure | Behavior | Status |
| --- | --- | --- |
| Invalid Adobe JSON (parse) | `SchemaError` with a path such as `elements[2].opacity` | Implemented |
| Proposal fails validation | Recorded; next attempt receives the report | Implemented |
| Retry budget exhausted | `NoFeasibleLayout` carrying all reports; no design returned; CLI aborts the batch (§7.2) | Implemented |
| Checkpoint missing/corrupt/incompatible/untrained | `CheckpointUnavailable` | Implemented |
| Scoring failure | `ModelInferenceError`; no partial score list | Implemented |
| Text embedding unavailable | Mask 0 for that text node; model-side acceptance rules in C.2 | Implemented |
| Empty candidate list to `score_candidates` | Returns `[]` | Implemented |
| Learned score unavailable → deterministic ranking | Specified in §6.4 | **Planned** |
| Zero valid candidates at the official boundary | `resize` raises (no draft is ever returned); the CLI aborts the batch today; recommended CLI policy in §7.5 | Implemented (raise); CLI policy **recommendation** awaiting approval (U10) |

---

## 9. Verification and acceptance criteria

### 9.1 Test modules

| Module | Covers | CI job ([CI.md](CI.md)) |
| --- | --- | --- |
| `tests/test_internal_schema.py` | S02 round trip, presence semantics, fixtures | Core (Python 3.10, 3.11) |
| `tests/test_layout_search.py` | Validator, retry loop, proportional repair, all three samples | Core |
| `tests/test_ci.py` | CI execution guard | Core |
| `tests/test_graph_features.py` | Graph tensors, topology, colors, embeddings, batching | Graph and ML / CPU |
| `tests/test_ranker.py` | Model, checkpoints, scoring failures | Graph and ML / CPU |

Ordinary `python -m pytest` skips the graph and ranker modules when torch/PyG are not installed; a skipped module is not evidence that they pass. CI's execution guard rejects such skips ([CI.md](CI.md)). This specification records no test counts or pass/fail results. Local results are not hosted-CI results;
run the commands in [CI.md](CI.md).

### 9.2 Acceptance criteria

Criteria for a change to be merge-ready, by component. Items marked (planned) have no code yet.

- **S02:** Appendix A.10 (round trip with and without target context, every field/enum, absent/null/default distinctions, extras and collisions, theme, groups).
- **Graph:** N=1 → one node and zero edges; N=4 → 12 directed edges; N=17 → 272; duplicate IDs rejected upstream; `group_id` of `None`, `ABSENT` and `""` yield correct `same_explicit_group`; negative `x/y`, zero width/height and rotated boxes give finite features and no mutation; reverse-edge relative displacements have opposite signs; every candidate of one source shares ID order and topology; output tensor shapes and dtypes exactly as in B.1.
- **graph_v4:** its first 57 node and 31 edge columns equal `graph_v3` on all three samples; shapes `[N,58]`/`[E,32]`; hand-computed text-fit values and size-order signs; font-size-only edits change exactly columns 53, 54, 57 (and edge 31); zero off text and for degenerate boxes; extreme font sizes stay finite; the sparse topology carries the extra edge column; unsupported versions are rejected; `engine/text_fit.py` imports no PyTorch.
- **Model:** shape tests on CPU (and CUDA when available); deterministic evaluation with dropout off; no NaN on all-zero embeddings; batched graphs keep separate pooling; checkpoint dimension/version mismatch rejected; numeric-only checkpoints behave as C.2; returned attention is never exposed as importance.
- **Output gate:** invalid-to-valid retries with diagnostics; exhaustion raises without emitting a draft; malformed schemas; frozen-field and extension changes; ID/order; bounds, rotation and strokes; image ratios; degeneracy; all three samples (`tests/test_layout_search.py`).
- **Generator (planned):** every candidate preserves ID count/order and frozen fields and has the target canvas; text `font_size` is searched together with boxes (a test that fails if a family never varies `font_size`, and one that rejects a `font_size` below 1); each family runs on warm-up/core/advanced samples; no candidate is accepted if it violates a known rule; intentional background overlaps are not globally forbidden; infeasible layouts are not falsely passed; diversity returns ≤ 3 options and no exact duplicates.
- **Preference UI (planned):** reversed display order does not invert canonical labels; same source and target in a comparison; no score leakage; tie/both-bad/skip recorded; repeats retained; labels survive restart; append-only storage with renderer/generator versions.
- **Evaluation (planned):** no source ID leaks across splits; report dataset sizes by source; compare the MLP and GNN to the deterministic baseline; final output comparison uses held-out designs.
- **Reporting:** state what was implemented, what was tested (command and observed result), what is unverified. A check not run is "not run".

---

## 10. Known limitations and unresolved decisions

### 10.1 Known limitations

- No trained checkpoint, human labels or measured aesthetic improvement; GNN output is not evidence of quality.
- Text fit, wrapping and legibility are not measured; the advanced sample (728×90) can yield very small text under the proportional repair.
- The text encoder `sentence-transformers/all-MiniLM-L6-v2` is optional and only proposed: it is **not** Adobe-approved (U8), and it has never been loaded or evaluated in this repository. The numeric-only graph and ranker path needs no encoder.
- Python 3.10 execution and the Apple reference hardware are untested locally; hosted CI has not been confirmed by this document.
- The vendored renderer's fidelity to the official scorer's renderer is unverified (U9).
- The root [README.md](../README.md) now describes the current implementation. Its run and Docker instructions were exercised once on one Windows host (Docker Desktop, 2026-10-28) as recorded there; they have not been run on the Apple reference machine or on Python 3.10.
- The schema is "v0.9 provisional"; a schema revision would require re-checking §8.1 and Appendix A.
- The main challenge brief referenced by RULES.md is not in the repository (U11); the copy examined is a PDF in the developer's `Downloads/` folder, and its provenance and version are unconfirmed (Appendix F).
- The proportional repair centers the whole composition at a uniform scale. On the 728×90 advanced sample this produces a very small portrait composition with near-illegible text; the output passes the validator but is not a good layout. This is the quality gap the planned generator exists to close.
- Source designs carry no semantic group labels, so any inferred grouping is uncertain by construction. A fully connected graph learns *potential* cross-element influence; it does not discover visual groups merely because attention exists.
- Text embeddings say nothing about image content; `asset_id` must not be read as a semantic category. Box geometry is not rendered glyph or intrinsic image geometry, so renderer fidelity matters for human labels.
- A 17-element 728×90 target may be infeasible or force very different aesthetics; a feasible solution is not guaranteed. Model quality may be limited by source-design diversity even with many labels, so a strong pooled-feature baseline is mandatory (Appendix E.4).

### 10.2 Decision register (U1–U13)

Status key. **Resolved**: settled by an explicit architectural direction, existing code, or an unambiguous
official rule. **Direction set**: the architecture is settled; the contract details are recommendations that need
team sign-off. **Recommended**: no settled direction exists; a recommendation is in §10.3 and needs team
approval. **Open (Adobe)**: depends on information Adobe has not given; do not change behavior until it does
(questions in [Appendix F](reference/F_submission_and_adobe_questions.md)).

| ID | Decision | Status | Where |
| --- | --- | --- | --- |
| U1 | Official definitions of off-canvas (rotation, stroke, tolerance, intentional bleed), image-proportion tolerance and its full-bleed exception, and whether fields other than `content`/`font_family`/`asset_id` (for example `rotation`, `opacity`, `z_index`) may change. The validator is currently stricter than the official text | **Open (Adobe)** | K6, K8, K9; Appendix F |
| U2 | `generate_candidates` returns `list[LayoutCandidate]`; `find_valid_layout` stays the iterative validation/repair loop; generation, validation/repair and ranking are separate. `LayoutConfig` fields and the composition flow | **Direction set** | §5.4 |
| U3 | Deterministic scorer on validated candidates only, primary until the GNN is trained, fallback when it fails. Signature, exception mapping, meaning of a missing score. Term definitions and **provisional** weights are in D.1.5 (untested, not final); calibration in D.1.6 | **Direction set**; interface and weights are recommendations awaiting calibration | §6.2, §6.4, D.1.5, D.1.6 |
| U4 | Final names of the selection function and the multi-option entry point | **Recommended** | §10.3 |
| U5 | Whether `candidate_id` is an opaque ID or a content hash; canonical JSON serialization for hashing | **Recommended** | §10.3 |
| U6 | `ValidationReport` extensions: structured codes, severity, "unevaluated" state, class tags | **Recommended** | §8.1.4, §10.3 |
| U7 | Group cohesion: `group_id` preservation is hard; spatial relationships are a strong soft objective; groups move as units first; controlled reflow allowed; containment scored only on reliable evidence. Numeric thresholds are unresolved | **Direction set** | §8.1.5, D.1.5 |
| U8 | Model eligibility and size limit, offline packaging of torch/PyG/encoder/checkpoint, text-encoder approval, and `linux/amd64` Docker vs the Apple M5 Pro reference machine | **Open (Adobe)** | §7.4, Appendix F |
| U9 | Whether `renderer/converter` matches the official scorer's renderer, hence whether it is a valid preview for human labels | **Open (Adobe)** | §1.4, Appendix F |
| U10 | Behavior when no layout is found: `resize` raises (implemented); CLI batch policy | **Recommended**; the official harness behavior is **Open (Adobe)** | §7.5, Appendix F |
| U11 | Canonical location and provenance of the main challenge brief | **Open (Adobe)** for provenance; which copy to cite is a team choice | §10.1, Appendix F |
| U12 | Planned `with_updates`/`with_layout` are structural-only and carry no policy checks | **Resolved**: A.8 already specifies it and the §2.3 rule makes `validate_candidate` the only policy checker | A.8, §2.3 |
| U13 | Typography features for the graph: `graph_v4` (text-fit node feature, size-order edge feature) is implemented opt-in alongside the unchanged default `graph_v3`, after an information-content evaluation (B.7.2). Which version becomes the default or the training contract | **Direction set**: `graph_v3` stays the default until a labeled ranking comparison shows a benefit (B.7.3); no improvement is claimed | B.4.3, B.7 |

### 10.3 Recommendations for open internal decisions

These are recommendations, not decisions. None changes implemented behavior.

**U4 — names.** Keep `select_options(candidates: list[ScoredCandidate], k: int = 3) -> list[ScoredCandidate]`
for the internal selector and `resize_with_options(source, target_canvas, k=3) -> list[dict]` for the
multi-option entry point, in a module other than `engine/resize.py`. *Why:* the names are already used
consistently across the spec, plan and Appendices D and E; the official adapter stays a single-output
`resize` (§1.3 item 9), and keeping options out of that module prevents a second Adobe-facing entry point.
`select_best_valid_candidate` (an early plan name) is not used.

**U5 — candidate identity.** Make `candidate_id` content-addressed: `"sha256:" + hex(sha256(canonical_json(candidate.design.to_adobe())))`,
where `canonical_json` is `json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)` encoded
as UTF-8. Provenance (`family`, `seed`, `generator_version`, config) stays in separate fields, not in the hash.
*Why:* Appendix E.1 already names `sha256_of_canonical_candidate_json` for preference records, identical
layouts from different families then deduplicate for free, and ids are stable across runs. Constraints to
accept: the hash depends on Python's `json` float formatting (use the same canonicalizer everywhere, and
record it in the manifest), and numeric type changes (`1` vs `1.0`) change the hash, which matches S02's
lossless numeric semantics. The source design gets the same hash scheme (`source_design_id`).

**U6 — `ValidationReport`.** Extend additively so existing code keeps working: keep `schema_ok`, `policy_ok`,
`geometry_ok`, `violations: tuple[str, ...]` and `is_valid`; add `issues: tuple[Issue, ...]` where
`Issue = (code: str, rule: str, element_id: str | None, field: str | None, message: str, rule_class: Literal["official", "internal"])`.
`code` is a stable machine-readable string (for example `GEOM_OUT_OF_BOUNDS`) and `rule` is the K-id of §8.1.1.
Make the three booleans `bool | None`, with `None` meaning "not evaluated" (for example after a parse failure),
and keep `is_valid` true only when all three are `True`. Defer an error/warning severity until a check needs
it (nothing is advisory today). *Why:* `find_valid_layout` hands the previous report to the proposer, and
free-text strings cannot drive a programmatic repair; tagging official versus internal rules stops internal
policy being reported as an Adobe rule (K4, K6, K8, K10). *Cost:* it changes a public dataclass, so it needs
regression tests and a spec update in the same change (AGENTS.md §7); do not do it before U1 is answered,
because U1 may change which rules exist.

**U10 — batch failure.** §7.5 recommends policy B (record the failure and continue, no output for the failed
task, nonzero exit after the batch) with `resize` still raising. The official handling of a missing output is
an Adobe question; keep the CLI unchanged until the team approves.

### 10.4 External references

- PyG [`GATv2Conv`](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATv2Conv.html) (supports `edge_dim`), [`Data`](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html), [`global_mean_pool`](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.pool.global_mean_pool.html).
- [SentenceTransformer](https://sbert.net/docs/package_reference/sentence_transformer/model.html); [`torch.load`](https://docs.pytorch.org/docs/2.11/generated/torch.load.html).
- OR-Tools [CP-SAT 2D no-overlap](https://github.com/google/or-tools/blob/stable/ortools/sat/cp_model.proto), relevant only to selected packing pairs (Appendix D).

### 10.5 History

Superseded decisions, version migrations and old milestone numbering: [reference/HISTORY.md](reference/HISTORY.md) (non-normative).
