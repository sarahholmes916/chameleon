# Adobe Chameleon — Implementation Plan

This document covers **ownership, dependencies, milestones and progress only**. Architecture, interfaces,
constraints and component status are defined in [TECHNICAL_SPEC.md](TECHNICAL_SPEC.md); this plan defines
none of them. A checked box means the item exists in the repository and a test module covers it (verified
by inspection when the box was updated); it is not a test result and not proof of contest compliance.

## 1. Developer ownership

### Developer 1 — ML, data and GNN

**Temporary exception:** implements the minimal shared S02 representation while Developer 2 is unavailable.
This is not a second independently developed schema.

After S02, primary ownership:

- **S03 graph construction:** node/edge features, optional frozen text embeddings, batching, missingness handling, tensor-shape tests.
- **S03/S05 ranker:** PyTorch Geometric GATv2 whole-layout scorer.
- **S05 data:** synthetic source designs and fixtures; dataset identifiers, splits, manifests, version metadata.
- **S05 annotation UI:** side-by-side comparison of validated, accurately rendered candidates (A/B/tie/both-bad, optional reasons, randomized placement, persisted records).
- **S05 training/evaluation:** pairwise labels, simple ranking baselines, then the GNN; source-design-disjoint holdout; comparison with the deterministic ranker; evaluation of the `graph_v3` ranker on font-size-differing candidate pairs before any graph change (spec B.7.3).
- **S05 ranking wrapper:** `rank_candidates` (spec §6.4): learned scoring with the deterministic fallback, the exception-to-fallback mapping and its tests. It calls Developer 2's `score_deterministic` and does not reimplement it.
- **Later experiments:** model-guided search/relationship prediction only after the basic ranker and computational pipeline are working and measurable.

Does **not** own by default: candidate geometry solver, deterministic scoring, full target-layout validation, preview renderer, or the production `resize.py` adapter. Use their public interfaces.

### Developer 2 — Computational layout and renderer

Begins when available, building on Developer 1's S02 foundation.

- Remaining **S02** controlled updates (`with_updates`, `with_layout`).
- **S04** source analysis, layout units, candidate generation and repair, validation extensions, deterministic scoring (implements and maintains `score_deterministic`, which `rank_candidates` uses as primary scorer and fallback, and its calibration, D.1.5–D.1.6), option diversification.
- **Rendering/preview:** faithful rendering for debugging and human comparison, or documented renderer limits agreed with Developer 1 before labeling.
- **S06 integration:** the thin competition-facing adapter in `engine/resize.py`.

Does **not** own by default: graph tensor definitions, GNN architecture, human-label data, annotation UI, training checkpoints.

### Shared responsibilities

- Verify contest constraints against the actual official files; coordinate interface changes in a PR (spec and tests updated in the same change) before either branch relies on them.
- **Scoring interface.** `score_candidates` (Developer 1), `score_deterministic` (Developer 2) and `rank_candidates` (Developer 1) are one coordinated interface: a change to any of their signatures, to the fallback mapping or to `objective_version` is made in a PR that updates the spec and tests and is reviewed by both developers.
- Cross-review pull requests for contract compatibility; integration and end-to-end tests; runtime measurement; held-out human evaluation.
- Independently label a subset of candidates to estimate annotation consistency.

## 2. Ownership by component

Paths marked *Planned* do not exist; confirm against the checkout before relying on them.

| Component / file | Primary owner | Consumer | Status |
| --- | --- | --- | --- |
| `engine/internal_schema.py` (S02) | Developer 1 now; extensions Developer 2 | all | Implemented |
| S02 `with_updates` / `with_layout` | Developer 2 | Developer 1 | Planned |
| `engine/graph_features.py`, `engine/text_embeddings.py` | Developer 1 | ranker | Implemented |
| `engine/ranker.py` | Developer 1 | integration | Implemented (untrained) |
| `engine/candidate_validator.py`, `engine/layout_search.py` | Developer 2 | all | Implemented (output gate, proportional repair) |
| `engine/candidate_generator.py`, `engine/layout_units.py`, `engine/layout_objective.py`, `engine/diversify.py` | Developer 2 | Developer 1 | Planned |
| Deterministic scorer (`score_deterministic`, D.1.5) and its calibration (D.1.6) | Developer 2 implements and maintains | Developer 1 (`rank_candidates`, baseline comparison) | Planned (location undecided) |
| Learned-then-deterministic ranking wrapper (`rank_candidates`, spec §6.4) | Developer 1; calls Developer 2's `score_deterministic` through its public interface | Developer 2 (selection, `resize`) | Planned |
| `renderer/converter/` | Developer 2 | Developer 1 (labeling) | Vendored; fidelity unverified |
| `annotation/app.py`, `annotation/records.py` | Developer 1 | Developer 2 may label | Planned |
| `training/dataset.py`, `training/train_ranker.py`, `training/evaluate.py` | Developer 1 | Developer 2 validates metrics | Planned |
| `engine/resize.py` | Developer 2 | Developer 1 reviews | Partial |
| `tests/`, docs | each owner | both review | — |

## 3. Dependencies and parallel work

| Developer 1 task | Needs Developer 2 output first? | Meanwhile |
| --- | --- | --- |
| S02 parser/serializer | No | Done |
| Node/edge feature extraction | No | Done (real sources; candidate fixtures) |
| GNN forward/backward | No | Done with synthetic/mock candidates for mechanics only |
| Annotation UI shell and record format | No | Placeholders; do not claim labels reflect real visuals |
| Synthetic source designs | No | Coherent design grammars and parser stress tests |
| Trustworthy aesthetic pairwise labels | **Yes** — validated candidates and reliable previews | Do not label known-invalid or unfaithfully rendered candidates as ordinary design-quality examples |
| Train/evaluate the final ranker | **Yes** — human labels and held-out candidate sets | Loss, batching, configs, simple baseline |
| End-to-end baseline-vs-GNN comparison | **Yes** | Agree on metric and holdout protocol now |

**Blocking handoffs**

1. Developer 1 → 2: stable S02 types, import path, tests, and documented deferred S02 features. *(done)*
2. Developer 2 → 1: a working `score_deterministic` (provisional weights, [spec D.1.5](reference/D_candidate_generation.md#d15-soft-objective-fully-defined-enough-to-implement)) so that `rank_candidates` can use it, and a working `generate_candidates` (recommended contract, awaiting approval: [spec §5.4](TECHNICAL_SPEC.md#54-planned-generator--validator--repair-integration-contract-u2)), reproducible candidate identity/versions, verified renderer and fixed preview API.
3. Developer 1 → 2: compatible ranker checkpoint, graph/encoder version manifest, `rank_candidates` with its fallback behavior (recommended contract, awaiting approval: [spec §6.4](TECHNICAL_SPEC.md#64-fallback-policy)), evaluation summary.
4. Both → integration: consistent `LayoutCandidate`, `ValidationReport`, known validation behavior, candidate diversity, single official output, documented failure cases.

## 4. Milestones

This is the only milestone numbering in the repository.

### M1 — Shared S02 foundation (Developer 1)

- [x] Locate and verify the schema, rules, samples and existing source files.
- [x] Typed representation and `ABSENT` handling in `engine/internal_schema.py`.
- [x] Lossless parsing/serialization, ID lookup, original groups, derived geometry.
- [x] Round-trip tests on all three fixtures, with and without target context.
- [x] Tests for null vs absent, nested extras, numeric preservation, missing assets, immutability, invalid input.
- [x] Public API documented ([Appendix A](reference/A_internal_representation.md)); deferred controlled-edit APIs listed.

### M2A — Graph foundation (Developer 1)

- [x] Node and edge features, missing-value masks.
- [x] Dense edges up to 64 elements and the deterministic sparse rule beyond; one graph per design/candidate/target.
- [x] Exact-ID matching.
- [x] Optional frozen text embeddings and graph-batch tests.
- [x] Tensor sizes, finiteness, determinism, GNN forward/backward on fixtures.
- [x] Candidate font size, text/fill color and alpha features (`graph_v3`).
- [x] `graph_v4` typography features (text-fit node feature, size-order edge feature) and `gatv2_ranker_v4`, opt-in alongside the unchanged default `graph_v3` ([spec B.7.2](reference/B_graph_features.md#b72-information-content-evaluation-of-the-proposed-features-2026-10-28)).

### M2B — Computed candidate pipeline (Developer 2)

- [ ] S02 controlled updates and/or immutable candidate-layout installation.
- [ ] Proportional baseline and composition templates; multiple reproducible candidates.
- [x] Allowlisted generator edit helper `layout_edit` (`engine/layout_edits.py`), derived from the validator's `mutable_fields`, with unit tests; `baseline_proposal` emits elements only through it ([spec §8.1.6](TECHNICAL_SPEC.md#816-stage-1-design-property-preservation-project-requirement)). Deviation: it works on Adobe element dicts and is independent of the still-planned S02 `with_updates`.
- [x] Property tests for the implemented baseline family on the three samples (only K6 fields differ; protected data identical; source unmodified).
- [x] Property tests for each implemented family — VERTICAL_STACK, HORIZONTAL_FLOW (both suites), BANNER_ZONES (both suites), HERO_LEFT/RIGHT (both suites): only K6 fields differ; each emits through `layout_edit`.
- [ ] Property tests for each remaining planned family (TWO_COLUMN, CONTROLLED_OVERLAY): deferred until each exists; each must use `layout_edit`.
- [x] Regression tests naming text, fill and stroke color, stroke width and removal, opacity, rotation, `z_index` and `font_size` on non-text elements, plus permitted changes still passing ([spec §8.1.6](TECHNICAL_SPEC.md#816-stage-1-design-property-preservation-project-requirement)).
- [x] Hard validation and violation reports ([spec §8.1](TECHNICAL_SPEC.md#81-constraint-specification-single-authoritative-section)).
- [x] Bounded retry with proportional geometric repair; exhaustion raises `NoFeasibleLayout`.
- [ ] Deterministic heuristic scoring (`score_deterministic`, [spec D.1.5](reference/D_candidate_generation.md#d15-soft-objective-fully-defined-enough-to-implement)) with provisional weights, and a verified preview renderer.
- [ ] Calibrate the scorer's weights and text-fit constants against rendered examples and human labels ([spec D.1.6](reference/D_candidate_generation.md#d16-calibrating-the-deterministic-objective-procedure-nothing-here-has-been-run)); until then results are labelled provisional.
- [ ] Demonstrate the full pipeline without a model or GPU.

**Exit:** at least one tested sample produces reproducible validated candidates and renderable previews; extreme-case infeasibility is documented, not hidden.

### M3 — Annotation and dataset (Developer 1 lead)

- [x] Phase 1 original source templates, source schema/quality checks, actual renderer previews, manifest, duplicate quarantine, review gallery/exclusion and approved-source S04 handoff ([SOURCE_DATA_PIPELINE.md](SOURCE_DATA_PIPELINE.md)); regression tests in the dedicated source/renderer CI job.
- [ ] Human approval of the 20-source pilot, then family-diverse expansion; no pilot source is automatically approved by structural validation.
- [x] Side-by-side tournament UI (A/B, tie, both-bad, skip, reason tags, randomized placement, bracket, undo, resume) with SQLite storage and JSONL/CSV export ([ANNOTATION_TOURNAMENT.md](ANNOTATION_TOURNAMENT.md)); uses a development-only candidate generator until S04 candidates exist. No human labels collected yet.
- [x] Records: source/candidate IDs, target, displayed order, label, generator/renderer versions (`preference.v1` export).
- [ ] Pilot of diverse human judgments; examine ambiguities and consistency.
- [ ] Source-design-grouped splits; untouched final test sources.
- [ ] Reproducible dataset manifest and loader.

**Exit:** labels trace to exact validated candidate JSON and reliable previews with no source leakage.

### M4 — Ranker, evaluation and integration (both)

- [x] GATv2 architecture; forward/backward on CPU (CUDA when available).
- [x] Versioned checkpoints; batched scoring with explicit failure behavior.
- [ ] Simple learned baseline; held-out pairwise metrics.
- [ ] GNN trained on human preferences.
- [ ] Compare GNN to deterministic ranking on the same valid candidate sets.
- [ ] Compare `graph_v3` and `graph_v4` rankers on human labels, including font-size-differing pairs (overflow, unreadably small, appropriate); `graph_v4` becomes the default only if the team's pre-written rule is met ([spec B.7.3](reference/B_graph_features.md#b73-still-required-before-the-default-changes-or-any-improvement-is-claimed)).
- [ ] Implement `rank_candidates` (Developer 1) with the deterministic fallback and its tests, then connect it to selection.
- [ ] Test validity, retention, extreme canvases, runtime, end-to-end human preference.
- [ ] Diverse top-K internal alternatives while `resize` returns one design.

**Exit:** reproducible end-to-end resize, real comparison data, explicit failure/fallback tests, no unsupported visual-quality claims.

## 5. Stage-one submission checklist

Deliverables per [Appendix F](reference/F_submission_and_adobe_questions.md). Unlike the milestone boxes above,
these were checked by inspection or a one-off run, stated on each line; none is a hosted-CI or Adobe-scoring result.

- [x] Engine code: `resize()` returns a validated proportional fit (basic quality; spec §10.1).
- [x] `Dockerfile` builds and runs with `--network none` (run once on one Windows host, 2026-10-28; Apple reference machine untested).
- [x] Root README: approach, how to run, tests, Docker, status.
- [ ] Viewer add-on: `FAQ.md:105` says required (not scored), the brief says finale-only. Not started; blocked on Adobe's answer (Appendix F, Q-viewer).
- [ ] Confirm the submission channel and code-freeze date (announced separately by Adobe).
- [ ] Answers to the Adobe questions in Appendix F (U1, U8, U9, U11, batch behavior).

## 6. Collaboration rules

- **Branches:** Developer 1 uses feature branches for S02/S03/S05; Developer 2 for S04/S06. Merge the S02 foundation before it is extended.
- **Shared contracts:** change them in a PR that updates [TECHNICAL_SPEC.md](TECHNICAL_SPEC.md) and tests together.
- **No duplicate models:** no alternative Adobe classes, parsers, candidate identity conventions or ad hoc feature ordering.
- **PR checklist:** what changed; functions exposed; test command and observed result; limitations; effect on the other owner.
- **Honest status:** mark only inspected, tested work complete.
- **Agent instructions:** [AGENTS.md](../AGENTS.md) covers workflow; progress lives here, not duplicated across prompts.
