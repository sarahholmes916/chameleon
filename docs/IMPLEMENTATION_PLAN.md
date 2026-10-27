# Adobe Chameleon — Implementation Plan & Responsibilities

**Status:** Working plan; checked items reflect verified implementation

**Version:** 0.1  
**Updated:** 2026-10-26  
**Location:** `docs/IMPLEMENTATION_PLAN.md`

> **Purpose:** Define who owns each part of the hybrid Adobe Chameleon resize engine, how the parts connect, and which work can begin independently. This is a coordination document, **not** a replacement for the technical specifications or proof that any module exists.

## 1. Project goal and shared architecture

Given a source Adobe design JSON and a requested target canvas, produce a resized design that respects the competition's actual rules. The engine should produce several internally valid, visually different candidate layouts, select the best for the official single-output `resize(...)` contract, and optionally expose multiple choices through a separate demo/UI.

```text
Adobe source JSON + target_canvas
              |
              v
     S02 shared representation
       (Canvas/Element/Design)
              |
      +-------+----------------------------+
      |                                    |
      v                                    v
 S04 deterministic                   S03 graph features
 source analysis / layout            + optional embeddings
 generation / validation                    |
      |                                     v
      |                               GNN layout ranker
      v                                     ^
 Valid LayoutCandidate objects -------------+
      |                                     |
      +---------------------+---------------+
                            v
                 rank valid candidates
           GNN if verified/available;
           otherwise deterministic scorer
                            |
                            v
              diversify internal choices
                            |
                            v
             selected compliant Adobe JSON
                 (canvas + elements)
```

**Non-negotiable architectural boundaries:**

- S02 is the shared source of truth for Adobe element and canvas data; neither developer creates another competing parser or `Element`/`Design` class.
- **The computational engine is permanent.** It generates candidates, checks constraints, supplies heuristic ranking, and runs without an ML checkpoint. The GNN does **not** replace geometric validity checks.
- **Initially, the GNN ranks full candidates, not individual objects.** It uses one graph per `(source, candidate, target)` with nodes for original elements and relational edges; attention coefficients are *not* element-importance scores.
- Human side-by-side preferences are training labels for quality ranking. Do not use computational heuristic scores as synthetic *preference* ground truth.
- All original required elements and immutable stage-one fields must be preserved according to authoritative Adobe rules. Internal graph metadata, grouping hypotheses, ranking scores, and UI data never appear in official Adobe output.
- The final competition-facing function returns **one** `{"canvas": ..., "elements": [...]}` design unless the official interface changes. Multiple options remain internal/demo functionality.

## 2. Developer ownership

### Developer 1 — ML, data, and GNN (you)

**Immediate responsibility:** implement the minimal, reliable **S02 shared representation** because Developer 2 is currently unavailable. This is a temporary ownership exception, not a second independently developed schema.

**After the S02 milestone, primary ownership:**

- **S03 graph construction:** node features, directed edge features, optional frozen text embeddings, batching, feature/missingness handling, and tensor-shape tests.
- **S03/S05 ranker:** build and train a PyTorch Geometric GNN (initial design: edge-aware GATv2), producing a scalar **whole-layout** quality score conditioned on source and target.
- **S05 data:** generate synthetic *source* designs and experimental layout fixtures; maintain dataset identifiers, splits, manifests, and version metadata.
- **S05 annotation UI:** build side-by-side comparisons of **validated, accurately rendered candidate layouts**, supporting A/B/tie/both-bad and optional reasons. Randomize screen placement and persist annotations.
- **S05 training/evaluation:** collect manual pairwise rankings; train simple ranking baselines then the GNN; evaluate on **source-design-disjoint** holdout sets and compare against deterministic ranking.
- **Later experiments:** model-guided search/relationship prediction **only after** the basic ranker and computational pipeline are working and measurable.

**Do not own by default:** candidate geometry solver, deterministic scoring, full target-layout validation, final preview renderer, or the production `resize.py` adapter. Use their public interfaces rather than implementing competing versions.

### Developer 2 — Computational layout and renderer (teammate)

**Begins when available, building on Developer 1's committed S02 foundation.**

**Primary ownership:**

- Finish remaining **S02** controlled updates, `with_layout`, and stage-one comparison/validation interfaces as needed by S04; preserve the original S02 contracts.
- **S04 source analysis:** grouping heuristics, layout-unit construction, background/foreground and deliberate-overlap evidence, layout-family eligibility.
- **S04 candidate generation:** deterministic proportional baseline, horizontal/vertical/compact/hero-style layouts, feasible region assignment, geometry optimization and repair.
- **S04 validation:** enforce authoritative schema/policy and applicable geometric constraints, report violations explicitly, and reject invalid candidates before aesthetic ranking.
- **S04 fallback scoring:** deterministic layout-quality metrics independent of PyTorch, text encoders, or trained checkpoints.
- **Rendering/preview pipeline:** faithfully render candidate JSON for debugging and human comparisons, or document renderer limits and coordinate with Developer 1 before labeling.
- **Option diversification:** remove near-duplicates and retain meaningfully different strong alternatives.
- **S06 integration:** own the thin competition-facing adapter in `engine/resize.py`, keeping the official response structure and fallback behavior.

**Do not own by default:** graph tensor definitions, GNN architecture, human-label training data, annotation UI, or training checkpoints.

### Shared responsibilities

- Verify contest constraints against the **actual** `RULES.md`, `FAQ.md`, v0.9 schema, samples, and existing `engine/resize.py`. Specifications are proposals where these have not been checked.
- Agree on and test shared interfaces; coordinate interface changes *before* either branch relies on them.
- Review each other's pull requests for contract compatibility and regressions.
- Integration, end-to-end tests, demo design, runtime measurement, and held-out human evaluation.
- Independently label a subset of candidates (if possible) to estimate annotation consistency.

## 3. Ownership by component

| Component / file (proposed path) | Primary owner | Collaborator / consumer | Deliverable |
|---|---|---|---|
| `engine/internal_schema.py` (S02 core) | Developer 1 **now** | Developer 2 later | Typed, lossless design representation and source round-trip tests |
| S02 controlled updates/layout installation | Developer 2 **later** | Developer 1 | Immutable candidate creation and stage-one checks |
| `engine/graph_features.py` | Developer 1 | Developer 2 uses indirectly | Versioned source/candidate graph tensors |
| `engine/text_embeddings.py` | Developer 1 | — | Optional frozen NLP embeddings and cache |
| `engine/ranker.py` | Developer 1 | Developer 2 uses scoring API | Whole-candidate GNN scorer |
| `engine/layout_units.py` | Developer 2 | Developer 1 uses through candidate data | Internal grouping/layout units |
| `engine/candidate_generator.py` | Developer 2 | Developer 1 consumes candidates | Diverse computational candidate layouts |
| `engine/layout_objective.py` | Developer 2 | — | Reproducible soft objectives |
| `engine/candidate_validator.py` | Developer 2 | Developer 1 consumes validity reports | Hard-constraint and geometry checks |
| Deterministic scorer / `engine/diversify.py` | Developer 2 | Developer 1 compares against it | Fallback ranking and diverse option selection |
| Rendering/preview adapter (path TBD) | Developer 2 | Developer 1 requires for annotation | Accurate source/candidate previews |
| `annotation/app.py`, `annotation/records.py` | Developer 1 | Developer 2 may label | Preference UI and durable records |
| `training/dataset.py`, `training/train_ranker.py` | Developer 1 | — | Grouped datasets, training, model artifacts |
| `training/evaluate.py` | Developer 1 | Developer 2 validates output metrics | Baselines, ablations, preference accuracy |
| `engine/resize.py` | Developer 2 | Developer 1 reviews integration | Single-design competition output and fallback |
| `tests/` and docs | Each owner for their modules | Both cross-review | Tests and updated contracts |

Paths after S02 are **proposed** by the v0.1 technical spec; inspect the actual repository before adding or moving files.

## 4. Shared APIs and handoff contracts

Follow the exact S02 and S03–S06 technical specs for authoritative types and field behavior. These signatures express *the integration boundary*, not permission to implement all functions immediately.

```python
# S02 (Developer 1 implements minimal shared foundation first)
source = Design.from_adobe(source_json, target_canvas=target_canvas_json)
unchanged_json = source.to_adobe()    # structurally equal to source_json

# S04 (Developer 2)
candidates: list[LayoutCandidate] = generate_candidates(source, target, config)
report: ValidationReport = validate_candidate(source, candidate.design, target)

# S03/S05 (Developer 1)
graph = build_candidate_graph(source, candidate.design, target)
scores: list[float] = score_candidates(source, valid_candidates, target)

# S04/S06 (Developer 2 + shared integration)
# If ranker is unavailable, use deterministic scores; revalidate before output.
selected: LayoutCandidate = select_best_valid_candidate(...)
result = selected.design.to_adobe()
```

**Contract rules:**

- Each `LayoutCandidate` wraps a `Design` at the target canvas and carries a stable internal ID, layout family, generator seed, and generator version; it is **not** the Adobe output itself.
- Maintain stable element IDs, element ordering, explicit `group_id` values, original source context, required-field handling, and unknown extension data.
- Match elements by **exact ID**, never position or guessed roles.
- Graph tensors in S03 v0.1 use `x_numeric[N,53]`, optional `text_vec[N,384]`, `text_mask[N,1]`, `edge_index[2,E]`, and `edge_attr[E,31]`. Feature order is locked by the technical spec; changes require versioning and tests.
- The initial computational baseline must import **no** ML runtime dependency. `model_missing`, `model_incompatible`, and inference failures use deterministic ranking, never unvalidated GNN output.
- A candidate failing hard constraints cannot become valid because it has a high GNN score. Explicitly handle **zero valid candidates**; never claim guaranteed feasibility on extreme targets.

## 5. Dependencies and parallel work

| Developer 1 task | Needs Developer 2 output first? | How to proceed in the meantime |
|---|---|---|
| Implement minimal S02 parser/classes/serializer | **No** | Start now using authoritative schema and fixtures |
| Build node/edge feature extraction | **No**, after S02 interface works | Use real source designs; create test fixtures for candidate geometry |
| Build GNN and test forward/backprop | **No** | Use synthetic/mock candidate objects solely to verify tensors/training mechanics |
| Build annotation UI shell and data record format | **No** | Use placeholders for UI behavior; do **not** claim labels reflect real visuals |
| Generate varied synthetic *source* designs | **No** | Focus on coherent design grammars and parser stress tests |
| Collect trustworthy **aesthetic** pairwise labels | **Yes**, for validated candidates and reliable previews | Wait for generator + renderer; don't label known-invalid or unfaithfully rendered candidates as ordinary design-quality examples |
| Train/evaluate final candidate ranker | **Yes**, for human labels and held-out candidate sets | Prepare loss, batching, experiment configs, and a simple model baseline |
| End-to-end baseline-vs-GNN comparison | **Yes** | Agree on metric and holdout protocol now |

**Shared blocking handoffs:**

1. **Developer 1 → Developer 2:** stable `Canvas`, `Element`, `StyleRun`, `Design`, `ABSENT`, parsing/serialization tests, module import path, and documented deferred S02 features.
2. **Developer 2 → Developer 1:** functioning `generate_candidates()`, `validate_candidate()`, reproducible `candidate_id`/versions, verified renderer and fixed preview API.
3. **Developer 1 → Developer 2:** compatible ranker checkpoint/scoring interface, graph/encoder version manifest, fallback exception behavior, and evaluation summary.
4. **Both → integration:** consistent `LayoutCandidate` and `ValidationReport`, known validation behavior, candidate diversity, single official output, and documented failure cases.

## 6. Milestones and acceptance criteria

Check boxes only after actual implementation and tests; unchecked items remain planned.

### M1 — Shared S02 foundation (Developer 1)

- [ ] Locate and verify original schema, rules, samples, and existing source files.
- [ ] Implement core typed representation and `ABSENT` handling in `engine/internal_schema.py`.
- [ ] Implement lossless parsing/serialization, stable ID lookup, original groups, and derived geometry.
- [ ] Pass structural round-trip tests on all three source fixtures, with and without target canvas context.
- [ ] Test explicit null vs absent, nested extras, numeric preservation, missing assets, immutable source, and invalid input paths.
- [ ] Commit and document exact public API; list deferred controlled-edit APIs for teammate.

**Exit:** graph code can import `Design` and iterate source elements; teammate can extend S02 without rewriting it. **Do not implement S04 as part of this milestone.**

### M2A — Graph foundation (Developer 1; after minimal S02)

- [x] Implement node feature and relational edge calculations from S03 (including missing-value masks).
- [x] Verify all-pairs directed edges for designs with up to 64 elements; one graph per design/candidate/target, not separate group graphs.
- [x] Match source and candidate nodes by stable ID, not order guesses or ID semantics.
- [x] Implement optional frozen text embeddings and graph-batch tests.
- [x] Test tensor sizes, finite values, determinism, and a GNN forward/backward pass with *test fixtures*.

Implemented in `engine/graph_features.py` and `engine/text_embeddings.py`;
see [the graph foundation handoff](graph_features.md) for the exact API,
dependencies, topology conventions, and test commands. The optional encoder
interface and frozen adapter are tested with deterministic test providers;
real pretrained weights were not downloaded or evaluated. The GATv2 exercise
in the graph test module is a small CPU compatibility test. The subsequently
implemented full ranker foundation is documented under M4 below.

**Exit:** graph tensors satisfy the versioned feature contract; the model code can run without real preference labels.

### M2B — Computed candidate pipeline (Developer 2; when available, **parallel with M2A**)

- [ ] Implement S02 controlled updates and/or immutable candidate-layout installation.
- [ ] Build simple proportional/scaled baseline and meaningful layout templates.
- [ ] Produce multiple reproducible candidate layouts with distinct composition families.
- [ ] Validate Adobe retention/policy/geometry rules and report violations.
- [ ] Implement deterministic heuristic scoring and correct preview renderer.
- [ ] Demonstrate full pipeline without any model or GPU dependency.

**Exit:** at least one tested sample produces reproducible validated candidates and renderable previews; extreme-case infeasibility is documented, not hidden.

**Implemented output-safety subset:** `engine/candidate_validator.py` and
`engine/layout_search.py` now gate `resize()` output and retry invalid proposals.
See [output validation](output_validation.md). Invalid candidates never become
selected output; exhausted search raises `NoFeasibleLayout` with diagnostics.
The geometric repair preserves the source composition proportionally. It does
not complete the diverse template generator, renderer checks, heuristic scorer,
or GNN selection integration listed above. Future ranking must filter invalid
candidates first and retry generation when the valid set is empty.

### M3 — Annotation and dataset (Developer 1 lead; Developer 2 supports)

- [ ] Build side-by-side UI: A/B, tie, both-bad, optional reason, randomized placement.
- [ ] Record source/candidate IDs, target, actual shown order, annotation, generator/renderer versions.
- [ ] Collect an initial small pilot of diverse human judgments; examine ambiguities and label consistency.
- [ ] Split train/validation/test by **source design**; reserve untouched sources for final evaluation.
- [ ] Create reproducible dataset manifest and training loader.

**Exit:** labels can be traced to exact validated candidate JSON and reliable previews, with no source leakage across splits.

### M4 — Ranker, evaluation, and integration (both)

- [x] Implement the S03 GATv2 architecture and test forward/backprop on CPU and CUDA.
- [x] Implement versioned checkpoints and batched scoring with explicit model/encoder failure behavior.
- [ ] Train simple learned ranking baseline before/alongside GNN; record held-out pairwise metrics.
- [ ] Train GNN with human preference comparisons; do not infer element-importance ground truth from attention.
- [ ] Compare GNN ranking to deterministic ranking using the *same valid candidate sets*.
- [ ] Connect optional learned scores to selection; preserve deterministic fallback if checkpoint is absent/incompatible.
- [ ] Test validity, element preservation, extreme canvases, runtime, and end-to-end human preferences.
- [ ] Keep diverse top-K internal/demo alternatives while official `resize(...)` returns only one design.

The architecture/scoring foundation is implemented in `engine/ranker.py` with
tests in `tests/test_ranker.py`; see [the ranker handoff](ranker.md). Test fixtures
verify mechanics and checkpoint provenance validation only. There is no human
preference training run, evaluated checkpoint, or production fallback integration.
These foundation tasks can precede M3; the remaining training/evaluation tasks
still require validated candidates, faithful previews, and human judgments.

**Exit:** reproducible end-to-end resize, actual comparison data, explicit failure/fallback tests, no unsupported visual-quality claims.

## 7. Collaboration rules

- **Branch ownership:** Developer 1 uses feature branches for S02/S03/S05; Developer 2 uses separate branches for S04/S06. Merge S02 foundation before teammate extends it.
- **Changes to shared contracts:** propose in a PR, update spec/config and tests in the same change, and coordinate before either developer builds on the altered signature.
- **No duplicate models:** no alternative Adobe data classes, schema parsers, candidate identity conventions, or ad hoc feature ordering in parallel branches.
- **PR handoff checklist:** what changed; imports/functions exposed; test command and observed result; remaining limitations; any effect on other owner's code.
- **Honest status:** mark only tested work complete; planned architecture is not evidence of an implemented model, renderer, or contest compliance.
- **Agent instructions:** `CLAUDE.md` should link to this file, but milestone ownership and evolving status live **here**, not duplicated across prompts.

## 8. Document references / authority

1. Official Adobe challenge rules and FAQ; Adobe v0.9 JSON Schema; existing `engine/resize.py` contract.
2. `docs/S02_internal_design_representation.md` (or actual repository location): exact shared data representation specification.
3. `docs/S03-S06_layout_engine_technical_spec.md` (or actual repository location): graph features, GNN ranking, candidate generation, preference training, and integration.
4. This file: task ownership, dependencies, handoffs, and milestones.
5. `CLAUDE.md`: coding-agent behavior and development conventions.

**If anything in this plan conflicts with the official schema/rules or actual checked-in code, investigate and update the affected spec/plan before relying on it.**
