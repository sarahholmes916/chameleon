# Chameleon: Data Generation, Annotation, and Evaluation Guide

**Purpose:** Produce reproducible, Adobe-compatible source designs; generate valid and deliberately invalid resize candidates; collect human preferences; and evaluate the S03/S05 ranker without leaking test designs into training.

**Scope:** Development and evaluation data, **not** the stage-one submission interface. This document is an implementation plan, not a claim that these utilities already exist.

**Source-of-truth order:** Before implementation, read the repository's actual `RULES.md`, `FAQ.md`, `chameleon_design_schema_v0.9.json`, `docs/TECHNICAL_SPEC.md` and related appendices. Where this guide proposes methods, thresholds or schemas not mandated by Adobe, it marks them as **team proposals**. Do not silently promote an internal rule to an Adobe rule. If official materials disagree, record the disagreement and ask Adobe rather than guessing.

## 1. What Adobe provides and actually evaluates

The hackathon starter kit provides:

| Input task | Elements | Source → target |
| --- | ---: | --- |
| `input/chameleon_sample_level1_warmup.json` | 4 | 1080×1080 → 1080×1350 |
| `input/chameleon_sample_level2_core.json` | 11 | 1080×1080 → 1600×400 |
| `input/chameleon_sample_level3_advanced.json` | 17 | 1080×1350 → 728×90 |

It also provides `brand_kit/assets/`, `brand_kit/fonts/` (Solstice Serif and Solstice Sans), a v0.9 schema, and a working `cr-renderer`-based preview path. The brief does **not** provide a large collection of independent labeled resize pairs, human preference scores, or ideal outputs. Stage-one grading uses **both output JSON validity and the rendered PNG**. The unseen tasks cover unfamiliar source designs and target sizes, and the brief explicitly discourages hard-coding individual cases.

**Official stage-one constraints relevant to data:**

- Input consists of `source` (a canvas and text/image/shape elements) and `target_canvas`; the engine preserves the element inventory/IDs and returns a resized design.
- Preserve content and font family for stage one; adapting text `font_size` is allowed. Do not regenerate, redraw, or replace image pixel content. The brief explicitly calls regeneration disqualifying.
- Preserve image proportions and make the layout legible. Ordinary accidental collisions are undesirable; intentional layering and overlap are legitimate.
- Keep stage-one runtime offline and containerizable. The permitted-model list and size budget require Adobe confirmation before shipping a trained GNN.
- The supplied Level 3 sample intentionally references a **missing** `holiday_watermark` asset; do not add or fabricate that asset. Record and test missing-asset behavior.
- The official brief mentions **Crello** as the underlying renderer's research data representation, but **Crello files are not Adobe v0.9 input JSON**. The brief does not supply a ready-to-use Crello training corpus or permission to treat it as one.

**Important:** Rules about off-canvas/bleed, rotated bounds and which fields are mutable have known ambiguities (project U1). Maintain separate flags for **confirmed Adobe violation** and **stricter internal rejection**, and preserve the official documents unmodified.

## 2. Two distinct kinds of data, with distinct consumers

```text
Original source designs + target canvases
  ├─ Source generator / curated designs (development-only)
  │       ↓
  ├─ S04 production candidate generator (when implemented)
  │       └─ controlled perturbation generator (S05 diagnostic fallback)
  │                       ↓
  ├─ structural validation ── invalid → diagnostics dataset
  │                       ↓ valid
  ├─ render JSON to PNG
  │                       ↓
  ├─ same-source, same-target A/B pairs → human annotations
  │                       ↓
  └─ S03 graph_v3 or graph_v4 → S05 GNN training and held-out evaluation
```

**S04 produces the candidates.** The GNN **does not** generate layouts: it scores complete valid candidate graphs assembled from `(source, candidate, target)`. During model development, a separate controlled perturbation script can make candidates before S04 is ready. Once S04 is available, include its real candidates so training matches the actual inference distribution.

Use **three segregated pools**:

1. **Source designs:** coherent, readable original layouts from which resizing starts.
2. **Visually good/bad but structurally valid resized candidates:** eligible for human A/B ranking labels. “Good/bad” is determined by judgment, **not by the generator's intended mutation**.
3. **Structurally invalid candidates:** one or more rule violations, used for validator and renderer diagnostics; **never mixed into the ordinary GNN preference-training pool**.

## 3. Generate a diverse collection of original source designs

### 3.1 Prioritize provenance, originality and clean conversion

Start with the three provided Adobe samples as **integration fixtures only**. They are not enough for reliable GNN evaluation. For more sources, prefer:

1. **Original designs created directly in v0.9 JSON**, using structured layout templates and supported assets/fonts.
2. **Manually curated designs made by the team**, converted to v0.9 only when their element geometry, text, image assets, fonts and groups can be represented faithfully.
3. **Externally sourced research designs, e.g., Crello, only as an optional research path** after checking dataset availability, license/asset use and conversion fidelity. Convert categories, normalized coordinates, rotation and image crops carefully. Reject anything whose content/assets cannot be represented legally and accurately. Do not assume Adobe endorses Crello as training data.

Do **not** treat a screenshot or flattened image as an editable element-level source design. Do **not** fabricate or substitute any original image asset to make a candidate render.

### 3.2 Template-based generation (recommended team method)

Implement a **development-only** script such as `scripts/generate_source_designs.py`. Use ~8–12 genuinely distinct composition families: centered editorial, image-left/text-right, image-right/text-left, product focal, full-bleed image with a safe text area, typographic poster, multi-card information layout, event flyer, discount/CTA, and asymmetric grid. These are suggestions, not mandated by Adobe.

For each family, vary meaningful dimensions while satisfying the schema:

| Axis | Examples to cover |
| --- | --- |
| Source shape | portrait, square, landscape |
| Element complexity | ~4–8, ~9–15, ~16–25 elements (initial ranges, not official limits) |
| Text | short/long headlines, multiple paragraphs, multiple type-size levels, line breaks |
| Images | none, one primary asset, multiple provided/legally usable assets |
| Groups | absent, CTA text+shape, badge, image/caption, cards |
| Layers | no overlap, deliberate background/decorative overlap, text over quiet image zones |
| Style | light/dark contrast, distinct palettes, alignment schemes |

Create source layouts using constrained blocks and meaningful margins; **never scatter every element with independent random coordinates**. Font sizes and boxes should accommodate content in the actual renderer. Use supported `font_family`, `asset_id`, text fields, shape fields and type-specific properties from the installed schema. Preserve valid `group_id`s, `z_index`/stacking, and applicable defaults.

For a pilot, generate **20 source designs**, render them, and have someone visually inspect each. Fix source-template problems before scaling to **~100 distinct source designs**; these are project starting targets, not Adobe dataset requirements. Diversify template families, content semantics and element arrangements rather than producing 100 recolors of one layout.

### 3.3 Source quality gate

For every newly generated source:

1. Parse and validate against the repository's **actual v0.9 schema** and source-specific checks; do not call `validate_candidate` with a nonexistent source/target contract.
2. Check unique element IDs, supported element types/fields, valid dimensions and asset references, with intentional missing-asset fixtures tracked separately.
3. Render with the same project renderer used for candidate previews. The renderer must use the provided font and asset resolution rules.
4. Review actual PNG appearance: all important text readable, no accidental clipping, obvious hierarchy, expected grouping, reasonable contrast; manually accept or reject questionable sources.
5. Store rejected designs with rejection reasons; **schema-valid does not mean visually good**.
6. Detect duplicate JSON/content and near-duplicate layouts; assign stable **source family IDs** so siblings cannot cross splits.

## 4. Choose resize tasks (source + target)

For each accepted source design, choose several target canvases across real-world aspect ratios, including similar-aspect warmups, square↔wide, portrait↔landscape, and extreme banners. The brief says scoring targets come from a catalogue of common real-world formats; **do not invent that a particular list is official** until it is supplied. Keep each target's exact width, height and provenance.

Cover all three difficulty levels with independent sources. Do not optimize only for the supplied 1080×1350→728×90 example. Track the ratio change and difficulty category in metadata, not in the official resize JSON.

**Task shape (official concepts, illustrative object):**

```json
{
  "source": {"canvas": {"width": 1080, "height": 1350}, "elements": []},
  "target_canvas": {"width": 1600, "height": 400}
}
```

The empty elements array above is **only an illustration of the outer structure**, not a valid training example. Read the actual sample tasks and schema for all mandatory fields and exact nesting.

## 5. Generate resized candidate sets

### 5.1 Primary candidates: S04 (when available)

Feed each source-target task to your teammate's `generate_candidates` implementation once its interface is agreed. Retain every proposal's source/task IDs, generator revision, layout family, random seed, geometry/font-size changes, validation result and any repair history. Do not assume S04 already exists.

### 5.2 Early diagnostic candidates: S05-controlled perturbations

Before S04 is ready, build a **separate development-only** generator seeded from valid baseline/candidate layouts. Create purposeful, bounded perturbations such as:

| Controlled mutation | Desired comparisons |
| --- | --- |
| Font-size ladder | tiny vs readable vs clipped/overflowing; both acceptable sizes |
| Text box edits | same font, changed wrap/available space |
| Alignment | aligned vs slightly misaligned vs conspicuously misplaced |
| Spacing | even vs cramped vs excessive gaps |
| Group cohesion | CTA/badge intact vs displaced member vs controlled reflow |
| Unwanted overlap | previously separate elements begin colliding; preserve intentional source overlaps |
| Relative hierarchy | headline shrunk or secondary copy dominating |
| Image placement | proportion-preserving but less effective composition |

Generate **subtle, medium and strong** variants. Include good–good and bad–bad pairs, not only obvious good–bad. The controlled generator must not quietly change content, font family, image asset bytes or other frozen stage-one fields. Every output must go through the normal structural validation process.

**Overlap policy:** A box intersection alone is not an Adobe violation. Backgrounds, deliberate source overlaps, group-contained text and button backgrounds may correctly overlap. Separate *intended visual defect* from *observed rule failure*. The rendered preview—not a naive collision count—determines whether an overlap actually degrades quality.

### 5.3 Invalid-case generator (separate diagnostics path)

Intentionally construct isolated violations, e.g. missing/duplicate IDs, missing required fields, changed immutable text, unexpected assets, altered image proportions, out-of-bounds geometry under the team's currently configured interpretation, invalid dimensions. Track each failure's **expected rule, actual validator response, and official-vs-internal status**. Test one violation at a time first, then combinations. Rendering invalid designs may fail; record the failure rather than silently dropping it.

Never use “invalid” as an automatic low GNN preference label; the production pipeline should reject these before the ranker. Maintain diagnostic tests for false acceptance and false rejection.

## 6. Validate, render and deduplicate

For each candidate:

1. Confirm the source/target task and candidate refer to the **same element inventory and IDs**.
2. Run the project validator, distinguishing hard structural failure from quality issues. Capture reason codes where available; do not assume the existing report has fields that are still only proposed (U6).
3. Accept into the **ranking pool** only if structurally valid under the currently documented policy; put rejected cases into diagnostics.
4. Render accepted candidates with the repository's actual `cr-renderer` integration; retain the exact JSON and PNG together. Record font/asset resolution and renderer revision.
5. Visually review a pilot to detect text clipping and renderer mismatches; do not infer legibility from JSON-only box tests.
6. Deduplicate canonical candidate JSONs and identical renders. A stable content hash can be a **local dataset ID** without prematurely deciding the production U5 candidate identity contract.
7. Record rendering failures and missing assets; do not substitute or generate replacement imagery.

**Renderer limitation:** The local renderer is supplied for previews, but parity with Adobe's evaluation renderer is an open question (project U9). Track the revision; do not claim equivalence without confirmation.

## 7. File layout and reproducible records

Recommended structure inside the **same Chameleon repository**:

```text
chameleon/
  annotation/                 # optional local Streamlit UI and SQLite access
  scripts/                    # source, controlled-candidate and render generators
  training/                   # dataset adapter, pairwise train/evaluate scripts
  data/                       # ignored generated outputs (except tiny safe fixtures)
    sources/
    tasks/
    candidates/valid/
    candidates/invalid/
    renders/
    manifests/
    annotations.db
  tests/                      # generator, data contracts and no-leakage tests
```

Keep code, schemas, manifest examples and fixed seeds in Git. Ignore bulk JSON/PNGs, SQLite database files, proprietary assets and any data whose license is unclear; do not commit copied or redistributable-ineligible external imagery.

**Proposed manifest record (dataset-only, not Adobe's engine JSON):**

```json
{
  "task_id": "task_001",
  "source_id": "src_001",
  "source_family_id": "template_product_left_001",
  "source_provenance": "original_synthetic",
  "target_canvas": {"width": 1600, "height": 400},
  "candidate_id": "sha256:<canonical-json-hash>",
  "candidate_origin": "controlled_typography",
  "perturbation": {"type": "font_scale", "parameter": 1.25},
  "generator_seed": 42,
  "split": "train",
  "validation_policy_version": "recorded_version",
  "structurally_valid": true,
  "candidate_path": "data/candidates/valid/....json",
  "preview_path": "data/renders/....png",
  "renderer_revision": "recorded_version"
}
```

Also preserve the generator/template version, asset/font references, missing-asset flag, validation findings and source acceptance status. Paths should be relative to the repository or dataset root, never machine-specific absolute paths.

## 8. Human A/B annotation UI

Build a **local, optional Streamlit app** (not part of the submission-time resizing engine). Keep its dependencies in separate annotation requirements so core/Docker inference is unaffected.

### Human preference page

- Compare **two structurally valid PNG candidates from the same source and target**, displayed at equal *physical canvas scale* with full-resolution zoom. Show original source preview on demand.
- Randomize left/right assignment, but store immutable candidate IDs and actual mapping.
- Label **A better / B better / tie / both bad / skip**; optionally record *reason tags*: typography, clipping, accidental overlap, hierarchy, alignment, spacing, groups, imagery, other.
- Keep generator identity, perturbation type and model/heuristic score **hidden from annotators** to avoid bias.
- Save every response immediately to SQLite; enforce a unique annotator–comparison key and durable resumability, prevent duplicate accidental submissions, and allow an audit trail for revisions.
- Record annotation timestamp, annotator pseudonym, pair ID, candidate IDs, randomized display order, label, reason tags, UI/renderer versions.
- Provide filters, progress, review and controlled re-labeling. Include repeated pairs with reversed A/B presentation to estimate individual consistency.

### Validation diagnostics page

Review **invalid** JSON tasks separately. Show the expected violation category, actual validator outcome, source/target IDs and any render/error output. Record false accepts and false rejects. Do **not** mix this page's classifications into preference labels.

### Pair selection

Only pair candidates within a single `(source_id, target_canvas)` task. Balance obvious differences, moderate differences and near-ties as a **sampling strategy**, not assumed labels. Deduplicate unordered pairs. Record selection probability/method; do not let the GNN's own score manufacture ground truth. Later, active-learning sampling is possible, but evaluate on a separately frozen, independently sampled test set.

## 9. Split before generation; avoid leakage

Assign **source families** to train/validation/test before making resize candidates, perturbations or pairs. A family includes recolors, text swaps, alternate crops and every templated sibling of a base composition. Keep all family members together.

- Use a **family-disjoint** split for model selection; optionally reserve entire layout template categories for an out-of-template test.
- Ensure the provided Adobe sample designs, if used for development, are clearly flagged and not presented as independent proof of generalization.
- For very small datasets, use grouped cross-validation instead of pretending a 3-source held-out test is statistically persuasive.
- Never tune model, text-fit constants, deterministic weights, candidate sampler or thresholds on the final held-out test set.
- Avoid propagating exact duplicate rendered designs and repeated pair records across splits.

A starting experiment could be ~20 manually reviewed source designs and ~100–200 valid candidate outputs for pipeline debugging. Later scale to ~100+ independent sources, many target shapes and meaningful S04 candidate diversity. **These are proposed development targets, not rules or proven sample-size requirements.**

## 10. Train and compare graph_v3 versus graph_v4

For each valid, labeled candidate, call the **existing S03 graph builder** with `(source, candidate, target)`. Use the same human-labeled pairs, split, training budget and comparable hyperparameter-search budget for `graph_v3`, `graph_v4`, and a pooled-feature baseline. The deterministic scorer is another comparison once implemented.

The ranker outputs a scalar per **whole-layout** graph. For decisive A/B comparisons, a typical pairwise ranking loss is:

```text
p(A preferred to B) = sigmoid(score(A) - score(B))
loss = -[y*log(p) + (1-y)*log(1-p)]
```

Map screen A/B choices back to stable candidate IDs before forming the training target. Do not coerce “tie,” “both bad,” or “skip” into fabricated winners; record separately until the loss/evaluation explicitly supports them. Invalid layouts remain excluded from ranking training.

**U13 decision:** `graph_v3` remains the default. `graph_v4` is optional and contains experimental typography features, including a text-fit estimate that depends on the currently uncalibrated glyph-width constant `k=0.5`; validate the estimate against rendering, and keep checkpoint/feature metadata versioned. The feature-information experiment does **not** demonstrate higher ranking accuracy. Train and compare both versions **only after human labels exist**, and pre-register the pass threshold before comparison. Do not swap the default based on untrained scores.

## 11. Evaluate beyond one accuracy number

Report, on source-family-held-out designs:

| Evaluation | Measure |
| --- | --- |
| Structural correctness | validator pass/fail rate; false accept/reject on invalid pool |
| Preference prediction | A/B pairwise accuracy for decisive labels; tie/both-bad separately |
| Error breakdown | font readability/overflow, accidental overlap, grouping, hierarchy, alignment, spacing |
| Real selection quality | human preference for top-1 chosen candidate from a genuine S04 candidate set |
| Robustness | source category, element count, source→target aspect-ratio change, missing assets |
| Statistical uncertainty | confidence intervals clustered by source family, not by highly correlated pairs |
| Operational fit | offline Docker test, inference memory/runtime and model eligibility |

Also compare GNN and deterministic baseline on **the same candidate pool**. Keep controlled-diagnostic performance separate from realistic S04-generated selection performance. A GNN that succeeds on typography ladders might still fail on composition-level reflow.

## 12. Suggested implementation sequence and acceptance checks

| Milestone | Implement | Completion evidence |
| --- | --- | --- |
| M0 — Audit | Map schema, sample JSONs, validator, renderer and supported fonts/assets | Documented contracts; no assumed unsupported fields |
| M1 — Sources | 20 original, diverse source JSONs + previews + manifest | Schema-valid, renderer works, manual quality review; duplicate/family tracking |
| M2 — Controlled pool | Valid visual perturbations and separate invalid diagnostic fixtures | Deterministic seeds, validation log, actual PNGs and meaningful variation |
| M3 — Annotation app | Streamlit A/B + diagnostics, SQLite and resume | Randomized-side label tests, durable data, no hidden labels/scores |
| M4 — Pilot labels | 50–100 human comparisons (starting target) | Consistency check, balanced category coverage and no split leakage |
| M5 — S04 integration | Import real S04 candidate sets | Same manifest and UI, with no S04 logic duplicated in S05 |
| M6 — Training/eval | v3/v4 + baseline, held-out source-family evaluations | Reproducible checkpoint and honest uncertainty report |

Tests should cover reproducible generation from seeds, strict no-mutation of input sources, legitimate schema/asset use, structural validation, render failures, unique IDs/deduplication, left/right label mapping, split isolation, persistence/resume and checkpoint-feature compatibility.

## 13. Provenance and open decisions

**Official material:** *Adobe University Hackathon 2026 — Chameleon Challenge*, especially §§2.2–2.5 (inputs/quality), §3.1 (hidden evaluation), §4 (offline and fixed materials), §7.1 (starter-kit structure and renderer), §7.2 (JSON+PNG judging, stage-one field restrictions, no asset regeneration), §7.3 (sample task list, Crello difference, brand-kit and intentional missing asset). Cross-check the repository's `RULES.md`, `FAQ.md`, and schema for authoritative field details.

**Project-specific design decisions:** S02 internal representation, S03 `graph_v3` default and optional `graph_v4`, S04 generator/repair/validation, S05 human-labeled whole-layout ranking, and ownership are described in `docs/TECHNICAL_SPEC.md` and its appendices. Some interfaces remain proposed and should be checked against actual code before implementation.

**Open checks before shipping:** U1 (bleed/field interpretations), U5 (production candidate identity), U6 (validation-report contract), U8 (approved models, sizes and runtimes), U9 (local renderer parity), and relevant integration contracts. None prevents beginning a reproducible **development dataset**, but they do prevent claims about full official compliance or model eligibility without verification.

---

**Non-negotiable distinction:** Generating an original source design for local research is not the same as modifying image content in a submitted resize result. For stage one, given any source design, the engine must preserve the original element content/assets and output only a permissible rearrangement. The dataset workflow must continually test this separation.
