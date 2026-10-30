# S04 v0.2 implementation and evaluation

Verified locally on 2026-10-30, Windows, Python 3.11.17. No human aesthetic preference evaluation
was performed. Runtime is local wall time, not a promise about Adobe hardware or hosted CI.

## Run

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
.\.venv\ci-core\Scripts\python.exe -m scripts.ci_tests core
.\.venv\Scripts\python.exe -m scripts.ci_tests sources
.\.venv\Scripts\python.exe -m pytest -q --disable-warnings -ra
.\.venv\Scripts\python.exe -m scripts.evaluate_s04 --output data/evaluation/s04_review --sources data/sources/pilot20_v3/sources --render-pool --retain 4 --measured-fit
```

Use a new output directory for each run. Omit `--measured-fit` to evaluate the production
dependency-free word-fit path. Measured fitting is an explicitly injected development option,
not an implicit production dependency. Output contains summary.json, per-task report.json,
original source JSON/PNG, retained candidate JSON/PNG, candidate_manifest.jsonl and HTML galleries.
The manifests can be imported by the existing annotation tools. To use automatic queue discovery,
choose an output folder below `data/candidates/` rather than `data/evaluation/`.
The source-family ID is read from the source dataset manifest; absent lineage stays `unknown`.
No heuristic winner is exported as a preference label. Reports contain scores; annotation manifests
contain identity/provenance, not scores. Final selection is performed before retention diversity.

## Observed benchmark

Both final runs assessed 85 proposals across six tasks: 85 valid, 0 invalid, 3 exact duplicates,
0 additional near duplicates, 24 retained previews, 0 rendering failures. All emitted families
had zero invalid-proposal rates on this small benchmark; artificial failure cases are tested separately.
Generation failures and invalid-proposal rates are distinct, reported respectively in failures and
family_failure_rates. The default raw budget was 64 per task, retention was set to 4.

| Task | Generated/valid | Word-fit generation seconds | Measured generation seconds | Winner text fits, word/measured |
| --- | ---: | ---: | ---: | --- |
| Adobe warmup, 1080x1350 | 11/11 | 0.036 | 0.163 | 1/2; 2/2 |
| Adobe core, 1600x400 | 19/19 | 0.332 | 0.503 | 3/4; 4/4 |
| Adobe advanced, 728x90 | 17/17 | 0.420 | 0.705 | 7/8; 8/8 |
| Generated centered editorial, 728x90 | 8/8 | 0.036 | 0.097 | 3/4; 4/4 |
| Generated split product, 728x90 | 15/15 | 0.153 | 0.273 | 4/4; 4/4 |
| Generated product pedestal, 728x90 | 15/15 | 0.203 | 0.374 | 5/6; 6/6 |

Artifacts (Git-ignored): `data/evaluation/s04_final_wordfit/index.html` and
`data/evaluation/s04_final_measured/index.html`. Earlier exploratory runs remain preserved separately.
The renderer version/commit, converter hash, Python/Skia/Pillow versions, weights and configuration
are recorded in each report. Font-fit means measured extents fit the box with a two-pixel inset;
it does not certify readability or overall visual quality.

Inspection of the measured six-winner contact sheet showed the editorial headline no longer clipped
as in the first exploratory run. Extreme banners still have tiny text, compressed image regions and
large empty areas. The advanced sample retains its deliberately missing asset and produces a renderer
warning/official placeholder; no asset was invented and no rendering exception was hidden.

## Verification

- Core-only guard: **630 passed**, no skips; preflight confirms no Torch/PyG installation/import.
- Renderer/source guard: **117 passed**, no skips (77 source, 35 annotation, 5 S04 renderer cases).
- Full local suite: **1,022 passed**, no skips, 28 warnings (Gradio deprecations and optional scatter acceleration).
- Targeted Ruff and `git diff --check`: passed.
- Python 3.10 hosted checks are not claimed as run. Current core CI matrix retains Python 3.10/3.11.
- Adobe rules/schema/FAQ/README, renderer, source assets, main entrypoint, graph_v3/v4 estimator/features,
  GNN architecture/checkpoints were not modified. Existing uncommitted annotation work was preserved.

## Changed files and purpose

| File | Change |
| --- | --- |
| `engine/layout_units.py` | Reachable, guarded ungrouped text association; explicit groups unchanged |
| `engine/candidate_generator.py` | Bounded banner gaps/wrapping, adaptive packed gaps, TWO_COLUMN and CONTROLLED_OVERLAY, version v0.2 |
| `engine/layout_typography.py` (new) | S04-only word fitting and injectable renderer measurement, shrinking only font_size |
| `engine/layout_objective.py` | v3 overlap penalty detects small text occluded by large elements; other terms/weights unchanged |
| `engine/candidate_pool.py` (new) | Immutable evaluated snapshots, per-family/candidate failure isolation, validation, hashing, scoring, diversity, budgets |
| `engine/resize.py` | Production pool winner, revalidation, legacy fallback only when no valid pool result |
| `scripts/evaluate_s04.py` (new) | Reproducible real-renderer benchmark, lineage, reports, galleries, annotation manifests |
| `tests/test_candidate_pool.py` (new) | Clustering, dense/extreme geometry, new families, identity, diversity, budgets, failures, typography and objective regressions |
| `tests/test_s04_rendering.py` (new) | Three real sample renders, missing-output and renderer-exception checks |
| `tests/test_candidate_generator.py` | Version and adaptive-spacing contract updates |
| `tests/test_layout_objective.py` | Objective version update |
| `tests/test_layout_search.py` | Fail all families explicitly when testing exhausted fallback |
| `tests/test_resize.py` | Valid best-score selection instead of pinned sample winner; independent-family failure regression |
| `scripts/ci_tests.py`, `.github/workflows/ci.yml` | New core and renderer test/lint coverage in existing jobs |
| `docs/TECHNICAL_SPEC.md`, `docs/reference/D_candidate_generation.md`, `docs/IMPLEMENTATION_PLAN.md` | APIs, implemented status, versioned heuristics and remaining work |
| `docs/S04_EVALUATION.md` (new) | This evidence and usage report |

## Limits and next work

- Weights are provisional. Coverage still rewards larger occupied area and may overvalue oversized
  elements; overlap eligibility still exempts same-group pairs. Calibrate with independent human judgments
  before changing weights or presenting quality gains. Baseline remains eligible to win.
- The word-fit approximation passed 23/28 winner fit checks; it is not calibrated or exact. The optional
  measured path passed 28/28 in this run, but cannot make infeasible boxes readable. SourceRenderer's
  measurements do not fully model every style-run/letter-spacing effect. Fonts may reach the 1px schema
  floor; this remains low quality, not invented contest invalidity.
- Near-dedupe is a deterministic proxy, not perceptual equivalence. New geometry/typography stays in the
  all-candidates pool even if filtered from retained alternatives. Family balance cannot invent diversity.
- Family functions eagerly compute finite lists before the assessed-proposal budget is consumed;
  this budget is not a strict computational-work cap. Lazy families and per-variant isolation inside
  each eager family are future improvements.
- No continuous/local geometry optimizer, human preference evaluation, GNN training, learned production
  ranking, or new checkpoint/feature format is implemented. The pool `.design` view supports existing
  graph/ranker consumers, but training adapters must filter validation status and group splits by source
  family; invalid records are diagnostics, not aesthetic comparison examples.
- Measured and unmeasured candidates have different generator provenance. Regenerate candidate sets
  deliberately rather than silently mixing them or reusing old labels for changed JSON.

---

# Shape stretching (generator v0.3, `shape_stretch_v1`, objective v4)

Verified locally on 2026-10-30, Windows. Design and rules: [D.1.7](reference/D_candidate_generation.md).
No human preference evaluation was performed; every score below is the provisional deterministic objective
and is **not** evidence of visual quality.

## Limitations of resizing before this change

- `engine/candidate_generator.py` scales each layout unit uniformly (`_place_slotted`); only units whose
  members all touch every source canvas edge stretch (`_place_fill`), and non-cover images canvas-fit.
  A background-like shape that does not touch all four edges, every button, badge, bar and panel therefore
  took the unit's single scale. On 728x90 (scale ~0.07) the advanced sample's CTA came out 56x12 px and its
  badges 34-45x7 px, with most of the banner empty.
- `_center_pill_labels` only shrank a pill label's font to fit the already-shrunk shape; nothing widened
  the shape around its label.
- The scorer had no aspect-distortion term for circles, no label-padding term, and counted a stretched
  panel as foreground coverage.

## What was implemented

| Piece | File |
| --- | --- |
| Deterministic role classification (background, panel, button, bar, proportion-sensitive, fixed) | `engine/shape_roles.py` (new) |
| Post-placement independent-axis stretching and the five variants | `engine/shape_stretch.py` (new) |
| Pool integration: reserved budget, interleaved bases, provenance fields, shape-aware retention | `engine/candidate_pool.py` |
| Objective v4: ellipse distortion in `E_scale`, label padding asymmetry in `E_group`, backdrop panels excluded from `E_cover`/`E_balance`/`E_space` | `engine/layout_objective.py` |
| Version bump `s04_generator_v0.3` | `engine/candidate_generator.py` |
| Before/after benchmark, shape metrics, comparison galleries, schematic fallback | `scripts/evaluate_s04.py` |
| Tests | `tests/test_shape_stretch.py` (core, 33 cases), `tests/test_shape_stretch_graph.py` (ML), updated `test_layout_objective.py`, `test_candidate_generator.py` |
| CI wiring | `scripts/ci_tests.py` (`core`, `ml` suites), `.github/workflows/ci.yml` lint paths, `ruff.toml` |

Unchanged: `engine/graph_features.py`, `engine/ranker.py`, `graph_v3`/`graph_v4`, checkpoints, human labels,
the schema, the renderer and the validator. `engine/resize.py` needed no change: it already returns the pool winner.

## Verification

- Full suite, repository `.venv`: **1059 passed**, 0 failed, 0 skipped (28 pre-existing deprecation warnings).
- `python -m scripts.ci_tests core` in `.venv/ci-core` (no torch): **663 passed**. `ml` in `.venv/ci-ml`:
  **278 passed, 1 skipped** (the allow-listed CUDA test). The new graph test ran in that job.
- Ruff over the exact CI lint path sets of all three jobs: passed. `git diff --check`: clean.
- GitHub Actions has **not** been run (nothing was pushed). Python 3.10 specifically was not exercised locally
  (the venvs are 3.11).

## Benchmark (real renderer, measured text fit)

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_s04 --output data/evaluation/s04_shape_stretch_v1 `
  --sources data/sources/pilot20_v3/sources --source-limit 8 --source-roles button,panel,bar_h,bar_v `
  --targets 728x90,1600x400,90x728 --measured-fit --render-pool --retain 6
```

27 tasks: the three Adobe samples at their own targets plus 8 generated designs (each has a button, panel or bar)
at 728x90, 1600x400 and 90x728. Artifacts (Git-ignored): `data/evaluation/s04_shape_stretch_v1/index.html`; each task
has `comparison.html` (original, proportional-only winner, best candidate per stretch variant, selected winner),
`report.json` and rendered PNGs. "Before" is the same engine with `max_stretch_bases=0`.

| Measure | Result |
| --- | --- |
| Candidates generated / valid | 462 / 462 (0 invalid, 0 generation failures, 0 render failures) |
| Stretch variants generated / valid | 95 / 95 (panel_fill 46, balanced 38, label_fit 6, axis_x 5, axis_y 0) |
| Tasks where a stretch variant won | 9 of 27 (label_fit 2, balanced 3, panel_fill 4) |
| Tasks with variants where the proportional layout still won | 15 |
| Tasks with no stretchable shape | 3 |
| Winner score change | mean +0.0021, min 0.0000, max +0.0143 (never lower, by construction: the base stays in the pool) |
| Button labels inside their button (winners) | 12 / 12, max left/right padding asymmetry 0.00004, top/bottom 0.002 |
| Proportion-sensitive shapes, max aspect change in winners | 2e-16 (circles stay circles) |
| Runtime, 27 tasks | 7.8 s before, 10.7 s after; slowest task 0.88 s |

Examples (see the galleries):

- **Adobe core 1600x400 (stretched wins, +0.0143):** badge and CTA buttons widen around their labels with equal
  padding on every side; label fonts 11.7 -> 14.8 and 15.9 -> 16.6 px. Visually confirmed in the rendered PNGs
  that the labels sit centered and unclipped; the CTA comes close to the subtitle above it.
- **Adobe advanced 728x90 (stretched wins, +0.0070):** all four buttons re-pad around their labels and the
  CTA becomes visibly larger. The three badges got *narrower* (aspect 5-7 -> 2.5-3.3), i.e. less wide, because their
  uniformly scaled shapes had excess horizontal padding. Their label font is still only 1.9 -> 2.3 px: shape
  stretching cannot fix text that the placement stage scaled to unreadable size.
- **Generated coupon / split-product (panel_fill, balanced win):** the dark card or panel widens to fill more of
  the banner. Whether that looks better than the smaller card is a human judgement.
- **Proportional wins:** the editorial, type-poster, schedule, information-card and image-panel (two of three
  targets) designs. Stretch variants were generated for most of them but scored equal to or below the proportional
  layout; the reports do not say why, and no per-case cause was investigated. The pedestal designs produced no
  variant differing from the base at 728x90 and 1600x400.

## Limitations found and remaining

- Stretching only reshapes shapes of a placed layout. It does not move text, so it cannot rescue layouts whose text
  is already tiny (the advanced banner still has ~2 px badge labels) or whose families leave large gaps.
- Roles are conservative guesses from geometry. Ambiguous mid-size rectangles, rotated shapes, ellipses and empty squares
  are never stretched; a designer's intent for any of them is unknown. The ribbon in the advanced sample is rotated
  and stays fixed.
- Bars rarely qualify: 5 `axis_x` and 0 `axis_y` variants in this benchmark. The official samples contain no bars or
  panels, so those paths are exercised by synthetic tests and the generated designs only.
- Only single-line button labels are fitted. Two-label buttons, icon+text buttons, and buttons whose label overflows
  in the source keep the base placement.
- `panel_fill` and `balanced` often win because they enlarge a backdrop that the objective still scores through
  `E_scale`/`E_group`. Backdrop panels no longer buy coverage or hide gaps, but this is the least certain claim
  here: confirm with human comparisons before trusting it. Every new constant (aspect/padding/growth limits,
  distortion range 2:1, padding gain 4, backdrop ratio 1.5, distinctness 0.1) is uncalibrated.
- The pool's wall-clock cost rose ~37% on this benchmark (stretch passes plus extra scoring).
- Cases still needing human judgement: whether a widened panel improves or crowds the composition; whether
  tighter badge pills read better than the proportional ones; CTA-to-subtitle spacing on 1600x400.

No commit or push was performed.

---

# Post-stretch local refinement (`layout_refinement_v1`, objective v5, word-fit v2)

Verified locally on 2026-10-30, Windows. Design: [D.1.8](reference/D_candidate_generation.md). No human preference
evaluation was performed. Every score is the provisional deterministic objective, **not** evidence of visual quality.

## What was implemented

| Piece | File |
| --- | --- |
| Bounded local search, 12 operators, tiers, invariants | `engine/layout_refinement.py` (new) |
| Pool integration, separate budgets, provenance, refinement log | `engine/candidate_pool.py` |
| Objective v5: worst-text readability share; ungrouped containment | `engine/layout_objective.py` |
| Word-fit v2: 1.4 em line, 3% advance safety, shared `SINGLE_LINE_EM` | `engine/layout_typography.py`, `engine/shape_stretch.py` |
| Three-way benchmark, quality metrics, lineage galleries | `scripts/evaluate_s04.py` |
| Tests | `tests/test_layout_refinement.py` (new, core), `tests/test_s04_rendering.py` (renderer), updated `test_layout_objective.py`, `test_candidate_pool.py`, `test_shape_stretch.py` |
| CI wiring | `scripts/ci_tests.py`, `.github/workflows/ci.yml` |

Unchanged: graph features, ranker, checkpoints, labels, schema, validator, renderer, `resize.py`.

## Is E_read adequate? (it was not, enough)

E_read already penalized small text (linear against a 12px reference) but averaged it, so one unreadable text among many
was diluted ~8x, and its gradient at tiny sizes was shallow (1.9 to 2.3px moved the score ~0.006). The size part is now
0.7 x mean + 0.3 x worst. Across the benchmark the mean E_read of winners went 0.397 (baseline) to 0.389 (stretch) to
0.355 (refined). The 12px reference and the 8px refiner preference are uncalibrated quality preferences.

## Text fitting against actual rendered text

The production path (word-fit, no renderer) was compared with the real renderer on every sample text at five sizes: the
estimate was 1.6% under at worst on width and never under on height, 1.08x / 1.02x over on average (v2). v1 used 1.3 em per
line against a measured 1.36 to 1.41 em; word-fit winners that failed the renderer check went from 9 to 4 across the
sample cases, all four at the 1px floor. With renderer measurement injected (`--measured-fit`) every non-floor winner text
fit. In the final benchmark, renderer-measured fit is 395/398 (baseline), 394/398 (stretch), 393/398 (refined) texts, and
every miss is at the 1px floor, where the renderer cannot fit text in a 2 to 3px box.

## Benchmark (real renderer, measured fit)

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_s04 --output data/evaluation/s04_refinement_v1 `
  --sources data/sources/pilot20_v3/sources --source-limit 20 --targets 728x90,1600x400,90x728 `
  --measured-fit --render-pool --retain 6
```

63 tasks: the three Adobe samples at their own sizes plus 20 generated designs (editorial, split product, pedestal, image
panel, poster, schedule, cards, coupon, bands, columns) at 728x90, 1600x400 and 90x728. Artifacts (Git-ignored):
`data/evaluation/s04_refinement_v1/index.html`; each task has `comparison.html` (original, baseline winner, stretch winner,
the refined winner's parent, best stretch variants, best refined candidate, final winner), `report.json`, PNGs.
"Baseline" = no stretching and no refinement; "stretch" = shape stretching only; "refined" = both. Same generator, scorer and budgets.

| Measure | Baseline | Stretch | Stretch + refinement |
| --- | ---: | ---: | ---: |
| Candidates generated / valid (final pool) | | | 1208 / 1208 |
| Winner structurally valid, IDs preserved | all | all | all |
| Winner score, mean over 63 tasks | 0.7389 | 0.7433 | 0.7672 |
| Texts below 8px / 398 | 252 | 252 | 220 |
| Mean smallest winner font (px) | 5.69 | 6.03 | 6.16 |
| Mean E_read | 0.397 | 0.389 | 0.355 |
| Mean E_align (worse is higher) | 0.231 | 0.258 | **0.308** |
| Mean E_space | 0.276 | 0.313 | 0.143 |
| Mean E_unwanted_overlap | 0.000 | 0.0014 | 0.0025 |
| Mean E_cover (lower covers more) | 0.839 | 0.822 | 0.755 |
| Mean E_group | 0.104 | 0.092 | 0.081 |
| Renderer text-fit ok / 398 | 395 | 394 | 393 |
| Renderer failures | 0 | 0 | 0 |

Refinement itself (125 attempts on the 63 tasks, 2 bases per task):
proposals evaluated **13,655**; rejected by validation 21; rejected by invariants (containment, overlap) 3,694;
duplicate states skipped 3,745; scoring errors 0; refined candidates added 123 (all valid); accepted improvements per attempt
0 to 8 (mean 4.7); the refined layout won **59 of 63** tasks; mean winner improvement over the stretch winner **+0.0239**
(max +0.089); no winner changed in 4 tasks. Operators in winners: `grow_text` 202, `snap` 36, `separate` 34,
`scale_image` 11, `fit_container` 2, `equalize_gaps` 1. Runtime for all 63 tasks: 17 s baseline, 120 s with stretch and
refinement; slowest task 3.75 s (was 0.65 s). Latency is not a scoring concern (RULES.md), but the pool is ~7x slower.

Where it helps most, by mean refinement gain: 728x90 targets (+0.041) over 1600x400 (+0.018) and 90x728 (+0.013);
templates centered_editorial (+0.045), split_product (+0.041), editorial_columns (+0.036) over product_pedestal (+0.006),
offer_coupon (+0.010), information_cards (+0.010); winner families HERO_RIGHT, CONTROLLED_OVERLAY and HERO_LEFT (~+0.04)
over VERTICAL_STACK and SCALE_BASELINE (~+0.012).

## Where the objective improves but the rendering may not

- **Adobe advanced, 728x90** (baseline 0.6455, stretch 0.6524, refined 0.6986): the refined winner comes from a different parent
  (BANNER_ZONES) with `grow_text` x5. Its text is much larger and the layout is visibly more legible than the 2 to 5px text of
  the baseline, but in the rendering the subtitle runs along the bottom edge, the CTA touches the ribbon and the arrangement
  looks more crowded. The objective rewards readability and coverage more than the crowding it costs.
- **Alignment gets worse in 39 of 63 tasks** relative to the stretch-only winner (E_align 0.258 to 0.308): growing text and
  pushing neighbors breaks source alignments faster than `snap` repairs them, and E_align's weight (0.0675) is small next to E_read (0.225).
  This is a limit of the objective, not something the optimizer was forced around.
- **Adobe core, 1600x400**: the visible improvement is from shape stretching; refinement only rescaled the product image
  (+0.001), a change a human may not notice or prefer.
- E_cover/E_space gains partly reflect larger text filling the canvas; "more coverage" is not a goal in itself.

## Compliance and validation

Every refined candidate in the benchmark (123) passed `validate_candidate` and kept all IDs in original order. Protected
fields cannot change: edits go through `layout_edit`, the validator re-checks every proposal before and after text fitting,
and `build_candidate_pool` re-validates and re-scores the final JSON before it can be a record (a tampered refinement result
is dropped, tested). `resize()` is unchanged and still revalidates the pool winner.

## Verification

- Full suite in the repository `.venv`, core guard in `.venv/ci-core` (no torch), ML guard in `.venv/ci-ml`, and Ruff over all
  three CI lint path sets are reported in the final summary of this change; GitHub Actions has not been run (nothing pushed),
  and Python 3.10 specifically was not exercised (local venvs are 3.11).
- The core suite grew from ~660 to 730 tests and now takes ~110 s (limit 600 s per job).

## Limitations and next work

- Scores are uncalibrated; every new constant (readable 8px, gap 3%, step schedule, growth ladder, font-shrink onset 20% and weight 0.2, worst-text
  share 0.3, implicit containment) needs human judgements before trust. Compare baseline/stretch/refined pairs from
  `comparison.html` in the annotation tournament.
- 220 of 398 winner texts are still below 8px: extreme banners simply do not have room. Refinement cannot create room by
  removing or hiding elements, which is correctly forbidden.
- Operators not implemented: restricted swaps/region shifts, whole-block text moves beyond neighbor pushing, two-label
  button refit. Greedy best-improvement search can stall in local optima, and it is budget-limited here: of 125 refinement attempts 85 stopped at the evaluation cap (120), 14 at the round cap (8) and only 26 converged, so larger budgets would probably score higher at proportionally more runtime (untested).
- Refinement is applied to at most 2 bases per pool, so a weak base that would refine very well can be missed.

No commit or push was performed.
