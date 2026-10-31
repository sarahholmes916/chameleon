# Creative pilot data, label review, training and blind evaluation

Development tooling and data built around the layout ranker. **None of it is part of the competition path** except the one
checkpoint and the seam described in [Shipping](#shipping-the-checkpoint). Specifications stay in
[TECHNICAL_SPEC.md](TECHNICAL_SPEC.md); this file says what exists, how to run it, and what each result does and does not show.
Generated data lives under `data/creative_pilot_v1/` (not meant to be committed in full).

## 1. What was built

| Area | Code | Output under `data/creative_pilot_v1/` |
|---|---|---|
| 22 new source composition families, 100 sources | `creative_pilot/families.py`, `sources.py` | `sources/` |
| Candidate generation for each source × {1600×400, 728×90, 90×728} | `creative_pilot/blocks.py`, `layout.py`, `operators.py`, `degrade.py`, `generate.py`, `run_candidates.py`, `consolidate.py` | `candidates/`, `candidate_manifest_all.jsonl`, `validation_report.json` |
| Proposed pairwise labels (AI-assisted, never human) | `creative_pilot/calibrate.py`, `labeling.py`, `visual_review.py`, `visual_tiers.py`, `splits.py` | `labels/proposals.jsonl`, `labeler_calibration.json`, `splits.json` |
| Human review of the proposals | `annotation/ai_review.py`, `annotation/ai_review_ui.py` (tab "Review AI labels" in `annotation/app.py`) | `review.db`, `ai_review_exports/` |
| Analysis of the original human rankings | `scripts/analyze_human_rankings.py` | `human_ranking_analysis.md`, `human_ranking_stats.json` |
| Label audit and GATv2 training | `training/data.py`, `build_dataset.py`, `train.py`, `weighting.py` | `training_v1/` |
| Blind visual comparison of ranking methods | `training/blind_eval.py`, `blind_eval_ui.py` | `blind_eval/` |
| Inference timing | `training/benchmark_inference.py` | `training_v1/inference_benchmark_cpu_1thread.json` |

Tests: `tests/test_creative_pilot.py`, `test_ai_review.py`, `test_training.py`, `test_blind_eval.py`. CI wiring is in [CI.md](CI.md).

## 2. Creative candidate data

- **Sources:** 100 generated designs (20 easy / 50 medium / 30 hard) in 22 families, none overlapping the 11 families of the original human rankings. They use the same three assets, two fonts and palette sampler as the original sources, so style leakage across datasets is possible.
- **Candidates:** per task, S04 baselines (`conventional`), creative restructurings (`creative`: 14 recipes, e.g. columns, headline-dominant, stagger, bleed hero, bands, zigzag), and controlled degradations (`mediocre`, `poor`: ten operators such as tiny text, flat hierarchy, crowding, misalignment). 3,280 candidates over 300 tasks: 1,180 creative, 600 conventional, 900 mediocre, 600 poor.
- **Rules kept:** only `x`, `y`, `width`, `height` and text `font_size` change, through `engine.layout_edits.layout_edit`; images keep their aspect ratio; every candidate passes `validate_candidate`; text is fitted with the real renderer's measurements (`creative_pilot/fitting.py`), not the production word-fit estimate.
- **`intended_category` is provenance only.** It records how a candidate was made and is never a quality label.
- Known limit: 121 S04 baseline candidates have text the renderer cannot fit; they are recorded, not removed.
- Regenerating would add a few candidates (a late float-margin fix); the shipped data was left as is so review IDs stay stable.

Commands (from the repo root, venv active):
```
python -m creative_pilot.sources --output data/creative_pilot_v1/sources
python -m creative_pilot.run_candidates --pilot-root data/creative_pilot_v1 --workers 10
python -m creative_pilot.consolidate --pilot-root data/creative_pilot_v1
python -m creative_pilot.calibrate          # fits the labeler on original families, tests on held-out ones
python -m creative_pilot.splits --pilot-root data/creative_pilot_v1
python -m creative_pilot.labeling --pilot-root data/creative_pilot_v1
```
(`visual_review` / `visual_tiers` produce and record the blind contact-sheet assessments used for a 192-pair subset.)

## 3. Proposed labels and how far to trust them

2,400 proposals (A better 839, B better 862, tie 32, both bad 60, no proposal/SKIP 607). Each records its method:
`rendered_metrics_model` 1,469, `claude_visual_contact_sheet_v1` 192, `controlled_degradation_checked` 124, `rule_both_unreadable` 8, none 607.
- The metrics model is a pairwise logistic model on measured layout features, fit on 7 original families and tested on 4 held-out ones: **77.5 % directional agreement with human labels (n = 71)**. Its confidence numbers are uncalibrated.
- Tie and both-bad predictions are unreliable. SKIP means "ambiguous, needs a human".
- **Splits are by whole composition family** (14 train / 4 val / 4 test families; 63 / 18 / 19 sources).
- Proposals are `synthetic_proposed` until a person acts. Nothing is marked human-verified automatically.

## 4. Reviewing proposals (Accept / Reject / Correct)

```
python -m annotation.app --db data/creative_pilot_v1/review.db --proposals data/creative_pilot_v1/labels/proposals.jsonl --port 7870
```
Open the **Review AI labels** tab (import is idempotent). Accept keeps the proposal as a human-verified label; Reject drops it from training but keeps it for audit; Correct supplies your own label (A better, B better, **A good·B bad**, **B good·A bad**, tie, both bad). Keys: `A`/Enter accept, `X`/`R` reject, `1`/`2`/`5`/`6` A/B/A-good/B-good, `3` tie, `4` both bad, `S` skip, `Z` undo. SKIP items are hidden from the queue. The "blind evaluation subset" (369 pairs) is held out for labelling without seeing the proposal.

The Export button (accordion "Recent reviews and exports") writes to `ai_review_exports/`: `synthetic_human_verified.jsonl` (use for training), `synthetic_human_rejected_audit.jsonl` (audit only, never train) and `synthetic_proposed_unverified.jsonl`. Rows use `preference.v1` plus provenance and, when you used the good/bad buttons, `candidate_a_quality` / `candidate_b_quality`. Reviews are stored append-only in `review.db`; undo voids, never deletes. The original tournament database and exports are only read.

## 5. Training (graph_v3, numeric-only GATv2)

```
python -m training.build_dataset      # combine + audit + cache graphs -> training_v1/
python -m training.train              # original-only / combined / synthetic-only x 5 seeds
python -m training.weighting          # synthetic loss weights 0.25 / 0.5 / 0.75
```
Needs the torch + PyG environment (CUDA optional; the run used an RTX 5090, torch 2.11.0+cu128). Combines the 367 original comparisons with the 1,520 you verified; only A/B-better rows (312 + 1,117 = 1,429) feed the pairwise loss. The audit checks file resolution, content hashes, Stage 1 validity, graph shapes, duplicates, label conflicts, preference cycles and family/source/candidate leakage between splits (all clean). Splits are by family (train 19 / val 6 / test 8). Settings follow Appendix E.3; GPU atomics make reruns statistically, not bit-wise, reproducible.

Headline numbers and caveats are in `training_v1/TRAINING_REPORT.md` and `MODEL_SELECTION.md`. In short: offline pairwise accuracy is above the deterministic objective (about +8 to +10 points) but only 0 to 3 points above a simple pooled-feature baseline on the original test labels, and the unweighted combined model did not improve on original-only on those labels. Down-weighting synthetic labels to 0.25 recovered original-label accuracy (0.767 vs 0.741 original-only, 0.720 combined), selected by a validation-only rule. 98 original test pairs from 4 families: differences of a few points are within noise.

## 6. Blind visual evaluation

```
python -m training.blind_eval build          # score pools, sample 30 disagreement tasks (already built; seed 11)
python -m training.blind_eval serve --port 7880
python -m training.blind_eval report         # reveal + tally
```
The gallery shows the source and 2–3 anonymous, position-shuffled layouts selected by the deterministic objective, the original-only GATv2 and the combined GATv2 from one shared pool per task (held-out families only). Judgments go to `blind_eval/results.db`; the answer key is in `blind_eval/private/` (do not open before judging). Result (30 tasks): objective, original-only and combined were each chosen best on 13 / 13 / 14 tasks; every head-to-head was close to even (all sign-test p ≥ 0.55). The weighted checkpoint was **not** one of the judged arms. Because the gallery is drawn from the 58 disagreement tasks, its rates describe disagreement cases only (all three methods agreed on 19 of 77).

## 7. Shipping the checkpoint

`engine/learned_selection.py` re-ranks the pool's validated, distinct candidates with `engine/checkpoints/ranker_synth_weight_0.25_best.pt` (SHA-256 pinned) and `resize()` re-validates the pick. Torch and PyG are imported lazily; if either is missing, the file hash is wrong, scoring fails, or `CHAMELEON_RANKER=off`, the deterministic winner is returned. Designs above 64 elements fall back. The Docker image installs CPU `torch==2.11.0` and `torch-geometric==2.8.1`; the built `linux/amd64` image (2.44 GB) ran the three sample inputs with `--network none` and matched the Windows results exactly. Not verified: GitHub Actions, the Apple M5 Pro reference machine, any quality gain over the deterministic objective.

## 8. Limits to keep in mind
- Every synthetic label began as an AI proposal; the reviewer agreed with 82 % of directional ones. Accuracy on them is partly in-distribution.
- The evaluated pools included creative layouts that the production engine does not generate; an S04-only-pool comparison has not been done.
- 458 tie / both-bad rows are unused. A future "acceptable layout" head could use them with a separate target.
- Teammate tournament labels (`data/rankings/teammate/`) were not used.
- No human evaluation of final designs beyond the 30-task blind comparison exists.
