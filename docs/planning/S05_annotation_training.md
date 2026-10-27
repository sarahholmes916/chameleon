# S05 — Preference annotation and ranker training (planning)

**Status:** Proposal, split from the former S03–S06 spec (§8–11). Nothing here is
implemented: there is no `annotation/` or `training/` directory, no human labels, and
no trained checkpoint. The scoring model it trains is documented in [ranker.md](../ranker.md).

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
  "feature_version": "graph_v3",
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
