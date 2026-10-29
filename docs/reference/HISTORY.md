# History of superseded decisions (non-normative)

Nothing in this file is a requirement. It records why current contracts look the way they do, so that
old documents and branches can be understood. The current contracts are in
[TECHNICAL_SPEC.md](../TECHNICAL_SPEC.md) and Appendices A–E.

## Feature and model versions

| Version | Change | Compatibility |
| --- | --- | --- |
| `graph_v1` | 53 numeric node fields (columns 0:53), 31 edge fields, text vector 384 + mask. | Rejected by current code; not padded. |
| `graph_v2` | Appended two typography columns 53:55 (`candidate_font_size_log`, `font_size_log_change`) so candidates with identical boxes but different font sizes are distinguishable. Columns 0:53 keep their meaning and order; the candidate size uses target-canvas normalization and the change feature is a difference of logs. Text vectors and edge features unchanged. Ranker architecture became `gatv2_ranker_v2`. | Rejected by current code. |
| `graph_v3` (default) | Appended `fill_alpha` (55) and `text_alpha` (56); color parsing extended from the fixtures' rgba syntax to the bounded parser in Appendix B.6. Numeric width 57, first GAT input width 90. Ranker architecture `gatv2_ranker_v3`. | Default. |
| `graph_v4` (opt-in) | Appended node column 57 `text_fit_height_ratio` and edge column 31 `size_order_preserved` (Appendix B.4.3). Numeric width 58, edge width 32, first GAT input width 91; ranker `gatv2_ranker_v4`. Two of the four fields first proposed were dropped after an information-content evaluation (B.7.2): the edge size log ratio (a linear function of two `graph_v3` columns) and a node validity flag (redundant). | Selectable with `feature_version="graph_v4"`; `graph_v3` graphs, architecture and checkpoints are unchanged and not interchangeable with v4. |

`ranker_checkpoint_v1` (container format) has not changed across these migrations; its architecture and
feature manifests are what reject old checkpoints.

## Superseded document structure

- The former single "S03–S06 Layout Engine Technical Specification v0.1" mixed implemented contracts,
  proposals, ownership tables and milestone tables. It was split (S03 contracts, then S04–S06 proposals)
  and then consolidated into `TECHNICAL_SPEC.md` plus Appendices A–E.
- `docs/internal_schema.md`, `docs/graph_features.md`, `docs/ranker.md` and `docs/output_validation.md`
  were folded into Appendices A, B, C and spec §8 respectively.

## Superseded milestone numberings

Three inconsistent schemes existed. The only current scheme is in
[IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md).

- `AGENTS.md` (old): M1 S02; M2 generator + validator + renderer; M3 candidate pairs + annotation;
  M4 graph builder + GNN; M5 diverse top-k + learned-guided generation.
- Old technical-spec table: M1 finish S02 and validator; M2 families and objective; M3 renderer QA and
  annotation UI; M4 solver repairs, diversity and GNN training; M5 `resize` adapter; M6 runtime stress
  tests.

## Superseded interface drafts

- Early drafts described `generate_candidates` as returning `Design` instances in one place and
  `LayoutCandidate` in another; the settled design is `LayoutCandidate` wrapping `Design`.
- Early S02 drafts proposed `Design.validate_stage_one(source=...)`; output policy checking is now
  `validate_candidate` only.
- Early plan sketches named the selection step `select_best_valid_candidate`; the spec's provisional
  name is `select_options` (open decision U4).
- Early drafts named fallback states `model_missing` / `model_incompatible`; the code raises
  `CheckpointUnavailable` and `ModelInferenceError` instead.
- An earlier draft named a `configs/graph_v3.yaml` file; no such file exists.
