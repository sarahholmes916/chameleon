# S06 — Integration, acceptance tests and risks (planning)

**Status:** Proposal, split from the former S03–S06 spec (§12–16). `resize()` currently
runs only the validated proportional repair ([output validation](../output_validation.md)).
Milestone ownership and status live in [IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md);
the milestone table in §14 is a historical numbering and should not be relied on.
Graph and model acceptance tests in §13 describe implemented behavior covered by
`tests/test_graph_features.py` and `tests/test_ranker.py`.

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

**Graph:** N=1 -> one node, zero explicit edges; N=4 -> 12 directed edges; N=17 -> 272 directed edges. Duplicate IDs must be rejected upstream. For `group_id=None`, `group_id=ABSENT`, and `group_id=''`, assert correct `same_explicit_group` behavior. For negative `x/y`, zero width/height, and rotated boxes, assert finite graph values and no stored-data mutations. Reverse edge relative-displacement signs must be opposites. Every candidate for the same source has the exact same ordered IDs and edge-index topology. Check `x_numeric.shape==(N,57)`, `text_vec==(N,384)`, `text_mask==(N,1)`, `edge_attr==(E,31)`, output scalar finite.

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
