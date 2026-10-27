# Validated output and retry policy

`resize()` now returns only a JSON snapshot that passes
`engine.candidate_validator.validate_candidate`. Invalid proposals are rejected
and generation/repair continues. There is no best-effort invalid return path.
The GNN is a scorer, not a generator or validator; the current repair loop is
deterministic and works without any model, PyTorch, assets, or network access.

## Implemented contract

`validate_candidate(source: Design, candidate: Design | Mapping, target: Canvas)`
returns a frozen `ValidationReport(schema_ok, policy_ok, geometry_ok, violations)`
with an `is_valid` property. The gate checks:

- Adobe schema validity, including conditional fields and finite JSON numbers.
- Exact original IDs once each, original array order, and target dimensions.
- Unchanged fields except `x`, `y`, `width`, `height`, and text `font_size`.
  This includes types, grouping, content, fonts, assets, style runs, extensions,
  optional-field absence/null, and the numeric types of immutable fields.
- Stored boxes and conservative rotated extents inside the target. Visible
  shape strokes expand those extents. This conservative stroke/rotation policy
  is an internal safety choice, not a claim about the official scorer's formula.
- For positive-area images other than source `fit="cover"`, relative source-box
  ratio error `abs((candidate_width * source_height) /
  (candidate_height * source_width) - 1) <= 0.05`. This operationalizes the
  approximate tolerance in `resize.py`; confirm any updated official definition.
  No exemption is inferred from semantic IDs or background-like appearance.
- As an internal anti-collapse safeguard, a positive dimension cannot become
  zero and an originally zero dimension stays zero. Schema-permitted source
  lines remain supported. Text font size remains at least the schema minimum 1.

Only output is subjected to target bounds checks. Negative source coordinates,
missing assets, and unknown permitted properties still round-trip through S02.

## Retry and failure behavior

```python
output = find_valid_layout(source, target, propose, max_attempts=8)
```

`propose(attempt_index, previous_report)` returns a new candidate; the first
index is zero and its previous report is `None`. Every proposal is snapshotted
and validated. Invalid candidates never leave this function as output. A later
valid candidate, including one on the final allowed attempt, returns immediately.

The current `baseline_proposal` first tries the old independent-axis scaling.
If invalid, it proportionally scales and translates the entire original
composition to fit measured extents, including rotation and fixed-width strokes.
Later attempts shrink further. Image ratios, relative group geometry, original
element order, and immutable fields are preserved. This is a geometric fallback,
not the planned diverse composition optimizer.

The retry budget is an internal reliability limit, **not an Adobe time limit**.
If all eight attempts fail, `NoFeasibleLayout` carries each attempt's report and
no design is returned. It means search exhausted its budget, not mathematical
proof of infeasibility. Unexpected generator exceptions propagate. Never turn
either failure into a return of the last or least-invalid draft. Unbounded
retry cannot guarantee success on impossible constraints and could hang forever.

`engine/main.py` remains unchanged: exceptions occur before writing new output
JSON or rendering PNG for the failing task and produce a nonzero process exit.
Existing output files from prior runs remain on disk; this change does not
delete them. Future CLI error handling must preserve the no-invalid-output rule.

## Ranking and verification boundary

Future candidate pools must filter invalid layouts before heuristic/GNN ranking
or presentation as valid options. If the pool is empty, generate/repair again.
Revalidate the exact selected JSON before export. A score never overrides a
failed check. The current bare graph and ranker APIs remain structural/scoring
tools and do not independently certify legality; learned selection is not yet
integrated into `resize()`.

This gate does not establish rendered text fit, legibility, aesthetic quality,
asset availability, or every possible official scoring condition. In particular,
fitting the advanced design into 728x90 can produce very small text. The proposed
renderer-aware checks and better layout families remain necessary quality work.

`tests/test_layout_search.py` covers invalid-to-valid retries, diagnostics,
exhaustion, malformed schemas, immutable fields/extensions, IDs/order, bounds,
rotation/strokes, image proportions, degeneracy, and all three supplied samples.
It is mandatory in both Python versions of the core CI job, without ML packages.
