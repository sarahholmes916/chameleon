# S04 — Candidate generation and computational fallback (planning)

**Status:** Proposal, split from the former S03–S06 spec (§6–7). Only the output gate
and a proportional repair are implemented; see [output validation](../output_validation.md).
No `candidate_generator`, `layout_units`, `layout_objective` or `diversify` module exists.
Interface names here (`generate_candidates`, `LayoutCandidate`, `select_options`) are
unresolved against the code; see [S03 spec §1](../S03_graph_ranker_spec.md#1-modules-and-stable-boundaries).
Official rules and the schema win over anything below.

## 6. Candidate generation policy

**Input:** immutable S02 `Design`, `Canvas target`. **Output:** 1–K candidate `Design` instances with exactly the original element IDs in the original order. No mutations of source IDs/types, text content, font families, asset IDs, group-field presence/values, or extension fields. Only contest-permitted element geometry and style fields may change; whitelist the exact allowed fields after checking authoritative RULES.md/FAQ.md.

### 6.1 Layout units

`LayoutUnit` is an internal optimization construct, **not** a new Adobe JSON group. First initialize one unit per source element. For identical explicit, non-null `group_id` strings, record a hard **membership constraint** (the group ID cannot change), but do not assume all group members form one rigid rectangle; some source groups include backgrounds and text overlay. Derive optional **soft clustering** evidence from: existing group membership, geometric containment, horizontal/vertical proximity, repeated anchor alignments. Do not use literal IDs to infer logos or CTAs. v0.1 may score these relations by fixed deterministic formulas; a learned relation head is optional v0.2.

Concrete geometric evidence: for source boxes `i,j`, define `normalized_center_distance = hypot(dx,dy)`, `same_group` as §3.3, `contains` as §3.3, and `aligned_left` when `|x_i-x_j|/SW <= .02`. For an inferred *soft* pairing, propose `association(i,j) = 0.40*same_group + 0.25*contains_any + 0.20*exp(-distance/0.20) + 0.15*aligned_left`; these coefficients are **starting heuristics**, not learned labels or Adobe rules. Exclude background-like containment relations from rigid units; only use scores to create candidate search preferences. Audit this formula on the three provided fixtures before trusting it.

### 6.2 Candidate families and generation budget

Initial composition-family enum and expected trial count:

| Family | Policy | Trials |
|---|---|---:|
| `SCALE_BASELINE` | Proportional affine scale + translation, then repair | 2 |
| `VERTICAL_STACK` | Order high-level soft units vertically with alignment variants | 4 |
| `HORIZONTAL_FLOW` | Put units into one or two horizontal bands | 4 |
| `TWO_COLUMN` | Allocate left/right regions when target ratio permits | 4 |
| `HERO_LEFT` | Largest source image region on left, other units to right | 3 |
| `HERO_RIGHT` | Mirrored hero arrangement | 3 |
| `CONTROLLED_OVERLAY` | Allow text/shape/image intentional layering inferred from source containment and z evidence | 4 |

These 24 trials are *proposal seeds*, not a guarantee of 24 valid or distinct outputs. Applicability checks suppress unhelpful families (e.g., two-column arrangement for very tall narrow targets). "Largest image" uses bounding-box area, **not** an inferred semantic product role. Every proposal must pass the same validator before learned ranking. The baseline can be used even with no NLP encoder/model checkpoint.

Defaults: `max_raw_candidates=32`, `max_valid_candidates=12`, deterministic RNG seed `42` for repeatable development, generator time budget initially `1.0s` on the reference CPU (must measure and adapt against submission time limits), and best 3 diverse internal options. Do not promise latency until benchmarked.

### 6.2a Deterministic template seeding and eligibility

For each family, first classify **background-like candidates for search only**: a shape/image with `area >= 0.75 * source_canvas_area` and source box covering the canvas center, or an image with effective `fit="cover"` and box area at least 75% of canvas. This classification is a heuristic hint and cannot justify removal, changing source z-order, or automatic full-bleed exceptions. All original elements still receive candidate boxes. Existing source containment/overlap and explicit group evidence have precedence over this hint when proposing overlay pairs.

Define candidate **layout units** as connected components of strong soft-pairing edges with association >= 0.75, but exclude background-like nodes from that grouping and cap each inferred unit at 6 nodes. Start with explicit same-group associations (where geometrically appropriate), then consider proximity edges in descending score with stable source-index tie-breaks. Unit membership exists only during candidate search; explicit Adobe `group_id` is never changed. Maintain each unit's source bounding rectangle and member-relative fractional positions. A shape that contains text can remain a separate overlay layer rather than be forced into the same rigid packing box.

For `HORIZONTAL_FLOW`, compute usable width `TW*(1-2m)` and height `TH*(1-2m)`, with initial margin fraction `m=0.04`; order units by source top coordinate then left coordinate (ties by source index), assign consecutive units left-to-right to evenly spaced region slots, and uniformly scale each unit to fit its assigned slot while preserving member-relative geometry. For `VERTICAL_STACK`, swap axes; for `TWO_COLUMN`, place alternating units by descending source unit area in the currently less-occupied column, breaking ties by source order. `HERO_LEFT/RIGHT` allocates an initial 45% width to the largest image-containing unit and 55% to the rest, then runs vertical sublayout on the remainder. `CONTROLLED_OVERLAY` starts with horizontal/vertical flow and separately reinstates candidate overlaps for source-overlapping pairs; it must retain original stacking semantics. All families run the same subsequent local geometry optimizer. These percentages/order policies are deterministic **seeding defaults** and must be tuned or rejected on evaluation; they are not claimed as inherently attractive designs.

Do not encode reading order as Adobe element-array order. The source-position sort here is only a repeatable computational starting guess; later semantic role predictions may replace it if evaluated. Retain candidate designs in original Adobe array order regardless of slot assignment.

**Overlap eligibility for v0.1's computed penalty:** eligible foreground pairs are two non-background-like, nonzero-area elements whose source-box IoU is <=0.02, neither contains the other in source, and which do not share an explicit non-null group. Ineligible does **not** mean 'should overlap'; it means 'no unconditional overlap penalty'. Source pairs already overlapping are exempt from this generic penalty, but visual diagnostics may still flag extreme target occlusion. These thresholds are heuristic and must not be used as official validity criteria.

### 6.3 Decision variables and geometric refinement

Each element has candidate variables `(x_i,y_i,w_i,h_i)`. Text elements also
have candidate `font_size_i >= 1` in pixels: font-size adaptation is explicitly
permitted by RULES.md and the Adobe schema. Search may shrink or enlarge font
size together with box geometry; changing a box alone is not typography fitting.
Preserve font family, literal content, font weight/style, alignment, line height,
letter spacing, max_lines, style runs, colors, and optional-field presence.
Rotation, image fit, and opacity also remain fixed under the current edit policy.
Do not infer headline/body/CTA roles from IDs.

The implemented proportional repair already scales text font size with its
uniform scale, floored at 1 px. The future S04 optimizer must explicitly search
font size alongside boxes: seed from proportional scaling, evaluate smaller and
larger size proposals, and retain only validated candidates. Renderer-aware
measurement should consider actual fonts, wrapping, style runs, line spacing,
and max_lines. Preserve all content when it overflows; try another box, font
size, or composition instead of rewriting/truncating it. Without dependable
measurement, mark text-fit diagnostics as unverified rather than claiming fit.
Preferred readability thresholds and preservation of source size hierarchy are
soft quality terms; the schema's 1 px minimum is the hard numeric floor, not a
readability guarantee. The graph/ranker must receive each candidate's actual
font size (columns 53:55), including when two candidates have identical boxes.

Candidate dimension minimums are not invented as hard contest rules. Preserve
zero-area lines and allowed zero-area elements under the S02 schema; do not
impose invalid positive-area requirements. Font-size search and renderer-aware
optimization remain proposed; graph_v3 feature extraction is implemented.

Two-stage solver:

1. **Discrete layout-family and region assignment** chooses a composition pattern and ordering of internal units. Use enumeration/beam search initially; optionally OR-Tools CP-SAT for discrete assignments and selective mutually exclusive foreground-rectangle packing.
2. **Continuous/local geometry refinement** adjusts positions, scale, gaps, and text font size, measuring constraint violations after each move; a deterministic coordinate-descent/random-restart implementation is sufficient for v0.1. CP-SAT is not required for rotations, arbitrary render-aware text, or all soft metrics.

Never add `NoOverlap2D` over every element. Background and decorative layers often intentionally overlap foreground content; only include confirmed conflicting foreground pairs in optional packing constraints.

### 6.4 Soft objective, fully defined enough to implement

First satisfy hard validation constraints. Among valid layouts minimize:

\[
J(L)=w_m E_{move}+w_a E_{align}+w_p E_{pair}+w_s E_{scale}+w_o E_{unwanted-overlap}+w_b E_{balance}
\]

Use starting weights `[w_m,w_a,w_p,w_s,w_o,w_b] = [0.10,0.15,0.25,0.15,0.25,0.10]` (sum 1.0); tune on held-out human preferences. Each term is averaged over its eligible set and bounded to `[0,1]`, so weights are comparable:

- `E_move`: average `min(1, ||center_c_norm - center_s_norm||_2 / sqrt(2))`; **ignore intentional family-wide directional reflow in this penalty** by computing it with low weight, not forcing source positions.
- `E_align`: average `min(1, |anchor_i-anchor_j|/0.1)` for source pairs with aligned corresponding anchors (`<=0.02` in source), excluding confirmed decorative/overlay pairs. Anchors are left, center-x, right, top, center-y, bottom, using appropriate axis normalization.
- `E_pair`: for pairs selected as soft associations, `min(1, max(0,d_c-d_s-0.05)/0.25)` where `d` is normalized center distance. This penalizes large increases in proximity but does not require rigid grouping.
- `E_scale`: mean `min(1, |log((area_c+1)/(area_s+1)) - log((target_area+1)/(source_area+1))|/3)` for scalable boxes. Exclude appropriate full-bleed backgrounds if corroborated by fit/geometry.
- `E_unwanted-overlap`: average candidate IoU for **eligible foreground pairs** that did not overlap in source, capped at 1; exclude allowed intentional overlays, zero-area elements, and background-like containment pairs. Eligibility must be explicitly logged for debugging.
- `E_balance`: Euclidean offset between *area-weighted* element center-of-mass and target center, normalized by half-diagonal, capped at 1; cap each object's contribution to avoid a single full-bleed shape dominating; ignore full-bleed backgrounds.

These are practical starting definitions, not a validated aesthetic metric. Add a preview-based text-legibility check if a dependable renderer is available; without a renderer, text-box area is an unreliable approximation of glyph fit. Avoid optimizing an unverified "readability" proxy as a hard constraint.

### 6.5 Hard validation and policy boundary

Validator checks in this order:

1. **Adobe JSON Schema:** correct root keys and allowed field types; conditional type-specific fields; values and nullability.
2. **Structural identity:** exact original ID multiset in original array order; original type, required policy, group presence/value and all extension data preserved; unchanged content, font families, asset references, and other fields frozen by authoritative challenge rules.
3. **Canvas:** `candidate.canvas == target`, dimensions positive integral under S02.
4. **Geometry:** finite numbers; permitted element size/position ranges and in-bounds/visible-extent rules as actually specified by Adobe; check rotated extents separately when required.
5. **Image policy:** exact permitted box-ratio distortion checks and full-bleed/`fit:'cover'` exceptions from current rules; do not infer from file name or ID.
6. **Rendering diagnostics:** warn on likely unreadable text, over-cropping, unexpected overlap, and missing assets; distinguish warnings from official disqualifying errors.

If the challenge provides a hard ~5% source-box proportion tolerance, implement it **after confirming the exact official definition and exception logic**, not by assuming every source bounding box is an asset's intrinsic ratio. S02 says the sample advanced design includes an image asset reference that may be missing: keep the element and asset_id regardless.

### 6.6 Repair/fallback and infeasibility

Reject each failed proposal and feed its violations into the next generation/repair attempt. Continue until at least one candidate passes every implemented hard check. All baseline and repaired candidates pass through the same gate; no model score, timeout, or requirement to return JSON grants an exception. Revalidate the selected output before serialization. Never return a least-violating or structurally-valid-but-policy-invalid draft.

The implemented `find_valid_layout(source, target, propose, max_attempts=8)` returns the first validated JSON snapshot and gives the prior `ValidationReport` to the proposer. The current deterministic proposer first tries scaling, then proportionally fits and translates the entire source composition, with progressively smaller retries. It is a basic geometric repair, not the full template optimizer. `resize()` uses this path without ML.

A finite search budget prevents hangs on impossible constraints or a broken generator. On exhaustion raise `NoFeasibleLayout` with all attempt reports; do not return a design. Unexpected generator failures also propagate. The unchanged CLI writes outputs only after `resize()` returns, so this failure produces no new JSON/PNG for the failed task. Existing files from earlier runs are not deleted. A successful return certifies the documented gate, not render legibility or official aesthetic acceptance. See [implemented checks and limitations](../output_validation.md).

## 7. Candidate option diversification

Among candidates passing all checked hard constraints:

1. Remove exact coordinate duplicates (`x,y,w,h` rounded to 0.5 px for **dedupe comparison only**, not output storage).
2. Compute layout distance between candidates A,B with matched IDs:

\[
D(A,B)=0.6\cdot\frac{1}{N}\sum_i\min\left(1,\frac{\lVert c_i^A-c_i^B\rVert_2}{0.25}\right)+0.4\cdot\frac{1}{N}\sum_i\min\left(1,\left|\log\frac{a_i^A+\epsilon}{a_i^B+\epsilon}\right|\right)
\]

Here centers are target-normalized, areas are target-area-normalized, and `epsilon=1e-6`. These choices are proposed heuristic defaults.
3. Sort by ranker score (or deterministic score if ranker unavailable), select best first, then greedily maximize `score_normalized + 0.25 * min_distance_to_selected`. Normalize scores by min-max among the current valid candidate set (all equal -> all 0.5).
4. Prefer different composition families as a tie-breaker. Return up to `k=3`; never fabricate three options if fewer distinct valid candidates exist.
5. The contest-facing `resize()` returns exactly the first chosen design in Adobe JSON; a separate `resize_with_options()` may expose a private list for the demo or annotation app.

---
