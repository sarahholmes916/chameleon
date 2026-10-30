# Appendix D — S04 candidate generation and diversification (planned design)

> **Appendix D of [TECHNICAL_SPEC.md](../TECHNICAL_SPEC.md).** Status: **partially implemented**.
> Implemented: `layout_units`, `layout_objective` (provisional weights), and `candidate_generator` with
> `SCALE_BASELINE`, `VERTICAL_STACK`, `HORIZONTAL_FLOW`, `HERO_LEFT`, `HERO_RIGHT`, `BANNER_ZONES`
> (engine/candidate_generator.py). Planned: `TWO_COLUMN`, `CONTROLLED_OVERLAY`, continuous geometry
> refinement, and diversification (D.2). Thresholds, weights and families are proposed engineering defaults
> to be verified experimentally, not Adobe rules. Constraints: spec §8; batch interface: spec §2.3.

## D.1 Candidate generation policy

**Input:** immutable S02 `Design`, `Canvas target`, a `LayoutConfig` (recommended fields: spec §5.4). **Output:** 1–K `LayoutCandidate` objects (spec §2.2), each wrapping a `Design` with exactly the original element IDs in the original order. What a candidate may change is defined only in spec §8.1; every candidate must pass `validate_candidate` before ranking.

### D.1.1 Layout units

`LayoutUnit` is an internal optimization construct, **not** a new Adobe JSON group. First initialize one unit per source element. For identical explicit, non-null `group_id` strings, record a hard **membership constraint** (the group ID cannot change; spec §8.1.5), but do not assume all group members form one rigid rectangle; some source groups include backgrounds and text overlay. Members of an explicit group are moved together as one unit first; controlled reflow is allowed where the unit cannot be placed validly or readably (spec §8.1.5). Derive optional **soft clustering** evidence from: existing group membership, geometric containment, horizontal/vertical proximity, repeated anchor alignments. Do not use literal IDs to infer logos or CTAs. v0.1 may score these relations by fixed deterministic formulas; a learned relation head is optional v0.2.

Concrete geometric evidence: for source boxes `i,j`, define `normalized_center_distance = hypot(dx,dy)`, `same_group` as in Appendix B.3.3, `contains` as in Appendix B.3.3, and `aligned_left` when `|x_i-x_j|/SW <= .02`. For an inferred *soft* pairing, propose `association(i,j) = 0.40*same_group + 0.25*contains_any + 0.20*exp(-distance/0.20) + 0.15*aligned_left`; these coefficients are **starting heuristics**, not learned labels or Adobe rules. Exclude background-like containment relations from rigid units; only use scores to create candidate search preferences. Audit this formula on the three provided fixtures before trusting it.

### D.1.2 Candidate families and generation budget

Initial composition-family enum and expected trial count:

| Family | Policy | Trials |
|---|---|---:|
| `SCALE_BASELINE` | Proportional affine scale + translation, then repair | 2 |
| `VERTICAL_STACK` | Order high-level soft units vertically with alignment variants | modes (implemented) |
| `HORIZONTAL_FLOW` | Put units into one or two horizontal bands | modes (implemented) |
| `TWO_COLUMN` | Allocate left/right regions when target ratio permits | 4 |
| `HERO_LEFT` | Largest image unit heroes left; packed reading-ordered column right (D.1.3) | modes × orders × small-variants (implemented) |
| `HERO_RIGHT` | Mirrored hero arrangement | modes × orders × small-variants (implemented) |
| `BANNER_ZONES` | One band of content-sized zones, brand first and CTA pill last (D.1.3); resize-gated above `FAMILY_ASPECT_CUTOFF` | modes × orders × small-variants (implemented) |
| `CONTROLLED_OVERLAY` | Allow text/shape/image intentional layering inferred from source containment and z evidence | 4 |

In every family a text element's `font_size` is proposed together with its box (D.1.4); a family that only moves or resizes boxes is incomplete. Per-family trial counts are *proposal seeds*, not a guarantee of valid or distinct outputs. Applicability checks suppress unhelpful families (e.g., two-column arrangement for very tall narrow targets); the one wide/narrow boundary is `FAMILY_ASPECT_CUTOFF = 2.0`. "Largest image" uses bounding-box area, **not** an inferred semantic product role. Every proposal must pass the same validator before learned ranking. The baseline can be used even with no NLP encoder/model checkpoint.

Defaults: `max_raw_candidates=32`, `max_valid_candidates=12`, deterministic RNG seed `42` for repeatable development, generator time budget initially `1.0s` on the reference CPU (must measure and adapt against submission time limits), and best 3 diverse internal options. Do not promise latency until benchmarked.

### D.1.3 Deterministic template seeding and eligibility

For each family, first classify **background-like candidates for search only**: a shape/image with `area >= 0.75 * source_canvas_area` and source box covering the canvas center, or an image with effective `fit="cover"` and box area at least 75% of canvas. This classification is a heuristic hint and cannot justify removal, changing source z-order, or automatic full-bleed exceptions. All original elements still receive candidate boxes. Existing source containment/overlap and explicit group evidence have precedence over this hint when proposing overlay pairs.

Define candidate **layout units** as connected components of strong soft-pairing edges with association >= 0.75, but exclude background-like nodes from that grouping and cap each inferred unit at 6 nodes. Start with explicit same-group associations (where geometrically appropriate), then consider proximity edges in descending score with stable source-index tie-breaks. Unit membership exists only during candidate search; explicit Adobe `group_id` is never changed. Maintain each unit's source bounding rectangle and member-relative fractional positions. A shape that contains text can remain a separate overlay layer rather than be forced into the same rigid packing box.

For `HORIZONTAL_FLOW`, compute usable width `TW*(1-2m)` and height `TH*(1-m-m*1.5)`, with initial margin fraction `m=0.04` (the bottom margin is 1.5× the sides so the last row never sits on the margin line); order units by source top coordinate then left coordinate (ties by source index), assign consecutive units left-to-right to evenly spaced region slots, and uniformly scale each unit to fit its assigned slot while preserving member-relative geometry. For `VERTICAL_STACK`, swap axes; for `TWO_COLUMN`, place alternating units by descending source unit area in the currently less-occupied column, breaking ties by source order. `HERO_LEFT/RIGHT` (implemented) splits usable width evenly between the largest non-small image unit and a packed reading-ordered column: stacked text blocks merge (gap below half a line-height, shared edge), text rows size to `estimate_text_height` exactly (ink-tight, so box gaps are visual gaps) with inner/outer rhythm gaps and exact vertical centering, pills render at natural size at most, low-opacity overlays ride image units they overlap (capped independents otherwise), smalls anchor opposite the hero (plus top-center on portrait targets), and label/block orders vary. Anchored smalls reserve their corner boxes; packed column and zone rects trim past them instead. `BANNER_ZONES` (implemented) lays one band of zones with widths proportional to source unit widths and packed rows inside: the smallest-area image unit opens the band (brand zone), the pill holding the last source-order text closes it, and content zones keep source order between them. `CONTROLLED_OVERLAY` starts with horizontal/vertical flow and separately reinstates candidate overlaps for source-overlapping pairs; it must retain original stacking semantics. All families run the same subsequent local geometry optimizer. These percentages/order policies are deterministic **seeding defaults** and must be tuned or rejected on evaluation; they are not claimed as inherently attractive designs.

Do not encode reading order as Adobe element-array order. The source-position sort here is only a repeatable computational starting guess; later semantic role predictions may replace it if evaluated. Retain candidate designs in original Adobe array order regardless of slot assignment.

**Overlap eligibility for v0.1's computed penalty:** eligible foreground pairs are two non-background-like, nonzero-area elements whose source-box IoU is <=0.02, neither contains the other in source, and which do not share an explicit non-null group. Ineligible does **not** mean 'should overlap'; it means 'no unconditional overlap penalty'. Source pairs already overlapping are exempt from this generic penalty, but visual diagnostics may still flag extreme target occlusion. These thresholds are heuristic and must not be used as official validity criteria.

### D.1.4 Decision variables and geometric refinement

Each element has candidate variables `(x_i,y_i,w_i,h_i)`. Text elements also
have candidate `font_size_i >= 1` in pixels: font-size adaptation is explicitly
permitted by RULES.md and the Adobe schema. Search may shrink or enlarge font
size together with box geometry; changing a box alone is not typography fitting.
Preserve font family, literal content, font weight/style, alignment, line height,
letter spacing, max_lines, style runs, colors, and optional-field presence.
Rotation, image fit, and opacity also remain fixed under the current edit policy.
Do not infer headline/body/CTA roles from IDs. Implemented: sizes are decided in descending source-size order under a running rank cap, and past the largest text each size is additionally capped at its exact source-size ratio to the largest decided size (the seed honors the ratio cap; the 1 px schema floor wins over the ratio). Pill labels are centered in their shape on both axes with the font fit to the padded shape inset. Non-text units cap their scale at the largest text's decided box-width scale, so buttons keep their source width proportion to the headline (fills, riders, and anchored smalls keep their own regimes).

The optimizer must explicitly search
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
optimization remain proposed.

Two-stage solver:

1. **Discrete layout-family and region assignment** chooses a composition pattern and ordering of internal units. Use enumeration/beam search initially; optionally OR-Tools CP-SAT for discrete assignments and selective mutually exclusive foreground-rectangle packing.
2. **Continuous/local geometry refinement** adjusts positions, scale, gaps, and text font size, measuring constraint violations after each move; a deterministic coordinate-descent/random-restart implementation is sufficient for v0.1. CP-SAT is not required for rotations, arbitrary render-aware text, or all soft metrics.

Never add `NoOverlap2D` over every element. Background and decorative layers often intentionally overlap foreground content; only include confirmed conflicting foreground pairs in optional packing constraints.

### D.1.5 Soft objective, fully defined enough to implement

First satisfy the hard constraints of spec §8.1. Among valid layouts minimize:

\[
J(L)=w_m E_{move}+w_a E_{align}+w_p E_{pair}+w_s E_{scale}+w_o E_{unwanted-overlap}+w_b E_{balance}+w_r E_{read}+w_g E_{group}+w_{sp} E_{space}
\]

The first six terms are defined below. The last three (`E_read`, `E_group`, `E_space`) are defined after them.

**Provisional starting weights (untested; not final; calibrate per D.1.6).** With all nine terms,
`[w_m,w_a,w_p,w_s,w_o,w_b,w_r,w_g,w_sp] = [0.05,0.075,0.125,0.075,0.125,0.05,0.25,0.20,0.05]` (sum 1.0). The six
original weights `[0.10,0.15,0.25,0.15,0.25,0.10]` are each multiplied by 0.5, so their relative proportions are
unchanged; the other half of the weight goes to readability (0.25, the largest single weight because the rendered
PNG is judged on legibility, RULES.md:18 and Brief §2.5, and no other term measures it),
group cohesion (0.20, because the schema describes a group as a unit that travels together) and spacing (0.05,
the weakest and noisiest evidence). If only the original six terms are implemented, use the original six-term
vector. Tune on held-out human preferences. Each term is averaged over its eligible set and bounded to `[0,1]`, so weights are comparable:

- `E_move`: average `min(1, ||center_c_norm - center_s_norm||_2 / sqrt(2))`; **ignore intentional family-wide directional reflow in this penalty** by computing it with low weight, not forcing source positions.
- `E_align`: average `min(1, |anchor_i-anchor_j|/0.1)` for source pairs with aligned corresponding anchors (`<=0.02` in source), excluding confirmed decorative/overlay pairs. Anchors are left, center-x, right, top, center-y, bottom, using appropriate axis normalization.
- `E_pair`: for pairs selected as soft associations, `min(1, max(0,d_c-d_s-0.05)/0.25)` where `d` is normalized center distance. This penalizes large increases in proximity but does not require rigid grouping.
- `E_scale`: mean `min(1, |log((area_c+1)/(area_s+1)) - log((target_area+1)/(source_area+1))|/3)` for scalable boxes. Exclude appropriate full-bleed backgrounds if corroborated by fit/geometry.
- `E_unwanted-overlap`: average candidate IoU for **eligible foreground pairs** that did not overlap in source, capped at 1; exclude allowed intentional overlays, zero-area elements, and background-like containment pairs. Eligibility must be explicitly logged for debugging.
- `E_balance`: Euclidean offset between *area-weighted* element center-of-mass and target center, normalized by half-diagonal, capped at 1; cap each object's contribution to avoid a single full-bleed shape dominating; ignore full-bleed backgrounds.

**Additional terms for the deterministic scorer** (spec §6.2; proposed, not implemented). Each is normalized to
`[0,1]` (0 best), averaged over its eligible set, and 0 when the eligible set is empty. All inputs are stored Adobe
fields; no asset or pixel is inspected. Constants marked *placeholder* are untested and are calibrated in D.1.6.

**Text-fit estimate (the single definition; Appendix B.4.3 and B.7 refer here).** It is implemented once, stdlib-only, in `engine/text_fit.py`; the graph_v4 node feature uses it and the deterministic scorer must import it from there, not re-implement it. For a text element with candidate box
`(w_c,h_c)`, candidate size `f_c`, effective `line_height` `L` (schema default 1.2) and content split on `\n` into
paragraphs: `lines = Σ max(1, ceil(len(par) · k · f_c / w_c))` and `est_height = lines · f_c · L`. `k` is an average
glyph advance in em, *placeholder* 0.5; it ignores letter spacing, bold, word-boundary wrapping and font family.
A text element with `w_c <= 0` or `h_c <= 0` counts as fully overflowing for `E_read` (the graph_v4 feature uses 0 and leaves the degenerate box visible instead, B.4.3).

- `E_read` (text readability *estimate*): eligible = text elements with non-empty content. Per element
  `r_i = max(e_size_i, e_over_i)`, with `e_size_i = clip(1 − f_c / f_ref, 0, 1)` (*placeholder* `f_ref` = 12 px, in
  target canvas pixels) and `e_over_i = clip((est_height − h_c) / h_c, 0, 1)`. Hierarchy `e_hier` is the fraction of
  source text pairs with different sizes whose candidate order is reversed (candidate sizes made equal count 0.5;
  0 with fewer than two distinct source sizes), the "preservation of source size hierarchy" of D.1.4.
  `E_read = 0.8 · mean(r_i) + 0.2 · e_hier` (the 0.8/0.2 split is *placeholder*). It does not penalize overlap or
  text over an image, and it does not estimate contrast, which would need pixels.
- `E_group` (group cohesion; spec §8.1.5): eligible = explicit groups with at least two members. For each group,
  with `o = ((cx − gx)/gw, (cy − gy)/gh)` the member-center offset from the group-box center in units of the group
  box (an axis with zero group size contributes 0), the group penalty is the mean of the defined components:
  (a) offset change, mean over members of `min(1, ||o_c − o_s||₂)`; (b) order flips, the fraction of member pairs
  that are not in a containment relation in the source and whose left/right or above/below center order reversed;
  (c) compactness change, `min(1, |ln r| / 3)` with `r = (A_c/A_s) / ((TW·TH)/(SW·SH))` and `A` the group-box area
  (0 when `A_s = 0`); (d) containment loss, mean over reliable container/inner pairs of
  `1 − area(inner_c ∩ outer_c) / area(inner_c)` (0 when `area(inner_c) = 0`). Until the evidence thresholds of
  spec §8.1.5 are set, (d) covers only pairs that share an explicit group. Components (b) and (d) are defined only
  when such pairs exist. `E_group` is the mean over groups. Moving a group as a scaled unit gives 0 for (a) and (b).
- `E_space` (spacing evenness): eligible = elements that are not background-like (D.1.3). Merge elements into
  blocks, so that intentional overlap and containment create no gap: elements that share an explicit non-null group,
  whose source boxes overlap (IoU above 0.02) or where one contains the other, join one block (union box). With
  fewer than three blocks the term is 0. Order blocks along the target's longer axis by candidate center; with
  `g_k = max(0, start_{k+1} − end_k) / L_axis` the normalized gaps, `E_space = min(1, std(g) / (mean(g) + 0.01))`.

**Overlap.** `E_unwanted-overlap` is the only overlap term and keeps its eligibility rules above, which already
exclude same-group pairs, containment pairs, source-overlapping pairs and background-like elements, so intentional
overlaps are not penalized. `E_read`, `E_group` and `E_space` neither penalize nor reward overlap. `E_pair` and
`E_group` both reward keeping associated elements close (`E_pair` includes the `same_group` term of D.1.1), so they
partly count the same thing; check this with the ablation in D.1.6 before trusting the weights.

These are practical starting definitions, not a validated aesthetic metric. Add a preview-based text-legibility check if a dependable renderer is available; without a renderer, text-box area is an unreliable approximation of glyph fit. Avoid optimizing an unverified "readability" proxy as a hard constraint.

### D.1.6 Calibrating the deterministic objective (procedure; nothing here has been run)

All constants and weights in D.1.5 are provisional starting values. Do not report them as tuned or treat a scorer
built on them as validated until the steps below are done and recorded.

1. **Rendered calibration set.** Using the vendored renderer, render candidates for the three sample designs on several target canvases. Each candidate set varies one concern at a time: a text-size ladder per text element (clearly too large and overflowing, appropriate, clearly too small); a group intact, scattered, and reflowed; even versus uneven spacing; intentional overlaps kept versus separated. Renderer fidelity to the scoring renderer is unconfirmed (spec U9); state this limit in every result.
2. **Calibrate the text-fit constants against rendering.** Compare `est_height` and `lines` with the rendered result for the brand fonts (Solstice Serif and Sans, regular and bold) and fit `k` per family and weight, using the renderer's own layout information if it exposes it, otherwise human judgments of "fits / overflows". Set `f_ref` from human judgments of the smallest size readable at the rendered resolution. Record the error of the estimate.
3. **Human labels only.** Collect side-by-side preferences with the Appendix E tool. Heuristic scores are never labels (spec §1.3 item 7); the scorer is evaluated against people, not against itself.
4. **Evaluate and tune.** Measure pairwise agreement of the scorer with human preferences, per concern (size, group, spacing, overlap) and overall. Run per-term ablations (set one weight to 0), including `E_pair` against `E_group` to detect double counting. Tune weights only on training sources, select on validation sources, and report once on untouched test sources (Appendix E.3, split by source design). Keep the weights summing to 1.0.
5. **Record.** Version the objective (`objective_version`) and store the weights, constants, calibration-set identity, renderer revision and the agreement numbers together; any later change bumps the version. Until step 4 is complete the weights are labelled provisional wherever they appear.

## D.2 Candidate option diversification

Among candidates passing all checked hard constraints:

1. Remove exact coordinate duplicates (`x,y,w,h` rounded to 0.5 px for **dedupe comparison only**, not output storage).
2. Compute layout distance between candidates A,B with matched IDs:

\[
D(A,B)=0.6\cdot\frac{1}{N}\sum_i\min\left(1,\frac{\lVert c_i^A-c_i^B\rVert_2}{0.25}\right)+0.4\cdot\frac{1}{N}\sum_i\min\left(1,\left|\log\frac{a_i^A+\epsilon}{a_i^B+\epsilon}\right|\right)
\]

Here centers are target-normalized, areas are target-area-normalized, and `epsilon=1e-6`. These choices are proposed heuristic defaults.
3. Sort by ranker score (or deterministic score if ranker unavailable), select best first, then greedily maximize `score_normalized + 0.25 * min_distance_to_selected`. Normalize scores by min-max among the current valid candidate set (all equal -> all 0.5).
4. Prefer different composition families as a tie-breaker. Return up to `k=3`; never fabricate three options if fewer distinct valid candidates exist.
5. The contest-facing `resize()` returns exactly the first chosen design in Adobe JSON; a separate `resize_with_options()` (provisional name, spec U4) may expose a private list for the demo or annotation app.

---
