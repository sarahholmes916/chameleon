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

For `HORIZONTAL_FLOW`, compute usable width `TW*(1-2m)` and height `TH*(1-m-m*1.5)`, with initial margin fraction `m=0.04` (the bottom margin is 1.5× the sides so the last row never sits on the margin line); order units by source top coordinate then left coordinate (ties by source index), assign consecutive units left-to-right to evenly spaced region slots, and uniformly scale each unit to fit its assigned slot while preserving member-relative geometry. For `VERTICAL_STACK`, swap axes; for `TWO_COLUMN`, place alternating units by descending source unit area in the currently less-occupied column, breaking ties by source order. `HERO_LEFT/RIGHT` (implemented) splits usable width evenly between the largest non-small image unit and a packed reading-ordered column: stacked text blocks merge (gap below half a line-height, shared edge), text rows size to `estimate_text_height` exactly (ink-tight, so box gaps are visual gaps) with exact inner/outer rhythm gaps (never clamped) and exact vertical centering within edge floors (bottom margin plus an inner em at each end; rows shrink, never gaps or marks), pills cap at the headline proportion, low-opacity overlays ride image units they overlap (capped independents otherwise), smalls anchor opposite the hero at equal mark insets on both axes, pinning the visible (rotated/stroke) union corner rather than the stored box corner (plus top-center on portrait targets), and label/block orders vary. Anchored smalls reserve their corner boxes; packed column and zone rects trim past them instead. `BANNER_ZONES` (implemented) lays one band of zones with widths proportional to source unit widths and packed rows inside: the smallest-area image unit opens the band (brand zone), the pill holding the last source-order text closes it, and content zones keep source order between them. `CONTROLLED_OVERLAY` starts with horizontal/vertical flow and separately reinstates candidate overlaps for source-overlapping pairs; it must retain original stacking semantics. All families run the same subsequent local geometry optimizer. These percentages/order policies are deterministic **seeding defaults** and must be tuned or rejected on evaluation; they are not claimed as inherently attractive designs.

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
Do not infer headline/body/CTA roles from IDs. Implemented: sizes are decided in descending source-size order under a running rank cap, and past the largest text each size is additionally capped at its exact source-size ratio to the largest decided size (the seed honors the ratio cap; the 1 px schema floor wins over the ratio). Pill labels are centered in their shape on both axes with the font fit to the padded shape inset. Text-chrome units (pills, shapes) cap at the smaller of the largest text's decided box-width and height scales, so buttons keep both source proportions to the headline; image units cap at the width scale only (fit units span their slots by regime; fills, riders, and anchored smalls keep their own regimes).

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
J(L)=w_m E_{move}+w_a E_{align}+w_p E_{pair}+w_s E_{scale}+w_o E_{unwanted-overlap}+w_b E_{balance}+w_r E_{read}+w_g E_{group}+w_{sp} E_{space}+w_c E_{cover}
\]

The first six terms are defined below. The last four (`E_read`, `E_group`, `E_space`, `E_cover`) are defined after them.

**Provisional starting weights (untested; not final; calibrate per D.1.6).** With all ten terms,
`[w_m,w_a,w_p,w_s,w_o,w_b,w_r,w_g,w_sp,w_c] = [0.045,0.0675,0.1125,0.0675,0.1125,0.045,0.225,0.18,0.045,0.10]` (sum 1.0). The six
original weights `[0.10,0.15,0.25,0.15,0.25,0.10]` were each multiplied by 0.5 when the second wave joined, and the nine v1 weights are now each multiplied by 0.9, so relative proportions are
unchanged; coverage takes the freed 0.10 while readability stays first (0.225, because the rendered
PNG is judged on legibility, RULES.md:18 and Brief §2.5, and no other term measures it),
group cohesion (0.18, because the schema describes a group as a unit that travels together) and spacing (0.045,
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
`E_cover` (canvas coverage): eligible = elements that are not background-like (D.1.3), read from the source. With `U` the union area of candidate boxes over the target area `A`, `E_cover = 1 − U/A`, clipped to `[0,1]`. Area-weighting means tiny elements barely register and the union never double-counts overlaps, so neither corner confetti nor stacked duplicates can game it; an empty eligible set scores 1.

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

### D.1.7 Shape stretching (S04 generator v0.3, `shape_stretch_v1`)

Layout units place every member with one uniform scale, so on extreme targets buttons, badges and bars
shrink to slivers. `engine/shape_stretch.py` is a post-placement pass over an already placed candidate:
eligible **shapes** may scale independently on each axis (`scale_x = new_w/old_w`, `scale_y = new_h/old_h`).
Only K6 fields change (`x`, `y`, `width`, `height`, and a button label's `font_size`), every edit goes through
`layout_edit`, and `validate_candidate` stays the authority. The base candidate is never replaced; the
proportional layout remains in the pool as the alternative.

**Roles** (`engine/shape_roles.py`, source geometry and `shape_type` only; IDs, `_note` text and underscore
fields are never read; ambiguity means FIXED):

| Role | Evidence | Treatment |
| --- | --- | --- |
| BACKGROUND | background-like rectangle (D.1.3) | fills its per-axis canvas-fraction region |
| PANEL | rectangle holding other elements, or >=15% of the canvas and not thin | fills its region while still holding its content (source padding scaled with the content) |
| BUTTON | rectangle <=25% of the canvas holding exactly one text label at least 25% of its height | resized around the label |
| BAR_H / BAR_V | rectangle aspect >=4 or <=1/4 (or a line), not holding anything | lengthens along its long axis only; thickness untouched |
| PROPORTION_SENSITIVE | any ellipse (circles included); empty near-square rectangles (aspect 0.8-1.25) | never stretched |
| FIXED | rotated, ambiguous, or other | never stretched |

**Button (label_fit).** Single-line labels only. Padding is measured from the source in ems of the label font
(clipped to 0.4-2.5 em horizontal, 0.15-1.2 em vertical); the new shape is natural label width plus that padding,
the label is centered in it, and the label font is tried from the largest allowed rung down to the base font
(growth at most x1.5, never above a text that was larger in the source, never above the source size). Widening is
tried before padding shrink, before shifting the button left or right. A pill (corner radius >= half the source
height) never grows taller than twice its radius, and the aspect ratio may change at most x3 from the source.

**Bars (axis_x / axis_y).** A bar follows the element it runs parallel to (overlap >=60% of the longer extent, within
15% of the canvas across); otherwise a bar spanning >=50% of the source canvas along its length maps to canvas
fractions; any other bar is left alone. It grows along a ladder (100/75/50/25% of the way) until it is not blocked.

**Rejection, not repair.** A move is rejected if the box leaves the canvas (stroke included), newly overlaps an
unrelated element that the old box did not touch (panels are blocked only by elements stacking beneath them), or
would shrink a bar. Stretches under half a pixel are not stretches.

**Variants** (each emitted only if it differs from the base and from earlier variants): `label_fit`, `panel_fill`,
`axis_x`, `axis_y`, and `balanced` (everything above, panels and bars log-interpolated at half strength; backgrounds
always fill). The pool derives them from the best `max_stretch_bases` (3) distinct valid bases, interleaved, up to
`max_stretch_candidates` (12), inside `max_raw_candidates`: a quarter of the raw budget (capped by the stretch cap)
is reserved for them. Each variant is validated, text-fitted and scored like any other candidate. Provenance
(`variant`, `base_candidate_id`, `shape_stretch_version`) lives on `PoolCandidate`, never in the Adobe JSON.
Retention treats a candidate as distinct from another if its mean layout distance exceeds the threshold **or** its
largest single-shape displacement exceeds 0.1, so a variant that only reshapes a few shapes is not discarded as a
near-duplicate of its base. Production selection is the global valid score winner, as before.

**Objective v4 (introduced with shape stretching; see D.1.8 for v5)**: no new term and no weight change. `E_scale` adds an aspect-distortion
penalty for ellipses (2:1 distortion scores 1; stretched buttons, panels and bars are not penalized for aspect).
`E_group` containment losses add label padding asymmetry beyond the source's (gain 4). `E_cover`, `E_balance` and the
`E_space` block boxes ignore **backdrop panels** (panel holding content at least 1.5x its area): without this a
panel-fill variant won on coverage and spacing alone by overlapping its neighbors, which was observed before the
change. All new constants are provisional and uncalibrated (D.1.6).

### D.1.8 Post-stretch local refinement (`layout_refinement_v1`, objective v5)

`engine/layout_refinement.py` improves an already valid candidate by bounded local search over the existing
deterministic scorer and `validate_candidate`. It replaces neither, adds nothing to the K6 allowlist, and always
returns a valid layout (the input when nothing improves).

**Algorithm.** Deterministic best-improvement local search with a coarse-to-fine step schedule
(`step_fractions` of each target axis, default 8/4/2/1%). Per step, proposals are generated in three priority tiers
and scored tier by tier; the best proposal beating the current score by more than `min_improvement` (1e-4) is accepted
and the search restarts from tier 1. A step ends when no tier improves. No randomness; ties break by tier, operator order,
proposal order; each geometric state is evaluated at most once (which also prevents oscillation).
Per proposal: apply via `layout_edit` -> `validate_candidate` -> text refit (`fit_candidate_text`, renderer measurement
if injected) -> revalidate -> invariants -> finite score.

| Tier | Detected problem | Operators |
| --- | --- | --- |
| 1 defects | text below `min_readable_font` (default 8px, a quality preference, not an Adobe rule) and smaller than its source size; text that overflows; unrelated units closer than `min_gap_fraction` (3% of the short side) or overlapping; a panel/button not holding its content | `grow_text`, `expand_text`, `separate`, `fit_container` |
| 2 polish | alignment drift for pairs aligned in the source; uneven gaps along the flow axis; off-center button label; container padding; bar length | `snap`, `equalize_gaps`, `recenter_label`, `fit_container`, `bar_extent` |
| 3 opportunistic | blank space around a unit; loose text boxes; text that can grow; images that can rescale; content off-center | `fill_space`, `tighten_text`, `grow_text`, `scale_image`, `center_content` |

**Readability first.** `grow_text` tries a font ladder (x1.1, 1.25, 1.5, 2.0; never above the source size or a text that
was larger in the source) with a larger box sized from measured natural width (single line) or area (multi-line), three
anchors (keep, center, bottom), a button's background resized with its label, and neighbors pushed away (`_resolve`, up
to three cascading passes). Font shrink is never a hard rejection: shrink relative to the starting layout is free up to `font_shrink_onset` (20%), then a
penalty `font_shrink_weight (0.2) * sum(((shrink - onset) / (1 - onset))^2)` is subtracted from the score the search compares
(reported scores stay unpenalized), so deeper shrinking costs progressively more but remains possible when it wins.

**Protection.** Units (explicit group; button shape and label; source-overlapping or contained foreground elements)
move rigidly, so group offsets and intentional overlaps survive. Backgrounds and backdrop panels never translate and are
only resized. Rejected, never repaired: leaving the canvas; breaking a containment that held at the start; enlarging an
unintentional overlap (intentional = same group, containment, IoU above 0.02, background-like, read from the source);
failing validation before or after text fitting; non-finite scores. Circles and ellipses are only translated; images keep
exact aspect when rescaled.

**Budgets** (`PoolConfig`, all separate; each is documented where it is enforced):
- `max_raw_candidates` (64) is the hard total of records in the pool.
- Generation uses `max_raw_candidates` minus the two reserves below. Families still build their finite lists eagerly, so
  this caps assessed proposals, not internal family computation.
- Shape stretching: `max_stretch_bases` (3) distinct valid bases, at most `max_stretch_candidates` (12) variants, reserving
  min(that, raw/4) records.
- Refinement: `max_refine_bases` (2) best valid layouts that show a diagnosable problem (distinct families first),
  one refined record per base, at most `max_refined_candidates` (4), reserving min(that, raw/8) records. The work cap is
  `RefinementConfig.max_evaluations` (120 validated-and-scored proposals per base) and `max_rounds` (8 accepted
  improvements per base); proposal generation itself is cheap geometry, capped at `max_proposals_per_tier` (24) per tier.
- 0 disables stretching or refinement. Retention (`max_retained_candidates`) never changes the production winner.

**Provenance** on `PoolCandidate`, never in the Adobe JSON: `parent_candidate_id`, `refinement_version`,
`refinement_operators`, `score_before_refinement`; the stretch `variant` and `base_candidate_id` are inherited from the
parent; `generator_version` gains `+layout_refinement_v1`; `CandidatePool.refinements` logs every attempt (proposals
evaluated, rejected invalid/invariant, duplicates, errors, accepted rounds, stop reason, scores). The candidate ID is the
hash of the exact refined JSON. Scores are diagnostics, not preference labels.

**Objective v5 (`objective_v5_readability`, superseded by v6 below)**, no new term or weight. `E_read` was checked: its linear 12px penalty
averaged over all texts, so one 1.9px text among eight moved the term by ~0.1 and a 1.9 to 2.3px gain moved the score by
~0.006. The size part is now 0.7 x mean + 0.3 x worst text (`READ_WORST_TEXT_SHARE`). `E_group` now also scores ungrouped
containment (a shape that held an element in the source should still hold it), which the refiner needs to value repairing
a panel. Both constants are provisional, like every weight here.

**Objective v6 (`objective_v6_relations_blank`)**, still no new term or weight (both changes live inside existing terms).
(a) `E_group` gains a *source-relation* component, weighted equally with the existing group components
(`RELATION_GROUP_SHARE`): pairs the SOURCE relates but does not explicitly group are rewarded for staying related. A
layered pair (IoU above the overlap threshold, or containment between non-shape elements) keeps its overlap fraction; a
pair that is aligned on an edge or center line and near in the source (`RELATION_NEAR_DISTANCE` 0.35) keeps both its
alignment and its closeness. This is a soft preference, never a validity or refinement requirement, and nothing is
inferred from IDs or text. (b) `E_cover` adds blank space beyond the source's own: with the source's covered fraction
measured as area and as the share of each axis crossed by some element, the candidate's shortfall is normalized and
`E_cover = raw + (1 - raw) * COVER_EXCESS_SHARE (0.5) * mean(shortfall)`, so a layout no emptier than the source is
unchanged. `fill_space` (tier 3) is the matching refinement operator: it scales a whole movable unit up about its center
(ratios 1.15, 1.35, 1.6, 2.0, or the largest that fits the canvas), fonts included but never above the source size,
keeping the unit's internal layering and alignment; `_resolve` pushes neighbors aside. All constants are provisional.

**Text-fit calibration** (`s04_word_fit_v3`): the production word-fit used 1.3 em per line; the real renderer measures
1.36 to 1.41 em, which caused 1 to 2px height misses (76px needed in a 75px box). It is now 1.4 em with an 8% advance safety (v3; the Linux CI renderer measured small bold text up to ~6.6% wider than Windows)
factor, shared by stretching and refinement through `SINGLE_LINE_EM`. `tests/test_s04_rendering.py` checks the estimate
against actual rendered text and that pool winners fit under the renderer (except below 2px, where the renderer draws 1px text and no box can
hold text). The production path stays renderer-free; measurement is an injected development/benchmark option.

**Known limits.** Not implemented: restricted swaps or region shifts; moving a text block as a whole to make room beyond
pushing its neighbors; two-label button refit; resizing explicit group members relative to each other. The objective
rewards some trade-offs a designer might refuse (see the evaluation report: alignment worsens while readability improves).

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


## D.3 Implemented S04 v0.2 revisions (2026-10-30)

These revisions supersede the older proposed defaults above where they differ.

- **Ungrouped clustering:** explicit grouping semantics and existing association weights are unchanged.
  Ungrouped text pairs use a separate 0.30 threshold with alignment <=0.02, normalized center distance
  <=0.06, no source overlap/containment, and vertical gap <=1.2 times the smaller font size.
  Inferred components remain capped at six. Shapes, images and overlay pairs are not rigidly clustered
  without group evidence by this path. These are internal heuristics, not Adobe rules.
- **Spacing:** banner zones bound total gap expenditure to 20% per axis, wrapping dense sets using
  aspect-aware columns. Packed containers reduce nominal gaps to at most 20% of available height;
  optical edge floors yield when they leave no room. No-area containers reject proposals. Positive
  source boxes cannot silently collapse and pass the existing validator.
- **New families:** TWO_COLUMN alternates units across two columns with adaptive row gaps.
  CONTROLLED_OVERLAY merges source-overlapping/contained foreground units (up to six members)
  before flow placement; it preserves their relative geometry. Backgrounds remain separate.
  Neither family changes Adobe group IDs or invents semantic roles.
- **Typography:** `layout_typography.py` fits whole words and explicit line breaks, accounts approximately
  for weight/letter spacing, preserves content and protected fields, respects max_lines, and shrinks
  only font_size down to schema minimum 1. It preserves candidate font order while fitting.
  Width constants are provisional, not calibrated glyph metrics. The shared graph_v4 estimator is
  untouched. An optional renderer measurement callback refines before validation/scoring; its version
  provenance is distinct. Font fitting is not proof of readable font size. Floor cases can still overflow.
- **Objective v3:** weights and the other nine terms are unchanged from v2. For eligible newly overlapping
  pairs, overlap penalty is max(IoU, intersection/text_area for each text member). This catches complete
  occlusion of small text by large images that IoU hid. Source overlays keep the existing exemptions.
  The tiny-font/hierarchy terms remain provisional. Coverage still risks rewarding oversized elements;
  no sample-specific winner forcing or aesthetic improvement claim is made.
- **Budgets:** PoolConfig defaults to 64 assessed raw proposals and 12 retained valid alternatives.
  Families are consumed round-robin. The limits are configurable resource choices, not competition
  limits or minimum success counts. Existing family functions eagerly construct finite lists, so the
  assessed-proposal cap does not bound all internal family computation. Generation errors do not count
  as design records; malformed yielded proposals do count. No wall-clock cutoff affects determinism.
- **Identity/diversity:** remove exact duplicates by canonical full Adobe JSON hash, preserving font
  differences. Near-duplicate distance is 0.9 times mean(0.55 center + 0.30 size + 0.15 font) plus 0.1
  mean relative group displacement. Center and group terms divide target-normalized displacement by
  0.25 and clip at 1; size is the clipped mean absolute log width/height ratio with +1 pixel stabilization;
  font is clipped absolute log size ratio (zero for non-text). All elements, including images, contribute.
  Threshold 0.025 is configurable. Center/size changes also capture alignment changes indirectly;
  this is a geometry proxy, not a perceptual metric. Greedy score-ordered near dedupe keeps the best
  representative. Retention favors the least represented family, then highest score and stable order.
  A sole family may fill remaining capacity; no valid alternatives are fabricated.
- **Production/training:** production picks the global valid score winner, independently of retention.
  Training consumers see all records/reports and the diverse retained subset. Rendering exports only
  valid retained JSON, version provenance and PNGs through the existing candidate manifest convention.
  Scores live in benchmark reports, never human preference labels or the annotation UI.

Renderer evaluation and limitations are recorded in [S04_EVALUATION.md](../S04_EVALUATION.md).
