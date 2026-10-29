# S05 Phase 1: original source datasets

Implemented development tooling for the source-design portion of
[CHAMELEON_DATA_GENERATION_GUIDE.md](CHAMELEON_DATA_GENERATION_GUIDE.md), sections 3, 6, 7 and 9.
These are **original sources**, not resized candidates or human preference labels. No S04 production
generator, scoring, training or feature-contract changes are included.

## Generate and preview

From the repository root, with Python 3.10+ (locally exercised on 3.11):

```powershell
# Recommended for a fresh installation: isolate rendering from the ML environment.
python -m venv .venv/sources
.\.venv\sources\Scripts\python.exe -m pip install -r requirements-sources.txt
.\.venv\sources\Scripts\python.exe -m scripts.generate_source_designs generate --count 20 --seed 42 --output data/sources/pilot20_v3
.\.venv\sources\Scripts\python.exe -m http.server 8000 --bind 127.0.0.1 --directory data/sources/pilot20_v3
```

Open `http://127.0.0.1:8000/gallery.html`. The static gallery also opens directly as a local HTML file.
Click each PNG for full resolution. On Linux/macOS use `.venv/sources/bin/python`.
The existing Windows `.venv/Scripts/python.exe` was used for this pilot and already has the dependencies;
substitute it for the fresh-environment executable above to use that installation.

Generation refuses a nonempty output directory. Reproduce into a new directory such as
`data/sources/pilot20_repeat`; do not overwrite an already reviewed dataset. `--count` is the number
of attempted source designs, not a promise of that many accepted outputs. Any validation, rendering
or duplicate rejection makes the generate command exit nonzero, after recording the batch results.
Unexpected template-programming errors propagate; there is no automatic resume of an interrupted run.

The renderer is pinned to cr-renderer commit `a17e1fb9a732959dd6c21ae168d5ee09d74bd64f`.
Dependency installation may use the network; generation and review use only local assets/fonts.
No pretrained encoder, GNN or model download is needed. The transitive Python environment is not a
complete lockfile: reproduce pixel hashes with the recorded renderer, Skia, Pillow and Python versions,
and recorded asset/font/converter hashes. Cross-platform pixel identity is not guaranteed.

## What is generated

`scripts/source_templates.py` contains ten composition families:

| Stable family ID | Category | Composition |
| --- | --- | --- |
| `centered_editorial` | editorial | centered headline, rule, narrow reading measure, footer |
| `split_product` | product advertisement | image panel beside text and a grouped action button |
| `product_pedestal` | product advertisement | central image over an ellipse/platform, 2–4 numbered feature groups |
| `image_panel` | promotion | background image, wordmark and product, opaque contrasting text panel |
| `type_poster` | event flyer | large stacked typography, decorative circle, two-column footer |
| `event_schedule` | event flyer | headline/date block above timed schedule rows |
| `information_cards` | announcement | introductory headline and 2–4 grouped information cards |
| `offer_coupon` | promotion | large offer, supporting copy and a perforated coupon strip |
| `announcement_bands` | announcement | full-width contrasting title band, introduction and statistics |
| `editorial_columns` | editorial | headline/image caption, pull quote and two body columns |

Each source is drawn from a **seeded variation profile** (`scripts/source_variation.json`, validated by
`scripts/source_variation.py`; generator `source_templates_v3`). Orientation cycles through portrait,
square and landscape by family index and variant; within it the short edge (800–1200 px) and aspect
ratio (portrait 0.68–0.9, landscape 1.2–1.65) are sampled. Also sampled per source: a complete palette
(background, ink, accent, panel) from hue/saturation/lightness ranges under one of four schemes
(monochromatic, analogous, complementary, neutral accent) in light or dark mode, rejecting any palette
whose text/accent pairs fall below 4.5:1; margins; heading/body font family, weight and scale; line
height; body alignment; image scale; corner radius; left-right mirroring (numbered sequences and
number/label pairs keep their reading order); repeated-element counts (cards, features, schedule rows,
statistics, perforations, discount); copy variants from `scripts/source_copy.json`; and a monotone
per-axis warp that jitters region boundaries without breaking ordering or containment. The recorded
profile, its hash and each source's sampled parameters are stored in `run.json`/`manifest.jsonl`, so a
source is reproducible from `(family, seed, variant, profile)`. `--profile` accepts a complete
replacement profile; unknown keys, out-of-range values and impossible palettes raise errors.
Remaining constants are deliberate: ten authored layouts, the three bundled assets, the two bundled
font families, the region geometry inside each template (as fractions of the canvas), per-role text
size ratios and group structure. Boxes are constrained regions, not independent random coordinates. Related pieces have explicit groups; layering is intentional
and encoded by `z_index`. Element IDs are opaque `eNNN` values. These templates author sources;
they do not infer semantic roles from supplied Adobe elements or modify the Adobe examples.

Only the three existing assets and exact available Solstice font faces are eligible. Images use
`contain` or `cover`, never stretched pixels or newly generated substitutes. The supplied assets are
**stand-ins**, as recorded in `brand_kit/README.md`; the missing holiday asset remains missing.
Shapes remain Adobe shape elements; any rasterization is performed by the unchanged official preview
integration, not by a new flattened-source generator.

## Validation and rendering gates

`scripts/source_dataset.py` applies the real draft-2020-12 JSON Schema with `jsonschema`, then
`Design.from_adobe` for the existing S02 identity and JSON checks. It does **not** call
`validate_candidate(source, source, target)` or impose target-retention rules on a new source.

Each finding records `code`, `rule_class` (`official` or `internal`) and `message`.
Schema/identity failures are distinguished from this pilot's stricter source-quality policy:
positive in-bounds boxes, no rotations, no unknown generated fields, available allowlisted assets,
exact bundled font faces, nonempty text of at least 12 px, multi-member explicit groups, and bounded
`rgba(...)` colors. The v0.9 schema permits more than this subset. Negative input coordinates,
zero-area elements, missing assets and optional extensions remain valid S02 inputs where allowed;
this source policy does not change that contract or claim these restrictions are Adobe rules.

For supported uniform-style text, `SourceRenderer.fit_text` copies the draft and uses the converter's
actual wrapping function and Skia font lookups/metrics, shrinking size by 1 px steps within each
constrained region until it fits or reaches the internal 12 px floor. It never rewrites/truncates
content. Widths and line extents are recorded, with a 2 px inset. Style runs, nonzero letter spacing
and max-line limits are excluded from these generated sources until their measurement is supported.
Measurement is a local font-fit check, not a proof of legibility, contrast or visual quality.

Every passing source is rendered with `renderer.converter.convert.render_design_to_png`, at full
canvas resolution. Import/render exceptions, warnings, corrupt/missing PNGs and wrong dimensions
are failures; there is no box-render fallback. The source JSON and any partial PNG go to quarantine,
with reasons in the manifest, and later sources continue. Warnings are rejected conservatively even
if the renderer produced an image. Structurally invalid sources are never rendered.

The initial pilot exposed the shape renderer's silent gray fallback for hex fills. The generator now
authors `rgba(...)` colors for text and shapes. Tests check real PNG background pixels as well as
schema validity, so a schema-valid gray substitution cannot silently pass that regression.

## Files and identity

All paths below are relative to the dataset root. `/data/` is ignored by Git; test fixtures live under
`tests/fixtures/`. Custom output directories outside `/data/` need their own ignore policy.

| Artifact | Purpose |
| --- | --- |
| `sources/*.json` | Original Adobe designs, with exactly `canvas` and `elements` |
| `previews/*.png` | Corresponding successful full-resolution renderer outputs |
| `quarantine/*` | Rejected source JSON and any failed/duplicate preview |
| `manifest.jsonl` | One generation record per attempt; preserved across review |
| `run.json` | Seed/count, generator code hash/version, schema hash, asset/font hashes, renderer provenance, dedupe/split policy |
| `summary.json` | Generation and review counts |
| `gallery.html` | Preview gallery, diagnostic details and review controls |
| `review.json` | Last applied explicit review snapshot; not a preference label |
| `approved.jsonl` | Only records explicitly approved after successful validation/rendering |
| `handoff_WIDTHxHEIGHT.jsonl` | Approved sources plus caller-selected S04 target and provenance |

Each manifest row records `source_design_id`, `template_family_id`, `source_family_id`, category,
seed, variant, sampled generation parameters, variation-profile hash, generator version, canvas, element count, asset/font references, validation status/findings,
renderer version/warnings/errors, text measurements, source/preview paths and PNG hash. Status is one
of `rendered`, `validation_failed`, `renderer_failed`, `duplicate`. Rendered means **pending review**,
not aesthetically approved. Internal metadata never appears in Adobe source JSON.

`source_design_id` is `sha256:` plus a hash of sorted-key, compact UTF-8 JSON with non-finite numbers
forbidden. The original element order and decoded numeric values are preserved. It is a local source
dataset convention, not a change to the unresolved production candidate-ID decision U5.

Duplicate detection first checks exact source hashes, then a deliberately conservative layout test:
same ordered element types, assets/fits/shapes/alignments and group-membership pattern; relative canvas
aspect difference at most 2%; every normalized box coordinate/size differs by at most 0.015. This
ignores copy, colors, IDs and font sizes, so recolors/copy changes alone are not new layouts. It is not
a perceptual metric and can over-reject useful sources. Exact decoded RGBA duplicates are also removed
after rendering. Duplicates record `duplicate_of` and remain quarantined. Only successful distinct
sources populate the comparison pool; a failed renderer does not prevent a later successful retry.

All variants retain the stable template family ID. `source_family_id` equals `template_family_id`,
conservatively grouping the entire composition family; changing the generator version must not silently
rename that lineage. Splits remain `unassigned` in Phase 1. Assign **whole families** to partitions
before creating resize candidates or labels. Twenty variants from ten families are not twenty
independent holdout groups. Split allocation and a family-disjoint loader remain future work.

## Human review and exclusion

1. Open the gallery, then each PNG at full resolution. Inspect clipping, wrapping, unwanted overlap,
   contrast, hierarchy, image treatment, small text and related groups.
2. Select `approved`, `rejected` or `pending`. Give a reason for rejection. Download the decisions file.
   Browser storage is a convenience; the downloaded file plus CLI application is the durable record.
3. Apply the file (substitute its actual download path):

```powershell
.\.venv\Scripts\python.exe -m scripts.generate_source_designs review --dataset data/sources/pilot20_v3 --decisions C:/Users/sarah/Downloads/review-decisions.json
```

Reviews are bound to the exact generation manifest hash. Unknown, duplicate or ineligible IDs are
rejected. Approval verifies source/PNG integrity, current source validity and unchanged bundled
assets/fonts. The CLI writes `review.json`, `approved.jsonl`, updated counts and gallery. It never edits
the original source. A review is a complete snapshot: omitted IDs return to pending, so changing an
approval to rejected/pending removes the source from the approved dataset. Previous downloads can be
retained for history; this phase is a single-reviewer workflow, not a multiuser annotation database.

Pending, rejected, invalid, duplicate and renderer-failed sources are **never** exported by the handoff.
Changing a source or PNG after review invalidates export. Use a new generated dataset for corrections.

## Noninvasive S04 handoff

```powershell
.\.venv\Scripts\python.exe -m scripts.generate_source_designs handoff --dataset data/sources/pilot20_v3 --target-width 1600 --target-height 400
```

This writes `handoff_1600x400.jsonl`. Each record has:

```text
source                 exact original {canvas, elements} JSON
target_canvas          {width, height}, explicitly selected by the caller
source_design_id       stable content hash
template_family_id     stable split-group identifier
generation_manifest    {path: "manifest.jsonl", sha256, record: original manifest row}
```

The target is an internal experiment choice, not an asserted official catalogue entry. Target size
does not affect the original source hash or family ID. A future S04 adapter passes only
`Design.from_adobe(task["source"])` and `Canvas.from_adobe(task["target_canvas"])` to its agreed API,
retaining the other fields as dataset provenance. No S04 batch generator is called or implemented here.
With no approved sources, export returns an empty task list rather than bypassing review.

## Verification and pilot observations (2026-10-29)

Commands actually used on Windows/Python 3.11.17:

```powershell
.\.venv\Scripts\python.exe -m scripts.ci_tests sources
.\.venv\ci-core\Scripts\python.exe -m scripts.ci_tests core
.\.venv\ci-core\Scripts\python.exe -m ruff check --no-cache scripts tests/test_source_designs.py
.\.venv\Scripts\python.exe -m scripts.generate_source_designs generate --count 20 --seed 42 --output data/sources/pilot20_v3
```

The source suite covers the actual schema, deterministic draft/fitted sources and PNGs, ten-family
coverage, all three orientations, literal assets/fonts, grouping/containment, duplicate handling,
render failures and quarantine, review integrity/exclusion, unchanged source handoff, overwrite
protection and Python 3.10 syntax. It requires the real renderer: missing dependencies fail, not skip.
The existing CI workflow has a separate source/renderer job so the core jobs remain renderer/ML-free.
Local checks do not establish hosted-CI success or official renderer parity (U9).

The `pilot20_v3` run produced 20 successful JSON/PNG pairs, zero generation rejections and zero
renderer warnings, with 20 pending human reviews and zero approved sources. A contact sheet of all 20
and full-resolution inspection of five designs found no clipped text or unwanted overlaps; sampled
minimum text is 1.7–2.4% of the short edge (about 15–26 px), so small footers and captions deserve
attention at intended display size, and some compositions (e.g. two-feature pedestals) are sparse.
That is not full human acceptance. Only the three placeholder brand assets are represented. A sweep of
300 drafts (10 families × seeds 0–29) found all valid, all text fitting after font fitting, none at the
12 px floor and no text/text overlaps. The earlier diagnostic runs (`pilot20`, `pilot20_v1`,
`pilot20_final`) are preserved separately; they used fixed palettes and sizes.

Superseded datasets: `pilot20_v2` was generated while the palettes were still being converted and
predates the final variation code (its `run.json` profile lacks the palette keys and its generator hash
no longer matches). Use `pilot20_v3` (generator `source_templates_v3`, seed 42, 20/20 rendered, 0
warnings, byte-identical on regeneration with the recorded environment); inspect it visually before
approving anything. v3 corrected mirrored layouts that reversed numbered sequences (03, 02, 01) and
split number/label pairs.

Before 100+ sources: complete human review, decide acceptable small-text thresholds and density,
expand original copy/layout diversity and authorized asset coverage, assess conservative near-dedupe
false positives, choose family-level split allocation, and confirm preview fidelity before collecting
preference labels. Do not count rejected attempts, recolors or family siblings as independent examples.
