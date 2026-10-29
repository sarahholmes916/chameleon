# Appendix F — Stage-one submission requirements and questions for Adobe

> **Appendix F of [TECHNICAL_SPEC.md](../TECHNICAL_SPEC.md).** This appendix records what the *official*
> documents say about submission, the viewer, Docker, offline execution and models, where they disagree, and
> the questions that only Adobe can answer (decisions U1, U8, U9, U11 and the batch behavior of U10). It
> defines no engine contract. Nothing here changes validator or CLI behavior. Constraints are defined only in
> spec §8; deployment facts in spec §7.4.

Date of examination: 2026-10-28. Facts below were read from the files named; nothing was inferred from
Adobe's behavior at scoring time.

## F.1 Official sources examined

| Short name | Location | Notes |
| --- | --- | --- |
| RULES | [RULES.md](../../RULES.md) | In the repository, unmodified. |
| FAQ | [FAQ.md](../../FAQ.md) | In the repository, unmodified. Standalone FAQ of the starter kit. |
| Schema | [chameleon_design_schema_v0.9.json](../../chameleon_design_schema_v0.9.json) | Titled provisional, v0.9. |
| Starter README | `README.md` (before the 2026-10-28 rewrite), `Dockerfile`, `engine/main.py`, `engine/resize.py` docstring | In the repository. |
| Brief | `Final Chameleon Challenge.pdf` in the developer's `Downloads/` folder (outside the repository; U11) | A second copy, `Final Chameleon Challenge (1).pdf`, extracts to byte-identical text. Sections cited as "Brief §n". Section 7.2 reproduces RULES and §7.3 reproduces an FAQ. |

The brief names a public starter repository (`github.com/jhaaj08/chameleon`, Brief §7.1). It was not fetched, so
whether the checked-in official files match it is unverified.

## F.2 What a stage-one submission must contain

| Item | Established by | Statement |
| --- | --- | --- |
| The engine (code) | FAQ.md:105; Brief §5.1; Brief §7.3 Q5 | "Your stage-one submission is your engine" |
| `Dockerfile` | FAQ.md:105; Brief §7.3 Q5 | Required; must build and run as in FAQ Q5 step 5 |
| Short read-me: approach and how to run it | FAQ.md:105; Brief §5.1; Brief §7.3 Q5 | Required |
| Viewer add-on | FAQ.md:105 only (see F.3) | "plus the required (but not scored) viewer add-on" |
| Where and how to submit | FAQ.md:107; Brief §7.3 Q5 | "announced separately ahead of the deadline" — not yet known |
| Deadline | FAQ.md:105; Brief §7.3 Q5 | "before code freeze" — no date in any document |
| Finale-only items | Brief §3.3, §5.1 | Full Express add-on, short pitch, demo video, adaptive copy/imagery |

## F.3 Viewer add-on: determination

**The documents disagree. Treat the viewer as required for stage one until Adobe says otherwise. It is not
scored.**

| Source | What it says |
| --- | --- |
| FAQ.md:105 | The stage-one submission is code, `Dockerfile`, a read-me, "plus the required (but not scored) viewer add-on." |
| FAQ.md:69 | The local preview tool is "separate from the Express add-on you'll build and submit as your viewer". |
| Brief §5.1 | Stage one: "the engine, with a short read-me explaining the approach and how to run it." Finale: "the full Express add-on running on the engine, a short pitch, and a brief demo video." |
| Brief §3.3 | Finalists "turn their engine into a full Adobe Express add-on" at the live finale. |
| Brief §7.3 Q5 | Stage-one submission is "code, Dockerfile, and a short read-me" — no viewer. |

- **Required vs optional vs finale-only:** the repository FAQ says required; the brief says finale-only. Neither calls it optional.
- **Scored?** Only FAQ.md:105 speaks to it: "not scored". The brief's scoring description (§3.1, §7.2: JSON checks plus the rendered PNG against a rubric) mentions no viewer.
- **What "the viewer add-on" is** is defined by no document: its scope, format, packaging, whether it must run inside the engine container, and where it is submitted are all unstated. FAQ.md:69 equates it with "the Express add-on you'll build", which the brief assigns to the finale (§3.3).
- **Why the documents may differ:** the two FAQ texts are not identical (FAQ.md Q3 and Q4 describe a separate `transformer-convertor` tool; Brief §7.3 Q3 says no separate tool is needed; only FAQ.md mentions a viewer). Which is newer is unknown: FAQ.md's last commit is a44a060 (2026-09-24); the brief PDF's file date is 2026-10-19.

The viewer is not implemented in this repository and was not started in this work.

## F.4 Docker, offline execution, models

| Topic | Established | Source |
| --- | --- | --- |
| Build command | `docker build --platform linux/amd64 -t my-chameleon-engine .` | FAQ.md:96; README; Brief §7.1 step 6 |
| Run command | `docker run --rm --network none -v input -v output my-chameleon-engine`; entry point `python -m engine.main` | FAQ.md:97-99; `Dockerfile:34` |
| Offline | Stage one runs offline; `--network none` "matches stage-one scoring exactly"; everything must install at build time; no runtime downloads | Brief §3.1, §4, §7.1; FAQ.md:101; `Dockerfile:19-23` |
| Local models | Allowed ("welcome and appropriate") with weights baked into the image at build time | Brief §3.1, §4, §7.1 step 6; README |
| Model size limit and approved-model list | Exist but are **not published**: "being set with the team", "being finalized with the science team" | Brief §3.1, §4 |
| Fine-tuned or adapted models | Weights must be publicly accessible (for example a public Hugging Face repo) | RULES.md:26; Brief §7.2 |
| Execution time | No hard limit; speed is a rubric row | RULES.md:11 |
| Reference hardware | "Apple M5 Pro", core count, RAM and disk "TBD"; the Brief §3.1 says stage one "runs on the team's own machine, offline" | RULES.md:30-32; Brief §3.1, §7.2 |
| Image content | Never regenerate, redraw or replace asset pixels (disqualifying) | RULES.md:48 |
| Content and font family | Fixed in stage one; font size may change | RULES.md:42-44 |
| Hand-tailoring | Solutions tuned to specific designs or sizes are not allowed | Brief §4 |
| Hidden evaluation | Hidden designs and target sizes from a catalogue of real formats | Brief §3.1; RULES.md:34-36 |

**Verified locally (2026-10-28, one Windows host, Docker Desktop 29.8.2):** `docker build --platform linux/amd64`
completed (about 75 s; the build needs network access for `apt-get` and `pip`, including `pip install` of
`cr-renderer` from GitHub); `docker run --rm --network none` with the three sample inputs mounted wrote an
`_output.json` and a real-rendered `_output.png` for each and exited 0. Docker printed a build warning,
`FromPlatformFlagConstDisallowed`, for Adobe's `Dockerfile` line 1; the file is unmodified.
**Not verified:** the Apple M5 Pro machine, any CPU/RAM/disk limit, whether `requirements.txt`'s unpinned
`cr-renderer` git dependency resolves to the same commit as in scoring, and a clean clone build.

## F.5 Questions for Adobe

Each entry gives: **Established** (official text), **Assumed** (what this project does or assumes now),
**Why it matters**, and the **Question**. Do not change validator or CLI behavior on an unconfirmed answer.

### Q-U1 — Off-canvas, bleed, rotation, stroke, image ratio, mutable fields

- **Established.** "Nothing off-canvas" and "images kept in proportion" are checked structurally (RULES.md:17; Brief §7.2). The `resize.py` docstring adds "Nothing should end up outside `target_canvas`'s bounds" and "matching the source aspect ratio within ~5% unless … full-bleed / `fit: "cover"`". The schema describes `rotation` as degrees clockwise about the center (schema `rotation`) and `stroke_width` as "shape only"; it says nothing about bleed. The Level 2 sample's decorative circle starts partly off-canvas (corner at `(-100,-100)`), and the starter README calls its self-check warning "expected". Brief §2.3 says the output has "new positions, sizes, and so on", and the Brief §1 figure caption lists "positions, stacking, type size, cropping" as what adapts.
- **Assumed.** The validator measures the union of the stored box and its rotated box, expands shapes by half the stroke width, allows no tolerance, rejects any bleed, applies 5 % to every non-`cover` image, and freezes every field except `x`, `y`, `width`, `height` and text `font_size` (K6, K8, K9; spec §8.1.1). Only `content`, `font_family` and `asset_id` are frozen by an official statement.
- **Why it matters.** The validator is stricter than the official text. It may reject layouts Adobe accepts (for example keeping a decorative circle's bleed, or changing `z_index` for stacking), which costs layout quality, and it may accept layouts Adobe rejects if Adobe's rotation or stroke rule is stricter.
- **Questions.** (1) Is "off-canvas" measured on the stored `x/y/width/height` box only, or on the rotated and stroke-expanded extent, and with what tolerance? (2) Is intentional bleed of a decorative element (for example the Level 2 circle) allowed in the output? Does a source element that already bleeds keep that exemption? (3) Is the 5 % image-proportion tolerance official, how is "full-bleed" determined (`fit: cover`, box equal to canvas, or something else), and is `fit: fill` exempt? (4) Besides `x`, `y`, `width`, `height` and text `font_size`, may `rotation`, `opacity`, `z_index`, `text_align`, `line_height`, `letter_spacing` or `fit` change in stage one? (5) Must output element order equal source order (K4)?

### Q-U8 — Model eligibility, size, packaging, text encoder, hardware

- **Established.** A locally hosted model is allowed; a size limit and an approved-model list exist but are unpublished (Brief §3.1, §4). Fine-tuned or adapted weights must be public (RULES.md:26). Weights must be baked into the image at build time (README; Brief §7.1). The scoring machine is "Apple M5 Pro" with limits TBD (RULES.md:30-32) while the image must be `linux/amd64` (FAQ.md:96).
- **Assumed.** The text encoder `sentence-transformers/all-MiniLM-L6-v2`, used frozen and unmodified, is acceptable; an untrained or numeric-only ranker needs no weights; PyTorch and PyG can be installed in the image; image size is not a limit. None of the encoder, `torch` or PyG is in `requirements.txt` or the image today (spec §7.4). The encoder was never loaded here (spec §10.1).
- **Why it matters.** If the approved list or the size limit excludes the encoder, or counts `torch`, PyG and weights, the learned ranker cannot ship as designed. The deterministic path is unaffected, which is why it is the permanent baseline.
- **Questions.** (1) What are the permitted open models and the size limit? Does the limit count model weights only or the whole image, and do pretrained embedding models such as `all-MiniLM-L6-v2` count as "models"? (2) Do weights we train ourselves from scratch (a small GNN ranker, not a fine-tune of an open base model) fall under RULES.md:26, and must they be public? (3) May the image include `torch`, `torch-geometric` and a model file, and is there an image-size or build-time limit? (4) May the build download weights from Hugging Face during `docker build`? (5) What are the CPU, RAM and disk limits of the reference machine, and is scoring run on arm64 or `linux/amd64`?

### Q-U9 — Renderer fidelity

- **Established.** The PNG is "your output JSON run through the official renderer" (RULES.md:18). FAQ.md:47 says the local tool renders "the same way scoring will — via cr-renderer"; Brief §7.3 Q3 says `python -m engine.main` "renders every task through cr-renderer". The starter README calls `renderer/converter/` a "vendored copy of `transformer-convertor`". Schema-to-Crello conversion "is on us" (FAQ.md:39).
- **Assumed.** The vendored converter plus the unpinned `cr-renderer` from GitHub (`requirements.txt`) produces the same image as the scoring pipeline, so human preference labels made on its PNGs transfer. Local text-fit estimates (D.1.5 `E_read`) will be calibrated against it.
- **Why it matters.** Annotation labels (Appendix E), readability estimates and any claim about text fit depend on the preview matching what the AI evaluator sees. An unpinned dependency can change the image between our build and scoring.
- **Question.** Is `renderer/converter/` in the starter repo identical to the converter used in scoring, and which `cr-renderer` commit, fonts and fallback behaviors (missing assets, text wrapping) does scoring use? Please pin the `cr-renderer` revision in the starter `requirements.txt`.

### Q-U11 — Canonical brief and FAQ

- **Established.** RULES.md:5, :26, :44 refer to "the main brief". The only copy available is the PDF in `Downloads/` (two copies, identical text). It embeds RULES (§7.2) and an FAQ (§7.3) that differ from the repository's `FAQ.md` in Q3, Q4 and Q5, and it names the standalone FAQ `CHAMELEON_FAQ.md`, which is not the file name in the repository.
- **Assumed.** The PDF is the current main brief; `FAQ.md` is the current FAQ.
- **Why it matters.** The two disagree on whether the viewer is a stage-one deliverable (F.3), and the brief is the only source for the model-size limit and the finale scope.
- **Question.** Which document is canonical for stage-one deliverables: `FAQ.md` or the brief PDF? What is the current version and date of each, and where is the official copy of the brief published?

### Q-viewer — Viewer add-on

- **Established.** See F.3.
- **Assumed.** Required but not scored; not implemented.
- **Why it matters.** A missing required deliverable could invalidate a submission; building an unneeded one costs time better spent on layout quality.
- **Questions.** (1) Is a viewer add-on part of the stage-one submission, or only of the finale? (2) If required, what must it do (render an output JSON, call the engine), in what format is it submitted, and must it run in the engine container?

### Q-batch — Missing output and process exit status

- **Established.** Scoring runs the container against hidden designs (Brief §3.1); the entry point processes every task in `input/` (`engine/main.py`). No document says how a missing output, a failed task or a nonzero exit is scored (spec §7.5).
- **Assumed.** A task with no valid layout should produce no output rather than an invalid one (spec §1 invariants).
- **Why it matters.** It decides whether the recommended per-task failure handling (spec §7.5) is safe, and whether a nonzero exit status could void a whole run.
- **Questions.** (1) Is a task with no output file scored as zero for that task only? (2) Is the container run once over a folder of tasks, or once per task? (3) Does a nonzero exit status affect scoring?
