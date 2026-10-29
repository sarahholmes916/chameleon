# Chameleon — design-resizing engine

Adobe North America Hackathon 2026, Chameleon challenge. Given a design (an Adobe v0.9 JSON document of
text, image and shape elements) and a target canvas, the engine returns the **same elements** — same `id`s,
same count — with new positions, sizes and (for text) font sizes so the design fits the new canvas.

This is a **stage-one engine**: it takes `input/*.json` and writes, for each task, a resized design
(`<task>_output.json`) and a rendered preview (`<task>_output.png`) into `output/`. Both are evaluated; see
[RULES.md](RULES.md).

## Where this stands

The pieces that exist today produce a **valid but basic** result. What `resize()` does now is take the source
design, scale and center the whole composition so everything fits, and check the result with a strict
validator before returning it. That keeps every element, never leaves the canvas, and never returns a layout
that fails the checks. It does **not** rearrange elements. On extreme canvases (for example 1080×1350 →
728×90) it gives a small, centered copy of the original with tiny text, which is correct JSON but not a good
design.

The candidate generator, the quality scorer, the diversified options, the annotation tool and the model
training described in the specification are **planned, not built**. Nothing in this README should be read as
saying they work.

| Component | State |
| --- | --- |
| Lossless parser/serializer for the Adobe JSON (`engine/internal_schema.py`) | Implemented, tested |
| Output validator and bounded retry/repair loop (`engine/candidate_validator.py`, `engine/layout_search.py`) | Implemented, tested |
| `resize()` — validated proportional fit (`engine/resize.py`) | Implemented (basic) |
| Graph features and an untrained GNN ranker (`engine/graph_features.py`, `engine/text_embeddings.py`, `engine/ranker.py`) | Implemented, tested; **no trained model exists; `resize()` does not use it** |
| Preview renderer (`renderer/`, Adobe's, vendored) | Provided; whether it matches the scoring renderer is unconfirmed |
| Candidate generator, layout templates, deterministic quality scorer, diversification | Planned |
| Human annotation tool, dataset, ranker training | Planned |
| Stage-one viewer add-on | Not built; whether it is required is unresolved ([Appendix F](docs/reference/F_submission_and_adobe_questions.md)) |

"Implemented, tested" means the code exists and a test module covers it; see [Tests](#tests) for how to run
them.

## Approach

Layout generation and validation are **computational and deterministic first**; a learned ranker is an
optional second stage that may only reorder layouts that already passed validation, and the deterministic
path is the permanent fallback. The whole design is in
[docs/TECHNICAL_SPEC.md](docs/TECHNICAL_SPEC.md):

```text
source JSON + target canvas
  -> lossless internal Design            (engine/internal_schema.py)
  -> proposal -> validator -> repair     (engine/layout_search.py, engine/candidate_validator.py)
  -> one validated design -> {canvas, elements} JSON       (engine/resize.py)
```

The validator is the only check that decides whether an output is acceptable: element retention, the
requested canvas, which fields may change, canvas bounds and image proportions. The authoritative list is the
constraint table in [spec §8.1.1](docs/TECHNICAL_SPEC.md#811-constraint-table), which labels each rule Official
(stated by Adobe) or Internal (this project's stricter reading). If no valid layout is found, `resize()` raises
`NoFeasibleLayout` instead of returning a draft.

## Installation

Requires Python 3.10 or newer, `git` and network access for the install (the renderer, `cr-renderer`, is
installed from its GitHub repository).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`requirements.txt` holds only what the engine needs to run and render (`Pillow`, `cr-renderer`). The parser,
validator and `resize()` use the Python standard library. PyTorch and PyTorch Geometric are **not** needed to
run the engine; they are only for the optional ranker code and its tests (see `requirements-graph.txt` and
[docs/CI.md](docs/CI.md)).

## Running the engine

```bash
python -m engine.main
```

**Verified:** on 2026-10-28, in a fresh virtual environment (Python 3.11, Windows) after `pip install -r requirements.txt`,
`python -m engine.main` processed all three samples, wrote an `_output.json` and a rendered `_output.png` for each,
and exited 0. Not verified: Python 3.10, macOS or Linux outside Docker.

This processes every `*.json` task in `input/` (three samples are included: warm-up, core, advanced), each a
`{"source": {...}, "target_canvas": {...}}` object, and writes `output/<task>_output.json` and
`output/<task>_output.png`. It then prints a quick self-check per task. That self-check is a local sanity
check, **not** the official scorer, and it is less strict than the engine's own validator.

If no valid layout can be found for a task, `resize()` raises `NoFeasibleLayout`, nothing is written for that
task, and **the program stops, so later tasks in the folder are not processed**. This is current behavior, not a
design goal; a per-task failure policy is recommended in [spec §7.5](docs/TECHNICAL_SPEC.md#75-batch-failure-policy-u10--analysis-and-recommendation)
and has not been approved or implemented.

## Tests

The test commands, the two CI environments and the execution guard are documented in
[docs/CI.md](docs/CI.md); use them rather than a bare `pytest`, because ordinary `pytest` silently skips the
graph and ranker tests when PyTorch is not installed.

```bash
pip install -r requirements-dev.txt
python -m scripts.ci_tests core        # parser, validator, retry loop, CI guard; no PyTorch needed
python -m scripts.ci_tests ml          # graph features and ranker; needs the CPU PyTorch + PyG environment in docs/CI.md
```

Test counts are deliberately not recorded in the documentation; run the commands. A local run is not a
hosted-CI result.

## Docker (how scoring runs the engine)

```bash
docker build --platform linux/amd64 -t my-chameleon-engine .
docker run --rm --network none \
  -v "$(pwd)/input:/app/input" \
  -v "$(pwd)/output:/app/output" \
  my-chameleon-engine
```

PowerShell: replace `$(pwd)` with `${PWD}` and the line-continuation `\` with a backtick.

- **Building needs network access** (`apt-get`, `pip`, and `cr-renderer` from GitHub). **Running does not:**
  `--network none` is exactly how stage-one scoring runs, so everything the engine needs must be installed at
  build time. Never add a step that downloads anything at run time.
- To check the image works offline, run the second command above after building; it must write one
  `_output.json` and one `_output.png` per input task.
- The image currently contains no PyTorch, PyG, text encoder or checkpoint, so the engine works offline as shipped.
  Shipping any model is an open decision ([spec §7.4](docs/TECHNICAL_SPEC.md#74-deployment-constraints),
  [Appendix F](docs/reference/F_submission_and_adobe_questions.md)).

**Verified:** on 2026-10-28, on one Windows host with Docker Desktop, the build completed and the offline run
wrote JSON and PNG for all three samples and exited 0. **Not verified:** the Apple M5 Pro reference machine,
resource limits, and a build from a fresh clone.

## Submission

What the official documents say a stage-one submission contains, where they disagree (notably whether a
"viewer add-on" is required), and the questions still open with Adobe are collected in
[Appendix F](docs/reference/F_submission_and_adobe_questions.md). In short: the engine, the `Dockerfile` and this
README are required; `FAQ.md` additionally lists a required-but-unscored viewer add-on that the challenge brief
assigns to the finale. That viewer is not built.

Rules that apply to everything here ([RULES.md](RULES.md)): never regenerate or redraw any image content
(disqualifying); in stage one text content and font family are fixed and only font size may change; the
engine must run offline.

## Known limitations and unresolved decisions

- Output quality is limited to proportional fitting (see above); extreme canvases give tiny, centered results.
- Text fit and legibility are not measured; the preview renderer's fidelity to the scoring renderer is unconfirmed.
- No trained ranker, no human preference labels, no measured improvement of any kind.
- The validator is stricter than Adobe's published text in places (bleed, rotation, stroke, which fields may
  change); the exact official rules are an open question for Adobe.
- Decisions U1–U13 (what is settled, recommended or waiting on Adobe) are in
  [spec §10.2](docs/TECHNICAL_SPEC.md#102-decision-register-u1u13).

## Documentation

| Document | Purpose |
| --- | --- |
| [docs/TECHNICAL_SPEC.md](docs/TECHNICAL_SPEC.md) | The authoritative specification: architecture, interfaces, constraints, status; Appendices A–F in [docs/reference/](docs/reference/) |
| [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) | Ownership, milestones and progress |
| [docs/CI.md](docs/CI.md) | Test and CI procedure |
| [AGENTS.md](AGENTS.md) | Workflow rules for coding agents |
| [RULES.md](RULES.md), [FAQ.md](FAQ.md), [chameleon_design_schema_v0.9.json](chameleon_design_schema_v0.9.json) | Official Adobe documents (not edited by us) |

## Repository layout

```text
engine/        resize.py (entry point), main.py (CLI), internal_schema.py, candidate_validator.py,
               layout_search.py, graph_features.py, text_embeddings.py, ranker.py
renderer/      Adobe's vendored preview renderer (transformer-convertor + cr-renderer)
brand_kit/     assets/ and fonts/ referenced by the sample designs
input/         task JSON files (three samples included)
output/        results (generated; not committed)
tests/         test modules (see docs/CI.md for how CI runs them)
scripts/       ci_tests.py, the CI execution guard
docs/          specification, plan, CI guide, appendices
Dockerfile, requirements*.txt, ruff.toml, pytest.ini, chameleon_design_schema_v0.9.json
```
