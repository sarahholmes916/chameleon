# Continuous integration

The single workflow is `.github/workflows/ci.yml`. It replaces the misplaced
root-level `workflows/ci.yml`, which GitHub Actions would not discover. It runs
on pull requests targeting `main`, pushes to `main`, and manual dispatches.
Repository permissions are read-only, checkout does not persist credentials,
and newer runs cancel outdated runs for the same PR/ref.

## Jobs and dependencies

| Stable check name | Environment | Required suites |
| --- | --- | --- |
| `Core / Python 3.10` | Ubuntu, Python 3.10, pytest and Ruff | S02, output validation/repair, and CI-guard tests |
| `Core / Python 3.11` | Ubuntu, Python 3.11, pytest and Ruff | S02, output validation/repair, and CI-guard tests |
| `Graph and ML / CPU` | Ubuntu, Python 3.11, CPU Torch 2.11.0, PyG 2.8.1 | Graph/embedding and ranker tests |

Jobs run independently. Core jobs have a 10-minute timeout; ML has 20 minutes.
There are no S04/S05 placeholder jobs, model downloads, or real training runs.

Core installs only `requirements-dev.txt`; S02 and the baseline resize module
use the standard library. It deliberately does not install `requirements.txt`,
which contains rendering dependencies, or `requirements-graph.txt`. The guard
fails if Torch/PyG are installed and imports S02 and `engine.resize` to check the
boundary. Core also tests the output gate and proportional repair; the full S04
template generator and scorer remain deferred.

ML first installs `torch==2.11.0` from the official CPU wheel index, then installs
`requirements-dev.txt` and `requirements-graph.txt`. The installed CPU build
satisfies the graph file's `torch>=2.6,<3` constraint; the second install does not
request an upgrade. PyG remains pinned by the project to `2.8.1`. The preflight
imports Torch/PyG and all three S03 modules, prints Python/Torch/PyG versions,
and rejects CUDA/ROCm builds. Both jobs run `pip check`.

This follows the official [PyTorch 2.11 CPU installation command](https://pytorch.org/get-started/previous-versions/#v2110)
and [PyG minimal installation guidance](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).
`torchvision`, `torchaudio`, `torch-scatter`, and pretrained encoder packages
are not needed by these tests. ML sets Hugging Face/Transformers offline mode
and hides CUDA devices; fixtures supply embedding providers and model weights.

The workflow uses the official [checkout v7](https://github.com/actions/checkout/tree/v7)
and [setup-python v7](https://github.com/actions/setup-python/tree/v7) actions.
Dependency installation always runs; no workflow cache is required or configured.

## Required execution and skips

`python -m scripts.ci_tests core` and `python -m scripts.ci_tests ml` run pytest
with a CI-only `ExecutionGuard`. Ordinary `python -m pytest` keeps the existing
optional `importorskip()` behavior for core-only local installations.

The guard requires collected tests and at least one real passing test call in
each expected module. It also accounts for every collected test and rejects
deselection, collection-only runs, collection errors/skips, unexpected runtime
skips, xfail/XPASS, and setup/call/teardown failures. No permanent test count is
hardcoded. Its expected modules live in `scripts/ci_tests.py::SUITES`.

Only this exact test may skip in CPU ML CI, during setup with the exact reason
`CUDA unavailable`:

```text
tests/test_ranker.py::test_cuda_forward_backward_and_cpu_eval_agree
```

The CPU forward/backward, inference, and batching tests still execute. A skipped
graph/ranker module cannot satisfy the guard even when another module passes.
`tests/test_ci.py` tests these false-green cases in isolated pytest subprocesses.
External pytest plugin auto-loading is disabled in the workflow for isolation;
the guard tests explicitly load pytest's bundled `pytester` plugin.

## Local commands

Run from the repository root. Use separate environments without
`--system-site-packages` so the core job cannot inherit an installed ML runtime.
On Windows, the ignored environments created during verification can be used:

```powershell
python -m venv .venv/ci-core
.\.venv\ci-core\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\ci-core\Scripts\python.exe -m scripts.ci_tests core

python -m venv .venv/ci-ml
.\.venv\ci-ml\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\ci-ml\Scripts\python.exe -m pip install -r requirements-dev.txt -r requirements-graph.txt
.\.venv\ci-ml\Scripts\python.exe -m scripts.ci_tests ml
```

On Linux/macOS, environment executables are under `bin/python` instead of
`Scripts/python.exe`. The CI ML build is intended for Ubuntu/Windows CPU wheels;
the core tests also run without any ML installation. For normal local work in
an existing CUDA environment, run `python -m pytest -q
tests/test_graph_features.py tests/test_ranker.py`; the strict CI ML command
deliberately requires a CPU build.

Ruff uses `ruff.toml` and selects `E4,E7,E9,F` for basic correctness, undefined
names, and import checks. Only E402 is exempted in the two optional ML test
modules because imports intentionally follow `pytest.importorskip()`.

```powershell
.\.venv\ci-core\Scripts\python.exe -m ruff check --no-cache engine/internal_schema.py engine/resize.py engine/candidate_validator.py engine/layout_search.py tests/test_internal_schema.py tests/test_ci.py tests/test_layout_search.py scripts
.\.venv\ci-core\Scripts\python.exe -m ruff check --no-cache engine/graph_features.py engine/text_embeddings.py engine/ranker.py tests/test_graph_features.py tests/test_ranker.py
```

Lint is scoped to the implemented core/S03 code and CI helpers. Renderer code,
formatting enforcement, and speculative future modules are outside this initial
lint scope. No existing application or model code was reformatted.

## Diagnosing failures and extending coverage

- A dependency/preflight failure means the environment cannot test the intended
  component. Fix installation or the dependency boundary; do not add Torch to core.
- `CI execution guard FAILED` identifies missing execution, filtering, or an
  unexpected skip. Fix the cause instead of allowing an entire suite to skip.
- Assertion/collection/lint failures retain nonzero exit codes and diagnostics.
  Missing-checkpoint tests use temporary fixtures; do not download a real model
  to make them pass.
- Add future S04 generator, validator, deterministic scorer, and lightweight
  integration test modules to `SUITES['core']` and the core lint paths. Their
  implementation must remain usable without ML. Add only their actual required
  dependencies, separately from graph requirements.
- Add S05 preference-record/split/leakage tests to core when they need no ML;
  add pairwise-loss and short training-mechanics tests to ML. Keep quality
  benchmarking and expensive training outside ordinary pull-request CI.
- New skips require explicit review of the exact test and reason. Update CPU
  pins deliberately and rerun the graph/ranker suites when changing versions.

## Verification boundary

Local verification uses isolated Windows environments on Python 3.11.17.
Observed results with pytest 9.1.1 and Ruff 0.17.3:

| Command/check | Observed result |
| --- | --- |
| Core environment: `python -m scripts.ci_tests core` | 131 passed: 107 S02 + 24 CI-guard tests; no skips; Torch/PyG absent |
| CPU ML environment: `python -m scripts.ci_tests ml` | 165 passed: 80 graph + 85 ranker; exactly one allowed CUDA skip |
| ML preflight invoked in the core environment | Exit 1 for missing Torch, as required |
| Both documented Ruff scopes | Passed |
| `python -m pip check` in both environments | No broken requirements |
| YAML parsing, Bash syntax, workflow paths/triggers/matrix | Passed local checks |

The CPU installation was actually tested with Torch `2.11.0+cpu` and PyG `2.8.1`;
it did not inherit the pre-existing CUDA environment. These counts describe
this verification run and are not hardcoded acceptance conditions.

Python 3.10 and Ubuntu execution, action startup, event triggering, and the
branch-protection check names still require an actual GitHub Actions run.
No hosted run, commit, push, pull request, or branch-protection change is made
by this CI setup task. Local results do not establish that a GitHub-hosted
check has passed.

After adding the output gate, local Python 3.11.17 verification ran the strict
core suite with **172 passed** (including 41 output validation/repair tests) and
the strict CPU ML suite with **165 passed, one allowed CUDA skip**. Core Ruff
and `git diff --check` also passed. These are local results; a new hosted CI run
is still required.
