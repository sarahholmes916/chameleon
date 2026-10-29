"""CI-only pytest execution guard; ordinary local pytest keeps optional skips."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import platform
import sys

import pytest


SUITES = {
    "core": ("tests/test_internal_schema.py", "tests/test_ci.py", "tests/test_layout_search.py",
             "tests/test_layout_edits.py", "tests/test_layout_units.py",
             "tests/test_layout_objective.py", "tests/test_candidate_generator.py"),
    "ml": ("tests/test_graph_features.py", "tests/test_ranker.py"),
    "sources": ("tests/test_source_designs.py",),
}
CPU_ALLOWED_SKIPS = {
    "tests/test_ranker.py::test_cuda_forward_backward_and_cpu_eval_agree": "CUDA unavailable",
}


class ExecutionGuard:
    """Require real passes from every module and account for every selected test.

    There is no fixed test count. Collection skips, deselection, unexpected
    skips/xfails/XPASS, and tests without a completed call cannot produce green CI.
    """

    def __init__(self, required_modules, allowed_skips=None):
        self.required_modules = set(required_modules)
        self.allowed_skips = dict(allowed_skips or {})
        self.selected: set[str] = set()
        self.passed: set[str] = set()
        self.skipped: set[str] = set()
        self.problems: list[str] = []

    @staticmethod
    def module(nodeid: str) -> str:
        return nodeid.split("::", 1)[0].replace("\\", "/")

    def pytest_collectreport(self, report):
        if report.failed or report.skipped:
            self.problems.append(f"collection {report.outcome}: {report.nodeid}")

    def pytest_collection_finish(self, session):
        self.selected = {item.nodeid for item in session.items}

    def pytest_deselected(self, items):
        self.problems.append(f"{len(items)} tests deselected; CI requires complete suites")

    def pytest_runtest_logreport(self, report):
        if getattr(report, "wasxfail", None) is not None:
            self.problems.append(f"unexpected xfail/XPASS: {report.nodeid}")
        elif report.skipped:
            reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else str(report.longrepr)
            expected = self.allowed_skips.get(report.nodeid)
            if report.when == "setup" and expected is not None and reason == f"Skipped: {expected}":
                self.skipped.add(report.nodeid)
            else:
                self.problems.append(f"unexpected skip: {report.nodeid}: {reason}")
        elif report.failed:
            self.problems.append(f"{report.when} failed: {report.nodeid}")
        elif report.when == "call" and report.passed:
            self.passed.add(report.nodeid)

    def pytest_sessionfinish(self, session, exitstatus):
        collected_modules = {self.module(nodeid) for nodeid in self.selected}
        passed_modules = {self.module(nodeid) for nodeid in self.passed}
        for module in sorted(self.required_modules):
            if module not in collected_modules:
                self.problems.append(f"required module collected no tests: {module}")
            elif module not in passed_modules:
                self.problems.append(f"required module executed no passing tests: {module}")
        incomplete = self.selected - self.passed - self.skipped
        if incomplete:
            self.problems.append(f"{len(incomplete)} collected tests did not complete a passing call")
        if self.problems and exitstatus == pytest.ExitCode.OK:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            if self.problems:
                reporter.write_sep("=", "CI execution guard FAILED", red=True)
                for problem in self.problems:
                    reporter.write_line(problem)
            else:
                reporter.write_sep("=", "CI execution guard: required suites executed")
                for module in sorted(self.required_modules):
                    count = sum(self.module(nodeid) == module for nodeid in self.passed)
                    reporter.write_line(f"{module}: {count} passing calls")
                reporter.write_line(f"Explicitly allowed CPU-only skips: {len(self.skipped)}")


def check_environment(suite: str) -> None:
    print(f"Python: {platform.python_version()}", flush=True)
    if suite == "core":
        for name in ("torch", "torch_geometric"):
            if importlib.util.find_spec(name) is not None:
                raise RuntimeError(f"core CI must run without {name}; use a clean core environment")
        importlib.import_module("engine.internal_schema")
        importlib.import_module("engine.resize")
        if any(name.split(".", 1)[0] in ("torch", "torch_geometric") for name in sys.modules):
            raise RuntimeError("core imports loaded an ML dependency")
        print("Core imports succeed without Torch/PyG; output validation and geometric repair are tested.", flush=True)
    elif suite == "ml":
        torch = importlib.import_module("torch")
        pyg = importlib.import_module("torch_geometric")
        print(f"Torch: {torch.__version__}; PyG: {pyg.__version__}; CUDA build: {torch.version.cuda}", flush=True)
        if torch.version.cuda is not None or torch.version.hip is not None:
            raise RuntimeError("ML CI requires a CPU PyTorch wheel, not a CUDA/ROCm build")
        for name in ("engine.graph_features", "engine.text_embeddings", "engine.ranker"):
            importlib.import_module(name)
    else:
        for name in ("jsonschema", "PIL", "cr_renderer", "scripts.source_dataset"):
            importlib.import_module(name)
        print("Source generation uses local brand assets/fonts and the actual renderer.", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=SUITES)
    args = parser.parse_args(argv)
    try:
        check_environment(args.suite)
    except (ImportError, RuntimeError) as exc:
        print(f"CI dependency check failed: {exc}", file=sys.stderr)
        return 1
    modules = SUITES[args.suite]
    allowed = CPU_ALLOWED_SKIPS if args.suite == "ml" else {}
    return int(pytest.main(["-q", "-ra", *modules], plugins=[ExecutionGuard(modules, allowed)]))


if __name__ == "__main__":
    raise SystemExit(main())
