"""Exercise CI's fail-closed guard in real, isolated pytest subprocesses."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import ci_tests


pytest_plugins = ["pytester"]
ROOT = Path(__file__).resolve().parents[1]


def guarded_run(pytester, second, *, allowed=None, extra_args=()):
    pytester.makeconftest(
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from scripts.ci_tests import ExecutionGuard\n"
        "def pytest_configure(config):\n"
        "    config.pluginmanager.register(ExecutionGuard(\n"
        f"        ('test_first.py', 'test_second.py'), {allowed!r}))\n"
    )
    pytester.makepyfile(test_first="def test_first():\n    assert True\n", test_second=second)
    return pytester.runpytest_subprocess("-q", "test_first.py", "test_second.py", *extra_args)


def test_required_modules_execute_without_a_fixed_test_count(pytester):
    result = guarded_run(pytester, "import pytest\n@pytest.mark.parametrize('n', range(4))\ndef test_case(n):\n    assert n >= 0\n")
    assert result.ret == 0
    result.assert_outcomes(passed=5)
    result.stdout.fnmatch_lines(["*test_first.py: 1 passing calls*", "*test_second.py: 4 passing calls*"])


@pytest.mark.parametrize("body", [
    "import pytest\npytest.importorskip('chameleon_missing_ci_dependency')\n",
    "import pytest\npytest.skip('module unavailable', allow_module_level=True)\n",
    "# An empty module must never satisfy the CI gate.\n",
])
def test_missing_or_skipped_module_fails_even_when_another_suite_passes(pytester, body):
    result = guarded_run(pytester, body)
    assert result.ret != 0
    result.stdout.fnmatch_lines(["*required module collected no tests: test_second.py*"])


@pytest.mark.parametrize("body", [
    "@pytest.mark.skip(reason='unexpected')\ndef test_skip():\n    pass\n",
    "def test_skip():\n    pytest.skip('runtime skip')\n",
    "@pytest.mark.xfail(reason='known issue')\ndef test_skip():\n    assert False\n",
    "@pytest.mark.xfail(reason='known issue')\ndef test_skip():\n    assert True\n",
])
def test_unexpected_skip_xfail_and_xpass_fail(pytester, body):
    result = guarded_run(pytester, "import pytest\ndef test_pass():\n    assert True\n" + body)
    assert result.ret != 0
    result.stdout.fnmatch_lines(["*CI execution guard FAILED*"])


def test_only_the_exact_cuda_skip_and_reason_are_allowed(pytester):
    body = "import pytest\ndef test_pass():\n    assert True\n@pytest.mark.skip(reason='CUDA unavailable')\ndef test_gpu():\n    pass\n"
    result = guarded_run(pytester, body, allowed={"test_second.py::test_gpu": "CUDA unavailable"})
    assert result.ret == 0
    result.assert_outcomes(passed=2, skipped=1)


def test_cuda_allowlist_does_not_allow_different_skip_reasons(pytester):
    body = "import pytest\ndef test_pass():\n    assert True\n@pytest.mark.skip(reason='broken dependency')\ndef test_gpu():\n    pass\n"
    result = guarded_run(pytester, body, allowed={"test_second.py::test_gpu": "CUDA unavailable"})
    assert result.ret != 0


def test_even_allowed_skips_cannot_replace_all_execution_in_a_module(pytester):
    body = "import pytest\n@pytest.mark.skip(reason='CUDA unavailable')\ndef test_gpu():\n    pass\n"
    result = guarded_run(pytester, body, allowed={"test_second.py::test_gpu": "CUDA unavailable"})
    assert result.ret != 0
    result.stdout.fnmatch_lines(["*required module executed no passing tests: test_second.py*"])


@pytest.mark.parametrize("extra_args", [("--collect-only",), ("-k", "not omitted")])
def test_collection_only_and_partial_selection_cannot_pass(pytester, extra_args):
    body = "def test_pass():\n    assert True\ndef test_omitted():\n    assert True\n"
    result = guarded_run(pytester, body, extra_args=extra_args)
    assert result.ret != 0


@pytest.mark.parametrize("body", [
    "raise RuntimeError('collection failed')\n",
    "def test_failure():\n    assert False\n",
    "import pytest\n@pytest.fixture\ndef fixture():\n    yield\n    raise RuntimeError('teardown failed')\ndef test_teardown(fixture):\n    assert True\n",
])
def test_collection_test_and_teardown_errors_preserve_failure(pytester, body):
    result = guarded_run(pytester, body)
    assert result.ret != 0


def test_core_environment_rejects_installed_torch(monkeypatch):
    monkeypatch.setattr(ci_tests.importlib.util, "find_spec", lambda _name: object())
    with pytest.raises(RuntimeError, match="core CI must run without torch"):
        ci_tests.check_environment("core")


def test_core_environment_imports_only_core_modules(monkeypatch):
    imported = []
    monkeypatch.setattr(ci_tests.importlib.util, "find_spec", lambda _name: None)
    monkeypatch.setattr(ci_tests.importlib, "import_module", imported.append)
    monkeypatch.setattr(ci_tests.sys, "modules", {})
    ci_tests.check_environment("core")
    assert imported == ["engine.internal_schema", "engine.resize"]


def test_core_environment_detects_ml_import_leak(monkeypatch):
    modules = {}
    monkeypatch.setattr(ci_tests.importlib.util, "find_spec", lambda _name: None)
    monkeypatch.setattr(ci_tests.importlib, "import_module", lambda _name: modules.update(torch=object()))
    monkeypatch.setattr(ci_tests.sys, "modules", modules)
    with pytest.raises(RuntimeError, match="loaded an ML dependency"):
        ci_tests.check_environment("core")


@pytest.mark.parametrize("cuda,hip", [("12.8", None), (None, "7.2")])
def test_ml_environment_rejects_gpu_wheels(monkeypatch, cuda, hip):
    fake = SimpleNamespace(__version__="test", version=SimpleNamespace(cuda=cuda, hip=hip))
    monkeypatch.setattr(ci_tests.importlib, "import_module", lambda _name: fake)
    with pytest.raises(RuntimeError, match="CPU PyTorch wheel"):
        ci_tests.check_environment("ml")


def test_ml_environment_imports_each_required_module_and_prints_versions(monkeypatch, capsys):
    fake = SimpleNamespace(__version__="test", version=SimpleNamespace(cuda=None, hip=None))
    imported = []

    def load(name):
        imported.append(name)
        return fake

    monkeypatch.setattr(ci_tests.importlib, "import_module", load)
    ci_tests.check_environment("ml")
    assert imported == ["torch", "torch_geometric", "engine.graph_features", "engine.text_embeddings", "engine.ranker"]
    assert "Torch: test; PyG: test" in capsys.readouterr().out


def test_missing_required_package_stops_ci_before_pytest(monkeypatch):
    def missing(_suite):
        raise ImportError("missing required package")

    monkeypatch.setattr(ci_tests, "check_environment", missing)
    monkeypatch.setattr(ci_tests.pytest, "main", lambda *_a, **_k: pytest.fail("pytest must not run"))
    assert ci_tests.main(["ml"]) == 1


def test_ci_helpers_use_python_310_syntax():
    for path in (ROOT / "scripts" / "ci_tests.py", Path(__file__)):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 10))
