"""Regression tests for rejecting bad drafts and returning only validated JSON."""

import copy
import json
from pathlib import Path

import pytest

from engine.candidate_validator import element_bounds, validate_candidate
from engine.internal_schema import Canvas, Design
from engine.layout_search import NoFeasibleLayout, baseline_proposal, find_valid_layout
from engine.resize import resize


def design_json():
    return {"canvas": {"width": 100, "height": 100}, "elements": [
        {"id": "opaque", "type": "image", "asset_id": "missing.png",
         "x": 10, "y": 20, "width": 20, "height": 10, "group_id": None,
         "extension": {"nested": [1, 2]}},
        {"id": "text", "type": "text", "x": 30, "y": 40, "width": 40, "height": 20,
         "content": "Literal content", "font_family": "Test", "font_size": 12,
         "color": "rgba(0,0,0,1)", "style_runs": [{"start": 0, "end": 3}]},
    ]}


@pytest.mark.parametrize("mutation", [
    lambda d: d["elements"].pop(),
    lambda d: d["elements"].append(copy.deepcopy(d["elements"][0])),
    lambda d: d["elements"].reverse(),
    lambda d: d["elements"][0].update(id="invented"),
    lambda d: d["elements"][0].update(asset_id="replacement.png"),
    lambda d: d["elements"][0].pop("group_id"),
    lambda d: d["elements"][0]["extension"]["nested"].append(3),
    lambda d: d["elements"][0]["extension"].update(nested=[True, 2]),
    lambda d: d["elements"][0]["extension"].update(nested=[1.0, 2]),
    lambda d: d["elements"][1].update(content="rewritten"),
    lambda d: d["elements"][1].update(font_family="replacement"),
    lambda d: d["elements"][1].update(style_runs=[]),
    lambda d: d["elements"][0].update(rotation=0),
    lambda d: d["elements"][0].update(x=-0.01),
    lambda d: d["elements"][0].update(x=95),
    lambda d: d["elements"][0].update(width=40),
    lambda d: d["elements"][0].update(width=0),
    lambda d: d["elements"][0].update(x=float("nan")),
    lambda d: d["elements"][0].update(width=float("inf")),
    lambda d: d["elements"][1].update(font_size=0.5),
    lambda d: d["canvas"].update(width=99),
    lambda d: d.update(theme={}),
])
def test_invalid_candidate_cannot_escape_and_next_attempt_gets_diagnostics(mutation):
    raw = design_json()
    source = Design.from_adobe(raw)
    broken = copy.deepcopy(raw)
    mutation(broken)
    calls = []

    def propose(attempt, previous):
        calls.append(attempt)
        if attempt == 0:
            assert previous is None
            return broken
        assert not previous.is_valid
        assert previous.violations
        return raw

    output = find_valid_layout(source, source.canvas, propose)
    assert calls == [0, 1]
    assert output == raw
    output["elements"][0]["extension"]["nested"].append(99)
    assert source.to_adobe() == raw


def test_exhaustion_raises_without_returning_last_invalid_candidate():
    source = Design.from_adobe(design_json())
    invalid = source.to_adobe()
    invalid["elements"][0]["x"] = -1
    attempts = []

    def propose(attempt, previous):
        attempts.append(attempt)
        return invalid

    with pytest.raises(NoFeasibleLayout) as error:
        find_valid_layout(source, source.canvas, propose, max_attempts=3)
    assert attempts == [0, 1, 2]
    assert len(error.value.reports) == 3
    assert all(not report.is_valid for report in error.value.reports)


def test_valid_candidate_on_last_attempt_is_returned():
    source = Design.from_adobe(design_json())
    assert find_valid_layout(source, source.canvas,
                             lambda i, previous: {} if i < 7 else source) == source.to_adobe()


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_bad_retry_budget_rejected(budget):
    source = Design.from_adobe(design_json())
    with pytest.raises(ValueError, match="max_attempts"):
        find_valid_layout(source, source.canvas, lambda *_: source, max_attempts=budget)


def test_unexpected_generator_error_is_not_silently_hidden():
    source = Design.from_adobe(design_json())

    def broken(*args):
        raise RuntimeError("generator failed")

    with pytest.raises(RuntimeError, match="generator failed"):
        find_valid_layout(source, source.canvas, broken)


def test_font_resize_is_allowed_and_cover_ratio_exception_uses_source_fit():
    raw = design_json()
    raw["elements"][0]["fit"] = "cover"
    source = Design.from_adobe(raw)
    candidate = copy.deepcopy(raw)
    candidate["elements"][0]["width"] = 40
    candidate["elements"][1]["font_size"] = 8
    assert validate_candidate(source, candidate, source.canvas).is_valid
    raw["elements"][0].pop("fit")
    source = Design.from_adobe(raw)
    assert not validate_candidate(source, candidate, source.canvas).is_valid


@pytest.mark.parametrize("width, valid", [(21, True), (21.01, False), (19, True), (18.99, False)])
def test_relative_image_ratio_tolerance(width, valid):
    source = Design.from_adobe(design_json())
    candidate = source.to_adobe()
    candidate["elements"][0]["width"] = width
    assert validate_candidate(source, candidate, source.canvas).is_valid is valid


def test_rotated_box_and_stroke_overflow_rejected_even_when_raw_box_fits():
    raw = {"canvas": {"width": 100, "height": 100}, "elements": [
        {"id": "shape", "type": "shape", "x": 0, "y": 40, "width": 100, "height": 20,
         "shape_type": "rectangle", "fill_color": "#fff", "rotation": 45,
         "stroke_color": "#000", "stroke_width": 40},
    ]}
    source = Design.from_adobe(raw)
    assert not validate_candidate(source, raw, source.canvas).is_valid
    output = resize(raw, raw["canvas"])
    assert validate_candidate(source, output, source.canvas).is_valid


def test_impossible_fixed_stroke_fails_instead_of_emitting_invalid_json():
    raw = {"canvas": {"width": 100, "height": 100}, "elements": [
        {"id": "line", "type": "shape", "x": 0, "y": 0, "width": 10, "height": 0,
         "shape_type": "line", "fill_color": "#fff", "stroke_color": "#000", "stroke_width": 50},
    ]}
    with pytest.raises(NoFeasibleLayout):
        resize(raw, {"width": 10, "height": 10})


@pytest.mark.parametrize("filename", sorted(Path("input").glob("*.json")), ids=lambda p: p.stem)
def test_real_samples_repair_invalid_baseline_and_preserve_source(filename):
    task = json.loads(filename.read_text(encoding="utf-8"))
    before = copy.deepcopy(task)
    source = Design.from_adobe(task["source"])
    target = Canvas.from_adobe(task["target_canvas"])
    assert not validate_candidate(source, baseline_proposal(source, target, 0), target).is_valid
    output = resize(task["source"], task["target_canvas"])
    assert validate_candidate(source, output, target).is_valid
    assert output == resize(task["source"], task["target_canvas"])
    assert task == before
    assert [e["id"] for e in output["elements"]] == [e.id for e in source.elements]
    for original, element in zip(source.elements, Design.from_adobe(output).elements):
        left, top, right, bottom = element_bounds(element)
        assert 0 <= left <= right <= target.width
        assert 0 <= top <= bottom <= target.height
        if original.type == "image":
            assert element.asset_id == original.asset_id
            if original.effective_value("fit") != "cover":
                assert element.width / element.height == pytest.approx(original.width / original.height)


def _raise_proposal(*args, **kwargs):
    raise RuntimeError("synthetic proposal failure")


def test_resize_boundary_propagates_search_failure(monkeypatch):
    import engine.resize as resize_module
    monkeypatch.setattr(resize_module, "baseline_proposal", lambda *_: {})
    monkeypatch.setattr(resize_module, "baseline_candidate", _raise_proposal)
    monkeypatch.setattr(resize_module, "horizontal_flow_candidates", _raise_proposal)
    monkeypatch.setattr(resize_module, "vertical_stack_candidates", _raise_proposal)
    raw = design_json()
    with pytest.raises(NoFeasibleLayout):
        resize(raw, raw["canvas"])


def test_cli_does_not_write_or_render_a_failed_task(monkeypatch, tmp_path):
    import engine.main as main_module
    import engine.resize as resize_module
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    raw = design_json()
    (input_dir / "task.json").write_text(json.dumps({"source": raw, "target_canvas": raw["canvas"]}))
    monkeypatch.setattr(main_module, "INPUT_DIR", input_dir)
    monkeypatch.setattr(main_module, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(resize_module, "baseline_proposal", lambda *_: {})
    monkeypatch.setattr(resize_module, "baseline_candidate", _raise_proposal)
    monkeypatch.setattr(resize_module, "horizontal_flow_candidates", _raise_proposal)
    monkeypatch.setattr(resize_module, "vertical_stack_candidates", _raise_proposal)
    rendered = []
    monkeypatch.setattr(main_module, "render_to_png", lambda *args: rendered.append(args))
    with pytest.raises(NoFeasibleLayout):
        main_module.main()
    assert list(output_dir.iterdir()) == []
    assert rendered == []


# --- Stage-1 property preservation (docs/TECHNICAL_SPEC.md K6 and 8.1.6) ---
# Only x, y, width, height and text font_size may change. These tests run the production validator.

def protected_design_json():
    # Every element sits well inside the canvas, so a rejected mutation can only fail the policy check.
    return {"canvas": {"width": 400, "height": 400}, "elements": [
        {"id": "headline", "type": "text", "x": 40, "y": 40, "width": 200, "height": 50,
         "content": "Hello", "font_family": "Solstice Sans", "font_size": 20, "color": "#112233"},
        {"id": "badge", "type": "shape", "x": 40, "y": 120, "width": 100, "height": 50,
         "shape_type": "rectangle", "fill_color": "#445566", "stroke_color": "#778899",
         "stroke_width": 2, "group_id": "g"},
        {"id": "photo", "type": "image", "x": 200, "y": 120, "width": 100, "height": 100,
         "asset_id": "a.png"},
    ]}


PROTECTED_CHANGES = [
    ("text color", 0, lambda e: e.update(color="#ffffff")),
    ("shape fill color", 1, lambda e: e.update(fill_color="#000000")),
    ("stroke color", 1, lambda e: e.update(stroke_color="#000000")),
    ("stroke width", 1, lambda e: e.update(stroke_width=5)),
    ("stroke removed", 1, lambda e: e.pop("stroke_color")),
    ("stroke set to null", 1, lambda e: e.update(stroke_color=None)),
    ("opacity", 0, lambda e: e.update(opacity=0.5)),
    ("rotation", 0, lambda e: e.update(rotation=3)),
    ("z_index", 0, lambda e: e.update(z_index=9)),
    ("font_size on a shape", 1, lambda e: e.update(font_size=9)),
    ("font_size on an image", 2, lambda e: e.update(font_size=9)),
]


@pytest.mark.parametrize("index,mutation", [c[1:] for c in PROTECTED_CHANGES],
                         ids=[c[0] for c in PROTECTED_CHANGES])
def test_protected_property_changes_fail_only_the_policy_check(index, mutation):
    raw = protected_design_json()
    source = Design.from_adobe(raw)
    candidate = copy.deepcopy(raw)
    mutation(candidate["elements"][index])
    report = validate_candidate(source, candidate, source.canvas)
    element_id = raw["elements"][index]["id"]
    assert not report.is_valid
    assert report.schema_ok and report.geometry_ok and not report.policy_ok
    assert report.violations == (f"{element_id}: immutable fields or optional-field presence changed",)


def test_unchanged_candidate_is_valid_control():
    raw = protected_design_json()
    report = validate_candidate(Design.from_adobe(raw), copy.deepcopy(raw), Canvas.from_adobe(raw["canvas"]))
    assert report.is_valid and report.violations == ()


@pytest.mark.parametrize("index,changes", [
    (0, dict(x=60, y=70)),
    (0, dict(width=150, height=60)),
    (0, dict(font_size=14)),
    (0, dict(x=50, y=50, width=180, height=40, font_size=12)),
    (1, dict(x=150, y=200, width=80, height=40)),
    (2, dict(x=20, y=250)),
    (2, dict(width=50, height=50)),
], ids=["move text", "resize text box", "text font size", "text box and size", "move and resize shape",
        "move image", "resize image keeping ratio"])
def test_permitted_changes_still_pass_validation(index, changes):
    raw = protected_design_json()
    source = Design.from_adobe(raw)
    candidate = copy.deepcopy(raw)
    candidate["elements"][index].update(changes)
    report = validate_candidate(source, candidate, source.canvas)
    assert report.is_valid and report.violations == ()
