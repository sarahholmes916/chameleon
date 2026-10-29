"""Generator-side safeguards: the allowlisted edit helper and property tests for generated candidates.

K6 (docs/TECHNICAL_SPEC.md section 8.1.1) is the only production definition of what a candidate may
change. The literal sets below are a deliberately independent copy used only by tests, so that any
drift of the production allowlist, or any generator that edits a protected field, fails here.

Implemented candidate families: the proportional baseline (`baseline_proposal`: attempt 0 is the
stretch, attempts 1+ fit the composition uniformly). Planned families (VERTICAL_STACK, HORIZONTAL_FLOW,
TWO_COLUMN, HERO_LEFT, HERO_RIGHT, CONTROLLED_OVERLAY; Appendix D.1.2) have no code, so their property
tests are deferred: register each one in FAMILIES below when it is implemented. No placeholder family
or skipped test stands in for them.
"""

import copy
import json
from pathlib import Path

import pytest

from engine import candidate_validator, layout_edits
from engine.candidate_validator import mutable_fields, validate_candidate
from engine.internal_schema import Canvas, Design
from engine.layout_edits import ProtectedFieldError, layout_edit
from engine.layout_search import NoFeasibleLayout, baseline_proposal, find_valid_layout
from engine.resize import resize


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = sorted((ROOT / "input").glob("*.json"))
SCHEMA_FIELDS = set(json.loads(
    (ROOT / "chameleon_design_schema_v0.9.json").read_text(encoding="utf-8"))["$defs"]["element"]["properties"])
EXTRA_TARGETS = ({"width": 1080, "height": 1080}, {"width": 300, "height": 250})

# Independent test-side statement of K6: x, y, width, height on every element, font_size on text.
LAYOUT_FIELDS = {"x", "y", "width", "height"}
TEXT_ONLY_FIELDS = {"font_size"}


def permitted(element_type):
    return LAYOUT_FIELDS | (TEXT_ONLY_FIELDS if element_type == "text" else set())


def strict(value):
    """Type-exact JSON text: distinguishes 1 from 1.0 and True from 1, at any depth."""
    return json.dumps(value, sort_keys=True)


def assert_only_permitted_changes(source_raw, candidate_raw):
    """Independent recursive comparison; deliberately not the production validator."""
    source_elements, candidate_elements = source_raw["elements"], candidate_raw["elements"]
    assert len(candidate_elements) == len(source_elements)
    assert [e["id"] for e in candidate_elements] == [e["id"] for e in source_elements]
    for source, candidate in zip(source_elements, candidate_elements):
        assert set(candidate) == set(source), f"{source['id']}: optional-field presence changed"
        allowed = permitted(source["type"])
        changed = {key for key in source if strict(source[key]) != strict(candidate[key])}
        assert changed <= allowed, f"{source['id']}: protected fields changed: {sorted(changed - allowed)}"
        if "font_size" in changed:
            assert source["type"] == "text"
        assert strict({k: v for k, v in source.items() if k not in allowed}) == strict(
            {k: v for k, v in candidate.items() if k not in allowed})


def sample_elements():
    return {
        "text": {"id": "t", "type": "text", "x": 1, "y": 2, "width": 30, "height": 10, "content": "Hi",
                 "font_family": "F", "font_size": 12, "color": "#fff", "group_id": None,
                 "style_runs": [{"start": 0, "end": 1}], "extension": {"nested": [1, 2, {"k": True}]}},
        "shape": {"id": "s", "type": "shape", "x": 1, "y": 2, "width": 30, "height": 10,
                  "shape_type": "rectangle", "fill_color": "#123", "stroke_color": "#456",
                  "stroke_width": 2, "opacity": 0.5, "rotation": 10, "z_index": 3},
        "image": {"id": "i", "type": "image", "x": 1, "y": 2, "width": 30, "height": 10,
                  "asset_id": "a.png", "fit": "cover", "extension": [None, 1.0]},
    }


# --- the allowlisted helper ---------------------------------------------------------------------

def test_layout_edit_applies_permitted_changes_and_copies_everything_else():
    source = sample_elements()["text"]
    before = copy.deepcopy(source)
    changes = {"x": 5.5, "y": 6, "width": 31, "height": 11, "font_size": 14}
    edited = layout_edit(source, changes)
    expected = copy.deepcopy(before)
    expected.update(changes)
    assert strict(edited) == strict(expected)            # type-exact, nested data included
    assert list(edited) == list(before)                   # key order is kept
    assert source == before and strict(source) == strict(before)  # the source element is untouched
    assert edited["extension"] is not source["extension"]
    assert edited["style_runs"] is not source["style_runs"]
    edited["extension"]["nested"].append("marker")
    edited["style_runs"].append({"start": 9, "end": 9})
    assert strict(source) == strict(before)               # no shared nested structure


def test_layout_edit_with_no_changes_returns_an_equal_independent_copy():
    source = sample_elements()["image"]
    edited = layout_edit(source, {})
    assert edited == source and edited is not source and edited["extension"] is not source["extension"]


@pytest.mark.parametrize("element_type", ["shape", "image"])
def test_font_size_is_rejected_on_non_text_elements(element_type):
    source = sample_elements()[element_type]
    before = copy.deepcopy(source)
    with pytest.raises(ProtectedFieldError, match="font_size"):
        layout_edit(source, {"font_size": 9})
    assert source == before


@pytest.mark.parametrize("field", [
    "id", "type", "content", "font_family", "color", "fill_color", "stroke_color", "stroke_width",
    "opacity", "rotation", "z_index", "group_id", "style_runs", "asset_id", "fit", "shape_type",
    "text_align", "line_height", "letter_spacing", "max_lines", "corner_radius", "font_weight",
    "some_extension_field",
])
def test_every_other_field_is_rejected_with_a_clear_error(field):
    source = sample_elements()["text"]
    before = copy.deepcopy(source)
    with pytest.raises(ProtectedFieldError) as error:
        layout_edit(source, {field: "changed"})
    message = str(error.value)
    assert field in message and "'t'" in message and "font_size" in message  # names the field, the element, the allowlist
    assert isinstance(error.value, ValueError)
    assert source == before


def test_a_mixed_edit_is_rejected_as_a_whole():
    source = sample_elements()["text"]
    before = copy.deepcopy(source)
    with pytest.raises(ProtectedFieldError, match="color"):
        layout_edit(source, {"x": 99, "color": "#000"})
    assert source == before


@pytest.mark.parametrize("element", [{"id": "a"}, {"id": "a", "type": 3}, {"id": "a", "type": None}])
def test_an_element_without_a_string_type_is_rejected(element):
    with pytest.raises(ValueError, match="type"):
        layout_edit(element, {"x": 1})


@pytest.mark.parametrize("element_type", ["text", "shape", "image"])
def test_helper_accepts_exactly_the_k6_fields_among_all_schema_fields(element_type):
    assert {"x", "y", "width", "height", "font_size"} <= SCHEMA_FIELDS
    source = sample_elements()[element_type]
    for field in sorted(SCHEMA_FIELDS):
        if field in permitted(element_type):
            assert layout_edit(source, {field: 7})[field] == 7
        else:
            with pytest.raises(ProtectedFieldError):
                layout_edit(source, {field: 7})


def test_k6_allowlist_is_pinned_to_the_documented_fields():
    assert mutable_fields("text") == LAYOUT_FIELDS | TEXT_ONLY_FIELDS
    assert mutable_fields("shape") == mutable_fields("image") == LAYOUT_FIELDS
    assert mutable_fields("unknown") == LAYOUT_FIELDS


def test_validator_and_helper_share_one_allowlist_definition(monkeypatch):
    raw = {"canvas": {"width": 100, "height": 100}, "elements": [
        {"id": "t", "type": "text", "x": 10, "y": 10, "width": 40, "height": 20,
         "content": "Hi", "font_family": "F", "font_size": 12, "color": "#fff"}]}
    source = Design.from_adobe(raw)
    moved = copy.deepcopy(raw)
    moved["elements"][0]["x"] = 20
    assert validate_candidate(source, moved, source.canvas).is_valid
    narrowed = lambda _element_type: frozenset({"y"})  # noqa: E731
    monkeypatch.setattr(candidate_validator, "mutable_fields", narrowed)
    assert not validate_candidate(source, moved, source.canvas).policy_ok   # the validator follows it
    monkeypatch.setattr(layout_edits, "mutable_fields", narrowed)
    with pytest.raises(ProtectedFieldError):
        layout_edit(raw["elements"][0], {"x": 20})                          # and so does the helper
    assert layout_edit(raw["elements"][0], {"y": 20})["y"] == 20


def test_the_helper_does_not_validate_values_so_the_validator_stays_the_authority():
    raw = {"canvas": {"width": 100, "height": 100}, "elements": [
        {"id": "t", "type": "text", "x": 10, "y": 10, "width": 40, "height": 20,
         "content": "Hi", "font_family": "F", "font_size": 12, "color": "#fff"}]}
    source = Design.from_adobe(raw)
    off_canvas = {"elements": [layout_edit(raw["elements"][0], {"x": 90})], "canvas": raw["canvas"]}
    assert not validate_candidate(source, off_canvas, source.canvas).is_valid
    not_finite = {"elements": [layout_edit(raw["elements"][0], {"x": float("nan")})], "canvas": raw["canvas"]}
    assert not validate_candidate(source, not_finite, source.canvas).is_valid


# --- property tests for the implemented candidate family ----------------------------------------

def baseline_candidates(source, target):
    # (label, candidate, must_be_fully_valid). Attempt 0 is the documented stretch, which fails the
    # geometry check by design on these samples; it must still never touch a protected field.
    return [(f"attempt{n}", baseline_proposal(source, target, n), n > 0) for n in range(8)]


FAMILIES = {"baseline": baseline_candidates}


@pytest.mark.parametrize("family", sorted(FAMILIES))
@pytest.mark.parametrize("target_index", [0, 1, 2], ids=["sample-target", "1080x1080", "300x250"])
@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.stem)
def test_generated_candidates_change_only_permitted_fields(path, target_index, family):
    task = json.loads(path.read_text(encoding="utf-8"))
    original = copy.deepcopy(task)
    target_raw = [task["target_canvas"], *EXTRA_TARGETS][target_index]
    source = Design.from_adobe(task["source"], target_canvas=target_raw)
    source_raw = source.to_adobe()
    target = Canvas.from_adobe(target_raw)
    candidates = FAMILIES[family](source, target)
    assert candidates
    for label, candidate, must_be_valid in candidates:
        assert_only_permitted_changes(source_raw, candidate)
        report = validate_candidate(source, candidate, target)
        assert report.schema_ok and report.policy_ok, (label, report.violations)
        if must_be_valid:
            assert report.is_valid, (label, report.violations)
        else:
            assert report.is_valid or not report.geometry_ok  # at most a geometry failure, never policy
    assert task == original                          # the caller's task data is untouched
    assert source.to_adobe() == source_raw           # and so is the source Design


@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.stem)
def test_generated_candidates_share_no_nested_structure_with_the_source(path):
    task = json.loads(path.read_text(encoding="utf-8"))
    source = Design.from_adobe(task["source"], target_canvas=task["target_canvas"])
    target = Canvas.from_adobe(task["target_canvas"])
    before = source.to_adobe()

    def poison(value):
        if isinstance(value, dict):
            for item in value.values():
                poison(item)
            value["poison"] = True
        elif isinstance(value, list):
            for item in value:
                poison(item)
            value.append("poison")

    for _label, candidate, _valid in baseline_candidates(source, target):
        poison(candidate)
    assert source.to_adobe() == before


def test_the_property_harness_detects_every_kind_of_protected_change():
    source = {"elements": [dict(sample_elements()["text"], id="t"), dict(sample_elements()["shape"], id="s")]}
    assert_only_permitted_changes(source, copy.deepcopy(source))
    for index, change in [
        (0, {"color": "#000"}), (1, {"fill_color": "#000"}), (1, {"stroke_width": 3}),
        (1, {"opacity": 0.6}), (0, {"font_family": "G"}), (0, {"content": "Hello"}),
        (1, {"font_size": 9}),                                   # font_size on a non-text element
    ]:
        candidate = copy.deepcopy(source)
        candidate["elements"][index].update(change)
        with pytest.raises(AssertionError):
            assert_only_permitted_changes(source, candidate)
    permitted_retype = copy.deepcopy(source)
    permitted_retype["elements"][0]["x"] = 1.0                   # 1 -> 1.0 on a permitted field is allowed
    assert_only_permitted_changes(source, permitted_retype)
    nested = copy.deepcopy(source)
    nested["elements"][0]["extension"]["nested"][2]["k"] = 1     # True -> 1 is a type change
    with pytest.raises(AssertionError):
        assert_only_permitted_changes(source, nested)
    removed = copy.deepcopy(source)
    del removed["elements"][1]["stroke_color"]
    with pytest.raises(AssertionError):
        assert_only_permitted_changes(source, removed)
    reordered = copy.deepcopy(source)
    reordered["elements"].reverse()
    with pytest.raises(AssertionError):
        assert_only_permitted_changes(source, reordered)


# --- defence in depth: the gate still decides -----------------------------------------------------

@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.stem)
def test_resize_output_changes_only_permitted_fields_and_passes_the_gate(path):
    task = json.loads(path.read_text(encoding="utf-8"))
    source = Design.from_adobe(task["source"], target_canvas=task["target_canvas"])
    output = resize(task["source"], task["target_canvas"])
    assert_only_permitted_changes(source.to_adobe(), output)
    assert validate_candidate(source, output, Canvas.from_adobe(task["target_canvas"])).is_valid


def test_search_loop_rejects_a_protected_change_even_when_geometry_is_fine():
    raw = {"canvas": {"width": 100, "height": 100}, "elements": [
        {"id": "t", "type": "text", "x": 10, "y": 10, "width": 40, "height": 20,
         "content": "Hi", "font_family": "F", "font_size": 12, "color": "#fff"}]}
    source = Design.from_adobe(raw)
    seen = []

    def propose(attempt, previous):
        seen.append(previous)
        if attempt == 0:
            return {"canvas": raw["canvas"], "elements": [dict(raw["elements"][0], color="#000")]}
        return raw

    assert find_valid_layout(source, source.canvas, propose) == raw
    assert seen[0] is None
    assert seen[1].schema_ok and seen[1].geometry_ok and not seen[1].policy_ok
    assert seen[1].violations == ("t: immutable fields or optional-field presence changed",)


def test_search_loop_never_returns_a_candidate_that_keeps_changing_a_protected_field():
    raw = {"canvas": {"width": 100, "height": 100}, "elements": [
        {"id": "t", "type": "text", "x": 10, "y": 10, "width": 40, "height": 20,
         "content": "Hi", "font_family": "F", "font_size": 12, "color": "#fff"}]}
    source = Design.from_adobe(raw)
    recolored = {"canvas": raw["canvas"], "elements": [dict(raw["elements"][0], color="#000")]}
    with pytest.raises(NoFeasibleLayout) as error:
        find_valid_layout(source, source.canvas, lambda *_: recolored, max_attempts=3)
    assert len(error.value.reports) == 3 and not any(report.policy_ok for report in error.value.reports)
