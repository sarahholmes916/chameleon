"""Lossless Adobe parsing and immutable shared-model contracts."""

import ast
from copy import copy, deepcopy
from dataclasses import FrozenInstanceError, fields, replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from engine.internal_schema import ABSENT, Canvas, Design, Element, SchemaError, StyleRun, Theme


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = sorted((ROOT / "input").glob("chameleon_sample_*.json"))
SCHEMA = json.loads((ROOT / "chameleon_design_schema_v0.9.json").read_text(encoding="utf-8"))
ELEMENT_FIELDS = SCHEMA["$defs"]["element"]["properties"]


def image_element(**changes):
    result = {
        "id": "arbitrary-id", "type": "image", "x": 10, "y": 20,
        "width": 30, "height": 40, "asset_id": "missing.png",
    }
    result.update(changes)
    return result


def text_element(**changes):
    result = image_element(
        type="text", content=" A\n\t\u00e9\u0301 \U0001f31e ",
        font_family="Exact Family", font_size=24, color="rgba(1,2,3,1)",
    )
    del result["asset_id"]
    result.update(changes)
    return result


def source_with(*elements):
    return {"canvas": {"width": 200, "height": 100}, "elements": list(elements)}


def assert_same_json(actual, expected):
    """Unlike ==, also catch int/float/bool conversion at every depth."""
    assert type(actual) is type(expected)
    assert actual == expected
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for name in expected:
            assert_same_json(actual[name], expected[name])
    elif isinstance(expected, list):
        for a, b in zip(actual, expected):
            assert_same_json(a, b)


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda p: p.stem)
@pytest.mark.parametrize("with_target", [False, True])
def test_all_samples_round_trip_and_context(sample, with_target):
    task = json.loads(sample.read_text(encoding="utf-8"))
    before = deepcopy(task)
    source = task["source"]
    target = task["target_canvas"] if with_target else None
    design = Design.from_adobe(source, target_canvas=target)

    assert_same_json(design.to_adobe(), source)
    assert_same_json(task, before)
    assert_same_json(design.canvas.to_adobe(), source["canvas"])
    assert_same_json(design.source_canvas.to_adobe(), source["canvas"])
    if with_target:
        assert_same_json(design.target_canvas.to_adobe(), target)
    else:
        assert design.target_canvas is None
    assert tuple(e.id for e in design.elements) == tuple(e["id"] for e in source["elements"])
    assert tuple(design.elements_by_id) == tuple(e.id for e in design.elements)
    assert all(e.required is True for e in design.elements)
    for element in design.elements:
        assert design.get_element(element.id) is element
        assert design.elements_by_id[element.id] is element
        element.normalized_bounds(design.source_canvas)
    expected_groups = {}
    for element in source["elements"]:
        if isinstance(element.get("group_id"), str):
            expected_groups.setdefault(element["group_id"], []).append(element["id"])
    assert {name: [e.id for e in members] for name, members in design.groups.items()} == expected_groups
    assert tuple(design.groups) == tuple(expected_groups)
    # Views and derived values must not materialize defaults or change the output.
    design.theme
    for element in design.elements:
        for name in ELEMENT_FIELDS:
            element.effective_value(name)
    assert_same_json(design.to_adobe(), source)


def test_absent_defaults_nulls_and_empty_values():
    assert copy(ABSENT) is ABSENT
    assert deepcopy(ABSENT) is ABSENT
    for value in (None, 0, False, "", []):
        assert ABSENT is not value
        assert ABSENT != value
    data = text_element(content="", style_runs=[], group_id=None, max_lines=None, stroke_color=None)
    element = Element.from_adobe(data)
    assert element.rotation is ABSENT
    assert element.opacity is ABSENT
    assert element.z_index is ABSENT
    assert element.group_id is None
    assert element.max_lines is None
    assert element.stroke_color is None
    assert element.style_runs == ()
    for name, definition in ELEMENT_FIELDS.items():
        if name not in data:
            assert element.effective_value(name) == definition.get("default", ABSENT)
    assert_same_json(element.to_adobe(), data)
    explicit = Element.from_adobe(image_element(rotation=0, opacity=0, group_id="", fit="contain"))
    assert explicit.effective_value("opacity") == 0
    assert explicit.effective_value("group_id") == ""
    assert explicit.to_adobe()["rotation"] == 0
    for name in ("required", "extras", "unknown"):
        with pytest.raises(KeyError):
            explicit.effective_value(name)


def test_schema_fields_enums_and_conditional_fields_are_complete():
    assert {f.name for f in fields(Element)} == set(ELEMENT_FIELDS) | {"extras", "required"}
    run_fields = SCHEMA["$defs"]["element"]["properties"]["style_runs"]["items"]["properties"]
    assert {f.name for f in fields(StyleRun)} == set(run_fields) | {"extras"}
    data = text_element(
        rotation=-12, opacity=0.5, z_index=-2, group_id="",
        font_weight="bold", font_style="italic", text_align="right",
        line_height=-1, letter_spacing=-0.5, max_lines=-1,
        style_runs=[{"start": 10, "end": 1, "font_weight": "regular", "font_style": "normal"}],
        asset_id="unchanged.png", fit="fill", shape_type="line", fill_color=" #ABC ",
        stroke_color="rgba(1, 2, 3, .2)", stroke_width=-2, corner_radius=-3,
    )
    assert set(data) == set(ELEMENT_FIELDS)
    assert_same_json(Element.from_adobe(data).to_adobe(), data)
    # Adobe does not forbid known fields merely because they describe another type.
    for name, definition in ELEMENT_FIELDS.items():
        for choice in definition.get("enum", ()):
            variant = dict(data, **{name: choice})
            assert_same_json(Element.from_adobe(variant).to_adobe(), variant)


def test_style_runs_order_extensions_and_unusual_indices():
    data = text_element(style_runs=[
        {"start": 6.0, "end": 99, "color": " first ", "custom": {"values": [False, None, 1.0]}},
        {"start": 4, "end": 2, "font_weight": "bold", "font_style": "italic"},
        {"start": 0, "end": 0},
    ])
    element = Element.from_adobe(data)
    assert isinstance(element.style_runs, tuple)
    assert [run.start for run in element.style_runs] == [6.0, 4, 0]
    assert element.style_runs[1].color is ABSENT
    assert_same_json(element.to_adobe(), data)
    for raw, run in zip(data["style_runs"], element.style_runs):
        assert_same_json(StyleRun.from_adobe(raw).to_adobe(), raw)
        with pytest.raises(FrozenInstanceError):
            run.end = 1


def test_extensions_internal_name_collisions_and_alias_isolation():
    data = text_element(
        required=False, extras={"nested": [0, False, "", None, {"n": 1.0}]},
        theme={"fonts": ["not a font"]}, center=[1, 2],
        style_runs=[{"start": 0, "end": 1, "extras": {"values": [1, {"a": 2}]} }],
    )
    original = source_with(data)
    before = deepcopy(original)
    design = Design.from_adobe(original, target_canvas={"width": 10, "height": 20})
    element = design.elements[0]
    assert element.required is True
    assert element.extras["required"] is False
    assert element.center == (25, 40)
    assert design.theme.fonts == ("Exact Family",)

    # Input mutation, including run extras, cannot modify the model.
    original["canvas"]["width"] = 1
    data["extras"]["nested"][4]["n"] = 999
    data["style_runs"][0]["extras"]["values"][1]["a"] = 999
    data["style_runs"].clear()
    assert_same_json(design.to_adobe(), before)
    exported = design.to_adobe()
    exported["elements"][0]["extras"]["nested"][4]["n"] = 100
    exported["elements"][0]["style_runs"][0]["extras"]["values"].append(5)
    exported["canvas"]["width"] = 5
    exported["elements"].clear()
    assert_same_json(design.to_adobe(), before)
    assert_same_json(json.loads(json.dumps(design.to_adobe(), allow_nan=False)), before)
    # Public views are recursively immutable.
    with pytest.raises(TypeError):
        element.extras["required"] = True
    with pytest.raises(TypeError):
        element.extras["extras"]["nested"][4]["n"] = 42
    with pytest.raises(TypeError):
        element.style_runs[0].extras["extras"]["values"][1]["a"] = 42


def test_numeric_representations_are_preserved_recursively():
    data = text_element(
        x=-0.0, y=2, width=0.0, height=3, z_index=4.0, max_lines=0.0,
        style_runs=[{"start": 0.0, "end": 1.0}], custom=[1, 1.0, -0.0, True],
    )
    source = {"canvas": {"width": 200.0, "height": 100}, "elements": [data]}
    target = {"width": 10, "height": 20.0}
    design = Design.from_adobe(source, target_canvas=target)
    assert_same_json(design.to_adobe(), source)
    assert_same_json(design.target_canvas.to_adobe(), target)
    assert json.dumps(design.to_adobe()["elements"][0]["x"]) == "-0.0"
    # JSON/Python integers have no artificial float-sized upper bound.
    huge = image_element(x=10**400, z_index=10**400, custom=10**400)
    assert_same_json(Element.from_adobe(huge).to_adobe(), huge)


@pytest.mark.parametrize("width,height,orientation", [(200, 100, "landscape"), (100, 200, "portrait"), (100, 100, "square")])
def test_canvas_geometry(width, height, orientation):
    canvas = Canvas(width=width, height=height)
    assert canvas.area == width * height
    assert canvas.aspect_ratio == width / height
    assert canvas.orientation == orientation
    assert_same_json(canvas.to_adobe(), {"width": width, "height": height})
    with pytest.raises(FrozenInstanceError):
        canvas.width = 1


@pytest.mark.parametrize("width,height,ratio", [(30, 40, 0.75), (0, 40, 0), (30, 0, None), (0, 0, None)])
def test_element_geometry_is_unrotated_unclipped_and_zero_safe(width, height, ratio):
    data = image_element(x=-100, y=200, width=width, height=height, rotation=-90, opacity=0)
    element = Element.from_adobe(data)
    assert element.center == (-100 + width / 2, 200 + height / 2)
    assert element.area == width * height
    assert element.aspect_ratio == ratio
    assert element.normalized_bounds(Canvas(width=200, height=100)) == (-0.5, 2, width / 200, height / 100)
    assert element.normalized_bounds(Canvas(width=100, height=200)) == (-1, 1, width / 100, height / 200)
    assert_same_json(element.to_adobe(), data)


def test_group_and_id_views_and_source_are_immutable():
    source = source_with(
        image_element(id="z", group_id="", z_index=20),
        image_element(id="a", group_id=None, z_index=-1),
        image_element(id="b"),
        image_element(id="c", group_id=""),
        image_element(id="d", group_id="other"),
    )
    design = Design.from_adobe(source)
    assert tuple(design.groups) == ("", "other")
    assert tuple(e.id for e in design.groups[""]) == ("z", "c")
    assert design.elements[1].group_id is None
    assert design.elements[2].group_id is ABSENT
    with pytest.raises(KeyError):
        design.get_element("missing")
    with pytest.raises(TypeError):
        design.elements_by_id["new"] = design.elements[0]
    with pytest.raises(TypeError):
        design.groups[""] = ()
    with pytest.raises(FrozenInstanceError):
        design.canvas = Canvas(width=1, height=1)
    with pytest.raises(FrozenInstanceError):
        design.elements[0].id = "new"
    with pytest.raises(FrozenInstanceError):
        design.elements[0].required = False
    assert_same_json(design.to_adobe(), source)


def test_independent_constructed_designs_retain_metadata_and_canvas_context():
    # Standard dataclass replacement establishes ownership isolation, not a
    # stage-one-approved editing interface or candidate validator.
    source = Design.from_adobe(source_with(image_element(custom={"a": [1]})), target_canvas={"width": 50, "height": 50})
    shifted = replace(source.elements[0], x=0)
    candidate = Design(canvas=source.target_canvas, source_canvas=source.source_canvas,
                       target_canvas=source.target_canvas, elements=[shifted])
    assert source.elements[0].x == 10
    assert source.canvas.width == 200
    assert candidate.get_element(shifted.id) is shifted
    assert candidate.canvas.width == 50
    assert candidate.source_canvas.width == 200
    assert candidate.to_adobe()["elements"][0]["custom"] == {"a": [1]}
    assert source.to_adobe()["elements"][0]["custom"] == {"a": [1]}
    with pytest.raises(TypeError):
        Element(**image_element(), required=False)


def test_theme_is_an_ordered_inventory_without_defaults_or_inference():
    design = Design.from_adobe(source_with(
        text_element(id="one", color="red", fill_color="blue", stroke_color=None,
                     style_runs=[{"start": 0, "end": 1, "color": "green"}, {"start": 0, "end": 1}],
                     fake_theme={"font_family": "Invented", "color": "purple"}),
        text_element(id="two", color="red", font_family="Second", fill_color="RED", stroke_color="blue"),
        image_element(id="logo", asset_id="does-not-exist.png", color="red"),
        image_element(id="cta", asset_id="does-not-exist.png"),
    ))
    theme = design.theme
    assert theme.fonts == ("Exact Family", "Second")
    assert theme.colors == ("red", "blue", "green", "RED")
    assert theme.asset_references == ("does-not-exist.png",)
    assert Theme.from_elements(()) == Theme(fonts=(), colors=(), asset_references=())
    assert "theme" not in design.to_adobe()
    assert Element.from_adobe(image_element()).fit is ABSENT
    with pytest.raises(FrozenInstanceError):
        theme.fonts = ()


@pytest.mark.parametrize("dimension", [0, -1, 1.5, True, False, None, "10", float("inf"), float("-inf"), float("nan")])
@pytest.mark.parametrize("name", ["width", "height"])
def test_bad_canvas_dimensions(dimension, name):
    data = {"width": 10, "height": 20, name: dimension}
    with pytest.raises(SchemaError, match=rf"canvas\.{name}"):
        Canvas.from_adobe(data)
    with pytest.raises(SchemaError, match=rf"canvas\.{name}"):
        Canvas(**data)


@pytest.mark.parametrize("name,value", [
    ("id", None), ("type", "video"), ("x", True), ("y", float("nan")),
    ("width", -1), ("height", float("inf")), ("rotation", None),
    ("opacity", -0.1), ("opacity", 1.1), ("opacity", False),
    ("z_index", 0.5), ("group_id", 123), ("content", None),
    ("font_family", []), ("font_size", 0.9), ("color", None),
    ("font_weight", "medium"), ("font_style", "oblique"), ("text_align", "justify"),
    ("line_height", "1.2"), ("letter_spacing", float("-inf")), ("max_lines", True),
    ("asset_id", 1), ("fit", "stretch"), ("shape_type", "triangle"),
    ("fill_color", None), ("stroke_color", 1), ("stroke_width", None),
    ("corner_radius", []), ("style_runs", None), ("style_runs", ()),
])
def test_invalid_known_fields_have_full_paths(name, value):
    data = source_with(image_element(id="first"), image_element(id="second"), text_element(**{name: value}))
    with pytest.raises(SchemaError) as error:
        Design.from_adobe(data)
    assert error.value.path == f"elements[2].{name}"


@pytest.mark.parametrize("kind,required", [
    ("text", ["id", "type", "x", "y", "width", "height", "content", "font_family", "font_size", "color"]),
    ("image", ["asset_id"]), ("shape", ["shape_type", "fill_color"]),
])
def test_missing_required_fields(kind, required):
    data = text_element(type=kind, asset_id="image.png", shape_type="rectangle", fill_color="red")
    for name in required:
        variant = dict(data)
        del variant[name]
        with pytest.raises(SchemaError) as error:
            Design.from_adobe(source_with(variant))
        assert error.value.path == f"elements[0].{name}"


@pytest.mark.parametrize("run,field_name", [
    ({"end": 1}, "start"), ({"start": 0}, "end"),
    ({"start": -1, "end": 1}, "start"), ({"start": True, "end": 1}, "start"),
    ({"start": 0, "end": 1.5}, "end"), ({"start": 0, "end": 1, "color": None}, "color"),
    ({"start": 0, "end": 1, "font_weight": "heavy"}, "font_weight"),
])
def test_style_run_validation_paths(run, field_name):
    with pytest.raises(SchemaError) as error:
        Design.from_adobe(source_with(text_element(style_runs=[run])))
    assert error.value.path == f"elements[0].style_runs[0].{field_name}"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), {1: "bad key"}, (1, 2), {1, 2}, b"bytes", object(), ABSENT])
def test_invalid_extension_values_are_rejected(bad):
    with pytest.raises(SchemaError) as error:
        Design.from_adobe(source_with(image_element(custom={"nested": [bad]})))
    assert error.value.path == "elements[0].custom.nested[0]"


def test_cyclic_extensions_rejected_and_shared_acyclic_values_copied():
    cyclic = []
    cyclic.append(cyclic)
    with pytest.raises(SchemaError, match="cyclic"):
        Element.from_adobe(image_element(custom=cyclic))
    shared = {"list": [1, None]}
    element = Element.from_adobe(image_element(custom=[shared, shared]))
    exported = element.to_adobe()
    exported["custom"][0]["list"].append(2)
    assert exported["custom"][1]["list"] == [1, None]
    assert element.to_adobe()["custom"] == [shared, shared]


def test_unknown_root_canvas_keys_and_wrong_shapes_are_rejected():
    cases = [
        ({"elements": [image_element()]}, "canvas"),
        ({"canvas": {"width": 10, "height": 10}}, "elements"),
        (dict(source_with(image_element()), theme={}), "theme"),
        ({"canvas": {"width": 10, "height": 10, "color": "red"}, "elements": [image_element()]}, "canvas.color"),
        ({"canvas": {"width": 10}, "elements": [image_element()]}, "canvas.height"),
        (source_with(), "elements"),
        (dict(source_with(), elements={}), "elements"),
        (dict(source_with(), elements=[None]), "elements[0]"),
        (None, "$"),
    ]
    for data, expected_path in cases:
        with pytest.raises(SchemaError) as error:
            Design.from_adobe(data)
        assert error.value.path == expected_path
    with pytest.raises(SchemaError) as error:
        Design.from_adobe(source_with(image_element()), target_canvas={"width": True, "height": 10})
    assert error.value.path == "target_canvas.width"
    with pytest.raises(SchemaError):
        Design.from_adobe({"source": source_with(image_element()), "target_canvas": {"width": 1, "height": 1}})


def test_duplicate_ids_and_extra_field_collisions_are_rejected():
    with pytest.raises(SchemaError) as error:
        Design.from_adobe(source_with(image_element(), image_element()))
    assert error.value.path == "elements[1].id"
    assert "duplicate" in str(error.value)
    with pytest.raises(SchemaError, match="schema field cannot be in extras"):
        Element(**image_element(), extras={"x": 99})
    with pytest.raises(SchemaError, match="schema field cannot be in extras"):
        StyleRun(start=0, end=1, extras={"color": "red"})


def test_direct_construction_validates_and_detaches_mutable_sequences():
    run_extras = {"values": [1]}
    run = StyleRun(start=0, end=1, extras=run_extras)
    runs = [run]
    element = Element(**text_element(), style_runs=runs)
    elements = [element]
    canvas = Canvas(width=100, height=100)
    design = Design(canvas=canvas, source_canvas=canvas, elements=elements)
    run_extras["values"].append(2)
    runs.clear()
    elements.clear()
    assert design.elements == (element,)
    assert element.style_runs == (run,)
    assert run.to_adobe()["values"] == [1]
    with pytest.raises(SchemaError):
        Element(**image_element(width=-1))
    with pytest.raises(SchemaError):
        Element(**image_element(), style_runs=[{"start": 0, "end": 1}])
    with pytest.raises(SchemaError):
        StyleRun(start=0, end=-1)
    with pytest.raises(SchemaError):
        Design(canvas=canvas, source_canvas=canvas, elements=[])


def test_core_requires_only_stdlib_and_imports_outside_working_directory(tmp_path):
    # -S excludes installed ML/rendering packages. Import must use the bundled
    # schema relative to the module, not relative to the process directory.
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from engine.internal_schema import Design; "
        "assert not any(n.split('.')[0] in {'torch', 'torch_geometric', 'PIL', 'pytest'} "
        "for n in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-S", "-c", code, str(ROOT)], cwd=tmp_path,
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    code = (ROOT / "engine/internal_schema.py").read_text(encoding="utf-8")
    ast.parse(code, feature_version=(3, 10))
