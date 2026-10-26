"""Lossless, immutable Adobe v0.9 models shared by layout and graph consumers.

Only parsing-level constraints are checked here. No target-layout validation,
asset loading, geometry repair, or semantic inference takes place. The bundled
Adobe schema is the source of field names, required fields, bounds and defaults.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Literal, TypeAlias, TypeVar, Union


Number: TypeAlias = int | float
# JSON Schema integers include integral floats; retain their decoded type.
Integer: TypeAlias = int | float
JSONValue: TypeAlias = Union[
    None, bool, int, float, str, list["JSONValue"], dict[str, "JSONValue"]
]
FrozenJSONValue: TypeAlias = Union[
    None, bool, int, float, str,
    tuple["FrozenJSONValue", ...], Mapping[str, "FrozenJSONValue"],
]
ElementType: TypeAlias = Literal["text", "image", "shape"]
FontWeight: TypeAlias = Literal["regular", "bold"]
FontStyle: TypeAlias = Literal["normal", "italic"]


class Absent(Enum):
    """Presence marker distinct from every JSON value, including null."""

    ABSENT = "ABSENT"


ABSENT = Absent.ABSENT


class SchemaError(ValueError):
    """Invalid input with a path into the decoded Adobe document."""

    def __init__(self, path: str, message: str) -> None:
        self.path = path or "$"
        super().__init__(f"{self.path}: {message}")


_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "chameleon_design_schema_v0.9.json"
with _SCHEMA_PATH.open(encoding="utf-8") as _schema_file:
    _SCHEMA = json.load(_schema_file)
_CANVAS_SCHEMA = _SCHEMA["$defs"]["canvas"]
_ELEMENT_SCHEMA = _SCHEMA["$defs"]["element"]
_RUN_SCHEMA = _ELEMENT_SCHEMA["properties"]["style_runs"]["items"]


def _path(parent: str, name: str) -> str:
    if name.isidentifier():
        return f"{parent}.{name}" if parent else name
    return f"{parent}[{name!r}]"


def _is_number(value: Any) -> bool:
    # Do not cast arbitrary-size integers to float just to check finiteness.
    return type(value) is int or (type(value) is float and math.isfinite(value))


def _json_tree(
    value: Any,
    path: str,
    *,
    freeze: bool = False,
    allow_frozen: bool = False,
    active: set[int] | None = None,
) -> Any:
    """Validate/copy a JSON tree, optionally freezing or thawing containers."""
    if value is None or type(value) in (str, bool) or _is_number(value):
        return value
    is_object = isinstance(value, Mapping)
    is_array = isinstance(value, list) or (allow_frozen and isinstance(value, tuple))
    if not is_object and not is_array:
        raise SchemaError(path, "expected a JSON value with finite numbers")
    if active is None:
        active = set()
    identity = id(value)
    if identity in active:
        raise SchemaError(path, "cyclic containers are not JSON values")
    active.add(identity)
    try:
        if is_object:
            result = {}
            for key, child in value.items():
                if not isinstance(key, str):
                    raise SchemaError(path, "object keys must be strings")
                result[key] = _json_tree(
                    child, _path(path, key), freeze=freeze,
                    allow_frozen=allow_frozen, active=active,
                )
            return MappingProxyType(result) if freeze else result
        items = [
            _json_tree(
                child, f"{path}[{index}]", freeze=freeze,
                allow_frozen=allow_frozen, active=active,
            )
            for index, child in enumerate(value)
        ]
        return tuple(items) if freeze else items
    finally:
        active.remove(identity)


def _validate_adobe(value: Any, schema: Mapping[str, Any], path: str) -> None:
    """Check the constraints used by the bundled v0.9 schema, without coercion.

    This is deliberately scoped to that schema, not a general JSON Schema engine.
    Optional fields' defaults are annotations, never inserted during validation.
    """
    if "$ref" in schema:
        schema = _SCHEMA["$defs"][schema["$ref"].removeprefix("#/$defs/")]
    declared = schema["type"]
    types = declared if isinstance(declared, list) else [declared]
    matches = {
        "null": value is None,
        "string": type(value) is str,
        "number": _is_number(value),
        "integer": _is_number(value) and (
            type(value) is int or value.is_integer()
        ),
        "object": isinstance(value, Mapping),
        "array": isinstance(value, list),
    }
    if not any(matches[kind] for kind in types):
        raise SchemaError(path, f"expected {' or '.join(types)} (finite if numeric)")
    if "enum" in schema and value not in schema["enum"]:
        raise SchemaError(path, f"expected one of {schema['enum']!r}")
    if _is_number(value):
        if "minimum" in schema and value < schema["minimum"]:
            raise SchemaError(path, f"must be >= {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            raise SchemaError(path, f"must be <= {schema['maximum']}")
    if isinstance(value, Mapping):
        properties = schema["properties"]
        required = list(schema.get("required", ()))
        # The v0.9 conditionals select additional required fields by type.
        for condition in schema.get("allOf", ()):
            discriminator = condition["if"]["properties"]["type"]["const"]
            if value.get("type") == discriminator:
                required.extend(condition["then"]["required"])
        for name in required:
            if name not in value:
                raise SchemaError(_path(path, name), "required field is missing")
        for name, child in value.items():
            if not isinstance(name, str):
                raise SchemaError(path, "object keys must be strings")
            child_path = _path(path, name)
            if name in properties:
                _validate_adobe(child, properties[name], child_path)
            elif schema.get("additionalProperties") is False:
                raise SchemaError(child_path, "additional property is not allowed")
            else:
                _json_tree(child, child_path)
    elif isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            raise SchemaError(path, f"must contain at least {schema['minItems']} item(s)")
        for index, child in enumerate(value):
            _validate_adobe(child, schema["items"], f"{path}[{index}]")


_RecordT = TypeVar("_RecordT", bound="_AdobeRecord")


class _AdobeRecord:
    """Shared schema-key handling; only the public dataclasses store state."""

    __slots__ = ()
    _schema: ClassVar[Mapping[str, Any]]
    _record_path: ClassVar[str]

    @classmethod
    def from_adobe(cls: type[_RecordT], data: Mapping[str, JSONValue]) -> _RecordT:
        """Parse a record, keeping extras separate from internal attributes."""
        _validate_adobe(data, cls._schema, cls._record_path)
        properties = cls._schema["properties"]
        values = {name: data[name] for name in properties if name in data}
        if "style_runs" in values:
            values["style_runs"] = tuple(
                StyleRun.from_adobe(run) for run in values["style_runs"]
            )
        if cls._schema.get("additionalProperties") is not False:
            values["extras"] = {
                name: value for name, value in data.items() if name not in properties
            }
        return cls(**values)

    def _known_fields(self) -> dict[str, JSONValue]:
        result = {}
        for name in self._schema["properties"]:
            value = getattr(self, name)
            if value is ABSENT:
                continue
            if name == "style_runs":
                value = [run.to_adobe() for run in value]
            result[name] = value
        return result

    def __post_init__(self) -> None:
        _validate_adobe(self._known_fields(), self._schema, self._record_path)
        if self._schema.get("additionalProperties") is not False:
            extras = self.extras
            if not isinstance(extras, Mapping):
                raise SchemaError(_path(self._record_path, "extras"), "expected a mapping")
            for name in extras:
                if name in self._schema["properties"]:
                    raise SchemaError(
                        _path(self._record_path, name), "schema field cannot be in extras"
                    )
            object.__setattr__(self, "extras", _json_tree(
                extras, self._record_path, freeze=True, allow_frozen=True,
            ))

    def to_adobe(self) -> dict[str, JSONValue]:
        """Return a fresh, mutable Adobe record with only present fields."""
        result = self._known_fields()
        result.update(_json_tree(
            getattr(self, "extras", {}), self._record_path, allow_frozen=True,
        ))
        return result


@dataclass(frozen=True, kw_only=True, slots=True)
class Canvas(_AdobeRecord):
    width: Integer
    height: Integer

    _schema: ClassVar[Mapping[str, Any]] = _CANVAS_SCHEMA
    _record_path: ClassVar[str] = "canvas"

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

    @property
    def area(self) -> Integer:
        return self.width * self.height

    @property
    def orientation(self) -> Literal["landscape", "portrait", "square"]:
        if self.width > self.height:
            return "landscape"
        if self.height > self.width:
            return "portrait"
        return "square"


@dataclass(frozen=True, kw_only=True, slots=True)
class StyleRun(_AdobeRecord):
    start: Integer
    end: Integer
    color: str | Absent = ABSENT
    font_weight: FontWeight | Absent = ABSENT
    font_style: FontStyle | Absent = ABSENT
    extras: Mapping[str, FrozenJSONValue] = field(default_factory=dict)

    _schema: ClassVar[Mapping[str, Any]] = _RUN_SCHEMA
    _record_path: ClassVar[str] = "style_run"


@dataclass(frozen=True, kw_only=True, slots=True)
class Element(_AdobeRecord):
    id: str
    type: ElementType
    x: Number
    y: Number
    width: Number
    height: Number
    rotation: Number | Absent = ABSENT
    opacity: Number | Absent = ABSENT
    z_index: Integer | Absent = ABSENT
    group_id: str | None | Absent = ABSENT
    content: str | Absent = ABSENT
    font_family: str | Absent = ABSENT
    font_size: Number | Absent = ABSENT
    font_weight: FontWeight | Absent = ABSENT
    font_style: FontStyle | Absent = ABSENT
    color: str | Absent = ABSENT
    text_align: Literal["left", "center", "right"] | Absent = ABSENT
    line_height: Number | Absent = ABSENT
    letter_spacing: Number | Absent = ABSENT
    max_lines: Integer | None | Absent = ABSENT
    style_runs: tuple[StyleRun, ...] | Absent = ABSENT
    asset_id: str | Absent = ABSENT
    fit: Literal["contain", "cover", "fill"] | Absent = ABSENT
    shape_type: Literal["rectangle", "ellipse", "line"] | Absent = ABSENT
    fill_color: str | Absent = ABSENT
    stroke_color: str | None | Absent = ABSENT
    stroke_width: Number | Absent = ABSENT
    corner_radius: Number | Absent = ABSENT
    required: bool = field(default=True, init=False)
    extras: Mapping[str, FrozenJSONValue] = field(default_factory=dict)

    _schema: ClassVar[Mapping[str, Any]] = _ELEMENT_SCHEMA
    _record_path: ClassVar[str] = "element"

    def __post_init__(self) -> None:
        if self.style_runs is not ABSENT:
            if not isinstance(self.style_runs, (tuple, list)):
                raise SchemaError("element.style_runs", "expected a sequence of StyleRun")
            for index, run in enumerate(self.style_runs):
                if not isinstance(run, StyleRun):
                    raise SchemaError(f"element.style_runs[{index}]", "expected StyleRun")
            object.__setattr__(self, "style_runs", tuple(self.style_runs))
        _AdobeRecord.__post_init__(self)

    def effective_value(
        self, field_name: str,
    ) -> Number | str | None | tuple[StyleRun, ...] | Absent:
        """Read a schema default without changing stored field presence."""
        specification = self._schema["properties"][field_name]  # KeyError if unknown
        value = getattr(self, field_name)
        return specification.get("default", ABSENT) if value is ABSENT else value

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)

    @property
    def area(self) -> Number:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float | None:
        return self.width / self.height if self.height != 0 else None

    def normalized_bounds(self, canvas: Canvas) -> tuple[float, float, float, float]:
        """Return unrotated x/y/width/height fractions without clipping."""
        return (
            self.x / canvas.width, self.y / canvas.height,
            self.width / canvas.width, self.height / canvas.height,
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class Theme:
    """Observed values only, in first-occurrence order; no semantic roles."""

    fonts: tuple[str, ...]
    colors: tuple[str, ...]
    asset_references: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("fonts", "colors", "asset_references"):
            values = getattr(self, name)
            if not isinstance(values, (tuple, list)) or not all(
                type(value) is str for value in values
            ):
                raise SchemaError(f"theme.{name}", "expected a sequence of strings")
            object.__setattr__(self, name, tuple(values))

    @classmethod
    def from_elements(cls, elements: Iterable[Element]) -> Theme:
        fonts: dict[str, None] = {}
        colors: dict[str, None] = {}
        assets: dict[str, None] = {}
        for element in elements:
            if isinstance(element.font_family, str):
                fonts[element.font_family] = None
            if isinstance(element.asset_id, str):
                assets[element.asset_id] = None
            for color in (element.color, element.fill_color, element.stroke_color):
                if isinstance(color, str):
                    colors[color] = None
            if element.style_runs is not ABSENT:
                for run in element.style_runs:
                    if isinstance(run.color, str):
                        colors[run.color] = None
        return cls(fonts=tuple(fonts), colors=tuple(colors), asset_references=tuple(assets))


@dataclass(frozen=True, kw_only=True, slots=True)
class Design:
    """Current layout plus immutable original/requested canvas context."""

    canvas: Canvas
    source_canvas: Canvas
    elements: tuple[Element, ...]
    target_canvas: Canvas | None = None
    elements_by_id: Mapping[str, Element] = field(init=False, repr=False, compare=False)
    groups: Mapping[str, tuple[Element, ...]] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in ("canvas", "source_canvas", "target_canvas"):
            value = getattr(self, name)
            if name == "target_canvas" and value is None:
                continue
            if not isinstance(value, Canvas):
                raise SchemaError(name, "expected Canvas")
        if not isinstance(self.elements, (tuple, list)):
            raise SchemaError("elements", "expected a sequence of Element")
        if len(self.elements) < _SCHEMA["properties"]["elements"]["minItems"]:
            raise SchemaError("elements", "must contain at least one element")
        by_id: dict[str, Element] = {}
        groups: dict[str, list[Element]] = {}
        for index, element in enumerate(self.elements):
            if not isinstance(element, Element):
                raise SchemaError(f"elements[{index}]", "expected Element")
            if element.id in by_id:
                raise SchemaError(f"elements[{index}].id", f"duplicate ID {element.id!r}")
            by_id[element.id] = element
            if isinstance(element.group_id, str):
                groups.setdefault(element.group_id, []).append(element)
        object.__setattr__(self, "elements", tuple(self.elements))
        object.__setattr__(self, "elements_by_id", MappingProxyType(by_id))
        object.__setattr__(self, "groups", MappingProxyType({
            name: tuple(members) for name, members in groups.items()
        }))

    @classmethod
    def from_adobe(
        cls,
        source: Mapping[str, JSONValue],
        *,
        target_canvas: Mapping[str, JSONValue] | None = None,
    ) -> Design:
        """Parse a design (not a task envelope), retaining source coordinates."""
        _validate_adobe(source, _SCHEMA, "")
        if target_canvas is not None:
            _validate_adobe(target_canvas, _CANVAS_SCHEMA, "target_canvas")
        canvas = Canvas.from_adobe(source["canvas"])
        return cls(
            canvas=canvas,
            source_canvas=canvas,
            target_canvas=(
                Canvas.from_adobe(target_canvas) if target_canvas is not None else None
            ),
            elements=tuple(Element.from_adobe(item) for item in source["elements"]),
        )

    def to_adobe(self) -> dict[str, JSONValue]:
        """Serialize current layout only, with no internal context/metadata."""
        return {
            "canvas": self.canvas.to_adobe(),
            "elements": [element.to_adobe() for element in self.elements],
        }

    def get_element(self, element_id: str) -> Element:
        """Look up an opaque ID; an unknown ID raises KeyError."""
        return self.elements_by_id[element_id]

    @property
    def theme(self) -> Theme:
        return Theme.from_elements(self.elements)
