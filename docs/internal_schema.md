# S02: Internal design representation

Status: the S02 foundation is implemented in
[engine/internal_schema.py](../engine/internal_schema.py). Parsing, immutable
models, derived views, and serialization are available. Controlled editing and
stage-one layout validation remain proposed APIs. Resize and rendering behavior
are unchanged.

## Implemented milestone and shared interfaces

Both future consumers should import the same models from
`engine.internal_schema`; no second JSON parser is needed for graph features.

| Public interface | Available behavior |
| --- | --- |
| `ABSENT`, `Absent` | Stable missing-value sentinel; compare with `is ABSENT`. |
| `SchemaError(ValueError)` | Parsing/construction failure with a `.path` and explanatory message. |
| `Canvas.from_adobe(data)`, `canvas.to_adobe()` | Validated integral pixel dimensions; `aspect_ratio`, `area`, and `orientation`. |
| `StyleRun.from_adobe(data)`, `run.to_adobe()` | Preserved indices, optional overrides, and extensions. |
| `Element.from_adobe(data)`, `element.to_adobe()` | All 28 Adobe fields, internal `required=True`, `effective_value(name)`, and derived geometry. |
| `Design.from_adobe(source, target_canvas=...)`, `design.to_adobe()` | Immutable original order, distinct canvas context, and lossless current-layout output. |
| `design.elements`, `elements_by_id`, `get_element(id)`, `groups` | Direct immutable iteration, indexed lookup, and explicit group membership. |
| `design.theme`, `Theme.from_elements(elements)` | Ordered, deduplicated observations of fonts, colors, and asset references. |

The core module has only standard-library dependencies. It reads the bundled
`chameleon_design_schema_v0.9.json` relative to its module location, so it does
not depend on the process working directory. Keep that file alongside the
`engine/` package when distributing the module; the existing Dockerfile already
copies both. Field constraints, conditional requirements, and effective defaults
come from this schema, while explicit dataclass annotations provide direct typed
access. The validator handles the constructs in v0.9; it is not a general-purpose
JSON Schema implementation.

All model dataclasses are frozen, keyword-only, and slotted. Element/run
`extras` expose recursively read-only mappings and tuples; `to_adobe()` converts
these back into fresh dictionaries and lists. Original integer/float types and
array order survive the conversion. Lists passed to direct model constructors
for elements or runs are detached into tuples. Direct constructors validate
state too; their `extras` may also reuse an existing frozen view.

The teammate can rely on the table above for source inspection and serialization.
Direct `Design` construction accepts explicit current/source/target canvases
and typed elements, but does not certify a layout against an original design.
`dataclasses.replace` likewise does not enforce cross-design retention policy.
The proposed `with_updates()`, `with_layout()`, and `validate_stage_one()` are
deliberately **not implemented** in this milestone. No placeholder candidate,
graph, ranker, solver, or rendering components are added.

Install test-only dependencies with `python -m pip install -r requirements-dev.txt`
and run `python -m pytest -q` from the repository root. The tests include all
three sample task envelopes, checking their `source` values with and without
target context, plus validation, immutability, and extension cases.

### Integration differences to resolve in later milestones

The implementation request's illustrative orchestration differs from
[S03–S06 §1](S03-S06_layout_engine_technical_spec.md#1-modules-and-stable-boundaries):
that spec declares `generate_candidates(source, target, config)` returning
`list[LayoutCandidate]`, and its `ValidationReport` exposes `schema_ok`,
`policy_ok`, and `geometry_ok` rather than `.is_valid`. Its §6 prose also calls
the outputs `Design` instances, whereas §1 wraps them in `LayoutCandidate`.
These are future generator/validator interface questions; S02 leaves them open
and adds no alternative candidate model. The underlying `Design` model and
`Design.from_adobe(source, target_canvas=...)` agree across these proposals.

The advanced fixture's suggestion to skip a missing asset conflicts with the
schema's same-ID contract; S02 preserves the element and reference. The core
fixture's out-of-range run and negative coordinates are schema-permitted input
and are preserved, as detailed below. No source-target feasibility or image
proportion checks are applied during parsing.

## Contract and scope

The internal model provides typed access to Adobe v0.9 designs while preserving
their JSON structure. Here, **normalized** means a consistent internal data
model; stored coordinates remain in pixels. Fractional coordinates relative to
a canvas are derived views, never replacements for the pixel values.

This specification is based on:

- [Adobe's design schema](../chameleon_design_schema_v0.9.json), including its
  conditional required fields and additional-property rules.
- [RULES.md](../RULES.md) and [FAQ.md](../FAQ.md), especially element retention,
  fixed text/font families, asset preservation, and the absence of semantic labels.
- [engine/resize.py](../engine/resize.py), whose boundary is
  `resize(source, target_canvas) -> {"canvas": ..., "elements": [...]}`.
- All three [warm-up](../input/chameleon_sample_level1_warmup.json),
  [core](../input/chameleon_sample_level2_core.json), and
  [advanced](../input/chameleon_sample_level3_advanced.json) sample tasks.

The sample files are task envelopes. Only `task["source"]` is an Adobe design;
`task["target_canvas"]` is a separate resize request. Fields such as `task_id`,
`difficulty`, and `notes` belong to the caller and are not design output fields.

An unchanged design must round-trip to the same parsed JSON value, including
field presence, explicit nulls, values, nested extension data, and array order.
This is structural fidelity, not preservation of file whitespace, escape
spellings, or object-key ordering. Preserve the `int`/`float` values supplied by
the JSON decoder without blanket float conversion or rounding. JSON number
spellings such as `1e2` cannot be recovered from a decoded dictionary.

Output contains exactly `canvas` and `elements` at its root. Internal theme,
required flags, source/target context, indexes, and computed geometry are never
introduced into Adobe JSON.

## Representation and presence

Use Python 3.10 standard-library tools: `dataclasses`, `enum`, `typing`, `copy`,
`json`, and `types.MappingProxyType`. Models are frozen,
keyword-only dataclasses. Use tuples for element and style-run sequences, and
explicit copy/replacement methods for edits. This keeps IDs, indexes, grouping,
and source snapshots consistent without tracking arbitrary list mutations.

The notation below uses:

- `Number`: `int | float`, excluding booleans and non-finite floats.
- `Integer`: also represented as `int | float`, but validated to have an
  integral value. This retains schema-valid JSON numbers such as `1080.0`
  without coercing their decoded Python type.
- `JSONValue`: recursively, null, boolean, finite number, string, list of JSON
  values, or dictionary with string keys and JSON values.
- `Absent`: a dedicated singleton sentinel type; `ABSENT` is its sole value.
  A one-member `Enum` is a suitable Python 3.10 implementation because identity
  survives copying. It is not a JSON value.
- `extras`: privately owned extension data, exposed only through defensive
  copies or a recursively protected view. The implementation uses protected
  mappings and tuples (`FrozenJSONValue`); it defaults to an empty mapping.

All optional Adobe fields default internally to `ABSENT`, including fields with
schema defaults. Do not use `None` or a default value to mean missing:

| Input state | Internal state | Serialized state |
| --- | --- | --- |
| No `rotation` key | `ABSENT` | No `rotation` key |
| `"rotation": 0` | `0` | `"rotation": 0` |
| No `group_id` key | `ABSENT` | No `group_id` key |
| `"group_id": null` | `None` | `"group_id": null` |
| `"style_runs": []` | Empty tuple | `"style_runs": []` |

Schema defaults are effective read values, not instructions to populate JSON.
An `effective_value(name)` accessor returns the stored value when present,
otherwise the schema default listed below. For an absent field with no schema
default, it returns `ABSENT`. An unknown field name raises `KeyError`. Reading
an effective value never modifies the model. Zero, `False`, empty strings,
empty arrays, and explicit nulls must not be dropped by truthiness checks.

Parsing and serialization recursively copy extension data. A frozen dataclass
alone does not protect nested dictionaries/lists: copying or read-only access
must also prevent callers from mutating model state through `extras`, and edits
must not alias the source model. Returned JSON is always a fresh mutable tree.

## Element

Use one flat `Element` dataclass with common fields and all schema-defined
type-specific fields. The `type` discriminator controls conditional validation.
This avoids losing data through subclass conversion and matches Adobe's flat
element objects. In particular, the schema describes some fields as "text
only" or "image only" but does not forbid their presence on other types. Keep
any such supplied fields and validate their declared value types; do not filter
them out according to the discriminator.

### Common and internal fields

| Field | Internal type / presence | Meaning or constraint |
| --- | --- | --- |
| `id` | `str`, required | Stable, opaque identifier; retain exactly. |
| `type` | `Literal["text", "image", "shape"]`, required | Retain the original type. |
| `x`, `y` | `Number`, required | Left/top position in pixels, relative to canvas top-left. Negative input coordinates are allowed. |
| `width`, `height` | `Number`, required | Pixel dimensions, each at least zero. |
| `rotation` | `Number \| Absent` | Degrees clockwise around the element center; effective default `0`. |
| `opacity` | `Number \| Absent` | Range `[0, 1]`; effective default `1`. |
| `z_index` | `Integer \| Absent` | Higher values paint on top; schema has no default. |
| `group_id` | `str \| None \| Absent` | Opaque group identifier; effective default `None`. |
| `required` | `bool`, internal | Set to `True` for every original element in stage one. Internal metadata, independent of any same-named JSON extension. |
| `extras` | Mapping of `str` to `JSONValue`, internal container | Preserve unrecognized element properties at their original flat JSON locations. |

`required` records retention policy, not whether an element has a visible role
or whether its asset exists. Every original element remains required, including
decorative shapes, zero-area elements, transparent elements, and images whose
files are missing. Stage one has no supported operation to add/remove elements
or set their internal `required` flag to `False`.

### Text fields

Every row is a field on `Element`. "Required for text" means conditional on
`type == "text"`; otherwise the field may be absent. Optional values use
`ABSENT` even where an effective default is listed.

| Field | Present value type / constraint | Presence and effective default |
| --- | --- | --- |
| `content` | `str` | Required for text; no default. |
| `font_family` | `str` | Required for text; no default. |
| `font_size` | `Number >= 1`, pixels | Required for text; no default. |
| `color` | `str`, preserved verbatim | Required for text; also described for shapes; no default. |
| `font_weight` | `Literal["regular", "bold"]` | Optional; `"regular"`. |
| `font_style` | `Literal["normal", "italic"]` | Optional; `"normal"`. |
| `text_align` | `Literal["left", "center", "right"]` | Optional; `"left"`. |
| `line_height` | `Number`, multiplier | Optional; `1.2`. |
| `letter_spacing` | `Number`, pixels | Optional; `0`. |
| `max_lines` | `Integer \| None` | Optional; `None`. |
| `style_runs` | `tuple[StyleRun, ...]` internally, array in JSON | Optional; no schema default. |

Do not add constraints absent from v0.9, such as a positive minimum for
`max_lines` or `line_height`. Preserve the content string, including Unicode and
whitespace, without normalization or rewriting.

Each `StyleRun` is a separate frozen dataclass:

| Field | Present value type / constraint | Presence |
| --- | --- | --- |
| `start`, `end` | `Integer >= 0` | Required; zero-based, end-exclusive indices into `content`. |
| `color` | `str` | Optional, `ABSENT` by default. |
| `font_weight` | `Literal["regular", "bold"]` | Optional, `ABSENT` by default. |
| `font_style` | `Literal["normal", "italic"]` | Optional, `ABSENT` by default. |
| `extras` | Mapping of `str` to `JSONValue` | Internal container for unknown run properties. |

Preserve run order, indices, overlaps, and explicit overrides. An absent run
override inherits the effective element value for interpretation; do not insert
inherited values into the run's output. The schema does not enforce
`start <= end <= len(content)`. Such checks may produce diagnostics but must
not reject or repair otherwise schema-valid input during parsing. The core
sample has `start=6, end=16` for the 15-character string `Chase the Light`;
round-tripping must preserve that run exactly.

### Image and shape fields

| Field | Present value type / constraint | Presence and effective default |
| --- | --- | --- |
| `asset_id` | `str` | Required for images; no default. |
| `fit` | `Literal["contain", "cover", "fill"]` | Optional; `"contain"`. |
| `shape_type` | `Literal["rectangle", "ellipse", "line"]` | Required for shapes; no default. |
| `fill_color` | `str` | Required for shapes; no default. |
| `stroke_color` | `str \| None` | Optional; `None`. |
| `stroke_width` | `Number`, pixels | Optional; `0`. |
| `corner_radius` | `Number`, pixels | Optional; `0`; described for rectangles. |

Shape `color` and `fill_color` remain independent keys; neither is an alias or
replacement for the other. Preserve color strings exactly without converting
between CSS spellings. Preserve `asset_id` exactly without resolving paths,
loading pixels, checking file existence, or substituting assets during parsing.
Image `fit` is distinct from element-box aspect ratio; this model does not infer
an image's intrinsic dimensions from its box.

Only `group_id`, `max_lines`, and `stroke_color` accept explicit null among
schema-defined element fields. Other optional fields are absent or have their
declared non-null type. Unknown properties may contain null at any depth.

### Derived geometry

Derived geometry is computed on demand from stored pixel bounds and never
serialized or cached as independent mutable state:

| API | Definition |
| --- | --- |
| `element.center` | `(x + width / 2, y + height / 2)` in pixels. |
| `element.aspect_ratio` | `width / height` if `height != 0`, otherwise `None`. Zero width with positive height yields `0`. |
| `element.area` | `width * height` in square pixels. |
| `element.normalized_bounds(canvas)` | `(x / canvas.width, y / canvas.height, width / canvas.width, height / canvas.height)`. |

These use the stored, unrotated box. Area means box area, including for ellipses
and lines; it does not measure painted pixels or account for opacity. Rotation
does not change the center. A future rotated-envelope calculation must have a
separate name and must not overwrite the stored bounds. Normalized values can
be negative or greater than one; do not clamp them. Requiring an explicit
`Canvas` argument makes the normalization reference unambiguous.

## Canvas

`Canvas` is a frozen value object with exactly `width: Integer` and
`height: Integer`, in pixels, both at least one. Validate these integral values
without rounding; reject fractional dimensions, booleans, and non-finite values.
Normally the stored values are Python integers. Retain integer-valued floats
when supplied, as explained above. The same integer-validation rule applies to
`z_index`, `max_lines`, and style-run indices.

| Property | Definition |
| --- | --- |
| `aspect_ratio: float` | `width / height`. |
| `area: Integer` | `width * height`. |
| `orientation` | `"landscape"` if width > height, `"portrait"` if height > width, otherwise `"square"`. |

Keep three explicit references on `Design`:

- `source_canvas`: immutable original dimensions, retained throughout resizing.
- `target_canvas: Canvas | None`: requested output dimensions, or `None` when
  parsing a design without a resize request.
- `canvas`: coordinate system of the design's current element layout. Initially
  equal to `source_canvas`; explicitly changed when a target layout is installed.

The source and target may share numeric values but have separate meanings.
Supplying a target does not scale elements or replace `canvas`. `to_adobe()`
always emits the current `canvas`, so merely parsing a source with target context
still round-trips the source. A completed resize must explicitly use
`canvas == target_canvas`; output validation checks this separately.

## Theme

`Theme` is a frozen, derived inventory of existing values:

```python
@dataclass(frozen=True, kw_only=True)
class Theme:
    fonts: tuple[str, ...]
    colors: tuple[str, ...]
    asset_references: tuple[str, ...]
```

`Theme.from_elements(elements)` traverses elements in original order. It collects
present `font_family` strings into `fonts`, and present `asset_id` strings into
`asset_references`. For colors, inspect each element's `color`, `fill_color`,
and non-null `stroke_color`, in that order, then each `style_runs` entry's
present `color` in run order. Collect known fields wherever supplied; do not
mine arbitrary extension objects, comments, IDs, or text for theme values.

Each inventory removes exact duplicates while preserving first occurrence.
Keep case, spacing, and CSS syntax unchanged; equivalent-looking colors with
different strings remain distinct. Font weight/style remain on elements/runs;
they do not create new font-family names. Missing fields, null strokes, and
inherited/default values do not add inventory entries. Asset references remain
listed even if their files are missing.

Do not assign primary/secondary colors, heading/body fonts, logo roles, brand
names, palette weights, or inferred asset categories. The sample identifiers
are arbitrary IDs, not semantic annotations. Do not inspect asset pixels or
invent brand-kit information. `Design.theme` is computed from its current
elements on access, avoiding stale metadata after replacement. It has no Adobe
serializer and is never included in the output JSON.

## Design, ordering, and grouping

`Design` combines the current `Canvas`, `tuple[Element, ...]`, and derived
`Theme`, with the separate source/target canvas context described above.

The element tuple is authoritative for ordering. Parse and serialize in input
order, without sorting by ID, type, group, position, or `z_index`. Stacking
information is preserved independently; this model does not invent a missing
`z_index` or define a new renderer ordering policy.

Expose `elements_by_id: Mapping[str, Element]` as a read-only lookup and
`get_element(element_id)` as a convenience that raises `KeyError` for an unknown
ID. Rebuild the mapping for each new `Design`; never serialize it. Reject
duplicate IDs with a clear error instead of silently overwriting a mapping
entry. This is a stage-one identity requirement beyond the JSON Schema's field
validation. The schema also requires at least one element.

Expose `groups: Mapping[str, tuple[Element, ...]]` as a derived read-only view.
Members with identical string `group_id` values share a group, in their original
relative order. Group keys follow first occurrence. Absent and null `group_id`
both mean ungrouped in this view, while retaining distinct serialized states.
An empty string is a valid group identifier, so do not use truthiness to decide
membership. Do not create parent elements, nested output groups, or group names
from apparent roles. Group membership and the original flat ordering remain
unchanged by stage-one layout edits.

An immutable source `Design` remains available to compare against candidate
layouts. Store source elements only once in that source instance; the candidate
need not duplicate a full original JSON snapshot. Replacing elements constructs
a new design, keeping source/target context and rebuilding derived lookups.

## Python APIs and deferred editing proposal

The signatures below describe the complete S02 interface proposal. All except
the three editing/validation methods marked deferred are now implemented in
`engine/internal_schema.py`. `Mapping`, `Iterable`, `Literal`, and
`Any` are available in Python 3.10; postponed annotations can handle forward
references. `JSONValue` is the recursive type described above.

```python
# Canvas
@classmethod
def from_adobe(cls, data: Mapping[str, JSONValue]) -> Canvas: ...
def to_adobe(self) -> dict[str, JSONValue]: ...

# StyleRun and Element each provide their own from_adobe/to_adobe pair.
# Element.from_adobe always initializes the INTERNAL required flag to True.
@classmethod
def from_adobe(cls, data: Mapping[str, JSONValue]) -> Element: ...
def to_adobe(self) -> dict[str, JSONValue]: ...
def effective_value(
    self, field_name: str,
) -> Number | str | None | tuple[StyleRun, ...] | Absent: ...
def normalized_bounds(self, canvas: Canvas) -> tuple[float, float, float, float]: ...
# Deferred: not available in the foundation milestone.
def with_updates(self, **changes: Any) -> Element: ...

# Theme
@classmethod
def from_elements(cls, elements: Iterable[Element]) -> Theme: ...

# Design
@classmethod
def from_adobe(
    cls,
    source: Mapping[str, JSONValue],
    *,
    target_canvas: Mapping[str, JSONValue] | None = None,
) -> Design: ...
def to_adobe(self) -> dict[str, JSONValue]: ...
def get_element(self, element_id: str) -> Element: ...
# Deferred: not available in the foundation milestone.
def with_layout(self, *, canvas: Canvas, elements: Iterable[Element]) -> Design: ...
def validate_stage_one(self, *, source: Design) -> None: ...
```

`with_updates` is a validated, copying wrapper around `dataclasses.replace`.
It accepts declared Adobe field names, rejects unknown names, and prohibits
changes to `id`, `type`, and internal fields. Setting an optional field to
`ABSENT` explicitly removes that JSON key; setting a nullable field to `None`
retains a JSON null. Required fields cannot become absent. Copy any supplied
style runs and keep extras unchanged. This general field-edit API does not by
itself certify that edits comply with stage-one policy.

`with_layout` installs already-computed pixel geometry; it performs no scaling
or role inference. It requires exactly the same ordered IDs, types, `required`
flags, group-field presence/values, and extension data as the current design,
and retains the original source/target canvas context.
`validate_stage_one(source=...)` compares
against the original source as well, including fixed content, font families,
and asset references. Keep this policy check separate from lossless parsing and
serialization; source fixtures need not satisfy target-layout constraints.

Illustrative future use (the candidate portion requires deferred APIs), with
no change to `resize.py` in this milestone:

```python
source_design = Design.from_adobe(source, target_canvas=target_canvas)
assert source_design.to_adobe() == source
assert all(element.required for element in source_design.elements)

# ID is supplied by the caller; its spelling does not imply a role.
element = source_design.get_element(element_id)
fractional_box = element.normalized_bounds(source_design.source_canvas)

# A later layout algorithm supplies these replacement elements in source order.
assert source_design.target_canvas is not None
candidate = source_design.with_layout(
    canvas=source_design.target_canvas,
    elements=laid_out_elements,
)
candidate.validate_stage_one(source=source_design)
result = candidate.to_adobe()
assert set(result) == {"canvas", "elements"}
```

## Serialization and validation boundaries

Use explicit schema-key allowlists and serializers, never `dataclasses.asdict`
as the output format. `asdict` would expose internal fields and cannot implement
absence handling or flatten extension data correctly.

Parsing splits element keys into schema-defined fields and `extras`. Style runs
use their own schema-key set and extras. The schema allows additional element
and style-run properties, but explicitly forbids extra root or canvas keys;
reject the latter with a path-specific error rather than silently discarding
them. Unknown element/run values are copied recursively without interpretation.

Serialization emits required Adobe fields, emits optional known fields only
when they are not `ABSENT`, serializes style runs back to arrays of flat objects,
and merges copied extras at their original level. Extras must not contain a
schema-defined key; reject conflicting manually constructed models rather than
letting extras overwrite an edited field. No `extras` wrapper is emitted.

An extension key can happen to be named `required`, `theme`, `center`, or
`extras`. Preserve that original JSON property in the extension container; it
does not set the corresponding internal state. "No internal fields in output"
means never add internal metadata, not discard an existing legal extension
whose name coincides with an internal attribute. Use the Adobe field allowlist,
not dataclass attribute names, when separating known fields from extras.

Validate schema-required fields, supported element types, enums, numeric bounds,
integer values, nullability, and JSON-compatible extras without coercion.
Report invalid data with a useful path such as `elements[2].opacity`. Do not
silently repair values, fill optional defaults, sort arrays, truncate runs,
clip positions, rewrite colors, or discard missing-asset elements.

Output policy is a separate concern: every original ID exactly once, preserved
order/type/grouping, all originals required, target dimensions selected,
unchanged stage-one content/font families/assets, in-bounds placement, and
appropriate image proportions. `resize.py` describes an approximately 5%
source-box proportion tolerance with full-bleed/`fit: "cover"` exceptions.
Any full-bleed decision must use explicit geometry/fit evidence, never an ID
such as `bg`. Detailed rotated-extent and image-fit checks belong to later
layout validation; S02 defines their inputs without implementing a solver or
changing rendering. `required=True` does not certify successful placement.

## Sample implications and future acceptance checks

| Fixture | Source -> target; element count | Representation checks |
| --- | --- | --- |
| Warm-up | `1080 x 1080 -> 1080 x 1350`; 4 | Text/image fields, source order, omitted common/text defaults. |
| Core | `1080 x 1080 -> 1600 x 400`; 11 | Negative `x/y`, opacity, `_note`, style runs, rounded shapes, two explicit groups. |
| Advanced | `1080 x 1350 -> 728 x 90`; 17 | `rotation=-12`, five explicit groups, long literal text, `_adversarial_note`, missing asset with omitted `fit`. |

The core sample's first decorative circle starts at `(-100, -100)`; parsing
must accept this even though a final target layout is expected to fit within
its canvas. The advanced sample's `solstice_holiday_badge_2025.png` reference
must remain present and required. The sample note's suggestion to skip a
missing asset does not authorize dropping the element from output: the
same-ID contract in the schema, rules, and resize boundary takes precedence.
Asset availability and renderer placeholders are outside this data model.

Foundation acceptance criteria (controlled editing checks remain deferred):

1. Round-trip each sample's `source` with and without target context; assert
   recursive equality, key presence, decoded numeric types, element/run order,
   unchanged caller-owned input, and unchanged source/target dimensions.
2. Exercise every schema field and enum, including fields absent from samples:
   explicit nulls, empty runs, `font_style`, line spacing, `max_lines`, all image
   fits, strokes, shape `color`, rectangles, ellipses, and zero-height lines.
3. Distinguish absent fields from explicit defaults/nulls/empty values. Reading
   effective defaults, theme, indexes, groups, or geometry adds no output keys.
4. Preserve nested unknown element/run properties, including names colliding
   with internal attributes. Reject schema-key collisions in constructed extras,
   duplicate IDs, invalid known values, and unknown root/canvas fields clearly.
5. Verify controlled edits change only requested Adobe values, optional removal
   works, serialization returns independent nested data, and neither input nor
   source state changes through a candidate or an exported dictionary.
6. Check canvas orientation/ratios, zero-size element geometry, normalization
   against explicitly chosen canvases, stable group membership, and ID lookup
   after replacement. Reject loss/reordering of elements in `with_layout`.
7. Check stable theme deduplication, run colors, null-stroke exclusion, and the
   retained missing asset reference, without inferred semantic roles.

The implemented foundation has tests in
[tests/test_internal_schema.py](../tests/test_internal_schema.py). Checks for
controlled edits, optional removal, `with_layout`, and full stage-one layout
validation remain requirements for a later implementation.
