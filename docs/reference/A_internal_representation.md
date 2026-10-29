# Appendix A — S02 internal design representation

> **Appendix A of [TECHNICAL_SPEC.md](../TECHNICAL_SPEC.md).** This appendix is the authoritative
> detail for S02 parsing, immutability, absence semantics, derived geometry and serialization.
> Implementation status, cross-component signatures and output-policy constraints are defined only in
> the main specification (§2, §3, §8); they are not repeated here.

Code: [engine/internal_schema.py](../../engine/internal_schema.py). Tests: [tests/test_internal_schema.py](../../tests/test_internal_schema.py).

## A.1 Public interface

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

## A.2 Scope

Direct `Design` construction accepts explicit current/source/target canvases and typed elements, but
does not certify a layout against an original design; `dataclasses.replace` likewise enforces no
cross-design retention policy. Retention and output policy are checked only by `validate_candidate`
(spec §8.1).

The advanced fixture's note suggesting that a missing asset be skipped conflicts with the schema's
same-ID contract; S02 preserves the element and reference. The core fixture's out-of-range style run
and negative coordinates are schema-permitted input and are preserved. No source-target feasibility
or image-proportion checks are applied during parsing.

### A.2.1 Contract and scope

The internal model provides typed access to Adobe v0.9 designs while preserving
their JSON structure. Here, **normalized** means a consistent internal data
model; stored coordinates remain in pixels. Fractional coordinates relative to
a canvas are derived views, never replacements for the pixel values.

This specification is based on:

- [Adobe's design schema](../../chameleon_design_schema_v0.9.json), including its
  conditional required fields and additional-property rules.
- [RULES.md](../../RULES.md) and [FAQ.md](../../FAQ.md), especially element retention,
  fixed text/font families, asset preservation, and the absence of semantic labels.
- [engine/resize.py](../../engine/resize.py), whose boundary is
  `resize(source, target_canvas) -> {"canvas": ..., "elements": [...]}`.
- All three [warm-up](../../input/chameleon_sample_level1_warmup.json),
  [core](../../input/chameleon_sample_level2_core.json), and
  [advanced](../../input/chameleon_sample_level3_advanced.json) sample tasks.

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

## A.3 Representation and presence

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

## A.4 Element

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

## A.5 Canvas

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

## A.6 Theme

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

## A.7 Design, ordering, and grouping

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


## A.8 Editing APIs (planned)

`Element.with_updates(**changes)` and `Design.with_layout(*, canvas, elements)` are **planned** and not
present in `engine/internal_schema.py`.

```python
def with_updates(self, **changes: Any) -> Element: ...                                  # Element
def with_layout(self, *, canvas: Canvas, elements: Iterable[Element]) -> Design: ...    # Design
```

`with_updates` is a validated, copying wrapper around `dataclasses.replace`. It accepts declared Adobe
field names, rejects unknown names, and prohibits changes to `id`, `type`, and internal fields. Setting
an optional field to `ABSENT` removes that JSON key; setting a nullable field to `None` retains a JSON
null. Required fields cannot become absent. Supplied style runs are copied and extras are unchanged.

`with_layout` installs already-computed pixel geometry; it performs no scaling or role inference. It
requires exactly the same ordered IDs, types, `required` flags, group-field presence/values and
extension data as the current design, and retains the original source/target canvas context. These are
structural construction checks for a well-formed `Design`; **neither method certifies output policy**.
Policy certification is `validate_candidate` only (spec §8.1).

Illustrative future use (requires the planned methods):

```python
source_design = Design.from_adobe(source, target_canvas=target_canvas)
assert source_design.to_adobe() == source
assert all(element.required for element in source_design.elements)
fractional_box = source_design.get_element(element_id).normalized_bounds(source_design.source_canvas)
candidate = source_design.with_layout(canvas=source_design.target_canvas, elements=laid_out_elements)
report = validate_candidate(source_design, candidate, source_design.target_canvas)
```

## A.9 Serialization and validation of input

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

Output policy (retention, bounds, image proportions, mutability of fields) is defined in spec §8.1.
`required=True` does not certify successful placement. Rotated-extent and image-fit checks belong to
`validate_candidate`; S02 defines their inputs only.

## A.10 Fixtures and S02 acceptance criteria

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

S02 acceptance criteria (items 5 and 6 depend on the planned editing APIs in A.8; the other items apply to the implemented foundation):

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
