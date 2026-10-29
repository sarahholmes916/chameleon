"""Generator-side allowlisted edits for candidate layouts (docs/TECHNICAL_SPEC.md section 8.1.6).

Candidate generators build element dicts through `layout_edit`, so a generator cannot produce a
protected-property change at all. This is defence in depth, not the authority: `validate_candidate`
still decides whether a candidate is valid. The allowlist is not defined here; it comes from
`engine.candidate_validator.mutable_fields` (K6).
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

from engine.candidate_validator import mutable_fields


class ProtectedFieldError(ValueError):
    """A generator tried to change a field that candidates may not change."""


def layout_edit(element: Mapping[str, Any], changes: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep copy of an Adobe element dict with `changes` applied.

    Only fields in `mutable_fields(element["type"])` may be changed: x, y, width, height, and
    font_size on text elements. Any other field, including font_size on a non-text element, raises
    `ProtectedFieldError` and nothing is returned. The source element is never mutated; every other
    property, including nested dictionaries, arrays and extension data, is copied unchanged. Values
    are not checked here, since the validator is the authority on whether a layout is acceptable.
    """
    element_type = element.get("type")
    if not isinstance(element_type, str):
        raise ValueError("element must have a string 'type'")
    allowed = mutable_fields(element_type)
    rejected = sorted(str(name) for name in changes if name not in allowed)
    if rejected:
        raise ProtectedFieldError(
            f"element {element.get('id')!r} (type {element_type!r}): cannot change {rejected}; "
            f"only {sorted(allowed)} may change"
        )
    edited = copy.deepcopy(dict(element))
    edited.update(copy.deepcopy(dict(changes)))
    return edited
