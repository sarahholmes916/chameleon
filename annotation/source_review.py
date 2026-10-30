"""Source-design review state, reusing scripts.source_dataset.apply_review and its review.json/approved.jsonl.

A decision here is the same durable record as the static gallery's download: it is bound to the manifest
hash, never auto-approves, and keeps every other source's existing decision.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts import source_dataset as sd

STATUSES = ("pending", "approved", "rejected")


def _decisions(root: Path) -> dict[str, dict]:
    """Current decisions; a review of a different manifest is treated as no review (all pending)."""
    path = root / "review.json"
    if not path.is_file():
        return {}
    review = json.loads(path.read_text(encoding="utf-8"))
    if review.get("manifest_sha256") != sd.file_hash(root / "manifest.jsonl"):
        return {}
    return {d["source_design_id"]: dict(status=d["status"], reason=d.get("reason", "")) for d in review["decisions"]}


def load_sources(root: Path | str) -> list[dict]:
    root = Path(root)
    decisions = _decisions(root)
    out = []
    for record in sd.read_records(root):
        decision = decisions.get(record["source_design_id"], dict(status="pending", reason=""))
        out.append(dict(
            source_design_id=record["source_design_id"], template_family_id=record["template_family_id"],
            canvas=record["canvas"], element_count=record["element_count"], gen_status=record["status"],
            eligible=record["status"] == "rendered", status=decision["status"] if record["status"] == "rendered" else "ineligible",
            reason=decision["reason"], source_path=root / record["source_path"],
            preview_path=(root / record["preview_path"]) if record["preview_path"] else None))
    return out


def set_decision(root: Path | str, source_design_id: str, status: str, reason: str = "") -> int:
    """Record one decision, preserving all others. Returns the number of approved sources."""
    root = Path(root)
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    current = _decisions(root)
    current[source_design_id] = dict(status=status, reason=reason)
    eligible = {r["source_design_id"] for r in sd.read_records(root) if r["status"] == "rendered"}
    decisions = [dict(source_design_id=sid, status=d["status"], reason=d["reason"])
                 for sid, d in current.items() if sid in eligible]
    return sd.apply_review(root, dict(manifest_sha256=sd.file_hash(root / "manifest.jsonl"), decisions=decisions))


def review_status(source_path: str | Path, source_design_id: str) -> str:
    """'approved'/'rejected'/'pending' for sources inside a generated dataset, else 'unknown'."""
    source_path = Path(source_path)
    if source_path.parent.name != "sources" or not (source_path.parent.parent / "manifest.jsonl").is_file():
        return "unknown"
    return _decisions(source_path.parent.parent).get(source_design_id, dict(status="pending"))["status"]
