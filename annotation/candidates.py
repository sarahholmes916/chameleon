"""Candidate-set manifest contract and validated import into the annotation database.

Any generator (the S05 dev generator now, S04 later) only has to write `candidate_manifest.jsonl` in this
format. The importer trusts nothing in a record: it re-hashes the source and candidate JSON, re-runs the
S02 `validate_candidate` against the real source, and checks the preview PNG size.

Record (one JSON object per line; paths relative to the repository root, or absolute):
  schema_version "candidate_manifest.v1", candidate_id "sha256:<hash of canonical candidate JSON>",
  source_design_id "sha256:<hash of canonical source JSON>", template_family_id,
  source_path, source_preview_path (optional), target_canvas {width,height}, candidate_path, preview_path,
  generator_id, generator_version, renderer_version, set_id, provenance {...}
"""

from __future__ import annotations

import json
from pathlib import Path

from engine.candidate_validator import validate_candidate
from engine.internal_schema import Canvas, Design
from annotation.store import now_utc, tx
from scripts.source_dataset import digest

MANIFEST_NAME = "candidate_manifest.jsonl"
MANIFEST_VERSION = "candidate_manifest.v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ("candidate_id", "source_design_id", "template_family_id", "source_path", "target_canvas", "candidate_path",
            "preview_path", "generator_id", "generator_version", "renderer_version", "set_id")


def resolve(path: str | Path, base: Path = REPO_ROOT) -> Path:
    p = Path(path)
    return p if p.is_absolute() else base / p


def candidate_id_of(candidate: dict) -> str:
    return "sha256:" + digest(candidate)


def check_record(record: dict, base: Path = REPO_ROOT) -> list[str]:
    """All reasons this record may not enter a tournament; empty means it is valid."""
    from PIL import Image

    missing = [k for k in REQUIRED if k not in record]
    if missing:
        return [f"missing fields: {missing}"]
    problems = []
    try:
        source = json.loads(resolve(record["source_path"], base).read_text(encoding="utf-8"))
        candidate = json.loads(resolve(record["candidate_path"], base).read_text(encoding="utf-8"))
        target = Canvas.from_adobe(record["target_canvas"])
        if "sha256:" + digest(source) != record["source_design_id"]:
            problems.append("source JSON does not match source_design_id")
        if candidate_id_of(candidate) != record["candidate_id"]:
            problems.append("candidate JSON does not match candidate_id")
        report = validate_candidate(Design.from_adobe(source), candidate, target)
        if not report.is_valid:
            problems.extend(f"validator: {v}" for v in report.violations)
        if set(candidate) != {"canvas", "elements"}:
            problems.append("candidate root must be exactly canvas and elements")
        with Image.open(resolve(record["preview_path"], base)) as image:
            image.load()
            if image.format != "PNG" or image.size != (target.width, target.height):
                problems.append(f"preview must be a PNG of {target.width}x{target.height}, got {image.format} {image.size}")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        problems.append(f"{type(exc).__name__}: {exc}")
    return problems


def import_manifest(conn, manifest: Path | str, base: Path = REPO_ROOT) -> dict:
    """Import every valid record; invalid ones are reported and never stored. Re-import is idempotent."""
    imported, rejected = [], []
    rows = [json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    with tx(conn):
        for record in rows:
            problems = check_record(record, base)
            if problems:
                rejected.append(dict(candidate_id=record.get("candidate_id"), problems=problems))
                continue
            target = record["target_canvas"]
            conn.execute(
                "INSERT OR IGNORE INTO candidates VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (record["candidate_id"], record["source_design_id"], record["template_family_id"], target["width"],
                 target["height"], str(record["source_path"]), record.get("source_preview_path"),
                 str(record["candidate_path"]), str(record["preview_path"]), record["generator_id"],
                 record["generator_version"], record["renderer_version"],
                 json.dumps(record.get("provenance", {}), sort_keys=True), record["set_id"], now_utc()))
            imported.append(record["candidate_id"])
    return dict(imported=imported, rejected=rejected)


def candidate_groups(conn) -> list[dict]:
    """Importable pools: candidates sharing one source design and one target canvas."""
    rows = conn.execute(
        "SELECT source_design_id, template_family_id, target_width, target_height, set_id, COUNT(*) n, "
        "MIN(source_path) source_path FROM candidates GROUP BY source_design_id, target_width, target_height, set_id "
        "ORDER BY set_id").fetchall()
    return [dict(r) for r in rows]


def group_candidate_ids(conn, source_design_id: str, width: int, height: int) -> list[str]:
    return [r["candidate_id"] for r in conn.execute(
        "SELECT candidate_id FROM candidates WHERE source_design_id=? AND target_width=? AND target_height=? "
        "ORDER BY candidate_id", (source_design_id, width, height))]
