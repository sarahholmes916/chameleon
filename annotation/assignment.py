"""Portable ranking assignment loader: scoped candidates, independent resumable labels."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

from annotation import candidates, store

PATH_FIELDS = ("source_path", "source_preview_path", "candidate_path", "preview_path")


def load_assignment(root: Path):
    root = root.resolve()
    config = json.loads((root / "assignment.json").read_text(encoding="utf-8"))
    for relative, expected in config["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("assignment path escapes its folder")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"assignment file changed: {relative}")
    rows = [json.loads(line) for line in (root / "candidate_manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        for field in PATH_FIELDS:
            if row.get(field):
                path = (root / row[field]).resolve()
                if not path.is_relative_to(root):
                    raise ValueError("candidate path escapes assignment folder")
                row[field] = str(path)
        problems = candidates.check_record(row)
        if problems:
            raise ValueError(f"{row['candidate_id']}: {problems}")
    return config, rows


def prepare_assignment(root: Path):
    root = root.resolve()
    config, rows = load_assignment(root)
    db = root / "rankings.db"
    conn = store.connect(db)
    try:
        previous = store.get_meta(conn, "assignment_id")
        if previous and previous != config["assignment_id"]:
            raise ValueError("database belongs to a different assignment")
        if not previous and conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]:
            raise ValueError("assignment requires an empty or matching database")
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "candidate_manifest.jsonl"
            manifest.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            imported = candidates.import_manifest(conn, manifest)
            if imported["rejected"]:
                raise ValueError(str(imported["rejected"]))
        # Rebase paths after moving an entire assignment; preserve every judgment.
        with store.tx(conn):
            for row in rows:
                conn.execute("UPDATE candidates SET source_path=?, source_preview_path=?, candidate_path=?, preview_path=? WHERE candidate_id=?",
                             tuple(row.get(k) for k in PATH_FIELDS) + (row["candidate_id"],))
            store.set_meta(conn, "assignment_id", config["assignment_id"])
        for task in config["tasks"]:
            tid = config["annotator_id"] + "_" + task["task_id"]
            if not conn.execute("SELECT 1 FROM tournaments WHERE tournament_id=?", (tid,)).fetchone():
                store.create_tournament(conn, task["candidate_ids"], size=len(task["candidate_ids"]), seed=task["bracket_seed"],
                                        annotator_id=config["annotator_id"], source_review_status="unreviewed",
                                        tournament_id=tid)
        store.set_meta(conn, "automatic_queue", "yes")
    finally:
        conn.close()
    return db, config["annotator_id"]
