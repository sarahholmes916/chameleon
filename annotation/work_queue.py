"""Persistent source/target queue; source approval is optional for annotation."""

from pathlib import Path

from annotation import candidates, source_review, store
from annotation.bracket import SIZES


def sync(conn, root: Path) -> list[str]:
    """Validate discovered batches; keep bad batches visible as diagnostics."""
    problems = []
    for path in sorted((root / "data/candidates").rglob(candidates.MANIFEST_NAME)):
        try:
            result = candidates.import_manifest(conn, path)
            if result["rejected"]:
                problems.append(f"{path.parent.name}: {len(result['rejected'])} candidates rejected")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            problems.append(f"{path}: {exc}")
    return problems


def rejected(conn, source_id: str) -> bool:
    return store.get_meta(conn, "queue_rejected_source:" + source_id) == "yes"


def reject_source(conn, tid: str) -> None:
    source_id = store.load_state(conn, tid).tournament["source_design_id"]
    store.set_meta(conn, "queue_rejected_source:" + source_id, "yes")


def next_task(conn, annotator="local_annotator") -> tuple[str | None, str]:
    if store.get_meta(conn, "assignment_id"):
        rows = conn.execute("SELECT tournament_id FROM tournaments ORDER BY source_design_id, target_width, target_height, tournament_id").fetchall()
        done = excluded = 0
        for row in rows:
            state = store.load_state(conn, row[0])
            if rejected(conn, state.tournament["source_design_id"]):
                excluded += 1
            elif state.complete:
                done += 1
            else:
                store.set_meta(conn, "last_tournament", row[0])
                return row[0], f"Queue: {done} completed sets; {excluded} excluded; {len(rows)} total. Progress saves automatically."
        return None, f"Queue finished: {done} completed sets; {excluded} excluded."
    groups = conn.execute(
        "SELECT source_design_id, target_width, target_height, MIN(source_path) source_path "
        "FROM candidates GROUP BY source_design_id, target_width, target_height "
        "ORDER BY source_path, target_width, target_height").fetchall()
    done = excluded = 0
    for group in groups:
        sid, width, height = group["source_design_id"], group["target_width"], group["target_height"]
        if rejected(conn, sid):
            excluded += 1
            continue
        rows = conn.execute(
            "SELECT tournament_id FROM tournaments WHERE source_design_id=? AND target_width=? AND target_height=? "
            "ORDER BY created_at_utc", (sid, width, height)).fetchall()
        if rows:
            states = [store.load_state(conn, row[0]) for row in rows]
            if any(state.complete for state in states):
                done += 1
                continue
            tid = rows[0][0]
        else:
            ids = candidates.group_candidate_ids(conn, sid, width, height)
            sizes = [size for size in SIZES if size <= len(ids)]
            if not sizes:
                excluded += 1
                continue
            status = source_review.review_status(candidates.resolve(group["source_path"]), sid)
            if status in ("rejected", "ineligible"):
                excluded += 1
                continue
            tid = store.create_tournament(conn, ids, size=max(sizes), seed=42,
                                          annotator_id=annotator, source_review_status=status)
        store.set_meta(conn, "last_tournament", tid)
        return tid, f"Queue: {done} completed sets; {excluded} excluded; {len(groups)} total. Progress saves automatically."
    return None, f"Queue finished: {done} completed sets; {excluded} excluded (rejected sources or fewer than 4 candidates)."
