"""Portable assignments, balanced pairs, isolated databases and relocation."""
import copy
import json
from pathlib import Path
import shutil

from PIL import Image
import pytest

from annotation import candidates, store, work_queue
from annotation.assignment import load_assignment, prepare_assignment
from scripts.package_rankings import select_tasks, split_tasks, write_package
from tests import test_annotation

world = test_annotation.world


@pytest.fixture
def task(world, tmp_path):
    rows = [json.loads(line) for line in world[2].manifest.read_text().splitlines()]
    original = tmp_path / "source.png"
    Image.new("RGB", (200, 200), "white").save(original)
    for r in rows:
        r["source_preview_path"] = str(original)
    for name, selected in (("old", rows[:4]), ("new", rows[4:])):
        path = tmp_path / name
        path.mkdir()
        (path / candidates.MANIFEST_NAME).write_text("".join(json.dumps(r) + "\n" for r in selected))
    return select_tasks(tmp_path / "old", tmp_path / "new")[0]


def test_assignment_portable_idempotent_and_separate(task, tmp_path):
    folder = tmp_path / "alice"
    write_package(folder, [task], set(), "alice", include_app=False)
    config, rows = load_assignment(folder)
    assert len(rows) == 4 and all(not candidates.check_record(r) for r in rows)
    db, name = prepare_assignment(folder)
    conn = store.connect(db)
    tid = store.list_tournaments(conn)[0]["tournament_id"]
    m = store.load_state(conn, tid).next_match()
    store.record_judgment(conn, tid, m.match_id, "LEFT", presented_left=m.left, presented_right=m.right,
                          token=m.token, annotator_id=name)
    conn.close()
    prepare_assignment(folder)
    relocated = tmp_path / "elsewhere/alice"
    shutil.copytree(folder, relocated)
    moved_db, _ = prepare_assignment(relocated)
    conn = store.connect(moved_db)
    assert conn.execute("SELECT COUNT(*) FROM judgments").fetchone()[0] == 1
    assert len(store.list_tournaments(conn)) == 1
    assert Path(conn.execute("SELECT preview_path FROM candidates LIMIT 1").fetchone()[0]).is_relative_to(relocated)
    conn.close()
    other = tmp_path / "bob"
    write_package(other, [task], set(), "bob", include_app=False)
    conn = store.connect(prepare_assignment(other)[0])
    assert conn.execute("SELECT COUNT(*) FROM judgments").fetchone()[0] == 0
    conn.close()


def test_assignment_halves_all_candidates_without_overlap(task):
    tasks = []
    for cohort, count, size in (("old", 60, 8), ("s04", 63, 6)):
        for i in range(count):
            t = copy.deepcopy(task)
            ids = [f"{cohort}_{i}_{j}" for j in range(size)]
            t.update(task_id=f"{cohort}_{i}", cohort=cohort, candidate_ids=ids,
                     rows=[{"candidate_id": cid} for cid in ids])
            tasks.append(t)
    result, shared = split_tasks(tasks)
    assert not shared
    sets = [{cid for t in part for cid in t["candidate_ids"]} for part in result]
    assert not sets[0] & sets[1]
    assert sets[0] | sets[1] == {cid for t in tasks for cid in t["candidate_ids"]}
    for part in result:
        assert len(part) == 62
        assert sum(len(t["candidate_ids"]) for t in part if t["cohort"] == "old") == 240
        assert sum(len(t["candidate_ids"]) for t in part if t["cohort"] == "s04") == 189
        assert sum(len(t["candidate_ids"]) - 1 for t in part) == 367
    assert split_tasks(tasks) == (result, shared)


@pytest.mark.parametrize("size", [2, 3, 5, 6, 7, 8])
def test_arbitrary_brackets_include_every_candidate_and_queue_every_set(world, size):
    conn = store.connect(world[0] / f"size{size}.db")
    candidates.import_manifest(conn, world[2].manifest)
    ids = [r[0] for r in conn.execute("SELECT candidate_id FROM candidates")][:size]
    store.set_meta(conn, "assignment_id", "test")
    tids = [store.create_tournament(conn, ids, size=size, seed=42, annotator_id="test",
                                   source_review_status="unreviewed") for _ in range(2)]
    for _ in tids:
        tid, _ = work_queue.next_task(conn)
        assert tid is not None
        seen, count = set(), 0
        while not store.load_state(conn, tid).complete:
            state = store.load_state(conn, tid)
            m = state.next_match()
            assert m is not None
            seen.update((m.a, m.b))
            store.record_judgment(conn, tid, m.match_id, "LEFT", presented_left=m.left,
                                  presented_right=m.right, token=m.token, annotator_id="test")
            count += 1
        assert count == size - 1
        assert seen == set(ids)
        assert store.progress(store.load_state(conn, tid)) == (size - 1, size - 1)
    assert work_queue.next_task(conn)[0] is None
    assert conn.execute("SELECT COUNT(*) FROM judgments").fetchone()[0] == 2 * (size - 1)
    conn.close()


def test_assignment_detects_changed_assets(task, tmp_path):
    folder = tmp_path / "alice"
    write_package(folder, [task], set(), "alice", include_app=False)
    config = json.loads((folder / "assignment.json").read_text())
    (folder / next(iter(config["files"]))).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        prepare_assignment(folder)
    assert not (folder / "rankings.db").exists()


def test_assignment_ui_cannot_import_other_datasets(task, tmp_path, monkeypatch):
    from annotation import app
    folder = tmp_path / "alice"
    write_package(folder, [task], set(), "alice", include_app=False)
    db, name = prepare_assignment(folder)
    monkeypatch.setattr(app.work_queue, "sync", lambda *_: pytest.fail("must not import unrelated data"))
    demo = app.build_app(db, assignment_root=folder, default_annotator=name)
    try:
        handlers = {getattr(fn.fn, "__name__", ""): fn for fn in demo.fns.values()}
        result = handlers["on_start_queue"].fn(name, "Fit")
        assert "3 comparisons remaining" in result[3]
        assert "1 brackets remaining" in result[3]
        assert result[18]["status"] == "ready"
        assert "Original source design" in result[19]
        assert "limited" in handlers["on_import"].fn("unrelated.jsonl")[0]
    finally:
        demo.close()


def test_assignment_progress_updates_and_excludes_rejected_sources(task, tmp_path):
    folder = tmp_path / "progress"
    write_package(folder, [task], set(), "alice", include_app=False)
    conn = store.connect(prepare_assignment(folder)[0])
    tid = store.list_tournaments(conn)[0]["tournament_id"]
    m = store.load_state(conn, tid).next_match()
    store.record_judgment(conn, tid, m.match_id, "TIE", presented_left=m.left,
                          presented_right=m.right, token=m.token, annotator_id="alice")
    assert "3 comparisons remaining" in work_queue.assignment_progress(conn)
    assert "1 saved responses still need" in work_queue.assignment_progress(conn)
    store.resolve_match(conn, tid, m.match_id, m.a, "alice")
    assert "2 comparisons remaining" in work_queue.assignment_progress(conn)
    store.undo_last(conn, tid, "alice")
    assert "3 comparisons remaining" in work_queue.assignment_progress(conn)
    work_queue.reject_source(conn, tid)
    progress = work_queue.assignment_progress(conn)
    assert "0 comparisons remaining" in progress
    assert "0 brackets remaining" in progress
    assert "1 excluded" in progress
    conn.close()
