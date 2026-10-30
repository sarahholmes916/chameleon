"""S05 tournament annotation: bracket logic, SQLite persistence, import validation, export, source review.

Uses a fake renderer so no real rendering is needed; the Gradio wiring test needs requirements-annotation.txt
and fails (never skips) if it is missing.
"""

import ast
import copy
import csv
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest
from PIL import Image

from annotation import bracket, candidates, export, source_review, store, views
from annotation.dev_candidates import GENERATOR_VERSION, POOR_KINDS, build_candidate_set, invalid_cases
from engine.candidate_validator import validate_candidate
from engine.internal_schema import Canvas, Design
from scripts import source_dataset as sd
from scripts.source_templates import make_source

TARGET = {"width": 1600, "height": 400}


class FakeRenderer:
    version = {"package": "fake", "commit": "0" * 40, "converter_sha256": "1" * 64}

    def fit_text(self, source):
        return copy.deepcopy(source), []

    def render(self, source, path):
        digest = hashlib.sha256(sd.canonical(source)).digest()
        image = Image.new("RGB", (source["canvas"]["width"], source["canvas"]["height"]), tuple(digest[:3]))
        for i, e in enumerate(source["elements"][:12]):  # make renders differ when layouts differ
            x, y = int(e["x"]) % image.width, int(e["y"]) % image.height
            image.putpixel((x, y), tuple(digest[3 + i:6 + i]))
        image.save(path)
        return []


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("annotation-world")
    source_path = root / "source.json"
    sd.write_json(source_path, make_source("information_cards", 42))
    result = build_candidate_set(source_path, TARGET, root / "set", renderer=FakeRenderer(), seed=7, count=8, poor=2)
    return root, source_path, result


@pytest.fixture
def db(tmp_path, world):
    conn = store.connect(tmp_path / "annotations.db")
    outcome = candidates.import_manifest(conn, world[2].manifest)
    assert len(outcome["imported"]) == 8 and not outcome["rejected"]
    return conn, outcome["imported"], tmp_path / "annotations.db"


def new_tournament(conn, ids, size=8, seed=1, **kw):
    return store.create_tournament(conn, ids, size=size, seed=seed, **kw)


def vote(conn, tid, choice="LEFT", tags=(), match=None, annotator="tester"):
    state = store.load_state(conn, tid)
    m = match or state.next_match()
    row, dup = store.record_judgment(conn, tid, m.match_id, choice, presented_left=m.left, presented_right=m.right,
                                     token=m.token, tags=tags, annotator_id=annotator)
    assert not dup
    return m, row


# ---------------------------------------------------------------- creation and seeding

def test_creation_and_deterministic_seeding(db):
    conn, ids, _ = db
    assert bracket.seed_entrants(ids, 8, 5) == bracket.seed_entrants(list(reversed(ids)), 8, 5)
    assert bracket.seed_entrants(ids, 8, 5) != bracket.seed_entrants(ids, 8, 6)
    assert sorted(bracket.seed_entrants(ids, 8, 5)) == sorted(ids)
    a, b = new_tournament(conn, ids, seed=3), new_tournament(conn, ids, seed=3)
    assert store.load_state(conn, a).entrants == store.load_state(conn, b).entrants
    for size, matches in ((4, 3), (8, 7)):
        tid = new_tournament(conn, ids, size=size)
        state = store.load_state(conn, tid)
        assert len(state.bracket()) == matches and len(state.entrants) == size
    assert [s for s in bracket.bracket_slots(16)].__len__() == 15
    with pytest.raises(ValueError):
        new_tournament(conn, ids, size=6)
    with pytest.raises(ValueError):
        new_tournament(conn, ids[:5], size=8)


def test_sixteen_bracket_structure():
    assert [bracket.round_name(r, 16) for r in (1, 2, 3, 4)] == ["Round of 16", "Quarterfinals", "Semifinals", "Final"]
    assert bracket.downstream(1, 5, 16) == [(2, 2), (3, 1), (4, 0)]
    assert bracket.feeders(3, 1) == ((2, 2), (2, 3))


# ---------------------------------------------------------------- advancement and completion

def test_full_tournament_advances_winners_and_completes(db):
    conn, ids, _ = db
    tid = new_tournament(conn, ids)
    state = store.load_state(conn, tid)
    assert [m.round for m in state.bracket()].count(1) == 4 and state.next_match().round == 1
    assert all(m.status == "waiting" for m in state.bracket() if m.round > 1)
    winners = {}
    for _ in range(7):
        m, row = vote(conn, tid, "LEFT")
        winners[m.match_id] = m.left
        assert row["label"] in ("A_BETTER", "B_BETTER")
    state = store.load_state(conn, tid)
    assert state.complete and state.next_match() is None and store.progress(state) == (7, 7)
    qf = [state.by_slot(1, p) for p in range(4)]
    assert [m.winner for m in qf] == [winners[m.match_id] for m in qf]
    sf = [state.by_slot(2, p) for p in range(2)]
    assert {sf[0].a, sf[0].b} == {qf[0].winner, qf[1].winner} and {sf[1].a, sf[1].b} == {qf[2].winner, qf[3].winner}
    final = state.by_slot(3, 0)
    assert {final.a, final.b} == {sf[0].winner, sf[1].winner} and state.champion == final.winner == winners[final.match_id]
    assert len(set(state.eliminated())) == 7 and state.champion not in state.eliminated()


def test_left_right_mapping_uses_candidate_ids_not_screen_positions(db):
    conn, ids, _ = db
    seen = set()
    for seed in range(6):
        tid = new_tournament(conn, ids, seed=seed)
        for choice in ("LEFT", "RIGHT"):
            m = store.load_state(conn, tid).next_match()
            seen.add(m.left == m.a)
            _, row = vote(conn, tid, choice)
            chosen = m.left if choice == "LEFT" else m.right
            state = store.load_state(conn, tid)
            assert state.matches[m.match_id].winner == chosen
            assert row["screen_left_id"] == m.left and {row["candidate_a_id"], row["candidate_b_id"]} == {m.a, m.b}
            assert (row["label"] == "A_BETTER") == (chosen == m.a)
            # reviewing a judged match must show what was shown at vote time, not a recomputed orientation
            assert state.matches[m.match_id].left == m.left == row["screen_left_id"]
    assert seen == {True, False}  # both orientations occur
    # presentation is reproducible but a stale/forged presentation is refused
    m = store.load_state(conn, tid).next_match()
    assert store.load_state(conn, tid).matches[m.match_id].left == m.left
    with pytest.raises(store.StaleMatch):
        store.record_judgment(conn, tid, m.match_id, "LEFT", presented_left=m.right, presented_right=m.left, token=m.token)


# ---------------------------------------------------------------- non-decisive responses

@pytest.mark.parametrize("choice", ["TIE", "BOTH_BAD", "SKIP"])
def test_nondecisive_response_is_saved_and_never_auto_advances(db, choice):
    conn, ids, _ = db
    tid = new_tournament(conn, ids)
    m, row = vote(conn, tid, choice, tags=["poor_spacing", "other"])
    assert row["label"] == choice and json.loads(row["reason_tags"]) == ["other", "poor_spacing"]
    state = store.load_state(conn, tid)
    cur = state.matches[m.match_id]
    assert cur.status == "unresolved" and cur.winner is None
    assert state.next_match().match_id != m.match_id  # moves on, never picks randomly
    with pytest.raises(store.StaleMatch):  # a fresh vote on the same match is refused
        store.record_judgment(conn, tid, m.match_id, "LEFT", presented_left=cur.left, presented_right=cur.right, token=cur.token)
    assert store.record_judgment(conn, tid, m.match_id, "LEFT", presented_left=m.left, presented_right=m.right,
                                 token=m.token)[1] is True  # replaying the old request is an idempotent no-op
    store.resolve_match(conn, tid, m.match_id, m.a, "tester")
    resolved = store.load_state(conn, tid).matches[m.match_id]
    assert resolved.status == "resolved" and resolved.winner == m.a
    assert resolved.judgment["label"] == choice  # original response preserved
    row = export.comparison_rows(conn, tid)[0]
    assert row["label"] == choice and row["preferred_candidate_id"] is None and not row["usable_for_pairwise_loss"]
    assert row["advancement_candidate_id"] == m.a and row["advancement_overrides_label"]
    with pytest.raises(store.StaleMatch):
        store.resolve_match(conn, tid, m.match_id, m.a)


def test_cannot_resolve_decisive_or_foreign_candidate(db):
    conn, ids, _ = db
    tid = new_tournament(conn, ids)
    m, _ = vote(conn, tid, "LEFT")
    with pytest.raises(store.StaleMatch):
        store.resolve_match(conn, tid, m.match_id, m.a)
    m2, _ = vote(conn, tid, "TIE")
    with pytest.raises(ValueError):
        store.resolve_match(conn, tid, m2.match_id, "sha256:not-in-match")


# ---------------------------------------------------------------- undo and invalidation

def test_undo_invalidates_downstream_matches_and_keeps_audit_trail(db):
    conn, ids, _ = db
    tid = new_tournament(conn, ids)
    for _ in range(6):  # 4 QF, SF1, SF2
        vote(conn, tid, "LEFT")
    state = store.load_state(conn, tid)
    assert state.by_slot(3, 0).status == "ready"
    qf0 = state.by_slot(1, 0)
    store.undo_match(conn, tid, qf0.match_id, "tester")
    state = store.load_state(conn, tid)
    assert state.by_slot(1, 0).status == "ready" and state.by_slot(2, 0).status == "waiting"
    assert state.by_slot(3, 0).status == "waiting" and state.by_slot(2, 1).status == "decided"  # other branch kept
    voided = conn.execute("SELECT match_id, void_reason, presentation_token FROM judgments WHERE voided_at_utc IS NOT NULL ORDER BY judgment_id").fetchall()
    assert [v["match_id"] for v in voided] == [qf0.match_id, state.by_slot(2, 0).match_id]
    assert voided[1]["void_reason"] == f"upstream_changed:{qf0.match_id}"
    assert len(export.comparison_rows(conn, tid)) == 4
    assert len(export.comparison_rows(conn, tid, include_voided=True)) == 6
    # redoing the match gets a fresh presentation token and a fresh active judgment
    m = state.by_slot(1, 0)
    assert m.token != voided[0]["presentation_token"]
    vote(conn, tid, "RIGHT", match=m)
    assert store.load_state(conn, tid).by_slot(2, 0).status == "ready"


def test_undo_last_orders_resolution_before_judgment(db):
    conn, ids, _ = db
    tid = new_tournament(conn, ids)
    assert store.undo_last(conn, tid) is None
    m, _ = vote(conn, tid, "TIE")
    store.resolve_match(conn, tid, m.match_id, m.a)
    assert store.undo_last(conn, tid) == m.match_id
    state = store.load_state(conn, tid)
    assert state.matches[m.match_id].status == "unresolved" and state.matches[m.match_id].judgment["label"] == "TIE"
    assert store.undo_last(conn, tid) == m.match_id
    assert store.load_state(conn, tid).matches[m.match_id].status == "ready"


# ---------------------------------------------------------------- persistence, restart, duplicates

def test_sqlite_persistence_and_resume_after_restart(db):
    conn, ids, path = db
    tid = new_tournament(conn, ids, seed=9, source_review_status="pending")
    played = [vote(conn, tid, "LEFT" if i % 2 else "RIGHT")[0] for i in range(3)]
    before = store.load_state(conn, tid)
    conn.close()
    again = store.connect(path)  # simulated restart
    after = store.load_state(again, tid)
    assert after.entrants == before.entrants and store.progress(after) == (3, 7)
    assert [after.matches[m.match_id].winner for m in played] == [before.matches[m.match_id].winner for m in played]
    assert after.next_match().match_id == before.next_match().match_id and after.next_match().left == before.next_match().left
    assert store.get_meta(again, "last_tournament") == tid
    for _ in range(4):
        vote(again, tid, "LEFT")
    assert store.load_state(again, tid).complete


def test_duplicate_submission_is_idempotent(db):
    conn, ids, _ = db
    tid = new_tournament(conn, ids)
    m = store.load_state(conn, tid).next_match()
    kwargs = dict(presented_left=m.left, presented_right=m.right, token=m.token)
    first, dup1 = store.record_judgment(conn, tid, m.match_id, "LEFT", **kwargs)
    second, dup2 = store.record_judgment(conn, tid, m.match_id, "RIGHT", **kwargs)  # rerun, even with another choice
    assert (dup1, dup2) == (False, True) and first["judgment_id"] == second["judgment_id"]
    assert conn.execute("SELECT COUNT(*) FROM judgments").fetchone()[0] == 1
    assert second["label"] == first["label"]
    with pytest.raises(sqlite3.IntegrityError):  # the database itself forbids a second active judgment
        conn.execute("INSERT INTO judgments(comparison_id,presentation_token,tournament_id,match_id,candidate_a_id,candidate_b_id,"
                     "screen_left_id,label,reason_tags,note,annotator_id,ui_version,created_at_utc) "
                     "VALUES('x','other-token',?,?,?,?,?,'TIE','[]','','a','u','t')", (tid, m.match_id, m.a, m.b, m.a))
    with pytest.raises(ValueError):
        vote(conn, tid, "LEFT", tags=["not_a_tag"], match=store.load_state(conn, tid).next_match())


# ---------------------------------------------------------------- validation, same source/target, import

def test_same_source_and_target_enforced(db, world, tmp_path):
    conn, ids, _ = db
    other = world[0] / "other.json"
    sd.write_json(other, make_source("offer_coupon", 42))
    second = build_candidate_set(other, {"width": 1200, "height": 628}, world[0] / "set2", renderer=FakeRenderer(), seed=1,
                                 count=8, poor=2)
    outcome = candidates.import_manifest(conn, second.manifest)
    with pytest.raises(ValueError, match="one source design and one target"):
        new_tournament(conn, ids[:4] + outcome["imported"][:4])
    assert len(candidates.candidate_groups(conn)) == 2


def _tamper(world, tmp_path, mutate_candidate=None, mutate_record=None, mutate_png=None):
    root, _, result = world
    record = json.loads(result.manifest.read_text().splitlines()[0])
    candidate = json.loads(candidates.resolve(record["candidate_path"]).read_text())
    preview = tmp_path / "p.png"
    Image.open(candidates.resolve(record["preview_path"])).save(preview)
    if mutate_candidate:
        mutate_candidate(candidate)
        sd.write_json(tmp_path / "c.json", candidate)
        record["candidate_path"] = str(tmp_path / "c.json")
        record["candidate_id"] = candidates.candidate_id_of(candidate)
    if mutate_png:
        mutate_png(preview)
    record["preview_path"] = str(preview)
    if mutate_record:
        mutate_record(record)
    return record


@pytest.mark.parametrize("name,kwargs,fragment", [
    ("offcanvas", dict(mutate_candidate=lambda c: c["elements"][1].update(x=c["canvas"]["width"] + 5)), "off-canvas"),
    ("immutable", dict(mutate_candidate=lambda c: next(e for e in c["elements"] if e["type"] == "text").update(content="changed")),
     "immutable"),
    ("wrong_size", dict(mutate_png=lambda p: Image.new("RGB", (10, 10)).save(p)), "preview must be a PNG"),
    ("bad_id", dict(mutate_record=lambda r: r.update(candidate_id="sha256:forged")), "does not match candidate_id"),
    ("bad_source_id", dict(mutate_record=lambda r: r.update(source_design_id="sha256:forged")), "source_design_id"),
])
def test_import_rejects_invalid_candidates(db, world, tmp_path, name, kwargs, fragment):
    conn, _, _ = db
    record = _tamper(world, tmp_path, **kwargs)
    problems = candidates.check_record(record)
    assert problems and any(fragment in p for p in problems), problems
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(json.dumps(record) + "\n")
    before = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    outcome = candidates.import_manifest(conn, manifest)
    assert outcome["rejected"] and not outcome["imported"]
    assert conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] == before


def test_import_is_idempotent_and_requires_all_fields(db, world, tmp_path):
    conn, ids, _ = db
    again = candidates.import_manifest(conn, world[2].manifest)
    assert sorted(again["imported"]) == sorted(ids) and conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] == 8
    assert "missing fields" in candidates.check_record({"candidate_id": "x"})[0]


# ---------------------------------------------------------------- dev generator

def test_dev_generator_valid_deterministic_distinct_and_bounded(world, tmp_path):
    root, source_path, result = world
    source = json.loads(source_path.read_text())
    design, canvas = Design.from_adobe(source), Canvas.from_adobe(TARGET)
    records = [json.loads(line) for line in result.manifest.read_text().splitlines()]
    assert len(records) == 8 and len({r["candidate_id"] for r in records}) == 8
    assert len({r["source_design_id"] for r in records}) == 1 and all(r["target_canvas"] == TARGET for r in records)
    assert sum(r["provenance"]["strategy"] in POOR_KINDS for r in records) >= 2
    assert {r["generator_version"] for r in records} == {GENERATOR_VERSION}
    for r in records:
        cand = json.loads(candidates.resolve(r["candidate_path"]).read_text())
        assert validate_candidate(design, cand, canvas).is_valid and set(cand) == {"canvas", "elements"}
        for before, after in zip(source["elements"], cand["elements"]):  # immutable fields untouched
            for key in set(before) - {"x", "y", "width", "height", "font_size"}:
                assert before[key] == after[key]
    again = build_candidate_set(source_path, TARGET, tmp_path / "again", renderer=FakeRenderer(), seed=7, count=8, poor=2)
    assert [json.loads(line)["candidate_id"] for line in again.manifest.read_text().splitlines()] == [r["candidate_id"] for r in records]
    with pytest.raises(FileExistsError):
        build_candidate_set(source_path, TARGET, tmp_path / "again", renderer=FakeRenderer())
    with pytest.raises(RuntimeError, match="candidates after"):
        build_candidate_set(source_path, TARGET, tmp_path / "tiny", renderer=FakeRenderer(), count=8, max_attempts=2)


def test_invalid_cases_stay_out_of_the_candidate_manifest(world):
    root, source_path, result = world
    rows = [json.loads(line) for line in (result.out_dir / "diagnostics/diagnostics.jsonl").read_text().splitlines()]
    assert len(rows) == 5 and {r["expected_category"] for r in rows} >= {"off_canvas", "identity", "immutable_field"}
    assert not any(r["false_accept"] for r in rows if r["name"] in ("shifted_off_canvas", "dropped_element", "changed_font_family", "wrong_canvas"))
    manifest_ids = {json.loads(line)["candidate_id"] for line in result.manifest.read_text().splitlines()}
    diag_ids = {candidates.candidate_id_of(json.loads(Path(r["path"]).read_text())) for r in rows}
    assert not manifest_ids & diag_ids and all(Path(r["path"]).parent.name == "diagnostics" for r in rows)
    assert invalid_cases(json.loads(source_path.read_text()), TARGET)[0]["name"] == "stretch_to_target"


def test_hidden_information_never_reaches_annotator_html(db):
    conn, ids, _ = db
    tid = new_tournament(conn, ids)
    state = store.load_state(conn, tid)
    page = views.bracket_html(conn, state, state.next_match().match_id) + views.header_md(state, state.next_match())
    page += views.viewer_html(store.get_candidate(conn, ids[0])["preview_path"], "Candidate A (left)")
    for secret in ("strategy", GENERATOR_VERSION, "dev_controlled", "tiny_text", "provenance", "score"):
        assert secret not in page
    assert views.ZOOMS["Fit"] is None and 'width:100%' in page


# ---------------------------------------------------------------- extra and repeated comparisons

def test_extra_and_repeat_comparisons_do_not_affect_the_bracket(db):
    conn, ids, _ = db
    tid = new_tournament(conn, ids)
    for _ in range(7):
        vote(conn, tid, "LEFT")
    champion = store.load_state(conn, tid).champion
    made = store.add_extra_matches(conn, tid, "eliminated_pairs", 3, seed=1)
    assert len(made) == 3
    state = store.load_state(conn, tid)
    compared = {frozenset((m.a, m.b)) for m in state.bracket()}
    for m in state.extras():
        assert {m.a, m.b} <= set(state.eliminated()) and frozenset((m.a, m.b)) not in compared
    vote(conn, tid, "TIE", match=state.next_match("extra"))
    reps = store.add_extra_matches(conn, tid, "repeats", 2)
    assert len(reps) == 2
    state = store.load_state(conn, tid)
    for rep in (m for m in state.extras() if m.kind == "repeat"):
        original = state.matches[rep.repeat_of]
        assert {rep.a, rep.b} == {original.a, original.b} and rep.left != original.judgment["screen_left_id"]
    vote(conn, tid, "LEFT", match=next(m for m in state.extras() if m.kind == "repeat"))
    assert store.load_state(conn, tid).champion == champion
    rows = export.comparison_rows(conn, tid)
    extra_rows = [r for r in rows if r["match_kind"] != "bracket"]
    assert len(rows) == 9 and len(extra_rows) == 2
    assert all(r["advancement_candidate_id"] is None and r["round"] is None for r in extra_rows)
    assert any(r["repeat_of"] for r in extra_rows)
    with pytest.raises(ValueError):
        store.add_extra_matches(conn, tid, "bogus", 1)


# ---------------------------------------------------------------- export

def test_export_has_seven_integral_decisions_and_manifest(db, tmp_path):
    conn, ids, _ = db
    tid = new_tournament(conn, ids, annotator_id="ann", source_review_status="pending")
    for i in range(7):
        vote(conn, tid, "LEFT" if i % 2 == 0 else "RIGHT", tags=["weak_hierarchy"] if i == 0 else [], annotator="ann")
    out = export.export_comparisons(conn, tmp_path / "out", tid)
    rows = [json.loads(line) for line in Path(out["jsonl"]).read_text().splitlines()]
    assert out["comparisons"] == 7 == len(rows)
    with open(out["csv"], newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 7 and list(csv_rows[0]) == list(export.CSV_FIELDS)
    manifest = {json.loads(line)["candidate_id"]: json.loads(line) for line in Path(out["candidate_manifest"]).read_text().splitlines()}
    state = store.load_state(conn, tid)
    assert set(manifest) == set(ids)
    for row, csv_row in zip(rows, csv_rows):
        assert row["schema_version"] == "preference.v1" and row["usable_for_pairwise_loss"] is True
        assert {row["candidate_a_id"], row["candidate_b_id"]} == {row["screen_left_candidate_id"], row["screen_right_candidate_id"]}
        assert row["preferred_candidate_id"] == (row["candidate_a_id"] if row["label"] == "A_BETTER" else row["candidate_b_id"])
        assert row["advancement_candidate_id"] == row["preferred_candidate_id"] and row["advancement_method"] == "vote"
        assert not row["advancement_overrides_label"]
        assert state.matches[row["match_id"]].winner == row["preferred_candidate_id"]
        assert row["target_width"] == 1600 and row["target_height"] == 400 and row["annotator_id"] == "ann"
        assert row["split_group_id"] == row["template_family_id"] and row["source_review_status"] == "pending"
        assert row["candidate_a_generator_version"] == GENERATOR_VERSION and row["renderer_version"].startswith("fake+")
        assert csv_row["comparison_id"] == row["comparison_id"] and csv_row["label"] == row["label"]
        assert Path(manifest[row["candidate_a_id"]]["preview_path"]).is_file()
    assert rows[0]["reason_tags"] == ["weak_hierarchy"] and csv_rows[0]["reason_tags"] == "weak_hierarchy"
    assert {r["round_name"] for r in rows} == {"Quarterfinals", "Semifinals", "Final"}
    assert not any("score" in key or "rank" in key or "reached" in key for key in rows[0])  # no advancement-derived quality


# ---------------------------------------------------------------- source review

@pytest.fixture
def source_root(tmp_path):
    class Plain:
        version = {"test_double": True}

        def fit_text(self, source):
            return copy.deepcopy(source), []

        def render(self, source, path):
            return FakeRenderer().render(source, path)

    root = tmp_path / "sources"
    sd.generate_dataset(root, count=3, renderer=Plain())
    return root


def test_source_review_state_is_preserved_across_decisions(source_root):
    sources = source_review.load_sources(source_root)
    assert {s["status"] for s in sources} == {"pending"}  # nothing is silently approved
    a, b, c = (s["source_design_id"] for s in sources)
    assert source_review.set_decision(source_root, a, "approved", "good") == 1
    assert source_review.set_decision(source_root, b, "rejected", "text too small") == 1
    assert source_review.set_decision(source_root, c, "approved", "") == 2
    state = {s["source_design_id"]: (s["status"], s["reason"]) for s in source_review.load_sources(source_root)}
    assert state == {a: ("approved", "good"), b: ("rejected", "text too small"), c: ("approved", "")}
    approved = [json.loads(line)["source_design_id"] for line in (source_root / "approved.jsonl").read_text().splitlines()]
    assert sorted(approved) == sorted([a, c])
    assert source_review.set_decision(source_root, a, "pending") == 1  # change one, keep the others
    assert source_review.load_sources(source_root)[1]["status"] == "rejected"
    with pytest.raises(ValueError):
        source_review.set_decision(source_root, b, "rejected", "")
    with pytest.raises(ValueError):
        source_review.set_decision(source_root, b, "maybe")
    source_path = Path(source_review.load_sources(source_root)[2]["source_path"])
    assert source_review.review_status(source_path, c) == "approved"
    assert source_review.review_status(source_root.parent / "elsewhere.json", c) == "unknown"
    # the existing static-gallery review format stays valid (same review.json / approved.jsonl)
    assert json.loads((source_root / "review.json").read_text())["manifest_sha256"] == sd.file_hash(source_root / "manifest.jsonl")


def test_tournament_records_review_status_of_its_source(db, source_root):
    conn, ids, _ = db
    tid = new_tournament(conn, ids, source_review_status=source_review.review_status(Path("x.json"), "sid"))
    assert store.load_state(conn, tid).tournament["source_review_status"] == "unknown"


# ---------------------------------------------------------------- UI wiring and style

def test_gradio_app_builds_and_initial_render_matches_outputs(db, tmp_path):
    from annotation import app
    conn, ids, path = db
    tid = new_tournament(conn, ids)
    demo = app.build_app(path)
    try:
        import gradio as gr
        assert isinstance(demo, gr.Blocks)
        load = next(fn for fn in demo.fns.values() if getattr(fn.fn, "__name__", "") == "on_load")
        result = load.fn()
        assert len(result) == len(load.outputs) == 21
        assert result[0] == tid and tid in result[3]
        assert "Candidate A" in result[5] and "Candidate B" in result[6] and result[18]["status"] == "ready"
    finally:
        demo.close()


def test_python310_syntax_for_annotation_modules():
    for path in sorted(Path(candidates.__file__).parent.glob("*.py")):
        ast.parse(path.read_text(encoding="utf-8"), feature_version=(3, 10))


def test_header_shows_overall_match_position_not_per_round(db):
    conn, ids, _ = db
    tid = new_tournament(conn, ids)
    state = store.load_state(conn, tid)
    assert "match 1 of 7" in views.header_md(state, state.next_match()) and "decided" not in views.header_md(state, state.next_match())
    for _ in range(6):
        vote(conn, tid, "LEFT")
    state = store.load_state(conn, tid)
    header = views.header_md(state, state.next_match())
    assert "Final · match 7 of 7 · 6 already decided (resumed)" in header and "of 1" not in header
