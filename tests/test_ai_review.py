"""AI-proposal review workflow: explicit verification only, audit trail, persistence, exports, existing labels untouched."""

import hashlib
import json

import pytest
from PIL import Image

from annotation import ai_review, store


def make_proposal(tmp_path, n, label="A_BETTER", conf=.8, blind=False, target=(728, 90)):
    for suffix in ("a", "b"):
        Image.new("RGB", target, (255, 255, 255)).save(tmp_path / f"{n}{suffix}.png")
        (tmp_path / f"{n}{suffix}.json").write_text(json.dumps({"canvas": {"width": target[0], "height": target[1]}, "elements": []}))
    return dict(
        comparison_id=f"c{n:030d}", proposal_id=f"prop_{n}", schema_version="ai_proposal.v1", task_id=f"t{n}",
        candidate_a_id=f"sha256:a{n}", candidate_b_id=f"sha256:b{n}", source_design_id=f"sha256:s{n}", template_family_id="fam", split_group_id="fam",
        split="train", target_width=target[0], target_height=target[1], pair_type="creative_vs_conventional", proposed_label=label,
        confidence=conf, confidence_kind="x", explanation=dict(summary="because", typography="t", hierarchy="h", balance_spacing="b",
                                                            composition="c", space_use="s", tradeoffs="x"),
        evaluation_method="rendered_metrics_model", method_version="v1", visually_assessed=False, review_status="synthetic_proposed",
        candidate_a_generator_version="g", candidate_b_generator_version="g", candidate_a_path=f"{n}a.json", candidate_b_path=f"{n}b.json",
        candidate_a_preview=f"{n}a.png", candidate_b_preview=f"{n}b.png", source_path=f"{n}a.json", source_preview_path=f"{n}a.png",
        screen_left_candidate_id=f"sha256:b{n}", blind_eval_subset=blind)


@pytest.fixture
def db(tmp_path):
    rows = [make_proposal(tmp_path, 1), make_proposal(tmp_path, 2, "SKIP", None), make_proposal(tmp_path, 3, "TIE", .4),
            make_proposal(tmp_path, 4, "B_BETTER", .9, target=(90, 728)), make_proposal(tmp_path, 5, blind=True)]
    path = tmp_path / "proposals.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    conn = store.connect(tmp_path / "review.db")
    result = ai_review.import_proposals(conn, path, tmp_path)
    return conn, tmp_path, result


def cid(n):
    return f"c{n:030d}"


def test_import_is_idempotent_and_validates_files(db):
    conn, tmp, result = db
    assert result["imported"] == 5 and not result["rejected"]
    again = ai_review.import_proposals(conn, tmp / "proposals.jsonl", tmp)
    assert again["imported"] == 0 and again["already_present"] == 5
    (tmp / "9a.png").write_bytes(b"")
    bad = make_proposal(tmp, 9)
    (tmp / "9a.png").unlink()
    (tmp / "bad.jsonl").write_text(json.dumps(bad) + "\n")
    assert ai_review.import_proposals(conn, tmp / "bad.jsonl", tmp)["rejected"]


def test_proposals_are_unverified_until_explicit_action(db):
    conn, _, _ = db
    assert ai_review.status_of(conn, cid(1)) == "synthetic_proposed"
    assert ai_review.progress(conn)["reviewed"] == 0
    assert ai_review.next_item(conn)["comparison_id"] == cid(1)       # merely viewing does not verify
    assert ai_review.status_of(conn, cid(1)) == "synthetic_proposed"


def test_accept_correct_reject_and_next(db):
    conn, _, _ = db
    ai_review.submit_review(conn, cid(1), "accept", "sarah")
    assert ai_review.status_of(conn, cid(1)) == "human_verified"
    assert ai_review.get_review(conn, cid(1))["human_label"] == "A_BETTER"
    ai_review.submit_review(conn, cid(3), "correct", "sarah", mode="guided", human_label="BOTH_BAD")
    assert ai_review.get_review(conn, cid(3))["human_label"] == "BOTH_BAD"
    assert ai_review.get_review(conn, cid(3))["proposed_label_at_review"] == "TIE"          # original proposal preserved
    ai_review.submit_review(conn, cid(4), "reject", "sarah")
    assert ai_review.status_of(conn, cid(4)) == "human_rejected"
    assert ai_review.get_item(conn, cid(3))["proposed_label"] == "TIE"                      # proposal row itself never changes
    assert ai_review.next_item(conn)["comparison_id"] == cid(2)                              # advances past reviewed ones
    p = ai_review.progress(conn)
    assert (p["accepted"], p["corrected"], p["rejected"], p["remaining"]) == (1, 1, 1, 2)


def test_tie_and_both_bad_corrections_and_invalid_actions(db):
    conn, _, _ = db
    ai_review.submit_review(conn, cid(1), "correct", "s", mode="guided", human_label="TIE")
    assert ai_review.get_review(conn, cid(1))["human_label"] == "TIE"
    with pytest.raises(ValueError):
        ai_review.submit_review(conn, cid(2), "accept", "s")           # cannot accept a SKIP
    with pytest.raises(ValueError):
        ai_review.submit_review(conn, cid(2), "correct", "s", human_label="MAYBE")
    with pytest.raises(ValueError):
        ai_review.submit_review(conn, cid(1), "reject", "s")            # already reviewed
    ai_review.submit_review(conn, cid(2), "correct", "s", mode="guided", human_label="BOTH_BAD")


def test_blind_subset_requires_blind_labelling(db):
    conn, _, _ = db
    with pytest.raises(ValueError):
        ai_review.submit_review(conn, cid(5), "accept", "s")
    with pytest.raises(ValueError):
        ai_review.submit_review(conn, cid(1), "accept", "s", mode="blind")
    ai_review.submit_review(conn, cid(5), "correct", "s", mode="blind", human_label="B_BETTER")
    assert ai_review.get_review(conn, cid(5))["verification_mode"] == "blind"
    assert ai_review.next_item(conn, filters=dict(exclude_blind=True))["comparison_id"] == cid(1)


def test_undo_revise_and_history_keep_audit_trail(db):
    conn, _, _ = db
    ai_review.submit_review(conn, cid(1), "accept", "s")
    assert ai_review.undo_review(conn, cid(1))
    assert ai_review.status_of(conn, cid(1)) == "synthetic_proposed"
    ai_review.submit_review(conn, cid(1), "correct", "s", mode="guided", human_label="B_BETTER")
    rows = conn.execute("SELECT decision, voided_at_utc FROM ai_reviews WHERE comparison_id=? ORDER BY review_id", (cid(1),)).fetchall()
    assert [r["decision"] for r in rows] == ["accept", "correct"] and rows[0]["voided_at_utc"] and rows[1]["voided_at_utc"] is None
    assert [h["decision"] for h in ai_review.history(conn, only="correct")] == ["correct"]


def test_filters_and_ordering(db):
    conn, _, _ = db
    assert ai_review.next_item(conn, filters=dict(target="90x728"))["comparison_id"] == cid(4)
    assert ai_review.next_item(conn, order="confident_first", filters=dict(exclude_blind=True))["comparison_id"] == cid(4)
    assert ai_review.next_item(conn, order="uncertain_first")["comparison_id"] == cid(2)    # no-proposal items are least confident
    assert ai_review.progress(conn, filters=dict(target="728x90"))["total"] == 4


def test_persistence_across_connections(db):
    conn, tmp, _ = db
    ai_review.submit_review(conn, cid(1), "accept", "s")
    conn.close()
    again = store.connect(tmp / "review.db")
    assert ai_review.status_of(again, cid(1)) == "human_verified"
    assert ai_review.next_item(again)["comparison_id"] == cid(2)


def test_exports_keep_label_sources_separate(db):
    conn, tmp, _ = db
    ai_review.submit_review(conn, cid(1), "accept", "s")
    ai_review.submit_review(conn, cid(3), "correct", "s", mode="guided", human_label="B_BETTER")
    ai_review.submit_review(conn, cid(4), "reject", "s")
    out = tmp / "out"
    v, r, u = ai_review.export_verified(conn, out), ai_review.export_rejected(conn, out), ai_review.export_unverified(conn, out)
    verified = [json.loads(l) for l in (out / "synthetic_human_verified.jsonl").read_text().splitlines()]
    assert {x["comparison_id"] for x in verified} == {cid(1), cid(3)} and v["rows"] == 2
    assert all(x["verification_status"] == "human_verified" and x["schema_version"] == "preference.v1" for x in verified)
    by = {x["comparison_id"]: x for x in verified}
    assert by[cid(3)]["label"] == "B_BETTER" and by[cid(3)]["proposed_label"] == "TIE" and by[cid(3)]["human_agreed_with_proposal"] is False
    assert by[cid(3)]["preferred_candidate_id"] == "sha256:b3" and by[cid(3)]["usable_for_pairwise_loss"] is True
    assert by[cid(1)]["usable_for_pairwise_loss"] is True and by[cid(1)]["candidate_a_path"] == "1a.json"
    rejected = [json.loads(l) for l in (out / "synthetic_human_rejected_audit.jsonl").read_text().splitlines()]
    assert [x["comparison_id"] for x in rejected] == [cid(4)] and rejected[0]["usable_for_pairwise_loss"] is False
    unverified = [json.loads(l) for l in (out / "synthetic_proposed_unverified.jsonl").read_text().splitlines()]
    assert {x["comparison_id"] for x in unverified} == {cid(2), cid(5)}
    assert all(x["usable_for_pairwise_loss"] is False and x["label_source"] == "synthetic_proposed_unverified" for x in unverified)
    assert not ({x["comparison_id"] for x in unverified} & {x["comparison_id"] for x in verified})
    assert (out / "synthetic_human_verified_candidates.jsonl").exists()


def test_existing_human_labels_are_only_read(tmp_path):
    src = tmp_path / "exports"
    src.mkdir()
    row = dict(comparison_id="x", label="A_BETTER", candidate_a_id="a", candidate_b_id="b")
    (src / "comparisons_all.jsonl").write_text(json.dumps(row) + "\n")
    (src / "comparisons_all_candidates.jsonl").write_text("{}\n")
    before = hashlib.sha256((src / "comparisons_all.jsonl").read_bytes()).hexdigest()
    ai_review.export_existing_human(src, tmp_path / "out")
    assert hashlib.sha256((src / "comparisons_all.jsonl").read_bytes()).hexdigest() == before
    copy = json.loads((tmp_path / "out" / "human_tournament_existing.jsonl").read_text())
    assert copy["label_source"] == "human_tournament" and copy["verification_status"] == "human_original"


def test_review_tab_builds_in_existing_app(tmp_path):
    from annotation.app import build_app
    build_app(tmp_path / "a.db")


def test_absolute_quality_is_recorded_and_exported(db):
    conn, tmp, _ = db
    ai_review.submit_review(conn, cid(1), "correct", "s", mode="guided", human_label="A_BETTER", a_quality="good", b_quality="bad")
    ai_review.submit_review(conn, cid(3), "correct", "s", mode="guided", human_label="BOTH_BAD")
    with pytest.raises(ValueError):
        ai_review.submit_review(conn, cid(4), "correct", "s", mode="guided", human_label="TIE", a_quality="great")
    ai_review.export_verified(conn, tmp / "q")
    rows = {json.loads(l)["comparison_id"]: json.loads(l) for l in (tmp / "q" / "synthetic_human_verified.jsonl").read_text().splitlines()}
    assert (rows[cid(1)]["candidate_a_quality"], rows[cid(1)]["candidate_b_quality"]) == ("good", "bad")
    assert (rows[cid(3)]["candidate_a_quality"], rows[cid(3)]["candidate_b_quality"]) == ("bad", "bad")
