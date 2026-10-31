"""Review of AI-proposed pairwise labels (separate tables; existing tournament data is never touched).

A proposal starts as `synthetic_proposed`. It becomes verified ONLY through an explicit human action:
  accept   -> `human_verified` with the human label equal to the proposal (not allowed when the proposal is SKIP)
  correct  -> `human_verified` with the human's own label (A_BETTER / B_BETTER / TIE / BOTH_BAD)
  reject   -> `human_rejected`: kept for audit, NEVER exported as a verified training label
The original proposal is stored once, unchanged, and every review (including undone ones) stays in `ai_reviews`.
Verification modes: fast_confirm (accept/reject without being blind), guided (corrected with the proposal visible),
blind (proposal hidden until the label was submitted).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from annotation.candidates import REPO_ROOT
from annotation.store import now_utc, tx

UI_VERSION = "ai_review_ui.v1"
LABELS = ("A_BETTER", "B_BETTER", "TIE", "BOTH_BAD")
DECISIVE = ("A_BETTER", "B_BETTER")
MODES = ("fast_confirm", "guided", "blind")
SCHEMA = """
CREATE TABLE IF NOT EXISTS ai_proposals(
  comparison_id TEXT PRIMARY KEY, proposal_json TEXT NOT NULL, candidate_a_id TEXT NOT NULL, candidate_b_id TEXT NOT NULL,
  source_design_id TEXT NOT NULL, template_family_id TEXT NOT NULL, task_id TEXT NOT NULL, split TEXT, target_width INTEGER NOT NULL,
  target_height INTEGER NOT NULL, proposed_label TEXT NOT NULL, confidence REAL, evaluation_method TEXT NOT NULL,
  blind_eval_subset INTEGER NOT NULL DEFAULT 0, assigned_to TEXT, seq INTEGER NOT NULL, imported_at_utc TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS ai_reviews(
  review_id INTEGER PRIMARY KEY AUTOINCREMENT, comparison_id TEXT NOT NULL REFERENCES ai_proposals,
  decision TEXT NOT NULL CHECK(decision IN ('accept','correct','reject')),
  human_label TEXT CHECK(human_label IS NULL OR human_label IN ('A_BETTER','B_BETTER','TIE','BOTH_BAD')),
  proposed_label_at_review TEXT NOT NULL, verification_mode TEXT NOT NULL, annotator_id TEXT NOT NULL,
  reason_tags TEXT NOT NULL DEFAULT '[]', note TEXT NOT NULL DEFAULT '', ui_version TEXT NOT NULL, created_at_utc TEXT NOT NULL,
  voided_at_utc TEXT, void_reason TEXT);
CREATE UNIQUE INDEX IF NOT EXISTS one_active_ai_review ON ai_reviews(comparison_id) WHERE voided_at_utc IS NULL;
"""


def ensure_schema(conn) -> None:
    conn.executescript(SCHEMA)
    have = {r[1] for r in conn.execute("PRAGMA table_info(ai_reviews)")}
    for col in ("candidate_a_quality", "candidate_b_quality"):  # additive migration; existing reviews keep NULL (= not stated)
        if col not in have:
            conn.execute(f"ALTER TABLE ai_reviews ADD COLUMN {col} TEXT")


def _resolve(path, base):
    p = Path(path)
    return p if p.is_absolute() else Path(base) / p


# --------------------------------------------------------------------------- import
def import_proposals(conn, path: Path | str, base: Path = REPO_ROOT) -> dict:
    """Idempotent. Rejects (and reports) proposals whose candidate or preview files are missing. Existing rows are kept."""
    ensure_schema(conn)
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    start = conn.execute("SELECT COALESCE(MAX(seq),0) FROM ai_proposals").fetchone()[0]
    imported, existing, rejected = 0, 0, []
    with tx(conn):
        for r in rows:
            missing = [k for k in ("candidate_a_preview", "candidate_b_preview", "candidate_a_path", "candidate_b_path")
                       if not _resolve(r[k], base).is_file()]
            if missing:
                rejected.append(dict(comparison_id=r["comparison_id"], missing=missing))
                continue
            if r["review_status"] != "synthetic_proposed" or r["proposed_label"] not in LABELS + ("SKIP",):
                rejected.append(dict(comparison_id=r["comparison_id"], missing=["invalid status/label"]))
                continue
            if conn.execute("SELECT 1 FROM ai_proposals WHERE comparison_id=?", (r["comparison_id"],)).fetchone():
                existing += 1
                continue
            start += 1
            conn.execute("INSERT INTO ai_proposals VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                r["comparison_id"], json.dumps(dict(r, _base=str(base)), sort_keys=True), r["candidate_a_id"], r["candidate_b_id"],
                r["source_design_id"], r["template_family_id"], r["task_id"], r.get("split"), r["target_width"], r["target_height"],
                r["proposed_label"], r.get("confidence"), r["evaluation_method"], int(bool(r.get("blind_eval_subset"))),
                r.get("assigned_to"), start, now_utc()))
            imported += 1
    return dict(imported=imported, already_present=existing, rejected=rejected)


# --------------------------------------------------------------------------- queries
def _where(filters: dict | None, annotator: str | None):
    f = filters or {}
    where, args = [], []
    if f.get("target") and f["target"] != "All":
        w, h = f["target"].split("x")
        where.append("p.target_width=? AND p.target_height=?")
        args += [int(w), int(h)]
    if f.get("family"):
        where.append("p.template_family_id=?")
        args.append(f["family"])
    if f.get("method"):
        where.append("p.evaluation_method=?")
        args.append(f["method"])
    if f.get("split"):
        where.append("p.split=?")
        args.append(f["split"])
    if f.get("conf_min") is not None:
        where.append("COALESCE(p.confidence,0)>=?")
        args.append(float(f["conf_min"]))
    if f.get("conf_max") is not None:
        where.append("COALESCE(p.confidence,0)<=?")
        args.append(float(f["conf_max"]))
    if f.get("exclude_skip"):
        where.append("p.proposed_label!='SKIP'")
    if f.get("exclude_blind"):
        where.append("p.blind_eval_subset=0")
    if f.get("blind_only"):
        where.append("p.blind_eval_subset=1")
    if annotator:
        where.append("(p.assigned_to IS NULL OR p.assigned_to=?)")
        args.append(annotator)
    return where, args


def next_item(conn, annotator: str | None = None, filters: dict | None = None, order: str = "sequence", skip: tuple = ()) -> dict | None:
    """Next proposal with no active review. order: sequence | uncertain_first | confident_first."""
    where, args = _where(filters, annotator)
    where.append("NOT EXISTS(SELECT 1 FROM ai_reviews r WHERE r.comparison_id=p.comparison_id AND r.voided_at_utc IS NULL)")
    if skip:
        where.append(f"p.comparison_id NOT IN ({','.join('?' * len(skip))})")
        args += list(skip)
    key = {"sequence": "p.seq", "uncertain_first": "COALESCE(p.confidence,0), p.seq", "confident_first": "COALESCE(p.confidence,0) DESC, p.seq"}[order]
    row = conn.execute(f"SELECT p.* FROM ai_proposals p WHERE {' AND '.join(where)} ORDER BY {key} LIMIT 1", args).fetchone()
    return _item(row)


def _item(row) -> dict | None:
    if row is None:
        return None
    d = json.loads(row["proposal_json"])
    d["blind_eval_subset"] = bool(row["blind_eval_subset"])
    return d


def get_item(conn, comparison_id: str) -> dict | None:
    return _item(conn.execute("SELECT * FROM ai_proposals WHERE comparison_id=?", (comparison_id,)).fetchone())


def get_review(conn, comparison_id: str) -> dict | None:
    r = conn.execute("SELECT * FROM ai_reviews WHERE comparison_id=? AND voided_at_utc IS NULL", (comparison_id,)).fetchone()
    return dict(r) if r else None


def status_of(conn, comparison_id: str) -> str:
    r = get_review(conn, comparison_id)
    if r is None:
        return "synthetic_proposed"
    return "human_rejected" if r["decision"] == "reject" else "human_verified"


def progress(conn, annotator: str | None = None, filters: dict | None = None) -> dict:
    where, args = _where(filters, annotator)
    base = f"FROM ai_proposals p {'WHERE ' + ' AND '.join(where) if where else ''}"
    total = conn.execute(f"SELECT COUNT(*) {base}", args).fetchone()[0]
    out = dict(total=total, accepted=0, corrected=0, rejected=0)
    q = (f"SELECT r.decision, COUNT(*) n FROM ai_proposals p JOIN ai_reviews r ON r.comparison_id=p.comparison_id AND r.voided_at_utc IS NULL "
         f"{'WHERE ' + ' AND '.join(where) if where else ''} GROUP BY r.decision")
    for r in conn.execute(q, args):
        out[{"accept": "accepted", "correct": "corrected", "reject": "rejected"}[r["decision"]]] = r["n"]
    out["reviewed"] = out["accepted"] + out["corrected"] + out["rejected"]
    out["remaining"] = total - out["reviewed"]
    return out


def distinct_values(conn, column: str) -> list:
    assert column in ("template_family_id", "evaluation_method", "split")
    return [r[0] for r in conn.execute(f"SELECT DISTINCT {column} FROM ai_proposals WHERE {column} IS NOT NULL ORDER BY 1")]


# --------------------------------------------------------------------------- actions
def submit_review(conn, comparison_id: str, decision: str, annotator_id: str, *, mode: str = "fast_confirm",
                  human_label: str | None = None, reason_tags=(), note: str = "",
                  a_quality: str | None = None, b_quality: str | None = None) -> dict:
    """Record ONE explicit human action. Raises ValueError on invalid combinations; never changes the proposal."""
    ensure_schema(conn)
    item = get_item(conn, comparison_id)
    if item is None:
        raise ValueError("unknown comparison")
    if mode not in MODES:
        raise ValueError("unknown verification mode")
    proposed = item["proposed_label"]
    if any(q not in (None, "good", "bad") for q in (a_quality, b_quality)):
        raise ValueError("quality must be good, bad or unset")
    if decision == "accept":
        if proposed not in LABELS:
            raise ValueError("a SKIP proposal cannot be accepted; correct it or reject it")
        human_label = proposed
    elif decision == "correct":
        if human_label not in LABELS:
            raise ValueError("correction needs A_BETTER, B_BETTER, TIE or BOTH_BAD")
    elif decision == "reject":
        human_label = None
    else:
        raise ValueError("decision must be accept, correct or reject")
    if item["blind_eval_subset"] and mode != "blind" and decision != "reject":
        # the blind subset exists for unbiased agreement measurement, so it must be labelled without the proposal shown
        raise ValueError("this comparison is in the blind evaluation subset: label it in blind mode")
    if mode == "blind" and decision != "correct":
        raise ValueError("blind mode records your own label (use the A/B/Tie/Both-bad buttons)")
    with tx(conn):
        if conn.execute("SELECT 1 FROM ai_reviews WHERE comparison_id=? AND voided_at_utc IS NULL", (comparison_id,)).fetchone():
            raise ValueError("already reviewed; undo it first to revise")
        conn.execute("INSERT INTO ai_reviews(comparison_id,decision,human_label,proposed_label_at_review,verification_mode,annotator_id,"
                     "reason_tags,note,ui_version,created_at_utc,candidate_a_quality,candidate_b_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                     (comparison_id, decision, human_label, proposed, mode, annotator_id, json.dumps(list(reason_tags)), note,
                      UI_VERSION, now_utc(), a_quality, b_quality))
    return get_review(conn, comparison_id)


def undo_review(conn, comparison_id: str, reason: str = "undone by annotator") -> bool:
    with tx(conn):
        cur = conn.execute("UPDATE ai_reviews SET voided_at_utc=?, void_reason=? WHERE comparison_id=? AND voided_at_utc IS NULL",
                           (now_utc(), reason, comparison_id))
    return cur.rowcount > 0


def last_review(conn, annotator_id: str | None = None) -> dict | None:
    q = "SELECT * FROM ai_reviews WHERE voided_at_utc IS NULL" + (" AND annotator_id=?" if annotator_id else "") + " ORDER BY review_id DESC LIMIT 1"
    r = conn.execute(q, (annotator_id,) if annotator_id else ()).fetchone()
    return dict(r) if r else None


def history(conn, limit: int = 100, only: str | None = None) -> list[dict]:
    where = "WHERE r.voided_at_utc IS NULL" + (" AND r.decision=?" if only else "")
    rows = conn.execute(f"SELECT r.*, p.target_width, p.target_height, p.template_family_id FROM ai_reviews r JOIN ai_proposals p USING(comparison_id) "
                        f"{where} ORDER BY r.review_id DESC LIMIT ?", ((only, limit) if only else (limit,))).fetchall()
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------- exports
def _base_row(p: dict) -> dict:
    a, b = p["candidate_a_id"], p["candidate_b_id"]
    left = a  # the review UI always shows A on the left and B on the right (A/B order is already randomised per comparison)
    return dict(schema_version="preference.v1", comparison_id=p["comparison_id"], tournament_id=None, match_id=None, match_kind="ai_review",
                round=None, round_name=None, bracket_position=None, repeat_of=None, source_design_id=p["source_design_id"],
                template_family_id=p["template_family_id"], split_group_id=p["split_group_id"], source_review_status="unreviewed",
                target_width=p["target_width"], target_height=p["target_height"], candidate_a_id=a, candidate_b_id=b,
                screen_left_candidate_id=left, screen_right_candidate_id=b if left == a else a,
                candidate_a_generator_version=p["candidate_a_generator_version"], candidate_b_generator_version=p["candidate_b_generator_version"],
                renderer_version=None, ui_version=UI_VERSION, voided=False, void_reason=None, split=p.get("split"),
                candidate_a_path=p["candidate_a_path"], candidate_b_path=p["candidate_b_path"], source_path=p["source_path"],
                proposed_label=p["proposed_label"], proposal_method=p["evaluation_method"], proposal_method_version=p["method_version"],
                proposal_confidence=p.get("confidence"), proposal_confidence_kind=p.get("confidence_kind"),
                proposal_explanation=p["explanation"]["summary"])


def _write(rows: list[dict], out: Path, stem: str, conn) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{stem}.jsonl").write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    if rows:
        fields = list(rows[0])
        with (out / f"{stem}.csv").open("w", newline="", encoding="utf-8") as h:
            w = csv.DictWriter(h, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})
    ids = sorted({c for r in rows for c in (r["candidate_a_id"], r["candidate_b_id"])})
    paths = {}
    for r in rows:
        paths[r["candidate_a_id"]] = r["candidate_a_path"]
        paths[r["candidate_b_id"]] = r["candidate_b_path"]
    (out / f"{stem}_candidates.jsonl").write_text("".join(json.dumps(dict(candidate_id=c, candidate_path=paths[c])) + "\n" for c in ids), encoding="utf-8")
    return dict(rows=len(rows), jsonl=str(out / f"{stem}.jsonl"), csv=str(out / f"{stem}.csv"))


def export_verified(conn, out_dir: Path | str, stem: str = "synthetic_human_verified") -> dict:
    """Human-verified (accepted or corrected) comparisons in preference.v1 form. Rejected ones are never included."""
    rows = []
    for r in conn.execute("SELECT r.*, p.proposal_json FROM ai_reviews r JOIN ai_proposals p USING(comparison_id) "
                          "WHERE r.voided_at_utc IS NULL AND r.decision IN ('accept','correct') ORDER BY r.review_id"):
        p = json.loads(r["proposal_json"])
        row = _base_row(p)
        label = r["human_label"]
        row.update(label=label, preferred_candidate_id={"A_BETTER": p["candidate_a_id"], "B_BETTER": p["candidate_b_id"]}.get(label),
                   usable_for_pairwise_loss=label in DECISIVE, source_rejected_in_queue=False, advancement_candidate_id=None,
                   advancement_method=None, advancement_overrides_label=False, reason_tags=json.loads(r["reason_tags"]), note=r["note"],
                   annotator_id=r["annotator_id"], created_at_utc=r["created_at_utc"], label_source="synthetic_proposal_human_verified",
                   verification_status="human_verified", verification_decision=r["decision"], verification_mode=r["verification_mode"],
                   verified_at_utc=r["created_at_utc"], human_agreed_with_proposal=label == r["proposed_label_at_review"],
                   candidate_a_quality=r["candidate_a_quality"] or ("bad" if label == "BOTH_BAD" else None),
                   candidate_b_quality=r["candidate_b_quality"] or ("bad" if label == "BOTH_BAD" else None))
        rows.append(row)
    return _write(rows, Path(out_dir), stem, conn)


def export_rejected(conn, out_dir: Path | str, stem: str = "synthetic_human_rejected_audit") -> dict:
    """Audit trail of rejected proposals. NOT training data."""
    rows = []
    for r in conn.execute("SELECT r.*, p.proposal_json FROM ai_reviews r JOIN ai_proposals p USING(comparison_id) "
                          "WHERE r.voided_at_utc IS NULL AND r.decision='reject' ORDER BY r.review_id"):
        row = _base_row(json.loads(r["proposal_json"]))
        row.update(label=None, preferred_candidate_id=None, usable_for_pairwise_loss=False, annotator_id=r["annotator_id"],
                   created_at_utc=r["created_at_utc"], label_source="synthetic_proposal_human_rejected", verification_status="human_rejected",
                   verification_mode=r["verification_mode"], note=r["note"], reason_tags=json.loads(r["reason_tags"]))
        rows.append(row)
    return _write(rows, Path(out_dir), stem, conn)


def export_unverified(conn, out_dir: Path | str, stem: str = "synthetic_proposed_unverified", weight_scale: float = .5) -> dict:
    """Weak-supervision file: still-unreviewed proposals. `usable_for_pairwise_loss` is hard-False so nothing includes them by accident;
    a future training script must opt in explicitly and may use `weak_supervision_weight` (an arbitrary default: confidence x scale)."""
    rows = []
    for r in conn.execute("SELECT p.proposal_json FROM ai_proposals p WHERE NOT EXISTS(SELECT 1 FROM ai_reviews r WHERE r.comparison_id=p.comparison_id "
                          "AND r.voided_at_utc IS NULL) ORDER BY p.seq"):
        p = json.loads(r["proposal_json"])
        row = _base_row(p)
        lab = p["proposed_label"]
        row.update(label=lab, preferred_candidate_id={"A_BETTER": p["candidate_a_id"], "B_BETTER": p["candidate_b_id"]}.get(lab),
                   usable_for_pairwise_loss=False, label_source="synthetic_proposed_unverified", verification_status="synthetic_proposed",
                   weak_supervision_weight=round((p.get("confidence") or 0) * weight_scale, 3) if lab in DECISIVE else 0.0,
                   annotator_id=None, created_at_utc=None)
        rows.append(row)
    return _write(rows, Path(out_dir), stem, conn)


def export_existing_human(src_dir: Path | str, out_dir: Path | str, stem: str = "human_tournament_existing") -> dict:
    """Normalised COPY of the existing human comparisons with explicit provenance. Originals are only read."""
    src = Path(src_dir)
    rows = []
    for line in (src / "comparisons_all.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            r.update(label_source="human_tournament", verification_status="human_original")
            rows.append(r)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{stem}.jsonl").write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (out / f"{stem}_candidates.jsonl").write_text((src / "comparisons_all_candidates.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    return dict(rows=len(rows), jsonl=str(out / f"{stem}.jsonl"))
