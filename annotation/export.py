"""Export human comparisons for the eventual pairwise GNN pipeline (JSONL + CSV) and the candidate manifest.

One row per active judgment. Bracket advancement is exported as separate descriptive columns and is
never a second preference label. No scores or "how far a candidate advanced" values are produced.
`usable_for_pairwise_loss` is true only for A_BETTER/B_BETTER; TIE is a weak/optional signal per
Appendix E.3; BOTH_BAD and SKIP carry no winner.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from annotation import store, work_queue
from annotation.bracket import round_name

SCHEMA = "preference.v1"
CSV_FIELDS = (
    "schema_version", "comparison_id", "tournament_id", "match_id", "match_kind", "round", "round_name",
    "bracket_position", "repeat_of", "source_design_id", "template_family_id", "split_group_id",
    "source_review_status", "target_width", "target_height", "candidate_a_id", "candidate_b_id",
    "screen_left_candidate_id", "screen_right_candidate_id", "label", "preferred_candidate_id",
    "usable_for_pairwise_loss", "source_rejected_in_queue", "advancement_candidate_id", "advancement_method", "advancement_overrides_label",
    "reason_tags", "note", "annotator_id", "created_at_utc", "candidate_a_generator_version",
    "candidate_b_generator_version", "renderer_version", "ui_version", "voided", "void_reason")


def comparison_rows(conn, tournament_id: str | None = None, *, include_voided: bool = False) -> list[dict]:
    where, args = [], []
    if tournament_id:
        where.append("j.tournament_id=?")
        args.append(tournament_id)
    if not include_voided:
        where.append("j.voided_at_utc IS NULL")
    sql = ("SELECT j.*, m.kind, m.round, m.position, m.repeat_of, t.size, t.source_design_id, t.template_family_id, "
           "t.source_review_status, t.target_width, t.target_height, "
           "ca.generator_version gva, cb.generator_version gvb, ca.renderer_version renderer, "
           "(SELECT winner_candidate_id FROM resolutions r WHERE r.judgment_id=j.judgment_id AND r.voided_at_utc IS NULL) res "
           "FROM judgments j JOIN matches m ON m.match_id=j.match_id JOIN tournaments t ON t.tournament_id=j.tournament_id "
           "JOIN candidates ca ON ca.candidate_id=j.candidate_a_id JOIN candidates cb ON cb.candidate_id=j.candidate_b_id"
           + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY j.judgment_id")
    rows = []
    for r in conn.execute(sql, args):
        a, b, label = r["candidate_a_id"], r["candidate_b_id"], r["label"]
        preferred = {"A_BETTER": a, "B_BETTER": b}.get(label)
        bracket_match = r["kind"] == "bracket"
        if not bracket_match:
            advancement, method = None, None
        elif preferred:
            advancement, method = preferred, "vote"
        elif r["res"]:
            advancement, method = r["res"], "manual_resolution"
        else:
            advancement, method = None, None
        rows.append(dict(
            schema_version=SCHEMA, comparison_id=r["comparison_id"], tournament_id=r["tournament_id"],
            match_id=r["match_id"], match_kind=r["kind"], round=r["round"] if bracket_match else None,
            round_name=round_name(r["round"], r["size"]) if bracket_match else None,
            bracket_position=r["position"] if bracket_match else None, repeat_of=r["repeat_of"],
            source_design_id=r["source_design_id"], template_family_id=r["template_family_id"],
            split_group_id=r["template_family_id"], source_review_status=r["source_review_status"],
            target_width=r["target_width"], target_height=r["target_height"], candidate_a_id=a, candidate_b_id=b,
            screen_left_candidate_id=r["screen_left_id"], screen_right_candidate_id=b if r["screen_left_id"] == a else a,
            label=label, preferred_candidate_id=preferred,
            usable_for_pairwise_loss=label in store.DECISIVE and not work_queue.rejected(conn, r["source_design_id"]),
            source_rejected_in_queue=work_queue.rejected(conn, r["source_design_id"]),
            advancement_candidate_id=advancement, advancement_method=method,
            advancement_overrides_label=method == "manual_resolution",
            reason_tags=json.loads(r["reason_tags"]), note=r["note"], annotator_id=r["annotator_id"],
            created_at_utc=r["created_at_utc"], candidate_a_generator_version=r["gva"],
            candidate_b_generator_version=r["gvb"], renderer_version=r["renderer"], ui_version=r["ui_version"],
            voided=r["voided_at_utc"] is not None, void_reason=r["void_reason"]))
    return rows


def export_comparisons(conn, out_dir: Path | str, tournament_id: str | None = None, *,
                       include_voided: bool = False, stem: str = "comparisons") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = comparison_rows(conn, tournament_id, include_voided=include_voided)
    jsonl, csv_path, manifest = out / f"{stem}.jsonl", out / f"{stem}.csv", out / f"{stem}_candidates.jsonl"
    jsonl.write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({**r, "reason_tags": ";".join(r["reason_tags"])})
    ids = sorted({c for r in rows for c in (r["candidate_a_id"], r["candidate_b_id"])})
    lines = []
    for cid in ids:
        c = store.get_candidate(conn, cid)
        lines.append(json.dumps(dict(
            candidate_id=cid, source_design_id=c["source_design_id"], template_family_id=c["template_family_id"],
            target_canvas=dict(width=c["target_width"], height=c["target_height"]), source_path=c["source_path"],
            source_preview_path=c["source_preview_path"], candidate_path=c["candidate_path"], preview_path=c["preview_path"],
            generator_id=c["generator_id"], generator_version=c["generator_version"], renderer_version=c["renderer_version"],
            set_id=c["set_id"], provenance=json.loads(c["provenance_json"])), sort_keys=True) + "\n")
    manifest.write_text("".join(lines), encoding="utf-8")
    return dict(comparisons=len(rows), jsonl=str(jsonl), csv=str(csv_path), candidate_manifest=str(manifest))
