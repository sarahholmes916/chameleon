"""Local Gradio tournament UI (Appendix E.1). Run:  python -m annotation.app

All state lives in SQLite (data/annotations.db by default; override with --db or
CHAMELEON_ANNOTATION_DB). Handlers re-read the database on every call, so refreshes and restarts lose
nothing, and every vote is idempotent per presentation token.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import gradio as gr

from annotation import candidates, export, source_review, store, views, work_queue
from annotation.bracket import SIZES
from annotation.candidates import REPO_ROOT

REASON_CHOICES = [("Text too small", "text_too_small"), ("Text overflow", "text_overflow"),
                  ("Unwanted overlap", "unwanted_overlap"), ("Poor spacing", "poor_spacing"),
                  ("Bad alignment", "bad_alignment"), ("Broken grouping", "broken_grouping"),
                  ("Weak hierarchy", "weak_hierarchy"), ("Other", "other")]
QUEUES = ["Bracket", "Extra & repeat comparisons"]


def build_app(db_path: Path | str) -> gr.Blocks:
    db_path = Path(db_path)

    def conn():
        return store.connect(db_path)

    # ------------------------------------------------------------ rendering of the main screen
    def tournament_choices(c):
        return [(f"{t['tournament_id']} · {t['template_family_id']} {t['target_width']}×{t['target_height']} · "
                 f"{t['size']} entrants", t["tournament_id"]) for t in store.list_tournaments(c)]

    def render(tid, focus_id, queue, zoom, message=""):
        c = conn()
        choices = tournament_choices(c)
        off = gr.update(interactive=False)
        if not tid:
            return (None, gr.update(choices=choices, value=None), gr.update(choices=[], value=None),
                    "### No tournament yet\nUse the **New tournament** tab to import a candidate set and start one.",
                    "", "", "", off, off, off, off, off, gr.update(visible=False), gr.update(visible=False), off, off,
                    gr.update(value=[]), message, None, "", [])
        state = store.load_state(c, tid)
        focus = state.matches.get(focus_id) if focus_id else None
        if focus is None:
            focus = state.next_match("extra" if queue == QUEUES[1] else "bracket")
            if focus is None and queue == QUEUES[0]:
                focus = next(iter(state.unresolved()), None)
        match_choices = [(f"{views.match_title(state, m)} · {m.status}", m.match_id) for m in state.matches.values()]
        left_html = right_html = source_html = ""
        presented = None
        buttons = dict(vote=False, resolve=False, redo=False)
        if focus is not None and focus.a and focus.b:
            lc, rc = store.get_candidate(c, focus.left), store.get_candidate(c, focus.right)
            decided = focus.judgment is not None
            cap_l = "Candidate A (left)" + (f" · {views.label_of(state, focus.left)}" if decided else "")
            cap_r = "Candidate B (right)" + (f" · {views.label_of(state, focus.right)}" if decided else "")
            if focus.winner and focus.kind == "bracket":
                cap_l += " ✔ advanced" if focus.winner == focus.left else ""
                cap_r += " ✔ advanced" if focus.winner == focus.right else ""
            left_html, right_html = views.viewer_html(lc["preview_path"], cap_l, zoom), views.viewer_html(rc["preview_path"], cap_r, zoom)
            src = lc["source_preview_path"]
            source_html = views.viewer_html(src, "Original source design", zoom) if src else "<i>No source preview recorded.</i>"
            presented = dict(match_id=focus.match_id, left=focus.left, right=focus.right, token=focus.token, status=focus.status)
            buttons = dict(vote=focus.status == "ready", resolve=focus.status == "unresolved" and focus.kind == "bracket",
                           redo=focus.judgment is not None)
            if focus.judgment is not None and not message:
                message = f"Recorded: **{views.LABELS[focus.judgment['label']]}**" + (
                    " — choose who advances below (the response itself is kept)." if focus.status == "unresolved" else "")
        elif state.complete:
            message = message or "Tournament complete. Review any match in the dropdown, or add extra comparisons."
        else:
            left_html = right_html = views.viewer_html(None, "", zoom)
        vote = gr.update(interactive=buttons["vote"])
        return (tid, gr.update(choices=choices, value=tid), gr.update(choices=match_choices, value=focus.match_id if focus else None),
                views.header_md(state, focus), views.bracket_html(c, state, focus.match_id if focus else None),
                left_html, right_html, vote, vote, vote, vote, vote,
                gr.update(visible=buttons["resolve"]), gr.update(visible=buttons["resolve"]),
                gr.update(interactive=buttons["redo"]), gr.update(interactive=any(
                    m.judgment is not None for m in state.matches.values())),
                gr.update(value=[]), message, presented, source_html, views.history_rows(state))

    # ------------------------------------------------------------ event handlers
    def on_load():
        c = conn()
        if store.get_meta(c, "automatic_queue") == "yes":
            tid, message = work_queue.next_task(c)
            return render(tid, None, QUEUES[0], "Fit", message)
        tid = store.get_meta(c, "last_tournament")
        if tid is None or c.execute("SELECT 1 FROM tournaments WHERE tournament_id=?", (tid,)).fetchone() is None:
            rows = store.list_tournaments(c)
            tid = rows[0]["tournament_id"] if rows else None
        return render(tid, None, QUEUES[0], "Fit")

    def on_start_queue(annotator, zoom):
        c = conn()
        problems = work_queue.sync(c, REPO_ROOT)
        store.set_meta(c, "automatic_queue", "yes")
        tid, message = work_queue.next_task(c, annotator or "local_annotator")
        if problems:
            message += "\n\nImport problems:\n" + "\n".join(problems)
        return render(tid, None, QUEUES[0], zoom, message)

    def queue_render(tid, queue, zoom, message, annotator):
        c = conn()
        if store.get_meta(c, "automatic_queue") == "yes" and (not tid or store.load_state(c, tid).complete):
            tid, progress = work_queue.next_task(c, annotator or "local_annotator")
            message += "\n\n" + progress
        return render(tid, None, queue, zoom, message)

    def on_reject_source(tid, annotator, zoom):
        if not tid:
            return render(None, None, QUEUES[0], zoom, "No source selected.")
        c = conn()
        work_queue.reject_source(c, tid)
        tid, message = work_queue.next_task(c, annotator or "local_annotator")
        return render(tid, None, QUEUES[0], zoom, "Source rejected for all targets. Its labels are excluded from training.\n\n" + message)

    def on_vote(choice):
        def handler(tid, presented, tags, annotator, queue, zoom):
            message = ""
            try:
                if not presented or presented.get("status") != "ready":
                    raise store.StaleMatch("This match is not waiting for a decision.")
                _, duplicate = store.record_judgment(
                    conn(), tid, presented["match_id"], choice, presented_left=presented["left"],
                    presented_right=presented["right"], token=presented["token"], tags=tags or [],
                    annotator_id=(annotator or "local_annotator").strip())
                names = dict(LEFT="Candidate A wins", RIGHT="Candidate B wins", TIE="Tie", BOTH_BAD="Both bad", SKIP="Skipped")
                message = ("Already recorded — duplicate click ignored." if duplicate else f"Saved: **{names[choice]}**." + (
                    " The match stays unresolved until you choose who advances." if choice not in ("LEFT", "RIGHT") else ""))
            except store.StaleMatch as exc:
                message = f"⚠️ {exc} Showing the current state."
            return queue_render(tid, queue, zoom, message, annotator)
        return handler

    def on_resolve(side):
        def handler(tid, presented, annotator, queue, zoom):
            message = ""
            try:
                if not presented:
                    raise store.StaleMatch("Nothing to resolve.")
                store.resolve_match(conn(), tid, presented["match_id"], presented[side], (annotator or "local_annotator").strip())
                message = "Advanced manually. Your original response is preserved and exported as the label."
            except store.StaleMatch as exc:
                message = f"⚠️ {exc}"
            return queue_render(tid, queue, zoom, message, annotator)
        return handler

    def on_redo(tid, presented, annotator, queue, zoom):
        try:
            store.undo_match(conn(), tid, presented["match_id"], (annotator or "local_annotator").strip())
            message = "Decision voided (kept in the audit trail); any dependent later matches were reset."
        except (store.StaleMatch, TypeError, KeyError) as exc:
            message = f"⚠️ {exc}"
        return render(tid, presented["match_id"] if presented else None, queue, zoom, message)

    def on_undo(tid, annotator, queue, zoom):
        undone = store.undo_last(conn(), tid, (annotator or "local_annotator").strip())
        return render(tid, undone, queue, zoom, "Undid the latest decision; dependent later matches were reset." if undone else "Nothing to undo.")

    def on_select(tid, queue, zoom):
        c = conn()
        if tid:
            store.set_meta(c, "last_tournament", tid)
        return render(tid, None, queue, zoom)

    def on_focus(tid, focus_id, queue, zoom):
        return render(tid, focus_id, queue, zoom)

    # ------------------------------------------------------------ new tournament / import
    def group_choices(c):
        seen = {}
        for g in candidates.candidate_groups(c):
            key = (g["source_design_id"], g["target_width"], g["target_height"])
            seen.setdefault(key, dict(g, n=0))["n"] += g["n"]
        return [(f"{g['template_family_id']} · source {g['source_design_id'][-8:]} · {g['target_width']}×{g['target_height']} · {g['n']} candidates",
                 json.dumps(list(k))) for k, g in seen.items()]

    def discovered_manifests():
        return sorted(str(p.relative_to(REPO_ROOT).as_posix()) for p in (REPO_ROOT / "data/candidates").rglob(candidates.MANIFEST_NAME))

    def discovered_datasets():
        return sorted(p.parent.relative_to(REPO_ROOT).as_posix() for p in (REPO_ROOT / "data/sources").glob("*/manifest.jsonl"))

    def on_refresh():
        c = conn()
        return (gr.update(choices=discovered_manifests()), gr.update(choices=group_choices(c)),
                gr.update(choices=discovered_datasets()), gr.update(choices=diagnostics_sets()),
                "Lists refreshed. New candidate manifests appear in the first dropdown; import one, then pick it below.")

    def on_import(path):
        if not path:
            return "Choose a candidate_manifest.jsonl.", gr.update()
        try:
            result = candidates.import_manifest(conn(), candidates.resolve(path))
        except (OSError, ValueError) as exc:
            return f"⚠️ {exc}", gr.update()
        text = f"Imported {len(result['imported'])} valid candidate(s); rejected {len(result['rejected'])}."
        for r in result["rejected"][:5]:
            text += f"\n- `{r['candidate_id']}`: {'; '.join(r['problems'])[:200]}"
        text += "\n\nNext: pick it in **Candidate set** below and click **Create tournament**."
        return text, gr.update(choices=group_choices(conn()))

    def on_create(group, size, seed, annotator, allow_unapproved):
        if not group:
            return "Pick a candidate set first.", gr.update(), None, gr.update()
        c = conn()
        source_id, w, h = json.loads(group)
        ids = candidates.group_candidate_ids(c, source_id, w, h)
        row = c.execute("SELECT source_path FROM candidates WHERE candidate_id=?", (ids[0],)).fetchone()
        status = source_review.review_status(candidates.resolve(row["source_path"]), source_id)
        if status != "approved" and not allow_unapproved:
            return (f"⚠️ This source's review status is **{status}**. Approve it in *Source review*, or tick the development "
                    "override (the status is recorded with the tournament).", gr.update(), None, gr.update())
        try:
            tid = store.create_tournament(c, ids, size=int(size), seed=int(seed), annotator_id=(annotator or "local_annotator").strip(),
                                          source_review_status=status)
        except ValueError as exc:
            return f"⚠️ {exc}", gr.update(), None, gr.update()
        return f"Created tournament `{tid}`.", gr.update(choices=tournament_choices(c), value=tid), tid, gr.Tabs(selected="tab_tournament")

    # ------------------------------------------------------------ source review
    def default_dataset():
        found = sorted((REPO_ROOT / "data/sources").glob("*/manifest.jsonl"))
        preferred = [p for p in found if p.parent.name == "pilot20_v3"]
        return (preferred or found or [None])[-1].parent.relative_to(REPO_ROOT).as_posix() if found else ""

    def sources_view(root, index):
        sources = source_review.load_sources(candidates.resolve(root))
        if not sources:
            return gr.update(choices=[], value=None), None, "No sources.", "pending", "", 0
        index = max(0, min(int(index or 0), len(sources) - 1))
        s = sources[index]
        counts = {k: sum(x["status"] == k for x in sources) for k in ("approved", "rejected", "pending", "ineligible")}
        choices = [(f"{i:02d} · {x['template_family_id']} · {x['status']}", i) for i, x in enumerate(sources)]
        summary = (f"**{s['template_family_id']}** · {s['canvas']['width']}×{s['canvas']['height']} · {s['element_count']} elements · "
                   f"`{s['source_design_id'][-12:]}`\n\nApproved {counts['approved']} · Rejected {counts['rejected']} · "
                   f"Pending {counts['pending']}" + (f" · Ineligible {counts['ineligible']}" if counts["ineligible"] else ""))
        return (gr.update(choices=choices, value=index), str(s["preview_path"]) if s["preview_path"] else None, summary,
                s["status"] if s["eligible"] else "pending", s["reason"], index)

    def on_sources_load(root):
        return sources_view(root, 0)

    def on_source_pick(root, index):
        return sources_view(root, index)

    def on_source_save(root, index, status, reason, advance):
        root_path = candidates.resolve(root)
        sources = source_review.load_sources(root_path)
        s = sources[int(index)]
        message = ""
        try:
            if status == "rejected" and not (reason or "").strip():
                raise ValueError("Give a reason for rejection.")
            source_review.set_decision(root_path, s["source_design_id"], status, reason or "")
            message = f"Saved: {s['template_family_id']} → {status}"
        except ValueError as exc:
            return (*sources_view(root, index), f"⚠️ {exc}")
        nxt = min(int(index) + 1, len(sources) - 1) if advance else int(index)
        return (*sources_view(root, nxt), message)

    # ------------------------------------------------------------ extras, export, diagnostics
    def on_extra(tid, mode, count):
        if not tid:
            return "Open a tournament first."
        try:
            made = store.add_extra_matches(conn(), tid, "eliminated_pairs" if mode.startswith("Pairs") else "repeats", int(count))
        except ValueError as exc:
            return f"⚠️ {exc}"
        return (f"Created {len(made)} extra match(es). Switch the Tournament tab's queue to *Extra & repeat comparisons*."
                if made else "No eligible pairs yet (finish more bracket matches, or nothing left to compare).")

    def on_export(tid, scope, include_voided):
        out = REPO_ROOT / "data/exports"
        result = export.export_comparisons(conn(), out, tid if scope.startswith("This") else None,
                                           include_voided=include_voided, stem=f"comparisons_{tid}" if scope.startswith("This") else "comparisons_all")
        return f"Exported {result['comparisons']} comparison(s):\n\n- `{result['jsonl']}`\n- `{result['csv']}`\n- `{result['candidate_manifest']}`"

    def diagnostics_sets():
        return sorted(str(p.relative_to(REPO_ROOT).as_posix()) for p in (REPO_ROOT / "data/candidates").glob("*/diagnostics/diagnostics.jsonl"))

    def on_diag(path):
        if not path:
            return []
        rows = [json.loads(line) for line in candidates.resolve(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        return [[r["name"], r["expected_category"], "; ".join(r["validator_violations"])[:160] or "(none)",
                 "FALSE ACCEPT" if r["false_accept"] else "rejected"] for r in rows]

    # ------------------------------------------------------------ layout
    with gr.Blocks(title="Chameleon tournament annotation", css=views.CSS, js=views.SHORTCUT_JS) as demo:
        tid_state = gr.State(None)
        presented_state = gr.State(None)
        gr.Markdown("# Chameleon layout tournament\nPick the better resized layout. Dev tool — scores and generator identity are never shown.")
        with gr.Tabs() as tabs:
            with gr.Tab("Tournament", id="tab_tournament"):
                gr.Markdown("Start the queue to load your generated layouts automatically. Source review is optional. "
                            "Finish each set to advance automatically; restart with the same button to resume.")
                start_queue = gr.Button("Start / resume all layouts", variant="primary")
                with gr.Row():
                    tournament_dd = gr.Dropdown(label="Tournament", choices=[], scale=3)
                    queue = gr.Radio(QUEUES, value=QUEUES[0], label="Queue", scale=2)
                    zoom = gr.Radio(list(views.ZOOMS), value="Fit", label="Zoom (both sides)", scale=3)
                    annotator = gr.Textbox(value="local_annotator", label="Annotator ID", scale=1)
                header = gr.Markdown()
                bracket_view = gr.HTML()
                with gr.Row():
                    with gr.Column(scale=1, min_width=220):
                        source_view = gr.HTML(elem_id="original-source-preview")
                        reject_source_btn = gr.Button("Reject this source for all targets")
                    with gr.Column(scale=2, min_width=300):
                        left_view = gr.HTML()
                    with gr.Column(scale=2, min_width=300):
                        right_view = gr.HTML()
                with gr.Row():
                    btn_left = gr.Button("A wins  ←", variant="primary", elem_id="btn-left", interactive=False, scale=3)
                    btn_tie = gr.Button("Tie  (T)", elem_id="btn-tie", interactive=False)
                    btn_bad = gr.Button("Both bad  (B)", elem_id="btn-bad", interactive=False)
                    btn_skip = gr.Button("Skip / can't judge  (S)", elem_id="btn-skip", interactive=False)
                    btn_right = gr.Button("→  B wins", variant="primary", elem_id="btn-right", interactive=False, scale=3)
                with gr.Row():
                    res_left = gr.Button("Advance A (resolve; label kept)", visible=False, variant="secondary")
                    res_right = gr.Button("Advance B (resolve; label kept)", visible=False, variant="secondary")
                tags = gr.CheckboxGroup(REASON_CHOICES, label="Optional reason tags (applied to the next response)", value=[])
                status = gr.Markdown()
                with gr.Row():
                    btn_undo = gr.Button("Undo last decision  (U)", elem_id="btn-undo", interactive=False)
                    btn_redo = gr.Button("Redo this match", interactive=False)
                    focus_dd = gr.Dropdown(label="Show match", choices=[], scale=3)
                gr.Markdown("Keyboard: ← A wins · → B wins · T tie · B both bad · S skip · U undo")
            with gr.Tab("New tournament", id="tab_new"):
                gr.Markdown("A tournament uses candidates of **one source and one target**. Import a `candidate_manifest.jsonl` "
                            "from the dev generator (or S04); every record is re-validated.")
                with gr.Row():
                    manifest_dd = gr.Dropdown(label="Candidate manifest", choices=discovered_manifests(), allow_custom_value=True, scale=4)
                    import_btn = gr.Button("Validate & import", scale=1)
                    refresh_btn = gr.Button("Refresh lists", scale=1)
                import_msg = gr.Markdown()
                group_dd = gr.Dropdown(label="Candidate set (source × target)", choices=[])
                with gr.Row():
                    size = gr.Radio(list(SIZES), value=8, label="Candidates in bracket")
                    seed = gr.Number(value=0, precision=0, label="Seeding seed")
                    new_annotator = gr.Textbox(value="local_annotator", label="Annotator ID")
                    allow = gr.Checkbox(False, label="Development override: allow a source that is not approved")
                create_btn = gr.Button("Create tournament", variant="primary")
                create_msg = gr.Markdown()
            with gr.Tab("Source review", id="tab_sources"):
                gr.Markdown("Review the original generated designs. These are **sources, not tournament candidates**; nothing is approved automatically.")
                with gr.Row():
                    dataset_box = gr.Dropdown(value=default_dataset(), choices=discovered_datasets(), allow_custom_value=True,
                                              label="Source dataset folder")
                    load_btn = gr.Button("Load")
                src_pick = gr.Dropdown(label="Source", choices=[])
                src_idx = gr.State(0)
                with gr.Row():
                    src_image = gr.Image(label="Preview (click ⤢ for full screen)", type="filepath", interactive=False,
                                         show_fullscreen_button=True, show_download_button=False, height=620)
                    with gr.Column():
                        src_summary = gr.Markdown()
                        src_status = gr.Radio(["pending", "approved", "rejected"], value="pending", label="Decision (approved = include in dataset)")
                        src_reason = gr.Textbox(label="Reason (required for rejection)")
                        with gr.Row():
                            src_save = gr.Button("Save", variant="primary")
                            src_save_next = gr.Button("Save & next")
                        src_msg = gr.Markdown()
            with gr.Tab("History & export", id="tab_history"):
                history = gr.Dataframe(headers=["Match", "Round", "Screen left | right", "Response", "Advanced", "Reason tags", "Time (UTC)"],
                                       interactive=False, wrap=True)
                with gr.Row():
                    extra_mode = gr.Radio(["Pairs of eliminated candidates", "Repeat judged pairs (consistency)"],
                                          value="Pairs of eliminated candidates", label="Extra comparisons")
                    extra_count = gr.Slider(1, 12, value=4, step=1, label="How many")
                    extra_btn = gr.Button("Create extra matches")
                extra_msg = gr.Markdown()
                with gr.Row():
                    export_scope = gr.Radio(["This tournament", "All tournaments"], value="This tournament", label="Export")
                    include_voided = gr.Checkbox(False, label="Include voided (undone) rows")
                    export_btn = gr.Button("Export JSONL + CSV")
                export_msg = gr.Markdown()
            with gr.Tab("Validator diagnostics", id="tab_diag"):
                gr.Markdown("Deliberately invalid outputs and what the S02 validator said. Separate from preference labels.")
                diag_dd = gr.Dropdown(label="Candidate set", choices=diagnostics_sets(), allow_custom_value=True)
                diag_table = gr.Dataframe(headers=["Case", "Expected violation", "Validator said", "Outcome"], interactive=False, wrap=True)

        outs = [tid_state, tournament_dd, focus_dd, header, bracket_view, left_view, right_view, btn_left, btn_tie, btn_bad,
                btn_skip, btn_right, res_left, res_right, btn_redo, btn_undo, tags, status, presented_state, source_view, history]
        demo.load(on_load, None, outs)
        start_queue.click(on_start_queue, [annotator, zoom], outs)
        reject_source_btn.click(on_reject_source, [tid_state, annotator, zoom], outs)
        common = [tid_state, presented_state, tags, annotator, queue, zoom]
        btn_left.click(on_vote("LEFT"), common, outs, show_progress="hidden")
        btn_right.click(on_vote("RIGHT"), common, outs, show_progress="hidden")
        btn_tie.click(on_vote("TIE"), common, outs, show_progress="hidden")
        btn_bad.click(on_vote("BOTH_BAD"), common, outs, show_progress="hidden")
        btn_skip.click(on_vote("SKIP"), common, outs, show_progress="hidden")
        res_left.click(on_resolve("left"), [tid_state, presented_state, annotator, queue, zoom], outs, show_progress="hidden")
        res_right.click(on_resolve("right"), [tid_state, presented_state, annotator, queue, zoom], outs, show_progress="hidden")
        btn_redo.click(on_redo, [tid_state, presented_state, annotator, queue, zoom], outs)
        btn_undo.click(on_undo, [tid_state, annotator, queue, zoom], outs)
        tournament_dd.input(on_select, [tournament_dd, queue, zoom], outs)
        focus_dd.input(on_focus, [tid_state, focus_dd, queue, zoom], outs)
        queue.input(on_focus, [tid_state, gr.State(None), queue, zoom], outs)
        zoom.input(on_focus, [tid_state, focus_dd, queue, zoom], outs)
        import_btn.click(on_import, [manifest_dd], [import_msg, group_dd])
        refresh_outs = [manifest_dd, group_dd, dataset_box, diag_dd, import_msg]
        refresh_btn.click(on_refresh, None, refresh_outs)
        demo.load(on_refresh, None, refresh_outs)
        demo.load(lambda: gr.update(choices=group_choices(conn())), None, [group_dd])
        create_btn.click(on_create, [group_dd, size, seed, new_annotator, allow], [create_msg, tournament_dd, tid_state, tabs]).then(
            on_focus, [tid_state, gr.State(None), queue, zoom], outs)
        src_out = [src_pick, src_image, src_summary, src_status, src_reason, src_idx]
        load_btn.click(on_sources_load, [dataset_box], src_out)
        demo.load(on_sources_load, [dataset_box], src_out)
        src_pick.input(on_source_pick, [dataset_box, src_pick], src_out)
        src_save.click(lambda r, i, s, why: on_source_save(r, i, s, why, False), [dataset_box, src_idx, src_status, src_reason], src_out + [src_msg])
        src_save_next.click(lambda r, i, s, why: on_source_save(r, i, s, why, True), [dataset_box, src_idx, src_status, src_reason], src_out + [src_msg])
        extra_btn.click(on_extra, [tid_state, extra_mode, extra_count], [extra_msg])
        export_btn.click(on_export, [tid_state, export_scope, include_voided], [export_msg])
        diag_dd.input(on_diag, [diag_dd], [diag_table])
    return demo


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path(os.environ.get("CHAMELEON_ANNOTATION_DB", store.DEFAULT_DB)))
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    db = args.db if args.db.is_absolute() else REPO_ROOT / args.db
    print(f"Annotation database: {db}")
    build_app(db).launch(server_name="127.0.0.1", server_port=args.port, inbrowser=not args.no_browser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
