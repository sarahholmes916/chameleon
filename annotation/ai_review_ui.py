"""Gradio tab "Review AI labels": accept / reject / (optionally) correct proposed pairwise labels, one click each."""

from __future__ import annotations

import html
from pathlib import Path

import gradio as gr

from annotation import ai_review, views
from annotation.candidates import REPO_ROOT

TARGETS = ["All", "1600x400", "728x90", "90x728"]
ORDERS = [("In batch order", "sequence"), ("Least confident first", "uncertain_first"), ("Most confident first (fast bulk accept)", "confident_first")]
NICE = {"A_BETTER": "A is better", "B_BETTER": "B is better", "TIE": "Tie", "BOTH_BAD": "Both bad", "SKIP": "No proposal (ambiguous)"}

REVIEW_JS = """() => {
  const click = id => { const el = document.querySelector('#' + id); if (el && !el.disabled) el.click(); };
  document.addEventListener('keydown', e => {
    const t = e.target.tagName; if (t === 'INPUT' || t === 'TEXTAREA' || e.ctrlKey || e.metaKey || e.altKey) return;
    const map = {a: 'rv-accept', enter: 'rv-accept', x: 'rv-reject', r: 'rv-reject', '1': 'rv-a', '2': 'rv-b', '3': 'rv-tie', '4': 'rv-bad', '5': 'rv-gb', '6': 'rv-bg', s: 'rv-skip', z: 'rv-undo'};
    const k = e.key.toLowerCase();
    if (map[k] && document.querySelector('#rv-accept')) { e.preventDefault(); click(map[k]); }
  });
}"""


def combined_js() -> str:
    return f"() => {{ ({views.SHORTCUT_JS})(); ({REVIEW_JS})(); }}"


def _img(path, base, caption):
    p = Path(path)
    return views.viewer_html(p if p.is_absolute() else Path(base) / p, caption, "Fit")


def _source(path, base):
    """Original source design, shown above the pair at a capped height so the comparison stays on screen."""
    p = Path(path)
    uri, w, h = views.image_uri(p if p.is_absolute() else Path(base) / p)
    return (f'<div class="cv"><div class="cap">Original source design ({w}×{h})</div>'
            f'<div style="text-align:center"><img src="{uri}" style="max-height:230px;max-width:100%;width:auto" alt="source"></div></div>')


def add_review_tab(conn_fn, default_annotator: str = "local_annotator"):
    """Call inside `with gr.Tabs():`. Returns the components dict (handlers are wired here)."""
    with gr.Tab("Review AI labels", id="tab_ai"):
        cur = gr.State(None)
        skipped = gr.State([])
        gr.Markdown("**Accept** keeps Claude's proposed label as a human-verified label. **Reject** drops it from training (kept for audit). "
                    "Correct only if you disagree and want to supply the label. Keys: **A/Enter** accept · **X** reject · **1** A · **2** B · **3** tie · **4** both bad · **5** A good/B bad · **6** B good/A bad · **S** skip · **Z** undo.")
        with gr.Row():
            annotator = gr.Textbox(default_annotator, label="Annotator ID", scale=1)
            target = gr.Dropdown(TARGETS, value="All", label="Target size", scale=1)
            order = gr.Dropdown(ORDERS, value="sequence", label="Order", scale=2)
            blind = gr.Checkbox(False, label="Blind mode (hide proposal, label yourself)")
            incl_blind = gr.Checkbox(False, label="Include blind-evaluation subset (you label these yourself)")
        progress_md = gr.Markdown()
        source_html = gr.HTML()
        with gr.Row():
            left = gr.HTML()
            right = gr.HTML()
        proposal_md = gr.Markdown()
        with gr.Row():
            accept = gr.Button("✔ Accept & Next", variant="primary", elem_id="rv-accept", scale=3, size="lg")
            reject = gr.Button("✘ Reject & Next", variant="stop", elem_id="rv-reject", scale=3, size="lg")
        gr.Markdown("**Correct (optional):**")
        with gr.Row():
            b_a = gr.Button("A better", elem_id="rv-a")
            b_b = gr.Button("B better", elem_id="rv-b")
            b_gb = gr.Button("A good · B bad", elem_id="rv-gb")
            b_bg = gr.Button("B good · A bad", elem_id="rv-bg")
            b_tie = gr.Button("Tie", elem_id="rv-tie")
            b_bad = gr.Button("Both bad", elem_id="rv-bad")
            b_skip = gr.Button("Skip for now", elem_id="rv-skip")
            b_undo = gr.Button("Undo last", elem_id="rv-undo")
        msg = gr.Markdown()
        with gr.Accordion("Recent reviews and exports", open=False):
            hist_only = gr.Dropdown([("All", ""), ("Corrected", "correct"), ("Rejected", "reject"), ("Accepted", "accept")], value="", label="Show")
            hist = gr.Dataframe(headers=["Comparison", "Decision", "Proposed", "Human label", "Mode", "Target", "Time"], interactive=False)
            hist_btn = gr.Button("Refresh history")
            export_btn = gr.Button("Export verified labels (accepted + corrected)")
            export_msg = gr.Markdown()

    def filters(tgt, ib, sk=False):
        return dict(target=tgt, exclude_blind=not ib, exclude_skip=not sk)

    def render(cid, tgt, ordr, bl, ib, who, skip, message=""):
        c = conn_fn()
        ai_review.ensure_schema(c)
        item = ai_review.next_item(c, None, filters(tgt, ib), ordr, tuple(skip)) if cid is None else ai_review.get_item(c, cid)
        prog = ai_review.progress(c, None, filters(tgt, ib))
        pm = (f"**Progress ({tgt}):** {prog['reviewed']}/{prog['total']} reviewed · {prog['accepted']} accepted · "
              f"{prog['corrected']} corrected · {prog['rejected']} rejected · **{prog['remaining']} left**")
        if item is None:
            return (None, pm, "<i>No more comparisons for this filter. 🎉</i>", "", "", message or "Done with this filter.", "", skip)
        base = item.get("_base", str(REPO_ROOT))
        hide = bl or item.get("blind_eval_subset")
        if hide:
            prop = "🙈 **Blind mode** — choose A better / B better / Tie / Both bad. The AI proposal is revealed after you submit." \
                   + ("  _(This comparison is in the blind evaluation subset.)_" if item.get("blind_eval_subset") else "")
        else:
            ex = item["explanation"]
            conf = "n/a" if item.get("confidence") is None else f"{item['confidence']:.2f}"
            prop = (f"### Proposed: **{NICE[item['proposed_label']]}**  ·  confidence {conf} _(uncalibrated)_\n"
                    f"{html.escape(ex['summary'])}  \n_Method: `{item['evaluation_method']}` — machine-generated, unverified._\n\n"
                    f"- Typography: {ex['typography']}\n- Hierarchy: {ex['hierarchy']}\n- Balance/spacing: {ex['balance_spacing']}\n"
                    f"- Composition: {ex['composition']}\n- Space use: {ex['space_use']}\n- Trade-offs: {ex['tradeoffs']}")
        tw, th = item["target_width"], item["target_height"]
        src = item.get("source_preview_path")
        src_html = _source(src, base) if src else "<i>No source preview.</i>"
        return (item["comparison_id"], pm, _img(item["candidate_a_preview"], base, f"Candidate A  ({tw}×{th})"),
                _img(item["candidate_b_preview"], base, f"Candidate B  ({tw}×{th})"), prop, message, src_html, skip)

    outs = [cur, progress_md, left, right, proposal_md, msg, source_html, skipped]
    inputs = [cur, target, order, blind, incl_blind, annotator, skipped]

    def on_review_load(tgt, ordr, bl, ib, who, skip):
        return render(None, tgt, ordr, bl, ib, who, skip)

    def act(decision, label=None, aq=None, bq=None):
        def run(cid, tgt, ordr, bl, ib, who, skip):
            if cid is None:
                return render(None, tgt, ordr, bl, ib, who, skip, "Nothing to review.")
            c = conn_fn()
            item = ai_review.get_item(c, cid)
            hidden = bl or item.get("blind_eval_subset")
            mode = "blind" if hidden else ("guided" if decision == "correct" else "fast_confirm")
            if hidden and decision != "correct":
                return render(cid, tgt, ordr, bl, ib, who, skip, "🙈 Blind: choose A better / B better / Tie / Both bad.")
            try:
                ai_review.submit_review(c, cid, decision, (who or default_annotator).strip() or default_annotator, mode=mode, human_label=label, a_quality=aq, b_quality=bq)
            except ValueError as exc:
                return render(cid, tgt, ordr, bl, ib, who, skip, f"⚠ {exc}")
            note = {"accept": "✔ Accepted", "reject": "✘ Rejected", "correct": f"✎ Corrected to {NICE.get(label, label)}"}[decision]
            if hidden:
                same = "agrees with" if label == item["proposed_label"] else "differs from"
                note += f" — your label {same} the AI proposal ({NICE[item['proposed_label']]})."
            return render(None, tgt, ordr, bl, ib, who, skip, note)
        return run

    def on_skip(cid, tgt, ordr, bl, ib, who, skip):
        skip = list(skip) + ([cid] if cid else [])
        return render(None, tgt, ordr, bl, ib, who, skip, "Skipped for now (it stays unverified).")

    def on_undo(cid, tgt, ordr, bl, ib, who, skip):
        c = conn_fn()
        last = ai_review.last_review(c, (who or default_annotator).strip() or default_annotator)
        if not last:
            return render(cid, tgt, ordr, bl, ib, who, skip, "Nothing to undo.")
        ai_review.undo_review(c, last["comparison_id"])
        return render(last["comparison_id"], tgt, ordr, bl, ib, who, skip, f"↶ Undid the {last['decision']} — revise it now.")

    def on_hist(only):
        rows = ai_review.history(conn_fn(), 200, only or None)
        return [[r["comparison_id"][:10], r["decision"], r["proposed_label_at_review"], r["human_label"] or "—", r["verification_mode"],
                 f"{r['target_width']}x{r['target_height']}", r["created_at_utc"][:19]] for r in rows]

    def on_export():
        out = Path(conn_fn().execute("PRAGMA database_list").fetchone()["file"]).parent / "ai_review_exports"
        c = conn_fn()
        v, r, u = ai_review.export_verified(c, out), ai_review.export_rejected(c, out), ai_review.export_unverified(c, out)
        return (f"Exported to `{out}`: **{v['rows']}** human-verified (`synthetic_human_verified.jsonl`), {r['rows']} rejected (audit only), "
                f"{u['rows']} still-unverified (weak supervision, opt-in only).")

    for comp in (target, order, blind, incl_blind):
        comp.input(lambda t, o, b, i, w, s: render(None, t, o, b, i, w, s), [target, order, blind, incl_blind, annotator, skipped], outs)
    accept.click(act("accept"), inputs, outs, show_progress="hidden")
    reject.click(act("reject"), inputs, outs, show_progress="hidden")
    for btn, lab in ((b_a, "A_BETTER"), (b_b, "B_BETTER"), (b_tie, "TIE"), (b_bad, "BOTH_BAD")):
        btn.click(act("correct", lab), inputs, outs, show_progress="hidden")
    b_gb.click(act("correct", "A_BETTER", "good", "bad"), inputs, outs, show_progress="hidden")
    b_bg.click(act("correct", "B_BETTER", "bad", "good"), inputs, outs, show_progress="hidden")
    b_skip.click(on_skip, inputs, outs, show_progress="hidden")
    b_undo.click(on_undo, inputs, outs, show_progress="hidden")
    hist_btn.click(on_hist, [hist_only], [hist])
    hist_only.input(on_hist, [hist_only], [hist])
    export_btn.click(on_export, None, [export_msg])
    return dict(load=(on_review_load, [target, order, blind, incl_blind, annotator, skipped], outs))
