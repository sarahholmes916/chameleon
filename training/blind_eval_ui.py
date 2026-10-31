"""Gradio front end for the blind layout comparison (see training/blind_eval.py).  Imported lazily: gradio is not an ML dependency."""

from __future__ import annotations

import html
import json
from pathlib import Path

from training import blind_eval as B

JS = """() => {
  document.addEventListener('keydown', e => {
    const t = e.target.tagName; if (t === 'INPUT' || t === 'TEXTAREA' || e.ctrlKey || e.metaKey || e.altKey) return;
    const map = {'1': 'be-1', '2': 'be-2', '3': 'be-3', t: 'be-tie', b: 'be-bad', z: 'be-undo'};
    const id = map[e.key.toLowerCase()]; const el = id && document.querySelector('#' + id);
    if (el && !el.disabled && el.offsetParent !== null) { e.preventDefault(); el.click(); }
  });
}"""


def task_html(task: dict) -> str:
    from annotation import views
    tw, th = task["target"]
    wide = tw >= th
    src_uri, sw, sh = views.image_uri(task["source_preview"])
    parts = [f'<div style="margin-bottom:6px"><b>Target canvas: {tw}×{th} px</b> &nbsp; · &nbsp; original source is {sw}×{sh} px</div>',
             f'<div style="text-align:center;margin-bottom:12px"><div style="font-size:12px;opacity:.7">Original source design</div>'
             f'<img src="{src_uri}" style="max-height:220px;max-width:100%;width:auto;border:1px solid #8884" alt="source"></div>']
    boxes = []
    for o in task["options"]:
        uri, w, h = views.image_uri(o["preview"])
        style = f"width:100%;max-width:{w}px" if wide else "max-height:620px;max-width:100%;width:auto"
        boxes.append(f'<div style="flex:1 1 0;min-width:0;margin:6px;text-align:center"><div style="font-weight:600;margin-bottom:3px">Option {o["position"]}</div>'
                     f'<img src="{uri}" style="{style};border:1px solid #8884" alt="option {o["position"]}"></div>')
    direction = "column" if wide else "row"
    parts.append(f'<div style="display:flex;flex-direction:{direction};align-items:{"stretch" if wide else "flex-start"}">{"".join(boxes)}</div>')
    return "".join(parts)


def reveal_markdown(private: dict, conn, annotator: str | None) -> str:
    result = B.tally(private, conn, annotator)
    n = result["judged_task_judgments"]
    lines = [f"### Results revealed — {n} tasks judged", f"Outcomes: {result['outcomes']}", "",
             "| Method | best (any) | best (alone) | best (shared) | tie | all bad | not chosen |", "|---|---|---|---|---|---|---|"]
    for m, c in result["per_method"].items():
        lines.append(f"| `{m}` | {c['best_any']} | {c.get('best_unique', 0)} | {c.get('best_shared', 0)} | {c.get('tie', 0)} | {c.get('all_bad', 0)} | {c.get('not_chosen', 0)} |")
    lines += ["", "**Head to head** (only tasks where the two methods picked different layouts)", "",
              "| Pair | first won | second won | tie / all bad | third option won | same pick | sign-test p |", "|---|---|---|---|---|---|---|"]
    for k, c in result["head_to_head"].items():
        p = c["sign_test_p"]
        lines.append(f"| {k} | {c.get('a_wins', 0)} | {c.get('b_wins', 0)} | {c.get('tie_or_all_bad', 0)} | {c.get('third_option_won', 0)} | {c.get('same_pick', 0)} | {'–' if p is None else f'{p:.3f}'} |")
    lines += ["", "**Per task**", "", "| Task | Target | Your choice | Methods behind each option |", "|---|---|---|---|"]
    for d in result["detail"]:
        opts = "; ".join(f"{k}: {', '.join(v)}" for k, v in d["options"].items())
        choice = d["choice"] if d["choice"] in (B.TIE, B.ALL_BAD) else "best = " + ", ".join(d["chosen_methods"])
        lines.append(f"| {html.escape(d['task_id'][:34])} | {d['target']} | {choice} | {opts} |")
    return "\n".join(lines)


def build_app(out: Path, default_annotator: str):
    import gradio as gr
    public = json.loads((out / "gallery_manifest.json").read_text(encoding="utf-8"))
    total = len(public["tasks"])
    private_path = out / "private" / "answer_key.json"

    def db():
        return B.connect(out / B.DB_NAME)

    def clean(annotator):
        return (annotator or default_annotator).strip() or default_annotator

    def view(annotator: str, note: str = ""):
        conn = db()
        task = B.next_task(public, conn, annotator)
        done = len(B.judged_ids(conn, annotator))
        conn.close()
        head = f"**{annotator}: {done} of {total} judged**" + (f" · {note}" if note else "")
        if task is None:
            return (None, head + " · 🎉 **All done.** Open the *Reveal results* tab.", "<i>No more tasks.</i>",
                    *[gr.update(visible=False)] * 3, gr.update(interactive=False), gr.update(interactive=False))
        n = len(task["options"])
        return (task["task_id"], head, task_html(task), *[gr.update(visible=i < n) for i in range(3)],
                gr.update(interactive=True), gr.update(interactive=True))

    def judge(choice):
        def run(task_id, annotator):
            annotator = clean(annotator)
            if task_id is None:
                return view(annotator, "nothing to judge")
            conn = db()
            try:
                B.record(conn, public, task_id, annotator, choice)
            except ValueError as exc:
                conn.close()
                return view(annotator, f"⚠ {exc}")
            conn.close()
            return view(annotator, "saved")
        return run

    def undo(task_id, annotator):
        annotator = clean(annotator)
        conn = db()
        undone = B.undo_last(conn, annotator)
        conn.close()
        return view(annotator, "undid the last judgment" if undone else "nothing to undo")

    def reveal(annotator, even_if_unfinished):
        annotator = clean(annotator)
        conn = db()
        finished = len(B.judged_ids(conn, annotator)) >= total
        if not finished and not even_if_unfinished:
            conn.close()
            return f"Finish all {total} tasks first (or tick the box to reveal early — that ends the blind)."
        text = reveal_markdown(json.loads(private_path.read_text(encoding="utf-8")), conn, annotator)
        conn.close()
        return text

    with gr.Blocks(title="Blind layout comparison", js=JS) as demo:
        gr.Markdown("# Which layout is best?\nEach task shows the original design and 2–3 anonymous resized layouts. Pick the best one, or say they are tied. "
                    "Judge only the final look. Keys: **1 / 2 / 3** best · **T** tie · **B** all bad · **Z** undo.")
        with gr.Tabs():
            with gr.Tab("Compare"):
                annotator = gr.Textbox(default_annotator, label="Annotator ID")
                cur = gr.State(None)
                progress = gr.Markdown()
                body = gr.HTML()
                with gr.Row():
                    b1 = gr.Button("Option 1 is best", elem_id="be-1", variant="primary", size="lg")
                    b2 = gr.Button("Option 2 is best", elem_id="be-2", variant="primary", size="lg")
                    b3 = gr.Button("Option 3 is best", elem_id="be-3", variant="primary", size="lg")
                with gr.Row():
                    tie = gr.Button("Tie / no clear difference", elem_id="be-tie")
                    bad = gr.Button("All unacceptable", elem_id="be-bad")
                    undo_b = gr.Button("Undo last", elem_id="be-undo")
                outs = [cur, progress, body, b1, b2, b3, tie, bad]
                ins = [cur, annotator]
                for btn, ch in ((b1, "1"), (b2, "2"), (b3, "3"), (tie, B.TIE), (bad, B.ALL_BAD)):
                    btn.click(judge(ch), ins, outs, show_progress="hidden")
                undo_b.click(undo, ins, outs, show_progress="hidden")
                demo.load(lambda a: view(clean(a)), [annotator], outs)
            with gr.Tab("Reveal results"):
                early = gr.Checkbox(False, label="Reveal even if I have not finished (ends the blind)")
                go = gr.Button("Reveal identities and tally")
                result = gr.Markdown()
                go.click(reveal, [annotator, early], [result])
    return demo


def launch(out: Path, port: int, default_annotator: str) -> int:
    build_app(out, default_annotator).launch(server_name="127.0.0.1", server_port=port, inbrowser=False)
    return 0
