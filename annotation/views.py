"""HTML builders for the tournament UI. Plain Python (no Gradio), so they are unit-testable.

Annotators see neutral labels only (L1..Ln by bracket slot). Generator identity, strategy names and any
score are never rendered.
"""

from __future__ import annotations

import base64
import html
import io
import json
from functools import lru_cache
from pathlib import Path

from annotation import bracket, store
from annotation.candidates import resolve

ZOOMS = {"Fit": None, "100% (actual pixels)": 1.0, "200%": 2.0}
LABELS = {"A_BETTER": "A better", "B_BETTER": "B better", "TIE": "Tie", "BOTH_BAD": "Both bad", "SKIP": "Skipped"}


@lru_cache(maxsize=256)
def _data_uri(path: str, mtime: float, thumb: int | None) -> tuple[str, int, int]:
    from PIL import Image
    with Image.open(path) as image:
        image.load()
        width, height = image.size
        if thumb:
            image = image.convert("RGB")
            image.thumbnail((thumb, thumb))
            buf, mime = io.BytesIO(), "image/jpeg"
            image.save(buf, "JPEG", quality=80)
        else:
            buf, mime = io.BytesIO(), "image/png"
            image.save(buf, "PNG")
    return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode()}", width, height


def image_uri(path: str | Path, thumb: int | None = None) -> tuple[str, int, int]:
    p = resolve(path)
    return _data_uri(str(p), p.stat().st_mtime, thumb)


def viewer_html(path: str | Path | None, caption: str, zoom: str = "Fit") -> str:
    """Scrollable viewer. Both sides use the same zoom, so equal-size canvases display at equal scale."""
    if path is None:
        return f'<div class="cv"><div class="cap">{html.escape(caption)}</div><div class="empty">No match</div></div>'
    uri, width, _ = image_uri(path)
    factor = ZOOMS.get(zoom)
    # Fit may shrink a large canvas, but must not magnify narrow raster previews.
    style = f"width:100%;max-width:{width}px" if factor is None else f"width:{round(width * factor)}px;max-width:none"
    return (f'<div class="cv"><div class="cap">{html.escape(caption)}</div>'
            f'<div class="scroll"><img src="{uri}" style="{style}" alt="{html.escape(caption)}"></div></div>')


def label_of(state: store.TournamentState, candidate_id: str | None) -> str:
    return "TBD" if candidate_id is None else f"L{state.entrants.index(candidate_id) + 1}" if candidate_id in state.entrants else "?"


def match_title(state: store.TournamentState, m: store.MatchView) -> str:
    size = state.tournament["size"]
    if m.kind == "bracket":
        n = (1 << bracket.round_count(size)) >> m.round
        return f"{bracket.round_name(m.round, size)} {m.position + 1}" if n > 1 else "Final"
    return ("Repeat " if m.kind == "repeat" else "Extra ") + str(m.position)


def header_md(state: store.TournamentState, focus: store.MatchView | None) -> str:
    t = state.tournament
    done, total = store.progress(state)
    lines = [f"**Tournament `{t['tournament_id']}`** · source `{t['source_design_id'][-12:]}` "
             f"({t['template_family_id']}) · target **{t['target_width']}×{t['target_height']}** · "
             f"{t['size']} candidates · bracket {done}/{total} matches"]
    if state.complete:
        lines.append(f"### 🏆 Champion: {label_of(state, state.champion)}")
    if focus is not None:
        if focus.kind == "bracket":
            order = [m.match_id for m in state.bracket() if m.status != "bye"]
            line = f"### {match_title(state, focus)} · match {order.index(focus.match_id) + 1} of {total}"
            if done:
                line += f" · {done} already decided" + (" (resumed)" if not state.complete else "")
        else:
            line = f"### {match_title(state, focus)} (extra comparison; does not affect the bracket)"
        lines.append(line)
    if t["source_review_status"] != "approved":
        lines.append(f"_Source review status: {t['source_review_status']} (development override)._")
    return "\n\n".join(lines)


def _card(state, conn, m: store.MatchView, current_id: str | None) -> str:
    cls = ["m"]
    if m.match_id == current_id:
        cls.append("cur")
    if m.status == "unresolved":
        cls.append("unres")
    rows = []
    for cid in (m.a, m.b):
        if cid is None:
            rows.append('<div class="c tbd">TBD</div>')
            continue
        uri, _, _ = image_uri(store.get_candidate(conn, cid)["preview_path"], thumb=72)
        win = " win" if m.winner == cid else (" lose" if m.winner and m.winner != cid else "")
        rows.append(f'<div class="c{win}"><img src="{uri}"><b>{label_of(state, cid)}</b></div>')
    note = {"unresolved": f"{LABELS.get(m.judgment['label'], '')} · unresolved" if m.judgment else "",
            "resolved": f"{LABELS.get(m.judgment['label'], '')} → manual advance" if m.judgment else "",
            "ready": "up next" if m.match_id == current_id else "", "decided": "", "waiting": "", "bye": "bye (no judgment)"}[m.status]
    return f'<div class="{" ".join(cls)}">{"".join(rows)}<div class="n">{html.escape(note)}</div></div>'


def bracket_html(conn, state: store.TournamentState, current_id: str | None) -> str:
    size = state.tournament["size"]
    columns = []
    for r in range(1, bracket.round_count(size) + 1):
        cards = "".join(_card(state, conn, state.by_slot(r, p), current_id) for p in range((1 << bracket.round_count(size)) >> r))
        columns.append(f'<div class="rd"><div class="rt">{bracket.round_name(r, size)}</div>{cards}</div>')
    champ = '<div class="c tbd">TBD</div>'
    if state.champion:
        uri, _, _ = image_uri(store.get_candidate(conn, state.champion)["preview_path"], thumb=96)
        champ = f'<div class="c win"><img src="{uri}"><b>{label_of(state, state.champion)}</b></div>'
    columns.append(f'<div class="rd"><div class="rt">Champion</div><div class="m champ">{champ}</div></div>')
    return f'<div class="bracket">{"".join(columns)}</div>'


def history_rows(state: store.TournamentState) -> list[list[str]]:
    rows = []
    for m in state.matches.values():
        if m.judgment is None:
            continue
        j = m.judgment
        right = j["candidate_b_id"] if j["screen_left_id"] == j["candidate_a_id"] else j["candidate_a_id"]
        advanced = label_of(state, m.winner) if m.kind == "bracket" and m.winner else ""
        how = "manual" if m.resolution is not None else ("vote" if advanced else "")
        rows.append([m.match_id.split(":", 1)[1], match_title(state, m), f"{label_of(state, j['screen_left_id'])} | {label_of(state, right)}",
                     LABELS[j["label"]], f"{advanced} ({how})" if advanced else "", ", ".join(json.loads(j["reason_tags"])),
                     j["created_at_utc"][:19].replace("T", " ")])
    return rows


CSS = """
.bracket{display:flex;gap:18px;overflow-x:auto;padding:6px 2px;align-items:stretch}
.rd{display:flex;flex-direction:column;justify-content:space-around;gap:8px;min-width:130px}
.rt{font-weight:600;font-size:12px;text-transform:uppercase;opacity:.7;text-align:center}
.m{border:2px solid #8883;border-radius:8px;padding:4px;background:#8881}
.m.cur{border-color:#2f7bff;box-shadow:0 0 0 2px #2f7bff55}.m.unres{border-color:#e0a020}.m.champ{border-color:#d4a017}
.c{display:flex;align-items:center;gap:6px;padding:2px 4px;border-radius:5px;font-size:13px}
.c img{height:26px;width:auto;max-width:70px;border-radius:3px}.c.tbd{opacity:.5;min-height:30px}
.c.win{background:#2e9e5b33;font-weight:700}.c.lose{opacity:.45}.n{font-size:11px;opacity:.75;text-align:center;min-height:14px}
.cv{border:1px solid #8884;border-radius:8px;overflow:hidden}.cap{padding:4px 10px;font-weight:700;background:#8882}
.scroll{overflow:auto;height:min(58vh,560px);background:#8881}.scroll img{display:block;height:auto}
.empty{padding:40px;text-align:center;opacity:.6}
#btn-left,#btn-right{min-height:54px;font-size:18px}
"""

SHORTCUT_JS = """() => {
  const click = id => { const el = document.querySelector('#' + id); if (el && !el.disabled) el.click(); };
  document.addEventListener('keydown', e => {
    const t = e.target.tagName; if (t === 'INPUT' || t === 'TEXTAREA' || e.ctrlKey || e.metaKey || e.altKey) return;
    const k = e.key.toLowerCase();
    const map = {arrowleft: 'btn-left', arrowright: 'btn-right', t: 'btn-tie', b: 'btn-bad', s: 'btn-skip', u: 'btn-undo'};
    if (map[k]) { e.preventDefault(); click(map[k]); }
  });
}"""

