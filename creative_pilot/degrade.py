"""Controlled degradation: realistic, still-valid design weaknesses applied to an existing good layout.

Each operator models one named failure designers recognise (tiny text, flat hierarchy, wasted space, crowding,
misalignment, scattered groups, dominant decor/imagery, lopsided mass, edge-hugging). They only move/resize boxes
and scale font sizes through `layout_edit`, so protected fields cannot change and images keep their aspect ratio
(images are only ever scaled uniformly or translated). `severity` in [0, 1] sets mediocre (~.3) vs poor (~.8).
The operator name is provenance for development galleries; it is never a quality label.
"""

from __future__ import annotations

import copy
import random
import statistics

from creative_pilot.blocks import SourceContext
from creative_pilot.metrics import is_background
from engine.layout_edits import layout_edit

OPERATORS = ("tiny_text", "flat_hierarchy", "whitespace", "crowd", "misalign", "scatter_group", "dominant_decor",
             "image_dominance", "lopsided", "edge_hug")
WEAKNESS = dict(tiny_text="unnecessary text reduction", flat_hierarchy="weak hierarchy", whitespace="excessive empty space",
                crowd="overcrowding", misalign="inconsistent alignment", scatter_group="awkward group placement",
                dominant_decor="overly dominant decorative shape", image_dominance="poor image/text balance",
                lopsided="unbalanced composition", edge_hug="poor spacing at canvas edges")


def _els(c):
    return {e["id"]: e for e in c["elements"]}


def _apply(source_candidate: dict, changes: dict) -> dict:
    out = copy.deepcopy(source_candidate)
    tw, th = out["canvas"]["width"], out["canvas"]["height"]
    for ch in changes.values():  # edges that touch the canvas keep a 2-micro-unit margin: the validator uses centre +/- half-size floats
        if "x" in ch and ch["width"] < tw - 1e-6 and ch["x"] + ch["width"] > tw - 1e-6:
            ch["x"] = round(max(0.0, tw - ch["width"] - .002), 3)
        if "y" in ch and ch["height"] < th - 1e-6 and ch["y"] + ch["height"] > th - 1e-6:
            ch["y"] = round(max(0.0, th - ch["height"] - .002), 3)
    out["elements"] = [layout_edit(e, changes[e["id"]]) if e["id"] in changes else e for e in out["elements"]]
    return out


def _round(box):
    return {k: round(v, 3) for k, v in box.items()}


def _clamp_box(x, y, w, h, tw, th):
    w, h = min(w, tw), min(h, th)
    x, y = min(max(x, 0.0), tw - w), min(max(y, 0.0), th - h)
    return x, y, w, h


def _content_ids(c, tw, th):
    return [e["id"] for e in c["elements"] if not is_background(e, tw, th)]


def tiny_text(c, ctx, rng, sev, tw, th, meas):
    f = 1.0 - (.35 + .35 * sev)  # .65 .. .30
    ch = {e["id"]: dict(font_size=round(max(1.0, e["font_size"] * f), 2)) for e in c["elements"] if e["type"] == "text"}
    return _apply(c, ch) if ch else None


def flat_hierarchy(c, ctx, rng, sev, tw, th, meas):
    texts = [e for e in c["elements"] if e["type"] == "text"]
    if len(texts) < 2:
        return None
    sizes = sorted(e["font_size"] for e in texts)
    target = statistics.median(sizes)
    ch = {}
    for e in texts:
        s = e["font_size"]
        want = s + (target - s) * (.6 + .4 * sev)
        if want > s:  # growing small text: only as far as it still fits its box
            while want > s and not meas.fits(e, e["width"], e["height"], want):
                want = (want + s) / 2
                if want - s < .2:
                    want = s
                    break
        ch[e["id"]] = dict(font_size=round(want, 2))
    out = _apply(c, ch)
    return out


def whitespace(c, ctx, rng, sev, tw, th, meas):
    f = 1.0 - (.28 + .34 * sev)  # .72 .. .38
    ids = _content_ids(c, tw, th)
    ch = {}
    for e in c["elements"]:
        if e["id"] not in ids:
            continue
        cx, cy = e["x"] + e["width"] / 2, e["y"] + e["height"] / 2
        nx, ny = tw / 2 + (cx - tw / 2) * f, th / 2 + (cy - th / 2) * f
        w, h = e["width"] * f, e["height"] * f
        d = dict(x=nx - w / 2, y=ny - h / 2, width=w, height=h)
        if e["type"] == "text":
            d["font_size"] = round(max(1.0, e["font_size"] * f), 2)
        ch[e["id"]] = _round(d) | ({"font_size": d["font_size"]} if "font_size" in d else {})
    return _apply(c, ch)


def crowd(c, ctx, rng, sev, tw, th, meas):
    f = 1.0 - (.25 + .4 * sev)
    ch = {}
    for e in c["elements"]:
        if is_background(e, tw, th):
            continue
        cx, cy = e["x"] + e["width"] / 2, e["y"] + e["height"] / 2
        nx, ny = tw / 2 + (cx - tw / 2) * f, th / 2 + (cy - th / 2) * f
        x, y, w, h = _clamp_box(nx - e["width"] / 2, ny - e["height"] / 2, e["width"], e["height"], tw, th)
        ch[e["id"]] = _round(dict(x=x, y=y, width=w, height=h))
    return _apply(c, ch)


def misalign(c, ctx, rng, sev, tw, th, meas):
    amp = (.04 + .1 * sev)
    ch = {}
    for u in ctx.units:
        dx = rng.uniform(-amp, amp) * tw if tw < th * 2 else rng.uniform(-amp, amp) * tw * .35
        dy = rng.uniform(-amp, amp) * th
        for i in u.ids:
            e = _els(c)[i]
            x, y, w, h = _clamp_box(e["x"] + dx, e["y"] + dy, e["width"], e["height"], tw, th)
            ch[i] = _round(dict(x=x, y=y, width=w, height=h))
    return _apply(c, ch)


def scatter_group(c, ctx, rng, sev, tw, th, meas):
    groups = [u for u in ctx.units if u.kind == "group" and len(u.ids) >= 3] or \
             [m for u in ctx.units for m in u.members if len(m.ids) >= 3]
    if not groups:
        return None
    g = rng.choice(groups)
    by_id = _els(c)
    members = [by_id[i] for i in g.ids if by_id[i]["type"] != "shape" or i != (g.frame or {}).get("id")]
    if len(members) < 2:
        return None
    x0 = min(e["x"] for e in members); y0 = min(e["y"] for e in members)
    x1 = max(e["x"] + e["width"] for e in members); y1 = max(e["y"] + e["height"] for e in members)
    stretch = 1.0 + 1.6 * (.3 + .7 * sev)
    ch = {}
    for e in members:
        nx = x0 + (e["x"] - x0) * (stretch if tw >= th else 1.0)
        ny = y0 + (e["y"] - y0) * (stretch if tw < th else 1.0)
        x, y, w, h = _clamp_box(nx, ny, e["width"], e["height"], tw, th)
        ch[e["id"]] = _round(dict(x=x, y=y, width=w, height=h))
    return _apply(c, ch)


def dominant_decor(c, ctx, rng, sev, tw, th, meas):
    cand = [d for d in ctx.decor if d.kind in ("accent", "backdrop")]
    if not cand:
        return None
    d = rng.choice(cand)
    by_id = _els(c)
    e = by_id[d.el["id"]]
    z = e["z_index"]
    # only grow a shape that sits beneath everything it will overlap, so nothing is hidden
    f = 1.8 + 2.2 * sev
    w, h = min(e["width"] * f, tw), min(e["height"] * f, th)
    x, y, w, h = _clamp_box(e["x"] + e["width"] / 2 - w / 2, e["y"] + e["height"] / 2 - h / 2, w, h, tw, th)
    for o in c["elements"]:
        if o["id"] == e["id"] or is_background(o, tw, th) or o["z_index"] > z:
            continue
        ow = max(0.0, min(x + w, o["x"] + o["width"]) - max(x, o["x"]))
        oh = max(0.0, min(y + h, o["y"] + o["height"]) - max(y, o["y"]))
        if ow * oh > .02 * o["width"] * o["height"]:
            return None
    return _apply(c, {e["id"]: _round(dict(x=x, y=y, width=w, height=h))})


def image_dominance(c, ctx, rng, sev, tw, th, meas):
    imgs = [e for e in c["elements"] if e["type"] == "image" and not is_background(e, tw, th)]
    if not imgs:
        return None
    e = max(imgs, key=lambda i: i["width"] * i["height"])
    f = min(1.5 + 1.4 * sev, tw / e["width"], th / e["height"])
    if f <= 1.15:
        return None
    w, h = e["width"] * f, e["height"] * f
    x, y, w, h = _clamp_box(e["x"] + e["width"] / 2 - w / 2, e["y"] + e["height"] / 2 - h / 2, w, h, tw, th)
    return _apply(c, {e["id"]: _round(dict(x=x, y=y, width=w, height=h))})


def lopsided(c, ctx, rng, sev, tw, th, meas):
    ids = _content_ids(c, tw, th)
    els = [e for e in c["elements"] if e["id"] in ids]
    if not els:
        return None
    x0 = min(e["x"] for e in els); y0 = min(e["y"] for e in els)
    x1 = max(e["x"] + e["width"] for e in els); y1 = max(e["y"] + e["height"] for e in els)
    horizontal = tw >= th
    slack = (tw - (x1 - x0)) if horizontal else (th - (y1 - y0))
    if slack < .05 * (tw if horizontal else th):
        return None
    shift = (slack / 2) * (.5 + .5 * sev) * rng.choice((-1, 1))
    if horizontal:
        lo, hi = -x0, tw - x1
        shift = max(lo, min(hi, shift))
    else:
        lo, hi = -y0, th - y1
        shift = max(lo, min(hi, shift))
    ch = {}
    for e in els:
        ch[e["id"]] = _round(dict(x=e["x"] + (shift if horizontal else 0), y=e["y"] + (0 if horizontal else shift),
                                  width=e["width"], height=e["height"]))
    return _apply(c, ch)


def edge_hug(c, ctx, rng, sev, tw, th, meas):
    ids = _content_ids(c, tw, th)
    els = [e for e in c["elements"] if e["id"] in ids]
    if not els:
        return None
    x0 = min(e["x"] for e in els); y0 = min(e["y"] for e in els)
    x1 = max(e["x"] + e["width"] for e in els); y1 = max(e["y"] + e["height"] for e in els)
    fx = (tw / max(x1 - x0, 1e-6)) if (x1 - x0) < tw else 1.0
    fy = (th / max(y1 - y0, 1e-6)) if (y1 - y0) < th else 1.0
    if fx < 1.04 and fy < 1.04:
        return None
    mix = .6 + .4 * sev
    fx, fy = 1 + (fx - 1) * mix, 1 + (fy - 1) * mix
    ch = {}
    for e in els:
        nx = (e["x"] - x0) * fx
        ny = (e["y"] - y0) * fy
        x, y, w, h = _clamp_box(nx, ny, e["width"], e["height"], tw, th)
        ch[e["id"]] = _round(dict(x=x, y=y, width=w, height=h))
    return _apply(c, ch)


FUNCS = dict(tiny_text=tiny_text, flat_hierarchy=flat_hierarchy, whitespace=whitespace, crowd=crowd, misalign=misalign,
             scatter_group=scatter_group, dominant_decor=dominant_decor, image_dominance=image_dominance,
             lopsided=lopsided, edge_hug=edge_hug)


def degrade(candidate: dict, ctx: SourceContext, op: str, severity: float, seed, meas) -> dict | None:
    """Return a degraded copy of `candidate`, or None when the operator does not apply."""
    tw, th = candidate["canvas"]["width"], candidate["canvas"]["height"]
    return FUNCS[op](candidate, ctx, random.Random(f"{seed}:{op}"), severity, tw, th, meas)
