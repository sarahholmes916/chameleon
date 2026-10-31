"""Source-aware decomposition of an Adobe design into layout units, decor and background.

Roles come only from geometry, type, font size, grouping and containment (never from IDs or `_note` text).
A *unit* is the thing an operator places: one text, one image, or one explicit group (card, button, cluster).
Decor (rules, accents, backdrops) is not placed independently; it follows an anchor unit (see operators.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field


def center(e):
    return e["x"] + e["width"] / 2, e["y"] + e["height"] / 2


def contains_center(shape, e):
    cx, cy = center(e)
    return shape["x"] <= cx <= shape["x"] + shape["width"] and shape["y"] <= cy <= shape["y"] + shape["height"]


def bbox(els):
    return (min(e["x"] for e in els), min(e["y"] for e in els),
            max(e["x"] + e["width"] for e in els), max(e["y"] + e["height"] for e in els))


@dataclass
class Unit:
    uid: str
    kind: str                       # text | image | group
    els: list
    role: str = "body"
    frame: dict | None = None       # card/button background shape inside a group
    texts: list = field(default_factory=list)
    images: list = field(default_factory=list)
    pills: list = field(default_factory=list)       # [(shape, [text, ...])]
    ornaments: list = field(default_factory=list)   # other shapes inside a group
    weight: float = 1.0
    members: list = field(default_factory=list)     # series: sibling group units placed as one equal-size set

    @property
    def box(self):
        return bbox(self.els)

    @property
    def z(self):
        return min(e["z_index"] for e in self.els)

    @property
    def ids(self):
        return [e["id"] for e in self.els]

    @property
    def all_texts(self):
        if self.members:
            return [t for m in self.members for t in m.all_texts]
        return list(self.texts) + [t for _, ts in self.pills for t in ts]

    @property
    def max_font(self):
        if self.members:
            return max(m.max_font for m in self.members)
        fonts = [t["font_size"] for t in self.texts] + [t["font_size"] for _, ts in self.pills for t in ts]
        return max(fonts) if fonts else 0.0


@dataclass
class Decor:
    el: dict
    kind: str                        # rule | accent | backdrop
    anchors: list                    # unit ids whose centres lie inside (backdrop) or nearest unit (rule/accent)


@dataclass
class SourceContext:
    source: dict
    width: int
    height: int
    bg: list
    units: list
    decor: list
    order: list                      # unit ids in reading order
    max_font: float

    def unit(self, uid):
        return next(u for u in self.units if u.uid == uid)

    @property
    def canvas_area(self):
        return self.width * self.height

    def role_units(self, role):
        return [u for u in self.units if u.role == role]


def _spans(e, W, H):
    return e["width"] >= .85 * W and e["height"] >= .85 * H and (
        e["type"] == "shape" or (e["type"] == "image" and e.get("fit", "contain") == "cover"))


def _group_unit(uid, members):
    shapes = [e for e in members if e["type"] == "shape"]
    texts = [e for e in members if e["type"] == "text"]
    images = [e for e in members if e["type"] == "image"]
    others = [e for e in members if e["type"] != "shape"]
    frame = None
    if shapes:
        cand = max(shapes, key=lambda s: s["width"] * s["height"])
        inside = [e for e in others if contains_center(cand, e)]
        if others and len(inside) >= .7 * len(others):
            frame = cand
    pills, pill_text_ids, ornaments = [], set(), []
    for s in shapes:
        if s is frame:
            continue
        inside = [t for t in texts if contains_center(s, t)]
        if inside:
            pills.append((s, inside))
            pill_text_ids.update(t["id"] for t in inside)
        else:
            ornaments.append(s)
    return Unit(uid, "group", sorted(members, key=lambda e: e["z_index"]), frame=frame,
                texts=[t for t in texts if t["id"] not in pill_text_ids], images=images, pills=pills,
                ornaments=ornaments)


def _signature(u):
    return (u.kind, u.role, len(u.texts), len(u.pills), len(u.images), u.frame is not None, len(u.ornaments))


def _make_series(units: list) -> None:
    """Replace same-structure, same-size, row/column-aligned sibling groups by one `series` unit (in place)."""
    by_sig: dict = {}
    for u in units:
        if u.kind == "group":
            by_sig.setdefault(_signature(u), []).append(u)
    for sig, us in by_sig.items():
        if len(us) < 2 or sig[1] == "button":
            continue
        ws = sorted(u.box[2] - u.box[0] for u in us)
        hs = sorted(u.box[3] - u.box[1] for u in us)
        mw, mh = ws[len(ws) // 2], hs[len(hs) // 2]
        sel = [u for u in us if abs((u.box[2] - u.box[0]) - mw) <= .3 * mw and abs((u.box[3] - u.box[1]) - mh) <= .3 * mh]
        if len(sel) < 2:
            continue
        cy = [(u.box[1] + u.box[3]) / 2 for u in sel]
        cx = [(u.box[0] + u.box[2]) / 2 for u in sel]
        if not (max(cy) - min(cy) <= .5 * mh or max(cx) - min(cx) <= .5 * mw):
            continue
        sel.sort(key=lambda u: (u.box[1], u.box[0]) if max(cy) - min(cy) > .5 * mh else (u.box[0], u.box[1]))
        first = units.index(sel[0])
        series = Unit("s_" + sel[0].uid, "series", sorted([e for u in sel for e in u.els], key=lambda e: e["z_index"]),
                      role="series", members=sel)
        for u in sel:
            units.remove(u)
        units.insert(min(first, len(units)), series)


def _make_bundles(units: list, loose_shapes: list, canvas_area: float) -> set:
    """An opaque backdrop plus the units whose centres lie inside it become one `bundle` placed together."""
    consumed = set()
    for sh in sorted(loose_shapes, key=lambda x: -x["width"] * x["height"]):
        if sh.get("opacity", 1) < .7 or sh["width"] * sh["height"] / canvas_area < .04:
            continue
        thin = min(sh["width"], sh["height"]) <= .02 * max(sh["width"], sh["height"])
        inside = [u for u in units if u.kind != "bundle" and contains_center(
            sh, dict(x=u.box[0], y=u.box[1], width=u.box[2] - u.box[0], height=u.box[3] - u.box[1]))]
        if thin or not inside:
            continue
        for u in inside:
            units.remove(u)
        first = min((u.box[1], u.box[0]) for u in inside)
        bundle = Unit("b_" + sh["id"], "bundle", sorted([sh] + [e for u in inside for e in u.els], key=lambda e: e["z_index"]),
                      role="panel", frame=sh, members=sorted(inside, key=lambda u: (u.box[1], u.box[0])))
        units.append(bundle)
        consumed.add(sh["id"])
    return consumed


def analyse(source: dict) -> SourceContext:
    W, H = source["canvas"]["width"], source["canvas"]["height"]
    els = source["elements"]
    bg = [e for e in els if _spans(e, W, H)]
    bg_ids = {e["id"] for e in bg}
    units, groups, loose_shapes = [], {}, []
    for e in els:
        if e["id"] in bg_ids:
            continue
        if e.get("group_id") is not None:
            groups.setdefault(e["group_id"], []).append(e)
        elif e["type"] == "text":
            units.append(Unit(f"u_{e['id']}", "text", [e], texts=[e]))
        elif e["type"] == "image":
            units.append(Unit(f"u_{e['id']}", "image", [e], images=[e]))
        else:
            loose_shapes.append(e)
    for gid, members in groups.items():
        units.append(_group_unit(f"g_{gid}", members))
    max_font = max([t["font_size"] for e in els for t in [e] if e["type"] == "text"] or [1.0])
    canvas_area = W * H
    # roles
    for u in units:
        if u.kind == "text":
            ratio = u.texts[0]["font_size"] / max_font
            u.role = "headline" if ratio >= .75 else "sub" if ratio >= .4 else "body"
        elif u.kind == "image":
            e = u.images[0]
            area = e["width"] * e["height"] / canvas_area
            u.role = "hero" if area >= .06 else "logo" if e["width"] / max(e["height"], 1) >= 2.5 else "photo"
        else:
            words = sum(len(t["content"].split()) for t in u.texts) + sum(len(t["content"].split()) for _, ts in u.pills for t in ts)
            fa = (u.frame["width"] * u.frame["height"] / canvas_area) if u.frame else 0
            u.role = "button" if (u.frame is not None and words <= 4 and fa < .05 and not u.images) else (
                "card" if u.frame is not None else "cluster")
    # headline: the largest-type text-bearing unit wins ties by area
    textual = [u for u in units if u.max_font > 0]
    if textual:
        head = max(textual, key=lambda u: (u.max_font, u.box[2] - u.box[0]))
        if head.role not in ("button",):
            head.role = "headline" if head.kind == "text" else head.role
    _make_series(units)
    consumed = _make_bundles(units, loose_shapes, canvas_area)
    loose_shapes = [sh for sh in loose_shapes if sh["id"] not in consumed]
    weights = dict(panel=2.0, series=2.4, headline=3.0, hero=2.2, card=1.5, sub=1.3, body=1.0, button=.8, logo=.7, photo=1.2, cluster=1.2)
    for u in units:
        length = sum(len(t["content"]) for t in u.texts) + sum(len(t["content"]) for _, ts in u.pills for t in ts) +             sum(len(t["content"]) for m in u.members for t in m.texts)
        u.weight = weights.get(u.role, 1.0) * (1 + min(length, 200) / 200 * .5)
    # decor: loose shapes, classified against units
    decor = []
    mn = min(W, H)
    for s in loose_shapes:
        area = s["width"] * s["height"] / canvas_area
        thin = min(s["width"], s["height"]) <= .02 * max(W, H) and max(s["width"], s["height"]) / max(min(s["width"], s["height"]), 1e-6) >= 6
        inside = [u.uid for u in units if contains_center(s, dict(zip(("x", "y"), u.box[:2]), width=u.box[2] - u.box[0],
                                                                  height=u.box[3] - u.box[1]))]
        if thin:
            kind, anchors = "rule", []
        elif inside and area >= .04:
            kind, anchors = "backdrop", inside
        else:
            kind, anchors = "accent", []
        if not anchors and units:
            sc = center(s)
            nearest = min(units, key=lambda u: ((u.box[0] + u.box[2]) / 2 - sc[0]) ** 2 + ((u.box[1] + u.box[3]) / 2 - sc[1]) ** 2)
            anchors = [nearest.uid]
        decor.append(Decor(s, kind, anchors))
    # reading order: row-major with rows formed by vertical overlap
    ordered = sorted(units, key=lambda u: ((u.box[1] + u.box[3]) / 2, u.box[0]))
    rows, row = [], []
    for u in ordered:
        if row:
            top = min(x.box[1] for x in row)
            bottom = max(x.box[3] for x in row)
            ov = min(bottom, u.box[3]) - max(top, u.box[1])
            if ov >= .5 * min(u.box[3] - u.box[1], bottom - top):
                row.append(u)
                continue
            rows.append(row)
            row = []
        row.append(u)
    if row:
        rows.append(row)
    order = [u.uid for r in rows for u in sorted(r, key=lambda x: x.box[0])]
    return SourceContext(source, W, H, bg, units, decor, order, max_font)
