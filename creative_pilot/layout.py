"""Unit and cell fitting with renderer-measured text.

A *cell* is a rectangle holding an ordered list of units, laid out either as a vertical stack or as a single
horizontal row. The type scale `k` (font = k * emphasis * source-font, optionally reshaped by a hierarchy
exponent) is searched as the LARGEST value at which the whole cell fits, so restructuring - not blind
shrinking - decides how big text ends up. All results are absolute boxes; only x/y/width/height/font_size
are ever produced (protected fields are untouched).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from creative_pilot.blocks import SourceContext, Unit
from creative_pilot.fitting import Measurer

MAX_K_ITER = 11


@dataclass
class UL:
    """Layout of one unit relative to its own top-left corner."""
    w: float
    h: float
    items: list = field(default_factory=list)   # dict(id, x, y, w, h, font?)
    min_font: float | None = None
    max_font: float | None = None
    need_w: float = 0.0          # width below which some word overflows its line


@dataclass
class Fit:
    ok: bool
    k: float = 0.0
    placements: dict = field(default_factory=dict)   # id -> dict(x, y, width, height, font_size?)
    used: tuple = (0.0, 0.0, 0.0, 0.0)               # bounding box of everything placed (x0, y0, x1, y1)
    min_font: float | None = None
    max_font: float | None = None
    slack: float = 0.0                               # unused fraction along the stacking axis


class Fitter:
    def __init__(self, meas: Measurer, ctx: SourceContext):
        self.m = meas
        self.ctx = ctx
        self.floor = 1.0   # soft readability floor (px) for this plan; set by operators.realize

    # ------------------------------------------------------------------ type scale
    def fs(self, f: float, k: float, emph: float = 1.0, gamma: float = 1.0) -> float:
        fm = self.ctx.max_font
        return max(self.floor, k * emph * fm * (f / fm) ** gamma)

    # ------------------------------------------------------------------ single units
    def _group_children(self, u: Unit):
        kids = [("text", t) for t in u.texts] + [("image", i) for i in u.images] + \
               [("pill", p) for p in u.pills] + [("orn", s) for s in u.ornaments]

        def key(c):
            e = c[1][0] if c[0] == "pill" else c[1]
            return e["y"], e["x"]
        return sorted(kids, key=key)

    def unit_layout(self, u: Unit, w: float, k: float, o: dict, mode: str = "stack", h_avail: float | None = None) -> UL | None:
        emph = o.get("emph", {}).get(u.uid, 1.0)
        gamma = o.get("gamma", 1.0)
        img_cap = o.get("img_cap", 1e9)
        if u.kind == "text":
            t = u.texts[0]
            s = self.fs(t["font_size"], k, emph, gamma)
            nw, h = self.m.need(t, w, s)
            return UL(w, h, [dict(id=t["id"], x=0, y=0, w=w, h=h, font=s)], s, s, nw)
        if u.kind == "image":
            e = u.images[0]
            if e.get("fit", "contain") == "cover":
                h = min(img_cap, max(w * o.get("cover_aspect", 1.0), 1.0))
                return UL(w, h, [dict(id=e["id"], x=0, y=0, w=w, h=h)])
            a = e["width"] / e["height"]
            h = min(w / a, img_cap)
            ww = h * a
            return UL(w, h, [dict(id=e["id"], x=(w - ww) / 2, y=0, w=ww, h=h)])
        if u.kind == "series":
            return self._series(u, w, k, o)
        if u.kind == "bundle":
            return self._bundle(u, w, k, o)
        return self._group_row(u, w, k, o, h_avail) if mode == "row" else self._group_stack(u, w, k, o)

    def _natural_w(self, u: Unit, k: float, o: dict) -> float:
        """Unwrapped width of a group's widest text plus padding (what a hugging frame needs)."""
        emph, gamma = o.get("emph", {}).get(u.uid, 1.0), o.get("gamma", 1.0)
        sizes = self._sizes(u, k, emph, gamma)
        best = 0.0
        for t in u.texts + [t for _, ts in u.pills for t in ts]:
            best = max(best, self.m.measure(t, 1e4, sizes[t["id"]])[1] + 3)
        pad = 0.0 if u.frame is None else max(2.0, .6 * max(sizes.values(), default=6.0))
        return best + 2 * pad

    def _series(self, u: Unit, w: float, k: float, o: dict) -> UL:
        """Sibling groups as one equal-size set: side by side when wide enough, otherwise stacked."""
        mem, n = u.members, len(u.members)
        ref = max((v for m in mem for v in self._sizes(m, k, 1.0, o.get("gamma", 1.0)).values()), default=6.0)
        gap = max(2.0, .45 * ref)
        nat = max(self._natural_w(m, k, o) for m in mem)
        cols = n if w >= n * nat + (n - 1) * gap else 1
        mw = (w - (n - 1) * gap) / n if cols > 1 else w
        oo = dict(o, hug=False)
        uls = [self._group_stack(m, mw, k, oo) for m in mem]
        H = max(ul.h for ul in uls)
        items, mins, maxs, x, y = [], [], [], 0.0, 0.0
        for m, ul in zip(mem, uls):
            for it in ul.items:
                it = dict(it, x=it["x"] + x, y=it["y"] + y)
                if m.frame is not None and it["id"] == m.frame["id"]:
                    it["h"] = H
                items.append(it)
            if ul.min_font is not None:
                mins.append(ul.min_font)
                maxs.append(ul.max_font)
            if cols > 1:
                x += mw + gap
            else:
                y += H + gap
        total = H if cols > 1 else n * H + (n - 1) * gap
        need = (n * max(ul.need_w for ul in uls) + (n - 1) * gap) if cols > 1 else max(ul.need_w for ul in uls)
        return UL(w, total, items, min(mins) if mins else None, max(maxs) if maxs else None, need)

    def _bundle(self, u: Unit, w: float, k: float, o: dict) -> UL | None:
        """A backdrop panel around its own member units, stacked in reading order with padding."""
        sizes = [self.fs(t["font_size"], k, 1.0, o.get("gamma", 1.0)) for t in u.all_texts]
        ref = (sum(sizes) / len(sizes)) if sizes else 6.0
        pad = max(2.0, min(.07 * w, .6 * max(sizes, default=6.0)))
        inner = max(w - 2 * pad, 2.0)
        gap = max(1.0, .3 * ref)
        items, y, mins, maxs, need = [], pad, [], [], 0.0
        for i, m in enumerate(u.members):
            if i:
                y += gap
            ul = self.unit_layout(m, inner, k, o, "stack")
            if ul is None:
                return None
            for it in ul.items:
                items.append(dict(it, x=it["x"] + pad, y=it["y"] + y))
            if ul.min_font is not None:
                mins.append(ul.min_font)
                maxs.append(ul.max_font)
            need = max(need, ul.need_w + 2 * pad)
            y += ul.h
        total = y + pad
        items.insert(0, dict(id=u.frame["id"], x=0, y=0, w=w, h=total))
        return UL(w, total, items, min(mins) if mins else None, max(maxs) if maxs else None, need)

    def _sizes(self, u: Unit, k, emph, gamma):
        out = {}
        for t in u.texts:
            out[t["id"]] = self.fs(t["font_size"], k, emph, gamma)
        for _, ts in u.pills:
            for t in ts:
                out[t["id"]] = self.fs(t["font_size"], k, emph, gamma)
        return out

    def _group_stack(self, u: Unit, w: float, k: float, o: dict) -> UL:
        emph, gamma = o.get("emph", {}).get(u.uid, 1.0), o.get("gamma", 1.0)
        sizes = self._sizes(u, k, emph, gamma)
        ref = (sum(sizes.values()) / len(sizes)) if sizes else 6.0
        if u.frame is not None and o.get("hug", True) and not u.images and sizes:
            w = min(w, max(self._natural_w(u, k, o), 12.0))
        pad = 0.0 if u.frame is None else max(2.0, min(.07 * w, .6 * max(sizes.values(), default=6.0)))
        inner = max(w - 2 * pad, 2.0)
        gap = max(1.0, .28 * ref)
        items, y = [], pad
        mins, maxs = [], []
        need_w = 0.0
        first = True
        for kind, child in self._group_children(u):
            if not first:
                y += gap
            first = False
            if kind == "text":
                s = sizes[child["id"]]
                nw, h = self.m.need(child, inner, s)
                need_w = max(need_w, nw + 2 * pad)
                items.append(dict(id=child["id"], x=pad, y=y, w=inner, h=h, font=s))
                mins.append(s)
                maxs.append(s)
                y += h
            elif kind == "image":
                a = child["width"] / child["height"]
                if child.get("fit", "contain") == "cover":
                    h = min(o.get("img_cap", 1e9), inner * .6)
                    items.append(dict(id=child["id"], x=pad, y=y, w=inner, h=h))
                else:
                    h = min(inner / a, o.get("img_cap", 1e9))
                    items.append(dict(id=child["id"], x=pad + (inner - h * a) / 2, y=y, w=h * a, h=h))
                y += h
            elif kind == "pill":
                shape, texts = child
                pp = max(2.0, .45 * max(sizes[t["id"]] for t in texts))
                t = texts[0]
                s = sizes[t["id"]]
                natural = self.m.measure(t, 1e4, s)[1] + 3
                tw = min(inner - 2 * pp, natural) if inner - 2 * pp > 4 else max(inner, 2.0)
                th = self.m.height(t, tw, s)
                pw = min(inner, tw + 2 * pp)
                px = pad + (inner - pw) / 2
                items.append(dict(id=shape["id"], x=px, y=y, w=pw, h=th + 2 * pp))
                for tt in texts:
                    items.append(dict(id=tt["id"], x=px + (pw - tw) / 2, y=y + pp, w=tw, h=th, font=sizes[tt["id"]]))
                mins.append(s)
                maxs.append(s)
                y += th + 2 * pp
            else:
                shape = child
                if shape["width"] / max(shape["height"], 1e-6) >= 3:
                    h = max(1.0, min(4.0, .12 * ref))
                    items.append(dict(id=shape["id"], x=pad, y=y, w=inner, h=h))
                else:
                    h = max(3.0, 1.2 * ref)
                    items.append(dict(id=shape["id"], x=pad, y=y, w=h, h=h))
                y += h
        total = y + pad
        if u.frame is not None:
            items.insert(0, dict(id=u.frame["id"], x=0, y=0, w=w, h=total))
        return UL(w, total, items, min(mins) if mins else None, max(maxs) if maxs else None, need_w)

    def _group_row(self, u: Unit, w: float, k: float, o: dict, h_avail: float | None) -> UL | None:
        """Children side by side, each as narrow as its own text allows at height h_avail."""
        if h_avail is None:
            return None
        emph, gamma = o.get("emph", {}).get(u.uid, 1.0), o.get("gamma", 1.0)
        sizes = self._sizes(u, k, emph, gamma)
        ref = (sum(sizes.values()) / len(sizes)) if sizes else 6.0
        pad = 0.0 if u.frame is None else max(1.5, min(.06 * h_avail, .5 * max(sizes.values(), default=6.0)))
        ih = h_avail - 2 * pad
        iw = w - 2 * pad
        if ih < 3 or iw < 3:
            return None
        gap = max(2.0, .5 * ref)
        kids = self._group_children(u)
        widths, ok = [], True
        for kind, child in kids:
            if kind == "text":
                mw = self.m.min_width(child, sizes[child["id"]], ih, iw)
                if mw is None:
                    return None
                widths.append(mw)
            elif kind == "image":
                a = child["width"] / child["height"]
                widths.append(min(ih * a, iw * .5) if child.get("fit", "contain") != "cover" else min(ih, iw * .4))
            elif kind == "pill":
                shape, texts = child
                t = texts[0]
                s = sizes[t["id"]]
                pp = max(2.0, .45 * s)
                mw = self.m.min_width(t, s, ih - 2 * pp, iw - 2 * pp)
                if mw is None:
                    return None
                widths.append(mw + 2 * pp)
            else:
                widths.append(max(2.0, min(ih, 4.0)))
        total = sum(widths) + gap * (len(widths) - 1)
        if total > iw + 1e-6:
            return None
        extra = iw - total
        text_idx = [i for i, (kd, _) in enumerate(kids) if kd in ("text", "pill")]
        if text_idx and extra > 0:
            grow = extra / len(text_idx)
            for i in text_idx:
                widths[i] += grow
        elif extra > 0:
            pad_left = extra / 2
        items, x = [], pad
        mins, maxs = [], []
        for (kind, child), cw in zip(kids, widths):
            if kind == "text":
                s = sizes[child["id"]]
                h = min(ih, self.m.height(child, cw, s))
                items.append(dict(id=child["id"], x=x, y=pad + (ih - h) / 2, w=cw, h=h, font=s))
                mins.append(s)
                maxs.append(s)
            elif kind == "image":
                a = child["width"] / child["height"]
                if child.get("fit", "contain") == "cover":
                    items.append(dict(id=child["id"], x=x, y=pad, w=cw, h=ih))
                else:
                    hh = min(ih, cw / a)
                    items.append(dict(id=child["id"], x=x, y=pad + (ih - hh) / 2, w=hh * a, h=hh))
            elif kind == "pill":
                shape, texts = child
                t = texts[0]
                s = sizes[t["id"]]
                pp = max(2.0, .45 * s)
                th = min(ih - 2 * pp, self.m.height(t, cw - 2 * pp, s))
                ph = th + 2 * pp
                items.append(dict(id=shape["id"], x=x, y=pad + (ih - ph) / 2, w=cw, h=ph))
                for tt in texts:
                    items.append(dict(id=tt["id"], x=x + pp, y=pad + (ih - ph) / 2 + pp, w=cw - 2 * pp, h=th, font=sizes[tt["id"]]))
                mins.append(s)
                maxs.append(s)
            else:
                items.append(dict(id=child["id"], x=x, y=pad, w=cw, h=ih))
            x += cw + gap
        if u.frame is not None:
            items.insert(0, dict(id=u.frame["id"], x=0, y=0, w=w, h=h_avail))
        return UL(w, h_avail, items, min(mins) if mins else None, max(maxs) if maxs else None)

    # ------------------------------------------------------------------ cells
    def _k_range(self, units, o):
        fonts = [t["font_size"] for u in units for t in u.all_texts]
        lo = 1.0 / self.ctx.max_font
        if not fonts:
            return lo, lo
        cap = o.get("size_cap", 1e9)
        emph = max([o.get("emph", {}).get(u.uid, 1.0) for u in units] + [1.0])
        hi = min(cap / (min(fonts) * min(1.0, emph)), 50.0, o.get("k_max", 1e9))
        return lo, max(hi, lo)

    def _gap(self, units, k, o):
        fonts = [t["font_size"] for u in units for t in u.all_texts] or [self.ctx.max_font * .3]
        return o.get("gap_k", .25) * k * sum(fonts) / len(fonts) + o.get("gap_px", 0.0)

    def fit_stack(self, units: list[Unit], rect, o: dict, valign: str = "top", halign: str = "left") -> Fit:
        x0, y0, w, h = rect
        if w < 3 or h < 3 or not units:
            return Fit(False)
        lo, hi = self._k_range(units, o)

        def evaluate(k):
            uls = [self.unit_layout(u, w, k, o, "stack") for u in units]
            gap = self._gap(units, k, o)
            total = sum(ul.h for ul in uls) + gap * (len(uls) - 1)
            fonts_ok = all((ul.max_font or 0) <= o.get("size_cap", 1e9) + 1e-6 and ul.need_w <= w + 1e-6 for ul in uls)
            return uls, gap, total, fonts_ok

        uls, gap, total, fonts_ok = evaluate(hi)
        k = hi
        if total > h + 1e-6 or not fonts_ok:
            a, b = lo, hi
            uls, gap, total, fonts_ok = evaluate(a)
            if total > h + 1e-6 or not fonts_ok:
                return Fit(False, a)
            for _ in range(MAX_K_ITER):
                mid = math.sqrt(a * b)
                r = evaluate(mid)
                if r[2] <= h + 1e-6 and r[3]:
                    a = mid
                    uls, gap, total, fonts_ok = r
                else:
                    b = mid
            k = a
        # stretch leftover into gaps only for 'spread'
        n = len(uls)
        leftover = max(0.0, h - total)
        if valign == "spread" and n > 1:
            gap = gap + min(leftover / (n - 1), 2.2 * gap + 8)
            used = sum(ul.h for ul in uls) + gap * (n - 1)
            leftover = max(0.0, h - used)
            start = y0 + leftover / 2
        else:
            start = y0 + (leftover / 2 if valign == "center" else leftover if valign == "bottom" else 0.0)
        return self._collect(uls, start, x0, w, gap, k, h, total, "stack", halign)

    def _collect(self, uls, start, x0, w, gap, k, h, total, axis, halign="left"):
        placements, y = {}, start
        mins, maxs = [], []
        for ul in uls:
            dx = (w - ul.w) * {"left": 0.0, "center": .5, "right": 1.0}.get(halign, 0.0) if ul.w < w else 0.0
            for it in ul.items:
                p = dict(x=x0 + dx + it["x"], y=y + it["y"], width=it["w"], height=it["h"])
                if "font" in it:
                    p["font_size"] = it["font"]
                placements[it["id"]] = p
            if ul.min_font is not None:
                mins.append(ul.min_font)
                maxs.append(ul.max_font)
            y += ul.h + gap
        xs = [p["x"] for p in placements.values()]
        ys = [p["y"] for p in placements.values()]
        fit = Fit(True, k, placements, (min(xs), min(ys), max(p["x"] + p["width"] for p in placements.values()),
                                         max(p["y"] + p["height"] for p in placements.values())),
                  min(mins) if mins else None, max(maxs) if maxs else None, slack=max(0.0, (h - total) / max(h, 1e-6)))
        return fit

    def fit_row(self, units: list[Unit], rect, o: dict, halign: str = "left", valign: str = "center") -> Fit:
        """Units side by side; every unit as narrow as its text allows at the shared type scale."""
        x0, y0, w, h = rect
        if w < 3 or h < 3 or not units:
            return Fit(False)
        lo, hi = self._k_range(units, o)

        def evaluate(k):
            widths = []
            for u in units:
                if u.kind == "image":
                    ul = self.unit_layout(u, 1e4, k, dict(o, img_cap=h), "stack")
                    widths.append(ul.items[0]["w"] if u.images[0].get("fit", "contain") != "cover" else min(h, w * .3))
                    continue
                # smallest width at which the unit's stack fits the cell height
                top = w
                ul = self.unit_layout(u, top, k, o, "stack")
                if ul.h > h + 1e-6 or ul.need_w > top + 1e-6:
                    return None

                def ok(width):
                    t = self.unit_layout(u, width, k, o, "stack")
                    return t.h <= h + 1e-6 and t.need_w <= width + 1e-6
                a, b = 4.0, top
                for _ in range(8):
                    mid = (a + b) / 2
                    if ok(mid):
                        b = mid
                    else:
                        a = mid
                widths.append(b)
            gap = self._gap(units, k, o) + 2
            total = sum(widths) + gap * (len(widths) - 1)
            return (widths, gap, total) if total <= w + 1e-6 else None

        r = evaluate(hi)
        k = hi
        if r is None:
            a, b = lo, hi
            if evaluate(a) is None:
                return Fit(False, a)
            for _ in range(MAX_K_ITER - 2):
                mid = math.sqrt(a * b)
                if evaluate(mid) is not None:
                    a = mid
                else:
                    b = mid
            k = a
            r = evaluate(k)
        widths, gap, total = r
        extra = max(0.0, w - total)
        if halign == "spread" and len(widths) > 1:
            gap += extra / (len(widths) - 1)
            start = x0
        else:
            start = x0 + (extra / 2 if halign == "center" else extra if halign == "right" else 0.0)
        placements, x = {}, start
        mins, maxs = [], []
        for u, uw in zip(units, widths):
            ul = self.unit_layout(u, uw, k, dict(o, img_cap=h), "stack")
            dy = (h - ul.h) / 2 if valign == "center" else (h - ul.h if valign == "bottom" else 0.0)
            for it in ul.items:
                p = dict(x=x + it["x"], y=y0 + dy + it["y"], width=it["w"], height=it["h"])
                if "font" in it:
                    p["font_size"] = it["font"]
                placements[it["id"]] = p
            if ul.min_font is not None:
                mins.append(ul.min_font)
                maxs.append(ul.max_font)
            x += uw + gap
        xs = [p["x"] for p in placements.values()]
        ys = [p["y"] for p in placements.values()]
        return Fit(True, k, placements, (min(xs), min(ys), max(p["x"] + p["width"] for p in placements.values()),
                                         max(p["y"] + p["height"] for p in placements.values())),
                   min(mins) if mins else None, max(maxs) if maxs else None, slack=extra / max(w, 1e-6))

    def fit_row_group(self, u: Unit, rect, o: dict) -> Fit:
        """A single group laid out internally as a row (number | title | body ...)."""
        x0, y0, w, h = rect
        lo, hi = self._k_range([u], o)

        def evaluate(k):
            return self.unit_layout(u, w, k, o, "row", h)

        k, ul = hi, evaluate(hi)
        if ul is None:
            a, b = lo, hi
            if evaluate(a) is None:
                return Fit(False, a)
            for _ in range(MAX_K_ITER - 1):
                mid = math.sqrt(a * b)
                if evaluate(mid) is not None:
                    a = mid
                else:
                    b = mid
            k = a
            ul = evaluate(k)
        return self._collect([ul], y0, x0, w, 0.0, k, h, ul.h, "row")

    def place_image(self, u: Unit, rect, align=(0.5, 0.5), fill: float = 1.0) -> Fit:
        """Contain-fit an image (aspect preserved exactly) or fill a cover image into `rect`."""
        x0, y0, w, h = rect
        e = u.images[0]
        if e.get("fit", "contain") == "cover":
            return Fit(True, 0, {e["id"]: dict(x=x0, y=y0, width=w, height=h)}, (x0, y0, x0 + w, y0 + h))
        a = e["width"] / e["height"]
        iw = min(w, h * a) * fill
        ih = iw / a
        x = x0 + (w - iw) * align[0]
        y = y0 + (h - ih) * align[1]
        return Fit(True, 0, {e["id"]: dict(x=x, y=y, width=iw, height=ih)}, (x, y, x + iw, y + ih))
