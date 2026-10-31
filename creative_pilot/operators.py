"""Composable, source-aware restructuring operators.

Every recipe turns the *roles* found by `blocks.analyse` into a different spatial structure: how many regions,
where the hero image goes, how much length each region gets, how regions are aligned or offset, which unit is
emphasised and where backdrops/rules/accents attach. Reading order of the source is preserved along the main
axis for sequential content; recipes never add, remove or duplicate elements and only emit x/y/width/height
and text font_size. Randomness comes from an explicit `random.Random` so every proposal is reproducible.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field

from creative_pilot.blocks import SourceContext, Unit
from creative_pilot.layout import Fit, Fitter

WIDE_RECIPES = ("cols", "headline_dominant", "center_stage", "two_row", "stagger", "cascade", "bleed_hero",
                "row_flow", "cta_forward")
TALL_RECIPES = ("bands", "rhythm", "zigzag", "sections", "bleed_hero", "cascade", "flow", "headline_dominant")
RECIPE_STYLE = dict(cols="conventional", row_flow="conventional", bands="conventional", flow="conventional")


@dataclass
class Cell:
    units: list
    rect: tuple
    mode: str = "stack"          # stack | row | rowgroup
    valign: str = "center"
    halign: str = "left"
    o: dict = field(default_factory=dict)


@dataclass
class Spec:
    recipe: str
    params: dict
    cells: list
    hero: tuple | None = None    # (unit, rect, align, fill)
    bleed_hero: bool = False
    bleed_cells: bool = False


@dataclass
class Plan:
    recipe: str
    params: dict
    placements: dict
    min_font: float | None
    max_font: float | None
    k_values: list


# ---------------------------------------------------------------------------------------------- helpers
def _ordered(ctx: SourceContext):
    return [ctx.unit(uid) for uid in ctx.order]


def pick_hero(ctx: SourceContext):
    heroes = [u for u in ctx.units if u.kind == "image" and u.role in ("hero", "photo")]
    if not heroes:
        return None
    return max(heroes, key=lambda u: u.images[0]["width"] * u.images[0]["height"])


def chunk(units: list, n: int, rng: random.Random, jitter: float = .25) -> list[list]:
    """Contiguous partition of an ordered list into n weight-balanced chunks (reading order preserved)."""
    n = max(1, min(n, len(units)))
    if n == 1:
        return [list(units)]
    w = [u.weight * (1 + rng.uniform(-jitter, jitter)) for u in units]
    total = sum(w)
    cuts, acc, target = [], 0.0, total / n
    for i, x in enumerate(w[:-1]):
        acc += x
        remaining_units = len(units) - i - 1
        remaining_chunks = n - len(cuts) - 1
        if (acc >= target * (len(cuts) + 1) and remaining_units >= remaining_chunks) or remaining_units == remaining_chunks:
            cuts.append(i + 1)
            if len(cuts) == n - 1:
                break
    out, start = [], 0
    for c in cuts + [len(units)]:
        out.append(units[start:c])
        start = c
    return [c for c in out if c]


def split_lengths(total: float, weights: list, gap: float, minimum: float) -> list | None:
    n = len(weights)
    avail = total - gap * (n - 1)
    if avail < minimum * n:
        return None
    s = sum(weights)
    lengths = [max(minimum, avail * w / s) for w in weights]
    over = sum(lengths) - avail
    if over > 1e-6:
        big = [i for i, v in enumerate(lengths) if v > minimum + 1e-9]
        shrink = sum(lengths[i] - minimum for i in big)
        for i in big:
            lengths[i] -= over * (lengths[i] - minimum) / max(shrink, 1e-9)
    return lengths


def chunk_weight(units):
    out = 0.0
    for u in units:
        chars = sum(len(t["content"]) for t in u.all_texts)
        out += max(8, chars) ** .5 * (1 + (.5 if u.images else 0)) * (1.4 if u.role == "headline" else 1.0)
    return out


def sample_params(recipe: str, wide: bool, rng: random.Random) -> dict:
    p = dict(m=rng.choice((.035, .05, .07, .09)), gamma=rng.choice((.6, .72, .85, 1.0, 1.0, 1.15)), floor=rng.choice((3.0, 4.0, 5.0, 6.0, 7.0, 7.0)),
             emph=round(rng.uniform(1.0, 1.7), 2), gap=round(rng.uniform(.015, .05), 3),
             gap_k=round(rng.uniform(.15, .45), 2), wmode=rng.choice(("weight", "weight", "equal", "headline", "taper", "jitter")),
             n=rng.randint(2, 5 if wide else 4), hero=rng.choice(("start", "end", "mid", "bleed_start", "bleed_end")),
             hero_scale=round(rng.uniform(.72, 1.0), 2), valign=rng.choice(("center", "top", "bottom", "spread")),
             halign=rng.choice(("left", "center")), amp=round(rng.uniform(.52, .78) if wide else rng.uniform(.74, .92), 2), img_frac=round(rng.uniform(.4, .7), 2),
             pad=round(rng.uniform(.02, .05), 3), cap=round(rng.uniform(.26, .42), 2), jitter=round(rng.uniform(.1, .45), 2),
             head_share=round(rng.uniform(.3, .5), 2), dominant_emph=round(rng.uniform(1.25, 2.0), 2))
    if recipe in ("bleed_hero", "bands", "rhythm", "sections", "flow", "cols", "row_flow") and rng.random() < .5:
        p["hero"] = rng.choice(("start", "end"))
    return p


# ---------------------------------------------------------------------------------------------- spec builders
def build_spec(ctx: SourceContext, T: tuple, recipe: str, p: dict, rng: random.Random) -> Spec | None:
    tw, th = T
    wide = tw >= th
    m = max(2.0, p["m"] * min(tw, th))
    ux, uy, uw, uh = m, m, tw - 2 * m, th - 2 * m
    if uw < 8 or uh < 8:
        return None
    main = _ordered(ctx)
    hero = pick_hero(ctx)
    others = [u for u in main if u is not hero]
    headline = next((u for u in others if u.role == "headline"), None) or (max(others, key=lambda u: u.max_font) if others else None)
    buttons = [u for u in others if u.role == "button"]
    L, C = (uw, uh) if wide else (uh, uw)       # main-axis length, cross-axis span
    d0, o0 = (ux, uy) if wide else (uy, ux)
    gap = max(2.0, p["gap"] * min(tw, th))
    o = dict(gamma=p["gamma"], gap_k=p["gap_k"], size_cap=max(6.0, p["cap"] * min(tw, th)), emph={})
    if headline is not None:
        o["emph"][headline.uid] = p["emph"] if recipe not in ("headline_dominant",) else p["dominant_emph"]

    def rect(d, dl, o_, ol):
        return (d, o_, dl, ol) if wide else (o_, d, ol, dl)

    if recipe == "two_row" and wide and th >= 200:
        return _two_row(ctx, T, p, rng, o, others, hero, headline, m, gap)
    if recipe == "two_row":
        recipe = "cols"

    slots = []   # ('hero',) | ('cell', units, emph_override)
    hero_slot_len = 0.0
    bleed_hero = False
    hero_pos = p["hero"] if hero is not None else None
    if recipe == "bleed_hero" and hero is not None:
        hero_pos = p["hero"] if p["hero"].startswith("bleed") else rng.choice(("bleed_start", "bleed_end"))
        bleed_hero = True
    elif recipe == "bleed_hero":
        recipe = "cols"
    if hero_pos is not None and hero_pos.startswith("bleed"):
        bleed_hero = True
    if recipe in ("flow", "row_flow") and hero_pos in ("mid",):
        hero_pos = "start"
    if recipe == "center_stage" and hero is not None:
        hero_pos = "mid"
    elif recipe == "center_stage":
        recipe = "cols"
    if hero is not None:
        e = hero.images[0]
        a = e["width"] / e["height"]
        if e.get("fit", "contain") == "cover":
            hl = p["img_frac"] * .55 * L
        else:
            hl = (C * a if wide else C / a) * p["hero_scale"]
        hl = min(hl, .46 * L if wide else .5 * L)
        hero_slot_len = max(hl, 6.0)
    # main cells
    n_cells = p["n"]
    cell_units: list[list] = []
    if recipe == "headline_dominant" and headline is not None and len(others) >= 2:
        rest = [u for u in others if u is not headline]
        idx = others.index(headline)
        before, after = others[:idx], others[idx + 1:]
        cell_units = ([before] if before and recipe != "cta_forward" else []) + [[headline]] + (
            chunk(after, max(1, min(n_cells - 1, len(after))), rng, p["jitter"]) if after else [])
    elif recipe == "cta_forward" and buttons:
        btn = buttons[-1]
        rest = [u for u in others if u is not btn]
        cell_units = chunk(rest, max(1, min(n_cells - 1, len(rest))), rng, p["jitter"]) + [[btn]]
    elif recipe in ("flow", "row_flow"):
        cell_units = [others]
    elif recipe == "center_stage" and hero is not None and len(others) >= 2:
        j = max(1, min(len(others) - 1, round(len(others) * rng.uniform(.3, .6))))
        left, right = others[:j], others[j:]
        cell_units = [left, "HERO", right]
    elif recipe in ("sections", "rhythm", "bands", "zigzag", "cascade", "stagger", "cols", "bleed_hero", "headline_dominant", "cta_forward"):
        cell_units = chunk(others, n_cells, rng, p["jitter"])
    else:
        cell_units = chunk(others, n_cells, rng, p["jitter"])
    if not cell_units and hero is None:
        return None
    # order slots: hero start/end/mid
    if "HERO" in cell_units:
        slots = [("hero",) if c == "HERO" else ("cell", c) for c in cell_units]
    else:
        slots = [("cell", c) for c in cell_units]
        if hero is not None:
            if hero_pos in ("start", "bleed_start"):
                slots.insert(0, ("hero",))
            elif hero_pos in ("end", "bleed_end"):
                slots.append(("hero",))
            else:  # mid
                slots.insert(max(1, len(slots) // 2) if len(slots) > 1 else 0, ("hero",))
    # lengths
    n_slots = len(slots)
    cell_ws = []
    for s in slots:
        if s[0] == "hero":
            cell_ws.append(None)
        else:
            w = chunk_weight(s[1])
            if p["wmode"] == "equal":
                w = 1.0
            elif p["wmode"] == "headline":
                w *= 1.8 if any(u.role == "headline" for u in s[1]) else 1.0
            elif p["wmode"] == "taper":
                w *= 1.0 - .15 * len(cell_ws)
            elif p["wmode"] == "jitter":
                w *= rng.uniform(.7, 1.4)
            cell_ws.append(max(w, .2))
    if recipe == "headline_dominant":
        cell_ws = [(None if x is None else (x * p["head_share"] * 5 if any(u.role == "headline" for u in s[1]) else x))
                   for x, s in zip(cell_ws, slots)]
    fixed = hero_slot_len
    n_free = sum(w is not None for w in cell_ws)
    min_cell = max(10.0, .12 * C) if wide else max(8.0, .16 * L / max(n_free, 1))
    free_L = L - fixed - gap * (n_slots - 1) - (0 if not bleed_hero else 0)
    if free_L < min_cell * max(n_free, 1):
        if hero is not None and hero_slot_len > 0 and n_free:
            hero_slot_len = max(6.0, L - gap * (n_slots - 1) - min_cell * n_free)
            free_L = L - hero_slot_len - gap * (n_slots - 1)
        if free_L < min_cell * max(n_free, 1):
            return None
    free_weights = [w for w in cell_ws if w is not None]
    lengths_free = split_lengths(free_L + gap * (len(free_weights) - 1), free_weights, gap * 0, min_cell) if free_weights else []
    if lengths_free is None:
        return None
    sizes, fi = [], 0
    for w in cell_ws:
        if w is None:
            sizes.append(hero_slot_len)
        else:
            sizes.append(lengths_free[fi])
            fi += 1
    avail_len = L - gap * (n_slots - 1)
    if sum(sizes) > avail_len:
        sizes = [s * avail_len / sum(sizes) for s in sizes]
    cells, hero_spec = [], None
    pos = d0
    n_cell_idx = 0
    n_main = sum(1 for s in slots if s[0] == "cell")
    for si, (slot, size) in enumerate(zip(slots, sizes)):
        start = pos
        # bleed: the hero slot touches the canvas edge
        if slot[0] == "hero":
            hs, hl2 = start, size
            if bleed_hero:
                if si == 0:
                    hs, hl2 = 0.0, size + d0
                    start = hs
                    pos = d0 + size
                else:
                    hl2 = size + ((tw if wide else th) - (d0 + L))
                    hs = start
            full = rect(hs, hl2, 0.0, (th if wide else tw)) if bleed_hero else rect(start, size, o0, C)
            hero_spec = (hero, full, (.5, .5), 1.0)
            if bleed_hero and si == 0:
                pos = d0 + size + gap
            else:
                pos = start + size + gap
            continue
        # cross-axis extent and offset
        ce, co = C, o0
        if recipe in ("stagger", "zigzag") and n_main > 1:
            ce = C * p["amp"]
            co = o0 if n_cell_idx % 2 == 0 else o0 + C - ce
        elif recipe == "cascade" and n_main > 1:
            ce = C * p["amp"]
            co = o0 + (C - ce) * n_cell_idx / (n_main - 1)
        r = rect(start, size, co, ce)
        va = p["valign"]
        if recipe in ("stagger",) and wide:
            va = "top" if n_cell_idx % 2 == 0 else "bottom"
        elif recipe == "cascade" and wide:
            va = "top"
        co2 = dict(o)
        co2["img_cap"] = max(4.0, p["img_frac"] * (r[3] if wide else r[3]))
        mode = "row" if (recipe == "row_flow" and wide) or (recipe == "flow" and wide) else "stack"
        cells.append(Cell(slot[1], r, mode, va if not (recipe in ("bands", "rhythm", "flow") and not wide) else p["valign"],
                          p["halign"], co2))
        n_cell_idx += 1
        pos = start + size + gap
    if hero_spec is None and hero is not None:
        return None
    return Spec(recipe, dict(p, m_px=round(m, 2), n_cells=len(cells), hero_pos=hero_pos), cells, hero_spec, bleed_hero,
                recipe in ("sections", "bleed_hero", "stagger") and p["pad"] > .03)


def _two_row(ctx, T, p, rng, o, others, hero, headline, m, gap) -> Spec | None:
    tw, th = T
    ux, uy, uw, uh = m, m, tw - 2 * m, th - 2 * m
    if headline is None or len(others) < 2:
        return None
    top_n = 1 if p["n"] % 2 else 2
    idx = others.index(headline)
    top = [headline] if top_n == 1 else others[idx:idx + 2]
    top = [u for u in others if u in top]
    rest = [u for u in others if u not in top]
    if not rest and hero is None:
        return None
    top_h = (p["head_share"] + .08) * uh
    bot_h = uh - top_h - gap
    if bot_h < 12:
        return None
    o_top = dict(o)
    o_bot = dict(o, img_cap=bot_h, size_cap=o["size_cap"] * .7)
    cells = [Cell(top, (ux, uy, uw, top_h), "row", "center", p["halign"], dict(o_top, img_cap=top_h))]
    hero_spec = None
    bottom_units = rest
    if hero is not None:
        a = hero.images[0]["width"] / hero.images[0]["height"]
        hw = min(bot_h * a * p["hero_scale"], .4 * uw) if hero.images[0].get("fit", "contain") != "cover" else .3 * uw
        if p["hero"] in ("start", "bleed_start", "mid"):
            hero_spec = (hero, (ux, uy + top_h + gap, hw, bot_h), (.5, .5), 1.0)
            cells.append(Cell(bottom_units, (ux + hw + gap, uy + top_h + gap, uw - hw - gap, bot_h), "row", "center", "spread", o_bot))
        else:
            hero_spec = (hero, (ux + uw - hw, uy + top_h + gap, hw, bot_h), (.5, .5), 1.0)
            cells.append(Cell(bottom_units, (ux, uy + top_h + gap, uw - hw - gap, bot_h), "row", "center", "spread", o_bot))
    else:
        cells.append(Cell(bottom_units, (ux, uy + top_h + gap, uw, bot_h), "row", "center", "spread", o_bot))
    cells = [c for c in cells if c.units]
    return Spec("two_row", dict(p, m_px=round(m, 2), n_cells=len(cells)), cells, hero_spec)


# ---------------------------------------------------------------------------------------------- realization
def realize(ctx: SourceContext, T: tuple, fitter: Fitter, spec: Spec, rng: random.Random) -> Plan | None:
    tw, th = T
    placements: dict = {}
    ks, mins, maxs = [], [], []

    def take(fit: Fit):
        if not fit.ok:
            return False
        placements.update(fit.placements)
        ks.append(fit.k)
        if fit.min_font is not None:
            mins.append(fit.min_font)
            maxs.append(fit.max_font)
        return True

    if spec.hero is not None:
        unit, r, align, fill = spec.hero
        if not take(fitter.place_image(unit, r, align, fill)):
            return None
    fitter.floor = spec.params.get("floor", 1.0)
    # The headline's cell is fitted first; every other cell's type scale is then bounded by it (25% tolerance),
    # so a lone small text in a roomy cell cannot outgrow the hierarchy.
    ref_unit = next((u for c in spec.cells for u in c.units if u.role == "headline"), None) or         max((u for c in spec.cells for u in c.units), key=lambda u: u.max_font, default=None)
    ordered = sorted(spec.cells, key=lambda c: 0 if ref_unit in c.units else 1)
    k_ref = None
    for cell in ordered:
        if k_ref is not None:
            cell.o = dict(cell.o, k_max=k_ref)
        if cell.mode == "row":
            fit = fitter.fit_row(cell.units, cell.rect, cell.o, cell.halign if cell.halign in ("left", "center", "right", "spread") else "left", "center")
            if not fit.ok and len(cell.units) == 1 and cell.units[0].kind == "group":
                fit = fitter.fit_row_group(cell.units[0], cell.rect, cell.o)
        else:
            fit = fitter.fit_stack(cell.units, cell.rect, cell.o, cell.valign, cell.halign)
        if not take(fit):
            fitter.floor = 1.0
            return None
        if k_ref is None:
            k_ref = 1.25 * fit.k * max(cell.o.get("emph", {}).get(u.uid, 1.0) for u in cell.units)
    fitter.floor = 1.0
    enforce_hierarchy(ctx, placements)
    mins = [placements[e["id"]]["font_size"] for e in ctx.source["elements"] if e["type"] == "text"]
    maxs = list(mins)
    for e in ctx.bg:
        placements[e["id"]] = dict(x=0.0, y=0.0, width=float(tw), height=float(th))
    if not attach_decor(ctx, T, placements, rng, spec):
        return None
    ids = {e["id"] for e in ctx.source["elements"]}
    if set(placements) != ids:
        return None
    return Plan(spec.recipe, spec.params, placements, min(mins) if mins else None, max(maxs) if maxs else None, ks)


def enforce_hierarchy(ctx: SourceContext, placements: dict) -> None:
    """No text may end up larger than any text that was strictly larger in the source (only ever shrinks)."""
    texts = sorted(((e["font_size"], e["id"]) for e in ctx.source["elements"] if e["type"] == "text"), reverse=True)
    cap, i = float("inf"), 0
    while i < len(texts):
        j = i
        while j < len(texts) and texts[j][0] == texts[i][0]:
            j += 1
        tier = [placements[t[1]] for t in texts[i:j]]
        for p in tier:
            p["font_size"] = min(p["font_size"], cap)
        cap = min([cap] + [p["font_size"] for p in tier])
        i = j


def _union(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def _placed_box(placements, ids):
    boxes = [(placements[i]["x"], placements[i]["y"], placements[i]["x"] + placements[i]["width"],
              placements[i]["y"] + placements[i]["height"]) for i in ids if i in placements]
    return _union(boxes) if boxes else None


def _overlap(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(w, 0) * max(h, 0)


def attach_decor(ctx: SourceContext, T: tuple, placements: dict, rng: random.Random, spec: Spec) -> bool:
    """Rules, accents and backdrops follow their anchor units; nothing is added, removed or re-ordered."""
    tw, th = T
    wide = tw >= th
    s_uniform = min(tw / ctx.width, th / ctx.height)
    unit_box = {u.uid: _placed_box(placements, u.ids) for u in ctx.units}
    pad_px = max(2.0, spec.params.get("pad", .03) * min(tw, th))
    z = {e["id"]: e["z_index"] for e in ctx.source["elements"]}
    for d in sorted(ctx.decor, key=lambda d: d.el["z_index"]):
        e = d.el
        anchors = [unit_box[a] for a in d.anchors if unit_box.get(a)]
        if d.kind == "backdrop" and anchors:
            x0, y0, x1, y1 = _union(anchors)
            grow = 1.0 if e.get("shape_type", "rectangle") != "ellipse" else 1.15
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            w, h = (x1 - x0) * grow + 2 * pad_px, (y1 - y0) * grow + 2 * pad_px
            x, y = cx - w / 2, cy - h / 2
            if spec.bleed_cells:
                if wide:
                    y, h = 0.0, float(th)
                else:
                    x, w = 0.0, float(tw)
            x2, y2 = min(x + w, tw), min(y + h, th)
            x, y = max(x, 0.0), max(y, 0.0)
            placements[e["id"]] = dict(x=x, y=y, width=max(x2 - x, 1.0), height=max(y2 - y, 1.0))
            # A backdrop may only sit behind its own anchors: other text was styled for another background.
            if e.get("opacity", 1) >= .7:
                own = {i for a in d.anchors for i in ctx.unit(a).ids}
                for o in ctx.source["elements"]:
                    if o["type"] != "text" or o["id"] in own:
                        continue
                    b = placements[o["id"]]
                    tb = (b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"])
                    if _overlap(tb, (x, y, x2, y2)) > .25 * max(b["width"] * b["height"], 1e-6):
                        return False
        elif d.kind == "rule" and anchors:
            ax0, ay0, ax1, ay1 = anchors[0]
            thick = max(1.0, min(4.0, min(e["width"], e["height"]) * s_uniform * 1.5))
            frac = min(1.0, e["width"] / max(1.0, (ctx.unit(d.anchors[0]).box[2] - ctx.unit(d.anchors[0]).box[0])))
            w = max(6.0, min((ax1 - ax0), (ax1 - ax0) * max(frac, .15)))
            below = (e["y"] + e["height"] / 2) >= (ctx.unit(d.anchors[0]).box[1] + ctx.unit(d.anchors[0]).box[3]) / 2
            y = ay1 + 1.0 if below else ay0 - thick - 1.0
            if y < 0 or y + thick > th:
                y = ay0 - thick - 1.0 if below else ay1 + 1.0
            y = min(max(y, 0.0), th - thick)
            placements[e["id"]] = dict(x=ax0, y=y, width=min(w, tw - ax0), height=thick)
        else:  # accent or backdrop without placed anchors
            base = max(min(e["width"], e["height"]) * s_uniform * 1.8, 6.0)
            base = min(base, .36 * min(tw, th))
            aspect = e["width"] / max(e["height"], 1e-6)
            aw, ah = (base * min(aspect, 3), base) if aspect >= 1 else (base, base / max(aspect, .33))
            aw, ah = min(aw, tw * .5), min(ah, th * .5)
            others = [(placements[i]["x"], placements[i]["y"], placements[i]["x"] + placements[i]["width"],
                       placements[i]["y"] + placements[i]["height"], z[i]) for i in placements if i != e["id"]
                      and i not in {b["id"] for b in ctx.bg}]
            best, best_cost = None, 1e18
            anchor_c = None
            if d.anchors and unit_box.get(d.anchors[0]):
                b = unit_box[d.anchors[0]]
                anchor_c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            xs = [0, (tw - aw) / 2, tw - aw] + ([max(0, min(tw - aw, anchor_c[0] - aw / 2))] if anchor_c else [])
            ys = [0, (th - ah) / 2, th - ah] + ([max(0, min(th - ah, anchor_c[1] - ah / 2))] if anchor_c else [])
            for cx in xs:
                for cy in ys:
                    box = (cx, cy, cx + aw, cy + ah)
                    cost = 0.0
                    for ob in others:
                        ov = _overlap(box, ob[:4])
                        cost += ov * (3.0 if ob[4] < z[e["id"]] else .3)
                    cost = cost / (aw * ah) + (0 if not anchor_c else .02 * math.hypot(cx + aw / 2 - anchor_c[0], cy + ah / 2 - anchor_c[1]) / max(tw, th))
                    if cost < best_cost:
                        best, best_cost = (cx, cy), cost
            placements[e["id"]] = dict(x=best[0], y=best[1], width=aw, height=ah)
    return True
