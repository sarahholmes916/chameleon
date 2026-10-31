"""Candidate assembly, geometric/typographic features and the structural descriptor used for diversity.

Features here are objective measurements (renderer-measured text fit, font sizes, empty space, overlap,
alignment, group cohesion, order agreement). They are NOT quality labels. `structure_distance` is a
composition-level distance: it ignores sub-percent coordinate noise by working on coarse occupancy grids.
"""

from __future__ import annotations

import math
import statistics

from engine.layout_edits import layout_edit

GRID_LONG = 32


def assemble(source: dict, placements: dict, T: tuple, meas=None, repair_text: bool = True) -> dict | None:
    """Adobe candidate from placements via `layout_edit` (protected fields cannot change). Repairs text fit."""
    tw, th = T
    elements = []
    for e in source["elements"]:
        p = placements.get(e["id"])
        if p is None:
            return None
        x, y, w, h = (round(float(p[k]), 3) for k in ("x", "y", "width", "height"))
        x, y = max(x, 0.0), max(y, 0.0)
        full_w, full_h = x == 0 and w >= tw, y == 0 and h >= th
        if full_w:
            w = float(tw)
        elif x + w > tw - 1e-9 and x + w - tw < .05:  # shave 1 micro-unit so float sums cannot leave the canvas
            w = math.floor((tw - x) * 1000 - 1) / 1000
        if full_h:
            h = float(th)
        elif y + h > th - 1e-9 and y + h - th < .05:
            h = math.floor((th - y) * 1000 - 1) / 1000
        changes = dict(x=x, y=y, width=w, height=h)
        if e["type"] == "text":
            changes["font_size"] = round(float(p.get("font_size", e["font_size"])), 2)
        elements.append(layout_edit(e, changes))
    candidate = {"canvas": {"width": tw, "height": th}, "elements": elements}
    if meas is not None and repair_text:
        repair_text_fit(candidate, meas)
    return candidate


def repair_text_fit(candidate: dict, meas) -> dict:
    """Shrink (never grow) any text whose renderer-measured box overflows, 3% at a time. Mutates `candidate`."""
    for el in candidate["elements"]:
        if el["type"] != "text":
            continue
        for _ in range(80):
            if meas.fit_report(el)["fits"] or el["font_size"] <= 1.0:
                break
            el["font_size"] = round(max(1.0, el["font_size"] * .97), 2)
    return candidate


def _rect(e):
    return e["x"], e["y"], e["x"] + e["width"], e["y"] + e["height"]


def _inter(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(w, 0.0) * max(h, 0.0)


def is_background(e, tw, th):
    return e["width"] >= .85 * tw and e["height"] >= .85 * th and (e["type"] == "shape" or e.get("fit") == "cover")


def occlusion_issues(candidate: dict, tw: int, th: int, min_fraction: float = .12) -> list:
    """A higher-z shape/image that substantially covers a lower-z text or image (content could be hidden)."""
    els = sorted(candidate["elements"], key=lambda e: e["z_index"])
    issues = []
    for i, low in enumerate(els):
        if low["type"] == "shape" or is_background(low, tw, th):
            continue
        area = max(low["width"] * low["height"], 1e-6)
        for up in els[i + 1:]:
            if up["type"] == "text" or is_background(up, tw, th):
                continue
            if up["type"] == "shape" and up.get("opacity", 1) < .4:
                continue
            frac = _inter(_rect(low), _rect(up)) / area
            if frac > min_fraction:
                issues.append((low["id"], up["id"], round(frac, 3)))
    return issues


def _grid(tw, th):
    if tw >= th:
        return GRID_LONG, max(2, round(GRID_LONG * th / tw))
    return max(2, round(GRID_LONG * tw / th)), GRID_LONG


def occupancy(candidate: dict, tw: int, th: int, kinds=("text", "image", "shape")) -> list:
    gx, gy = _grid(tw, th)
    cells = [0.0] * (gx * gy)
    for e in candidate["elements"]:
        if e["type"] not in kinds or is_background(e, tw, th):
            continue
        x0, y0, x1, y1 = _rect(e)
        for j in range(gy):
            for i in range(gx):
                cell = (i * tw / gx, j * th / gy, (i + 1) * tw / gx, (j + 1) * th / gy)
                v = _inter(cell, (x0, y0, x1, y1)) / ((cell[2] - cell[0]) * (cell[3] - cell[1]))
                if v > cells[j * gx + i]:
                    cells[j * gx + i] = v
    return cells


def layout_features(source: dict, candidate: dict, meas=None) -> dict:
    tw, th = candidate["canvas"]["width"], candidate["canvas"]["height"]
    area = tw * th
    els = candidate["elements"]
    texts = [e for e in els if e["type"] == "text" and e["content"].strip()]
    fonts = sorted(e["font_size"] for e in texts)
    content = [e for e in els if e["type"] != "shape" and not is_background(e, tw, th)]
    f = dict(n_text=len(texts), min_font=fonts[0] if fonts else None, median_font=statistics.median(fonts) if fonts else None,
             max_font=fonts[-1] if fonts else None,
             fonts_lt6=sum(x < 6 for x in fonts), fonts_lt8=sum(x < 8 for x in fonts),
             fonts_lt_target=sum(x < .08 * min(tw, th) for x in fonts) if fonts else 0)
    f["font_ratio_max_min"] = (fonts[-1] / fonts[0]) if fonts and fonts[0] > 0 else None
    f["font_rel_min"] = fonts[0] / min(tw, th) if fonts else None
    f["font_rel_median"] = statistics.median(fonts) / min(tw, th) if fonts else None
    if meas is not None:
        reports = [meas.fit_report(e) for e in texts]
        f["text_fit_failures"] = sum(not r["fits"] for r in reports)
    # empty space from element boxes on a coarse grid
    occ = occupancy(candidate, tw, th)
    gx, gy = _grid(tw, th)
    f["empty_cell_fraction"] = sum(v < .05 for v in occ) / len(occ)
    f["content_area_fraction"] = min(sum(max(e["width"], 0) * max(e["height"], 0) for e in content) / area, 3.0)
    rows = [sum(occ[j * gx:(j + 1) * gx]) / gx for j in range(gy)]
    cols = [sum(occ[j * gx + i] for j in range(gy)) / gy for i in range(gx)]
    f["empty_row_fraction"] = sum(r < .03 for r in rows) / gy
    f["empty_col_fraction"] = sum(c < .03 for c in cols) / gx
    # the longest empty stretch along the long axis of the canvas
    line = cols if tw >= th else rows
    longest = run = 0
    for v in line:
        run = run + 1 if v < .03 else 0
        longest = max(longest, run)
    f["longest_empty_stretch"] = longest / len(line)
    # overlaps among content elements
    ov = 0.0
    for i, a in enumerate(content):
        for b in content[i + 1:]:
            ov += _inter(_rect(a), _rect(b))
    f["content_overlap_fraction"] = ov / max(sum(e["width"] * e["height"] for e in content), 1e-6)
    f["occluded_pairs"] = len(occlusion_issues(candidate, tw, th))
    # alignment: distinct left edges / centre lines among texts (fewer = more systematic)
    lefts = {round(e["x"] / max(tw, th) * 100) for e in texts}
    f["distinct_left_edges"] = len(lefts)
    # visual-weight centroid offset from canvas centre (0 = balanced)
    wts = [(e["width"] * e["height"] * (2.0 if e["type"] == "image" else 1.0), e["x"] + e["width"] / 2, e["y"] + e["height"] / 2)
           for e in content]
    tot = sum(w for w, _, _ in wts) or 1.0
    cx = sum(w * x for w, x, _ in wts) / tot
    cy = sum(w * y for w, _, y in wts) / tot
    f["balance_offset"] = math.hypot((cx - tw / 2) / tw, (cy - th / 2) / th)
    # group cohesion: group bbox area relative to members' own area (large = scattered)
    groups = {}
    for e in els:
        if e.get("group_id") is not None:
            groups.setdefault(e["group_id"], []).append(e)
    spread = []
    for g in groups.values():
        x0 = min(e["x"] for e in g); y0 = min(e["y"] for e in g)
        x1 = max(e["x"] + e["width"] for e in g); y1 = max(e["y"] + e["height"] for e in g)
        spread.append(((x1 - x0) * (y1 - y0)) / max(sum(e["width"] * e["height"] for e in g), 1e-6))
    f["group_spread_max"] = max(spread) if spread else None
    # reading order agreement along the main axis (Kendall-style on text elements)
    src = {e["id"]: e for e in source["elements"] if e["type"] == "text"}
    ids = [e["id"] for e in texts if e["id"] in src]
    key = (lambda e: (e["x"] + e["width"] / 2)) if tw >= th else (lambda e: (e["y"] + e["height"] / 2))
    pairs = agree = 0
    cand = {e["id"]: e for e in texts}
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            sa, sb = src[a], src[b]
            # compare row-major source order with candidate main-axis order
            s_order = (sa["y"] + sa["height"] / 2, sa["x"]) < (sb["y"] + sb["height"] / 2, sb["x"])
            c_order = key(cand[a]) < key(cand[b])
            pairs += 1
            agree += s_order == c_order
    f["order_agreement"] = agree / pairs if pairs else 1.0
    # hierarchy: does the biggest source text remain the biggest here?
    if texts and src:
        top = max(src.values(), key=lambda e: e["font_size"])["id"]
        f["headline_rank"] = sorted((-e["font_size"], e["id"]) for e in texts).index((-cand[top]["font_size"], top)) if top in cand else None
    return f


def descriptor(candidate: dict, source: dict) -> dict:
    """Composition-level descriptor: coarse occupancy by kind, per-element centres, font hierarchy, focal point."""
    tw, th = candidate["canvas"]["width"], candidate["canvas"]["height"]
    els = candidate["elements"]
    texts = [e for e in els if e["type"] == "text"]
    fonts = [e["font_size"] for e in texts]
    fmax = max(fonts) if fonts else 1.0
    focal = None
    if texts:
        big = max(texts, key=lambda e: (e["font_size"], e["width"] * e["height"]))
        focal = ((big["x"] + big["width"] / 2) / tw, (big["y"] + big["height"] / 2) / th)
    return dict(
        occ_text=occupancy(candidate, tw, th, ("text",)), occ_image=occupancy(candidate, tw, th, ("image",)),
        occ_shape=occupancy(candidate, tw, th, ("shape",)),
        centres=[((e["x"] + e["width"] / 2) / tw, (e["y"] + e["height"] / 2) / th) for e in els if not is_background(e, tw, th)],
        sizes=[(e["width"] / tw, e["height"] / th) for e in els if not is_background(e, tw, th)],
        hierarchy=[f / fmax for f in sorted(fonts, reverse=True)], focal=focal,
        font_scale=(statistics.median(fonts) / min(tw, th)) if fonts else 0.0)


def structure_distance(a: dict, b: dict) -> float:
    """0 = same composition, 1 = very different. Sub-percent coordinate perturbations score ~0."""
    def l1(x, y):
        return sum(abs(p - q) for p, q in zip(x, y)) / max(len(x), 1)
    occ = (l1(a["occ_text"], b["occ_text"]) * 2 + l1(a["occ_image"], b["occ_image"]) + l1(a["occ_shape"], b["occ_shape"]) * .5) / 3.5
    cen = sum(math.hypot(p[0] - q[0], p[1] - q[1]) for p, q in zip(a["centres"], b["centres"])) / max(len(a["centres"]), 1)
    siz = sum(abs(p[0] - q[0]) + abs(p[1] - q[1]) for p, q in zip(a["sizes"], b["sizes"])) / max(len(a["sizes"]), 1) / 2
    ha, hb = a["hierarchy"], b["hierarchy"]
    hier = l1(ha, hb) if len(ha) == len(hb) else 1.0
    foc = math.hypot(a["focal"][0] - b["focal"][0], a["focal"][1] - b["focal"][1]) if a["focal"] and b["focal"] else 0.0
    fa, fb = a.get("font_scale") or 1e-6, b.get("font_scale") or 1e-6
    fsc = min(1.0, abs(math.log(fa / fb)))
    return min(1.0, 1.6 * occ + .9 * cen + .5 * siz + .35 * hier + .4 * foc + .45 * fsc)
