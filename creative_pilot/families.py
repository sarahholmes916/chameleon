"""Twenty NEW source composition families for the creative pilot (development data only).

None of these reuse the ten template families of `scripts/source_templates.py`, so no family overlaps the
existing human-labelled data. Each family is a different spatial/hierarchical structure (not a recolour or
mirror of another): see `FAMILY_NOTES`. They reuse `Composition` for schema-only emission, seeded palettes,
and the real-font text fitting done by `scripts.source_dataset`.

Difficulty bands follow the three official samples: level 1 has ~4 plain elements, level 2 ~11 elements with
groups and translucent/overlapping shapes, level 3 ~17 elements with many groups and dense layering. Rotation
and the intentionally missing asset of the level-3 sample are NOT generated (the source gate rejects them).
"""

from __future__ import annotations

import random

from scripts.source_templates import ASSETS, Composition
from scripts.source_variation import load_profile, sample_parameters, validate_profile

GENERATOR_VERSION = "creative_sources_v1"
BG_IMG, LOGO_IMG, PRODUCT_IMG = ASSETS

EYEBROWS = ("SOLSTICE / SEASONAL EDITION", "NEW THIS WEEK", "LIMITED RELEASE", "THE GARDEN SERIES",
            "FIELD NOTES / VOL. 7", "SPRING COLLECTION", "OPEN STUDIO WEEKEND", "MEMBERS PREVIEW")
HEADS = ("Slow mornings, bright ideas", "The season starts here", "Made for long evenings",
         "Bring the outside in", "Fresh ideas in full bloom", "A better way to gather", "Light, air and room to grow",
         "Small batches, big flavor", "Everything glows in July", "Find your golden hour",
         "Good things take their time", "Open the windows")
SHORT = ("Golden hour", "Open air", "Summer edit", "New season", "Slow down", "Made to share", "Just in", "Sunday best")
BODIES = ("An invitation to pause, share a story and enjoy the season together.",
          "Handmade in small batches and finished by people who care about the details.",
          "Join us for a weekend of workshops, tastings and conversation under the open sky.",
          "Good company, fresh perspectives and a new reason to step outside and stay a little longer.",
          "Thoughtfully designed essentials that make every ordinary day feel a little more special.")
CTAS = ("SHOP THE EDIT", "RESERVE A SEAT", "LEARN MORE", "DISCOVER NOW", "GET TICKETS", "EXPLORE")
STATS = ((("12k", "members"), ("48", "workshops"), ("96%", "return rate"), ("7", "studios")),
         (("3x", "faster"), ("24h", "delivery"), ("100%", "recycled"), ("5 min", "setup")))
STEPS = (("Gather", "Bring your ideas and a friend."), ("Make", "Hands-on sessions led by local makers."),
         ("Share", "Swap stories over a long table."), ("Grow", "Take something new back home."),
         ("Return", "Come back next season."))
TEASERS = (("Design", "Rooms that breathe"), ("Food", "Long tables, slow suppers"), ("Travel", "Places worth the detour"),
           ("Makers", "Meet the people behind it"), ("Garden", "What to plant in May"))
LINEUP = (("5:00", "Doors and welcome"), ("5:45", "Garden stage: Ada Rowe"), ("6:30", "Makers market opens"),
          ("7:15", "Headliner: The Slow Parade"), ("9:00", "Lantern walk"), ("9:45", "Closing set"))
PRODUCTS = (("Botanical No. 1", "$24"), ("Evening Blend", "$18"), ("Golden Hour", "$32"), ("Studio Set", "$46"))

FAMILY_NOTES = {
    "hero_image_caption": ("image-dominant", 1, "full-bleed image, caption chip and title strip"),
    "giant_type_stack": ("typography-dominant", 1, "oversized headline filling the canvas, tiny footer"),
    "logo_tagline_minimal": ("unconventional negative space", 1, "small off-centre lockup in empty field"),
    "offset_product_spotlight": ("offset typography and imagery", 1, "product low-right over circle, headline high-left"),
    "asymmetric_editorial": ("asymmetric editorial", 2, "edge image column, offset serif title, narrow measure"),
    "layered_promo_circles": ("layered promotional", 2, "three translucent circles under product and headline"),
    "diagonal_staggered_blocks": ("staggered content blocks", 2, "three panels stepping diagonally with an anchor image"),
    "stat_dashboard": ("multi-panel information", 2, "headline over a row of big-number stat panels"),
    "feature_trio": ("three-column features", 2, "left headline and CTA, three numbered feature columns"),
    "timeline_sequence": ("sequence / timeline", 2, "markers joined by a rule with numbered steps"),
    "poster_split_diagonal": ("geometric promotional poster", 2, "two large colour fields, headline vs image"),
    "quote_callout": ("typography-dominant callout", 2, "giant quote mark, quote, attribution and logo"),
    "price_tag_product": ("product ad, unusual hierarchy", 2, "price badge overlapping the product, features beneath"),
    "magazine_cover": ("magazine cover", 2, "masthead, centre image, side cover lines"),
    "image_tile_gallery": ("image-led grid", 2, "image tiles of different scale with captions"),
    "sidebar_notice": ("dense notice with sidebar", 2, "main column of text plus a tall sidebar panel"),
    "dense_magazine_grid": ("dense magazine layout", 3, "masthead, hero, headline and three teasers plus sidebar"),
    "multi_panel_info": ("multi-panel information", 3, "header band over a grid of six titled panels and footer CTA"),
    "competing_focal_points": ("multiple competing focal points", 3, "two products, two headlines, central badge"),
    "event_lineup_poster": ("complex grouped layout", 3, "giant date, lineup rows, side band and ornaments"),
    "product_catalog_cards": ("product cards", 3, "header, three product cards with prices and buttons, footer"),
    "layered_geometric_poster": ("overlapping geometric shapes", 3, "many translucent shapes under headline, stats and CTA"),
}
FAMILIES = {name: dict(composition=v[0], difficulty=v[1], description=v[2]) for name, v in FAMILY_NOTES.items()}


class ClampedComposition(Composition):
    """Shrink (never shift) a box that would leave the canvas, so ratio-sized discs stay in bounds."""

    def add(self, kind, box, **kwargs):
        x, y, w, h = box
        x, y = min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)
        return super().add(kind, (x, y, max(min(w, 1 - x), 1e-3), max(min(h, 1 - y), 1e-3)), **kwargs)


def ghost(c: Composition, box, color, opacity, *, ellipse=False, group=None):
    element = c.shape(box, color, ellipse=ellipse, group=group)
    element["opacity"] = round(opacity, 2)
    return element


def pill(c, label, box, group, fill=None, ink=None):
    x, y, w, h = box
    shape = c.shape(box, fill or c.ink, group=group)
    shape["corner_radius"] = round(min(shape["width"], shape["height"]) / 2, 3)
    c.text(label, (x + .02, y + .15 * h, w - .04, .7 * h), .024, bold=True, color=ink or c.bg, align="center", group=group)


def build(family: str, c: Composition, r: random.Random, p: dict) -> None:
    m = p["margin"]
    span = 1 - 2 * m
    head, short, body, cta, eyebrow = r.choice(HEADS), r.choice(SHORT), r.choice(BODIES), r.choice(CTAS), r.choice(EYEBROWS)
    c.shape((0, 0, 1, 1), c.bg)

    if family == "hero_image_caption":
        c.image(BG_IMG, (0, 0, 1, 1), cover=True)
        ghost(c, (0, .66, 1, .34), c.bg, .9)
        c.text(head, (m, .70, .62, .22), .062, serif=True)
        c.text(body, (.70, .72, .26, .18), .022)
        c.shape((m, .10, .16, .048), c.bg, group="g_chip")
        c.text(eyebrow.split(" / ")[0], (m + .015, .115, .13, .02), .016, bold=True, group="g_chip")
    elif family == "giant_type_stack":
        c.text(short.upper() if len(short) < 10 else "OPEN\nAIR", (m, .05, span, .55), .26, bold=True, color=c.ink)
        c.shape((m, .64, .30, .02), c.accent)
        c.text(head, (m, .70, .60, .17), .05, serif=True)
        c.text(eyebrow, (m, .91, .5, .04), .018)
    elif family == "logo_tagline_minimal":
        c.image(LOGO_IMG, (.12, .16, .42, .14))
        c.shape((.12, .34, .06, .006), c.accent)
        c.text(head, (.12, .39, .40, .24), .05, serif=True)
        c.text(cta, (.68, .88, .24, .045), .02, bold=True)
    elif family == "offset_product_spotlight":
        c.shape((.46, .30, .52, .66), c.panel, ellipse=True)
        c.image(PRODUCT_IMG, (.52, .38, .42, .56))
        c.text(head, (m, .08, .56, .32), .075, serif=True)
        c.text(body, (m, .50, .34, .20), .026)
        pill(c, cta, (m, .78, .30, .075), "g_cta")
    elif family == "asymmetric_editorial":
        c.image(BG_IMG, (0, 0, .24, 1), cover=True)
        c.text(eyebrow, (.30, .06, .6, .04), .018, bold=True)
        c.shape((.30, .12, .05, .006), c.accent)
        c.text(head, (.30, .17, .62, .30), .082, serif=True)
        c.text(body, (.30, .52, .36, .20), .026)
        c.text("Words by the editors  /  Photographs by the studio", (.30, .88, .5, .04), .016)
    elif family == "layered_promo_circles":
        ghost(c, (.05, .08, .56, .56), c.accent, .35, ellipse=True)
        ghost(c, (.30, .30, .62, .62), c.panel, .75, ellipse=True)
        ghost(c, (.60, .04, .34, .34), c.ink, .12, ellipse=True)
        c.image(PRODUCT_IMG, (.34, .22, .44, .56))
        c.text(head, (m, .66, .60, .22), .066, serif=True)
        pill(c, cta, (.66, .86, .28, .07), "g_cta")
        c.text(eyebrow, (m, .05, .5, .04), .018, bold=True)
    elif family == "diagonal_staggered_blocks":
        for i in range(3):
            x, y = .06 + .22 * i, .08 + .27 * i
            g = f"g{i}"
            c.shape((x, y, .44, .22), c.panel if i != 1 else c.accent, group=g)
            title, copy = STEPS[(i + p["feature_count"]) % 5]
            c.text(title, (x + .02, y + .02, .40, .07), .036, serif=True, bold=True,
                   color=c.ink if i != 1 else c.bg, group=g)
            c.text(copy, (x + .02, y + .10, .40, .10), .021, color=c.ink if i != 1 else c.bg, group=g)
        c.image(PRODUCT_IMG, (.74, .52, .22, .30))
        c.text(head, (.06, .88, .6, .08), .034, serif=True)
    elif family == "stat_dashboard":
        c.text(eyebrow, (m, .05, span, .04), .018, bold=True)
        c.text(head, (m, .11, .66, .26), .066, serif=True)
        stats = STATS[p["card_count"] % 2]
        n = 3
        for i in range(n):
            x, w = m + i * span / n, span / n - .02
            g = f"g{i}"
            c.shape((x, .46, w, .34), c.panel if i != 0 else c.ink, group=g)
            num, lab = stats[i]
            c.text(num, (x + .02, .50, w - .04, .15), .08, bold=True, color=c.accent if i != 0 else c.bg, group=g)
            c.text(lab, (x + .02, .68, w - .04, .08), .024, color=c.ink if i != 0 else c.bg, group=g)
        c.text(body, (m, .86, .7, .08), .02)
    elif family == "feature_trio":
        c.text(head, (m, .08, .40, .42), .07, serif=True)
        pill(c, cta, (m, .58, .30, .075), "g_cta")
        for i in range(3):
            x = .50 + i * .17
            g = f"g{i}"
            c.shape((x, .10, .005, .78), c.accent)
            c.text(f"0{i + 1}", (x + .015, .10, .12, .08), .04, bold=True, color=c.accent, group=g)
            c.text(STEPS[i][0], (x + .015, .22, .13, .08), .028, serif=True, bold=True, group=g)
            c.text(STEPS[i][1], (x + .015, .33, .13, .4), .02, group=g)
    elif family == "timeline_sequence":
        c.text(head, (m, .06, span, .14), .06, serif=True)
        c.shape((m + .02, .36, span - .04, .008), c.accent)
        n = p["stat_count"] + 1
        for i in range(n):
            x, w = m + i * span / n, span / n - .015
            g = f"g{i}"
            c.shape((x, .335, .045, .045 * p["canvas"][0] / p["canvas"][1]), c.accent, ellipse=True)
            c.text(STEPS[i][0], (x, .46, w, .08), .03, serif=True, bold=True, group=g)
            c.text(STEPS[i][1], (x, .56, w, .22), .02, group=g)
        c.text(eyebrow, (m, .90, .5, .04), .017, bold=True)
    elif family == "poster_split_diagonal":
        c.shape((0, 0, .58, 1), c.ink)
        c.shape((.58, 0, .42, .62), c.accent)
        c.image(PRODUCT_IMG, (.60, .08, .36, .46))
        c.text(head, (m, .12, .46, .46), .078, serif=True, color=c.bg)
        c.text(body, (m, .66, .42, .16), .022, color=c.bg)
        c.text(eyebrow, (.62, .70, .34, .08), .02, bold=True)
        pill(c, cta, (.62, .84, .30, .07), "g_cta", fill=c.ink, ink=c.bg)
    elif family == "quote_callout":
        c.text("“", (m, .02, .25, .30), .30, serif=True, bold=True, color=c.accent)
        c.text(head + ".", (m + .02, .28, .60, .40), .07, serif=True)
        c.shape((m + .02, .74, .06, .006), c.accent)
        c.text("A. Solstice, founder", (m + .02, .78, .45, .05), .026, bold=True)
        c.shape((.68, .12, .26, .26 * p["canvas"][0] / p["canvas"][1]), c.panel, ellipse=True)
        c.image(PRODUCT_IMG, (.70, .15, .22, .22 * p["canvas"][0] / p["canvas"][1]))
        c.image(LOGO_IMG, (.62, .86, .32, .08))
    elif family == "price_tag_product":
        c.image(PRODUCT_IMG, (.28, .08, .62, .62))
        c.shape((m, .52, .40, .24), c.accent, group="g_price")
        c.text(PRODUCTS[0][1].replace("$", "$ "), (m + .02, .55, .36, .18), .11, bold=True, color=c.bg, group="g_price")
        c.text(PRODUCTS[0][0], (m, .08, .30, .20), .045, serif=True)
        for i, feat in enumerate(("Small batch", "Plant based", "Made to share")):
            c.text(feat, (m, .82 + i * .05, .6, .045), .022)
        pill(c, cta, (.64, .86, .30, .075), "g_cta")
    elif family == "magazine_cover":
        c.text("SOLSTICE", (m, .03, span, .16), .12, serif=True, bold=True, align="center")
        c.shape((m, .20, span, .005), c.ink)
        c.image(BG_IMG, (.26, .24, .48, .56), cover=True)
        for i, (k, t) in enumerate(TEASERS[:2]):
            g = f"gl{i}"
            c.text(k.upper(), (m, .26 + i * .26, .18, .04), .016, bold=True, color=c.accent, group=g)
            c.text(t, (m, .31 + i * .26, .18, .17), .028, serif=True, group=g)
        for i, (k, t) in enumerate(TEASERS[2:4]):
            g = f"gr{i}"
            c.text(k.upper(), (.78, .26 + i * .26, .17, .04), .016, bold=True, color=c.accent, group=g)
            c.text(t, (.78, .31 + i * .26, .17, .17), .028, serif=True, group=g)
        c.text(head, (m, .84, span, .1), .05, serif=True, align="center")
    elif family == "image_tile_gallery":
        c.image(BG_IMG, (m, .08, .46, .56), cover=True, group="g_main")
        c.text("Main hall, morning light", (m, .66, .46, .05), .02, group="g_main")
        c.image(PRODUCT_IMG, (.56, .08, .2, .22), group="g_a")
        c.text("Table no. 4", (.56, .32, .2, .04), .018, group="g_a")
        c.image(LOGO_IMG, (.56, .42, .38, .12))
        c.image(PRODUCT_IMG, (.80, .08, .14, .2), group="g_b")
        c.text("Small batch", (.80, .30, .14, .04), .016, group="g_b")
        c.text(head, (m, .78, .6, .14), .05, serif=True)
    elif family == "sidebar_notice":
        c.shape((.70, 0, .30, 1), c.ink)
        c.text("AT A GLANCE", (.73, .06, .24, .05), .02, bold=True, color=c.bg)
        for i, (k, v) in enumerate(STEPS[:4]):
            c.text(k, (.73, .15 + i * .19, .24, .05), .03, serif=True, bold=True, color=c.bg, group=f"s{i}")
            c.text(v, (.73, .21 + i * .19, .24, .11), .019, color=c.bg, group=f"s{i}")
        c.text(eyebrow, (m, .06, .56, .04), .018, bold=True)
        c.text(head, (m, .13, .58, .30), .07, serif=True)
        c.text(body + " " + r.choice(BODIES), (m, .50, .52, .26), .026)
        c.text(cta, (m, .86, .4, .05), .026, bold=True, color=c.accent)
    elif family == "dense_magazine_grid":
        c.text("THE SOLSTICE REVIEW", (m, .03, .6, .06), .036, serif=True, bold=True)
        c.text("Issue 14  /  Summer", (.68, .04, .26, .04), .018)
        c.shape((m, .10, span, .004), c.ink)
        c.image(BG_IMG, (m, .13, .50, .40), cover=True, group="g_hero")
        c.shape((m, .43, .50, .1), c.bg, group="g_hero")
        c.text(short, (m + .01, .44, .48, .08), .04, serif=True, group="g_hero")
        c.text(head, (.60, .13, .34, .22), .052, serif=True)
        c.text(body, (.60, .37, .34, .17), .02)
        for i in range(3):
            x, w = m + i * .30, .27
            g = f"g{i}"
            c.shape((x, .60, w, .004), c.accent)
            c.text(TEASERS[i][0].upper(), (x, .62, w, .035), .015, bold=True, color=c.accent, group=g)
            c.text(TEASERS[i][1], (x, .66, w, .10), .026, serif=True, group=g)
            c.text(r.choice(BODIES), (x, .77, w, .13), .017, group=g)
        c.shape((.60, .55, .34, .035), c.panel)
        c.text(cta, (m, .93, .5, .035), .017, bold=True)
    elif family == "multi_panel_info":
        c.shape((0, 0, 1, .2), c.ink)
        c.text(head, (m, .035, .66, .13), .05, serif=True, color=c.bg)
        c.text(eyebrow, (.72, .07, .24, .06), .016, bold=True, color=c.accent)
        for i in range(6):
            col, row = i % 3, i // 3
            x, y = m + col * span / 3, .26 + row * .27
            w = span / 3 - .02
            g = f"g{i}"
            c.shape((x, y, w, .24), c.panel if (col + row) % 2 == 0 else c.bg, group=g)
            c.text(TEASERS[i % 5][0], (x + .015, y + .02, w - .03, .06), .028, serif=True, bold=True, group=g)
            c.text(STEPS[i % 5][1], (x + .015, y + .09, w - .03, .13), .019, group=g)
        pill(c, cta, (m, .84, .3, .08), "g_cta", fill=c.accent, ink=c.bg)
        c.text("Free entry  /  All ages", (.60, .87, .34, .04), .02)
    elif family == "competing_focal_points":
        ghost(c, (0, 0, .52, 1), c.panel, .9)
        ghost(c, (.48, .1, .52, .8), c.accent, .22)
        c.image(PRODUCT_IMG, (.04, .12, .38, .38))
        c.image(PRODUCT_IMG, (.58, .22, .36, .36))
        c.text(short, (.05, .54, .42, .14), .07, serif=True, bold=True)
        c.text(r.choice(SHORT), (.56, .62, .40, .14), .07, serif=True, bold=True)
        c.shape((.42, .40, .16, .16 * p["canvas"][0] / p["canvas"][1]), c.ink, ellipse=True, group="g_vs")
        c.text("vs", (.44, .445, .12, .07), .04, bold=True, color=c.bg, align="center", group="g_vs")
        c.text(body, (.05, .74, .40, .12), .02)
        pill(c, cta, (.30, .89, .40, .07), "g_cta")
        c.text(eyebrow, (m, .04, .5, .04), .017, bold=True)
    elif family == "event_lineup_poster":
        c.shape((.78, 0, .22, 1), c.ink)
        ghost(c, (.62, .70, .4, .4 * p["canvas"][0] / p["canvas"][1]), c.accent, .5, ellipse=True)
        c.text("SAT\n14", (m, .04, .40, .30), .13, bold=True, color=c.accent)
        c.text(head, (.42, .06, .34, .28), .05, serif=True)
        for i in range(5):
            y = .40 + i * .105
            g = f"g{i}"
            c.shape((m, y - .01, .68 - m, .003), c.ink)
            c.text(LINEUP[i][0], (m, y + .005, .14, .06), .03, bold=True, color=c.accent, group=g)
            c.text(LINEUP[i][1], (.22, y + .005, .50, .06), .028, group=g)
        c.text("GARDENS\nOPEN", (.80, .08, .18, .2), .036, bold=True, color=c.bg)
        c.text("Free entry\nAll welcome", (.80, .88, .18, .08), .018, color=c.bg)
        c.text(eyebrow, (m, .93, .6, .04), .016)
    elif family == "product_catalog_cards":
        c.text(head, (m, .04, .62, .14), .052, serif=True)
        c.text(eyebrow, (.70, .06, .25, .08), .016, bold=True, color=c.accent)
        for i in range(3):
            x, w = m + i * span / 3, span / 3 - .02
            g = f"g{i}"
            c.shape((x, .22, w, .56), c.panel, group=g)
            c.image(PRODUCT_IMG, (x + .02, .24, w - .04, .30), group=g)
            c.text(PRODUCTS[i][0], (x + .02, .56, w - .04, .06), .026, serif=True, bold=True, group=g)
            c.text(PRODUCTS[i][1], (x + .02, .63, w * .4, .06), .028, bold=True, color=c.accent, group=g)
            c.shape((x + .02, .70, w - .04, .06), c.ink, group=g)
            c.text("ADD", (x + .02, .712, w - .04, .035), .016, bold=True, color=c.bg, align="center", group=g)
        c.shape((0, .86, 1, .14), c.ink)
        c.text("Free delivery over $40  /  Returns within 30 days", (m, .90, span, .05), .022, color=c.bg)
    else:  # layered_geometric_poster
        ghost(c, (.02, .04, .50, .50 * p["canvas"][0] / p["canvas"][1]), c.accent, .45, ellipse=True)
        ghost(c, (.30, .20, .60, .36), c.panel, .85)
        ghost(c, (.62, .02, .32, .32 * p["canvas"][0] / p["canvas"][1]), c.ink, .14, ellipse=True)
        ghost(c, (.04, .60, .36, .28), c.ink, .9)
        c.text(head, (.34, .24, .52, .28), .062, serif=True)
        c.text(eyebrow, (.34, .08, .5, .04), .018, bold=True)
        stats = STATS[p["card_count"] % 2]
        for i in range(2):
            g = f"g{i}"
            x = .06 + i * .16
            c.text(stats[i][0], (x, .63, .14, .12), .06, bold=True, color=c.bg, group=g)
            c.text(stats[i][1], (x, .76, .14, .06), .018, color=c.bg, group=g)
        c.image(LOGO_IMG, (.52, .62, .40, .13))
        pill(c, cta, (.52, .82, .34, .08), "g_cta", fill=c.accent, ink=c.bg)
        c.text(body, (.04, .91, .86, .05), .019)


def make_creative_source(family: str, seed: int, variant: int, *, profile: dict | None = None) -> tuple[dict, dict]:
    if family not in FAMILIES:
        raise ValueError(f"unknown creative family {family}")
    profile = load_profile() if profile is None else validate_profile(profile)
    index = list(FAMILIES).index(family)
    parameters, rng = sample_parameters(family, seed, variant, index, profile)
    c = ClampedComposition(parameters, rng)
    build(family, c, rng, parameters)
    parameters = dict(parameters, creative_family=family, difficulty=FAMILIES[family]["difficulty"])
    return {"canvas": {"width": c.width, "height": c.height}, "elements": c.elements}, parameters
