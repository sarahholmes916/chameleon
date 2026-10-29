"""Original source compositions, not S04 resize candidates (development only)."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.source_variation import load_profile, sample_parameters, validate_profile, warp


GENERATOR_VERSION = "source_templates_v3"
FAMILIES = {
    "centered_editorial": "editorial",
    "split_product": "product_advertisement",
    "product_pedestal": "product_advertisement",
    "image_panel": "promotion",
    "type_poster": "event_flyer",
    "event_schedule": "event_flyer",
    "information_cards": "announcement",
    "offer_coupon": "promotion",
    "announcement_bands": "announcement",
    "editorial_columns": "editorial",
}
ASSETS = ("solstice_bg_gradient.png", "solstice_logo_lockup.png", "solstice_bottle.png")
COPY_PATH = Path(__file__).with_name("source_copy.json")


def rgba(hex_color: str) -> str:
    """The vendored shape renderer supports rgba(), not hex literals."""
    r, g, b = (int(hex_color[i:i+2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},1)"


class Composition:
    """Place related blocks in canvas fractions; emit only schema fields."""

    def __init__(self, parameters: dict, rng):
        width, height = parameters["canvas"]
        self.width, self.height = width, height
        self.unit = min(width, height)
        self.bg, self.ink, self.accent, self.panel = parameters["palette"]
        self.parameters, self.rng = parameters, rng
        self.copy = json.loads(COPY_PATH.read_text(encoding="utf-8"))
        self.elements: list[dict] = []

    def add(self, kind: str, box: tuple, *, group: str | None = None, mirror: bool = True, **fields) -> dict:
        """`mirror=False` keeps a block's internal order (e.g. number left of label) when the layout is mirrored."""
        x, y, w, h = box
        left, right = (warp(v, self.parameters["x_knots"]) for v in (x, x+w))
        top, bottom = (warp(v, self.parameters["y_knots"]) for v in (y, y+h))
        if self.parameters["mirrored"] and mirror:
            left, right = 1-right, 1-left
        x, y, w, h = left, top, right-left, bottom-top
        element = dict(id=f"e{len(self.elements):03d}", type=kind,
                       x=round(x*self.width, 3), y=round(y*self.height, 3),
                       width=round(w*self.width, 3), height=round(h*self.height, 3),
                       z_index=len(self.elements), **fields)
        if group is not None:
            element["group_id"] = group
        self.elements.append(element)
        return element

    def shape(self, box, color=None, *, ellipse=False, group=None):
        element = self.add("shape", box, group=group,
                        shape_type="ellipse" if ellipse else "rectangle",
                        fill_color=rgba(color or self.panel))
        if group and not ellipse:
            element["corner_radius"] = round(min(element["width"], element["height"])*self.parameters["corner_radius"]*5, 3)
        return element

    def text(self, content, box, size=.032, *, serif=False, bold=False,
             color=None, align="left", group=None, mirror=True):
        content = self.rng.choice(self.copy.get(content, [content]))
        heading = serif or size >= .045
        scale = self.parameters["font_scale"]*(self.parameters["heading_scale"] if heading else 1)
        return self.add("text", box, group=group, mirror=mirror, content=content,
                        font_family=self.parameters["heading_font" if heading else "body_font"],
                        font_size=max(12, round(size*self.unit*scale, 2)),
                        font_weight="bold" if bold else self.parameters["heading_weight"] if heading else "regular",
                        color=rgba(color or self.ink),
                        text_align=self.parameters["body_alignment"] if align == "left" else align,
                        line_height=self.parameters["line_height"])

    def image(self, asset, box, *, cover=False, group=None):
        if not cover:
            x, y, w, h = box
            scale = self.parameters["image_scale"]
            box = (x+w*(1-scale)/2, y+h*(1-scale)/2, w*scale, h*scale)
        return self.add("image", box, group=group, asset_id=asset,
                        fit="cover" if cover else "contain")

    def button(self, label, box, group="g_action"):
        x, y, w, h = box
        self.shape(box, self.ink, group=group)
        self.text(label, (x+.02, y+.018, w-.04, h-.027), .024,
                  bold=True, color=self.bg, align="center", group=group)


def make_source(family: str, seed: int, variant: int = 0, *, profile: dict | None = None) -> dict:
    """Deterministic source draft. The dataset runner fits text with the real fonts."""
    return make_source_with_metadata(family, seed, variant, profile=profile)[0]


def make_source_with_metadata(family: str, seed: int, variant: int = 0, *, profile: dict | None = None) -> tuple[dict, dict]:
    if family not in FAMILIES:
        raise ValueError(f"Unknown source template: {family}")
    profile = load_profile() if profile is None else validate_profile(profile)
    index = list(FAMILIES).index(family)
    parameters, rng = sample_parameters(family, seed, variant, index, profile)
    c = Composition(parameters, rng)
    margin = parameters["margin"]
    span = 1-2*margin
    title = rng.choice(("Make room for slow afternoons", "A little more golden hour",
                        "Gather around something good"))
    body = rng.choice(("An invitation to pause, share a story and enjoy the season together.",
                       "Good company. Fresh perspectives. A new reason to step outside and stay a little longer."))
    body = rng.choice(c.copy[body])
    c.shape((0, 0, 1, 1), c.bg)

    if family == "centered_editorial":
        c.text(f"FIELD NOTES  /  VOL. {rng.randint(1, 24):02d}", (margin, .08, span, .05), .021, align="center")
        c.shape((.46, .19, .08, .005), c.accent)
        c.text(title, (.14, .29, .72, .28), .079, serif=True, align="center")
        c.text(body, (.21, .63, .58, .18), .027, align="center")
        c.text("READ. REFLECT. REPEAT.", (.15, .9, .7, .045), .018, align="center")
    elif family == "split_product":
        c.shape((0, 0, .47, 1), c.panel)
        c.image(ASSETS[2], (.08, .22, .30, .60), group="g_product")
        c.text("THE BOTANICAL COLLECTION", (.06, .09, .35, .07), .021, group="g_product")
        c.text("Small batch.\nBig occasion.", (.53, .18, .41, .27), .064, serif=True)
        c.text(body, (.53, .52, .39, .20), .027)
        c.button("EXPLORE THE COLLECTION", (.53, .82, .40, .09))
    elif family == "product_pedestal":
        c.text("SOLSTICE / SEASONAL EDITION", (margin, .05, span, .045), .02, align="center")
        c.text("Meet your golden hour", (.1, .13, .8, .12), .055, serif=True, align="center")
        c.shape((.24, .30, .52, .43), c.panel, ellipse=True)
        c.shape((.23, .68, .54, .06), c.accent)
        c.image(ASSETS[2], (.34, .27, .32, .44))
        count = parameters["feature_count"]
        labels = rng.sample(("Botanical notes", "Small batches", "Made to share", "Seasonal selection", "A fresh ritual"), count)
        for j, label in enumerate(labels):
            x = margin+j*span/count
            c.text(f"{(count-j if parameters['mirrored'] else j+1):02d}", (x, .81, span/count-.02, .05), .027, bold=True, align="center", group=f"g{j}")
            c.text(label, (x, .88, span/count-.02, .06), .022, align="center", group=f"g{j}")
    elif family == "image_panel":
        c.image(ASSETS[0], (0, 0, 1, 1), cover=True)
        if parameters["show_logo"]:
            c.image(ASSETS[1], (.07, .06, .3, .09))
        c.image(ASSETS[2], (.63, .22, .28, .58))
        c.shape((.055, .28, .52, .59), c.panel, group="g_panel")
        c.text("THE WEEKEND EDIT", (.09, .32, .43, .05), .021, color=c.ink, group="g_panel")
        c.text("Bring a little\nsunshine home.", (.09, .42, .43, .24), .056,
               serif=True, color=c.ink, group="g_panel")
        c.text("Discover the seasonal collection.", (.09, .73, .43, .09), .025,
               color=c.ink, group="g_panel")
    elif family == "type_poster":
        c.shape((.82, .08, .10, .10), c.accent, ellipse=True)
        c.text("AN OPEN-AIR GATHERING", (margin, .07, .69, .06), .023)
        c.text("STAY\nFOR THE\nSUNSET", (margin, .24, span, .48), .115, bold=True)
        c.shape((margin, .78, span, .007), c.accent)
        c.text("SATURDAY  /  5 PM\nRiverside gardens", (margin, .84, .53, .12), .027)
        c.text("Music. Stories.\nEveryone welcome.", (.63, .84, .30, .12), .023)
    elif family == "event_schedule":
        c.text("THE GARDEN SESSIONS", (margin, .06, span, .06), .022, bold=True)
        c.text("An evening\nwell spent", (margin, .17, .56, .25), .065, serif=True)
        c.shape((.73, .18, .20, .20), c.accent, group="g_date")
        month = rng.choice(("MAY", "JUN", "JUL", "AUG", "SEP"))
        c.text(f"{month}\n{rng.randint(1, 28):02d}", (.75, .20, .16, .16), .044, align="center", color=c.bg, group="g_date")
        count = parameters["schedule_count"]
        start_hour = rng.randint(12, 16)
        events = ["Doors & welcome"]+rng.sample(("Live acoustic set", "A shared table", "Meet the makers", "Creative workshop", "Garden conversations"), count-1)
        for j, name in enumerate(events):
            time = f"{start_hour+j:02d}:00"
            y, row = .48+j*.40/count, .40/count
            c.shape((margin, y, span, .002), c.accent)
            c.text(time, (margin, y+row*.16, .23, row*.7), .03, bold=True, group=f"g{j}")
            c.text(name, (.33, y+row*.16, .60, row*.7), .029, group=f"g{j}")
        c.text("Riverside gardens  /  Free entry", (margin, .91, span, .05), .02)
    elif family == "information_cards":
        c.text("A NEW WAY TO SPEND YOUR SATURDAY", (margin, .07, span, .07), .022)
        c.text("Come curious. Leave inspired.", (margin, .19, span, .17), .056, serif=True)
        count = parameters["card_count"]
        labels = rng.sample((("Make", "Hands-on workshops\nfor all experience levels."),
                             ("Meet", "Local makers and\nnew conversations."),
                             ("Discover", "Fresh ideas to take\nback into your week."),
                             ("Share", "Bring a story.\nFind a new perspective.")), count)
        for j, (name, copy) in enumerate(labels):
            n = f"{(count-j if parameters['mirrored'] else j+1):02d}"
            x, w = margin+j*span/count, span/count-.015
            group = f"g{j}"
            c.shape((x, .43, w, .41), c.panel, group=group)
            c.text(n, (x+.02, .46, w-.04, .06), .026, bold=True, group=group)
            c.text(name, (x+.02, .57, w-.04, .08), .036, serif=True, group=group)
            c.text(copy, (x+.02, .70, w-.04, .12), .022, group=group)
        c.text("THE COMMUNITY STUDIO  /  OPEN DAY", (margin, .91, span, .04), .018)
    elif family == "offer_coupon":
        c.shape((.06, .09, .88, .81), c.panel)
        c.text("THE SEASONAL THANK-YOU", (.11, .14, .78, .06), .022)
        discount = parameters["discount"]
        c.text(f"{discount}%", (.11, .28, .46, .25), .17, bold=True, color=c.accent)
        c.text("A little treat\nfor your next gathering.", (.61, .31, .27, .23), .036, serif=True)
        c.text("Enjoy the collection with friends.", (.11, .59, .76, .10), .029)
        count = parameters["perforations"]
        for j in range(count):
            c.shape((.11+j*.78/count, .74, .30/count, .003), c.ink)
        code = rng.choice(("GATHER", "SHARE", "SEASON"))+str(discount)
        c.text(f"USE CODE: {code}", (.11, .80, .50, .06), .027, bold=True, group="g_offer")
        c.text(f"Through {rng.choice(('May', 'June', 'July', 'August'))} {rng.randint(10, 28)}", (.65, .80, .22, .06), .02, group="g_offer")
    elif family == "announcement_bands":
        c.shape((0, .13, 1, .39), c.ink)
        c.text("NEWS FROM THE STUDIO", (margin, .055, span, .045), .02)
        c.text("Our doors\nare open.", (margin, .19, span, .28), .082, serif=True, color=c.bg)
        c.text("A place to make, learn and connect.", (margin, .59, span, .10), .034)
        count = parameters["stat_count"]
        for j, label in enumerate(rng.sample(("creative spaces", "weekly sessions", "local makers"), count)):
            slot = span/count
            x = margin+((count-1-j) if parameters["mirrored"] else j)*slot  # pair stays number-then-label
            c.text(str(rng.randint(2, 24)), (x, .75, slot*.30, .13), .066, bold=True, group=f"g{j}", mirror=False)
            c.text(label, (x+slot*.36, .80, slot*.60, .09), .025, group=f"g{j}", mirror=False)
        c.text("Visit the community studio this weekend.", (margin, .93, span, .04), .019)
    else:  # editorial_columns: a pull quote and two deliberately separated columns
        c.text("THE SLOW LIVING JOURNAL", (margin, .055, span, .045), .02)
        c.shape((margin, .13, span, .003), c.accent)
        c.text("The art of\nbeing here", (margin, .19, .52, .22), .062, serif=True)
        c.image(ASSETS[0], (.67, .19, .26, .24), cover=True, group="g_image")
        c.text("A study in evening light", (.67, .45, .26, .055), .016, group="g_image")
        c.text("Small rituals make space for something bigger.", (margin, .48, .53, .12), .032, serif=True)
        c.text(body+"\n\nStart with an open window and a moment of quiet.",
               (margin, .66, .40, .25), .023)
        c.text("Set the table. Put on a favorite record. Invite a friend.\n\nThe best occasions can begin with a very simple plan.",
               (.54, .66, .39, .25), .023)
    return {"canvas": {"width": c.width, "height": c.height}, "elements": c.elements}, parameters
