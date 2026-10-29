"""Seeded design parameters and declared-color checks; no ML or asset modification."""

from __future__ import annotations

import colorsys
import copy
import json
import math
import random
from pathlib import Path


PROFILE_PATH = Path(__file__).with_name("source_variation.json")
MIN_CONTRAST = 4.5  # Internal source-quality threshold, not an Adobe rule.


def load_profile(path: Path = PROFILE_PATH) -> dict:
    defaults = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    supplied = json.loads(path.read_text(encoding="utf-8")) if path != PROFILE_PATH else defaults
    return validate_profile(supplied, defaults)


def validate_profile(profile: dict, defaults: dict | None = None) -> dict:
    """Strict complete profile; reject typos and ranges outside tested geometric limits."""
    defaults = defaults or json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    if not isinstance(profile, dict) or profile.keys() != defaults.keys():
        raise ValueError("Variation profile must have exactly the documented keys")
    if profile["version"] != "source_variation_v1":
        raise ValueError("Unsupported variation profile version")
    limits = {
        "short_edge": (720, 1200), "portrait_ratio": (.65, .95), "landscape_ratio": (1.1, 1.7),
        "margin": (.045, .08), "font_scale": (.85, 1.15), "heading_scale": (.9, 1.15),
        "line_height": (1.15, 1.4), "image_scale": (.8, 1.0), "corner_radius": (0, .025),
        "card_count": (2, 4), "schedule_count": (2, 5), "feature_count": (2, 4),
        "stat_count": (2, 3), "perforations": (6, 16), "discount": (10, 40),
        "axis_jitter": (0, .025),
    }
    limits.update({name: (0, 1) for name in (
        "hue", "background_saturation", "neutral_saturation", "light_background_lightness",
        "dark_background_lightness", "light_ink_lightness", "dark_ink_lightness",
        "panel_lightness_offset", "accent_saturation", "light_accent_lightness", "dark_accent_lightness")})
    for name, (low, high) in limits.items():
        value = profile[name]
        if (not isinstance(value, list) or len(value) != 2
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in value)
                or not low <= value[0] <= value[1] <= high):
            raise ValueError(f"{name} must be an ordered numeric range within [{low}, {high}]")
        if name in ("short_edge", "card_count", "schedule_count", "feature_count", "stat_count", "perforations", "discount"):
            if any(type(x) is not int for x in value):
                raise ValueError(f"{name} requires integer bounds")
    options = {
        "schemes": {"monochromatic", "analogous", "complementary", "neutral_accent"},
        "modes": {"light", "dark"},
        "heading_fonts": {"Solstice Serif", "Solstice Sans"},
        "heading_weights": {"regular", "bold"},
        "body_fonts": {"Solstice Serif", "Solstice Sans"},
        "body_alignments": {"left", "right"},
    }
    for name, choices in options.items():
        value = profile[name]
        if not isinstance(value, list) or not value or any(not isinstance(x, str) or x not in choices for x in value):
            raise ValueError(f"Invalid choices for {name}")
    for name in ("mirror_probability", "logo_probability"):
        if type(profile[name]) not in (int, float) or not 0 <= profile[name] <= 1:
            raise ValueError(f"{name} must be a probability")
    return copy.deepcopy(profile)


def luminance(color: str) -> float:
    channels = [int(color[i:i+2], 16)/255 for i in (1, 3, 5)]
    linear = [c/12.92 if c <= .04045 else ((c+.055)/1.055)**2.4 for c in channels]
    return sum(a*b for a, b in zip(linear, (.2126, .7152, .0722)))


def contrast(a: str, b: str) -> float:
    light, dark = sorted((luminance(a), luminance(b)), reverse=True)
    return (light+.05)/(dark+.05)


def hls(hue, lightness, saturation):
    return "#"+"".join(f"{round(c*255):02X}" for c in colorsys.hls_to_rgb(hue % 1, lightness, saturation))


def sample_palette(rng: random.Random, scheme: str, mode: str, profile: dict) -> tuple[str, ...]:
    # Sample complete palettes and check every pairing used by the templates.
    # Bounded rejection avoids silently substituting one of the old fixed palettes.
    for _ in range(512):
        hue = rng.uniform(*profile["hue"])
        saturation = rng.uniform(*profile["background_saturation" if scheme != "neutral_accent" else "neutral_saturation"])
        accent_hue = hue + {"monochromatic": 0, "analogous": rng.choice((-.09, .09)),
                            "complementary": .5, "neutral_accent": rng.random()}[scheme]
        dark = mode == "dark"
        bg_l = rng.uniform(*profile["dark_background_lightness" if dark else "light_background_lightness"])
        bg = hls(hue, bg_l, saturation)
        panel_l = max(0, min(1, bg_l+rng.uniform(*profile["panel_lightness_offset"])*(1 if dark else -1)))
        panel = hls(hue, panel_l, saturation)
        ink = hls(hue, rng.uniform(*profile["light_ink_lightness" if dark else "dark_ink_lightness"]), saturation)
        accent = hls(accent_hue, rng.uniform(*profile["light_accent_lightness" if dark else "dark_accent_lightness"]),
                     rng.uniform(*profile["accent_saturation"]))
        if min(contrast(ink, bg), contrast(ink, panel), contrast(accent, bg), contrast(accent, panel)) >= MIN_CONTRAST:
            return bg, ink, accent, panel
    raise ValueError("Could not sample a contrasting palette")


def sample_parameters(family: str, seed: int, variant: int, family_index: int, profile: dict) -> tuple[dict, random.Random]:
    rng = random.Random(f"{family}:{seed}:{variant}")
    def real(name):
        return round(rng.uniform(*profile[name]), 6)
    def integer(name):
        return rng.randint(*profile[name])
    orientation = ("portrait", "square", "landscape")[(family_index+variant) % 3]
    ratio = 1 if orientation == "square" else real(orientation+"_ratio")
    short = integer("short_edge")
    canvas = (short, round(short/ratio)) if ratio <= 1 else (round(short*ratio), short)
    scheme, mode = rng.choice(profile["schemes"]), rng.choice(profile["modes"])
    palette = sample_palette(rng, scheme, mode, profile)
    jitter = real("axis_jitter")
    params = dict(canvas=canvas, palette=palette, palette_scheme=scheme, palette_mode=mode,
                  margin=real("margin"), font_scale=real("font_scale"), heading_scale=real("heading_scale"),
                  line_height=real("line_height"), image_scale=real("image_scale"), corner_radius=real("corner_radius"),
                  heading_font=rng.choice(profile["heading_fonts"]), body_font=rng.choice(profile["body_fonts"]),
                  heading_weight=rng.choice(profile["heading_weights"]),
                  body_alignment=rng.choice(profile["body_alignments"]),
                  mirrored=rng.random() < profile["mirror_probability"],
                  show_logo=rng.random() < profile["logo_probability"],
                  x_knots=[0]+[v+rng.uniform(-jitter, jitter) for v in (.25, .5, .75)]+[1],
                  y_knots=[0]+[v+rng.uniform(-jitter, jitter) for v in (.25, .5, .75)]+[1])
    for name in ("card_count", "schedule_count", "feature_count", "stat_count", "perforations", "discount"):
        params[name] = integer(name)
    return params, rng


def warp(value: float, knots: list[float]) -> float:
    """Shared monotone axis map: preserves ordering, overlap and containment."""
    segment = min(3, max(0, int(value*4)))
    fraction = value*4-segment
    return knots[segment]+fraction*(knots[segment+1]-knots[segment])


def text_contrasts(source: dict) -> list[dict]:
    """Check declared opaque backgrounds only. Unknown/image-backed regions stay explicit."""
    import re
    def hex_color(element, key):
        match = re.fullmatch(r"rgba\((\d+),(\d+),(\d+),1\)", element.get(key, ""))
        if match is None or element.get("opacity", 1) != 1:
            return None
        return "#"+"".join(f"{int(c):02X}" for c in match.groups())
    def bounds(e):
        return e["x"], e["y"], e["x"]+e["width"], e["y"]+e["height"]
    layers = sorted(source["elements"], key=lambda e: e.get("z_index", 0))
    checks = []
    for i, element in enumerate(layers):
        if element["type"] != "text":
            continue
        x, y, right, bottom = bounds(element)
        foreground = hex_color(element, "color")
        check = dict(element_id=element["id"], ratio=None, background_id=None)
        for under in reversed(layers[:i]):
            ux, uy, ur, ub = bounds(under)
            if min(right, ur) <= max(x, ux) or min(bottom, ub) <= max(y, uy):
                continue
            background = hex_color(under, "fill_color")
            radius = under.get("corner_radius", 0)
            inside_rounding = all(
                math.hypot(px-max(ux+radius, min(px, ur-radius)), py-max(uy+radius, min(py, ub-radius))) <= radius
                for px, py in ((x, y), (right, y), (x, bottom), (right, bottom)))
            if (under.get("shape_type") == "rectangle" and ux <= x and uy <= y and ur >= right and ub >= bottom
                    and foreground and background and inside_rounding):
                check.update(ratio=contrast(foreground, background), background_id=under["id"])
            # Partial overlap, images, curved/translucent backgrounds need visual review.
            break
        checks.append(check)
    return checks
