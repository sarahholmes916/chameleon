"""S04-only conservative word fitting; shared graph_v4 estimator is unchanged."""
import copy
import math

TYPOGRAPHY_VERSION = "s04_word_fit_v2"
# Height of the first line in ems. v1 used 1.3; the official renderer measures 1.36-1.41 em
# (ascent + descent of the font at the truncated integer size), which made 1-2px height misses
# (e.g. 76px needed, 75px box). Calibrated against renderer measurements in
# tests/test_s04_rendering.py; still an estimate, never a glyph-exact guarantee.
SINGLE_LINE_EM = 1.4
# Safety factor on summed glyph advances (v1: none); the renderer's widths exceeded the estimate
# by up to ~3% on bold/mixed text.
ADVANCE_SAFETY = 1.03


def estimated_fit(element):
    size = element["font_size"]
    width, height = element["width"] - 2, element["height"] - 2
    spacing = element.get("letter_spacing", 0)
    weight = 1.05 if element.get("font_weight") == "bold" else 1.0
    # Provisional widths, not measured glyph metrics or a graph feature change.
    def advance(word):
        return ADVANCE_SAFETY * sum((0.8 if c in "MW@" else 0.3 if c in "il.,'! " else 0.6) * size * weight
                                    + spacing for c in word)
    lines, widest = 0, 0.0
    for paragraph in element["content"].split("\n"):
        lines += 1
        current = 0.0
        for word in paragraph.split(" "):
            word_width = advance(word)
            addition = word_width + (advance(" ") if current else 0)
            if current and current + addition > width:
                lines += 1
                current = word_width
            else:
                current += addition
            widest = max(widest, current)
    required = size * (SINGLE_LINE_EM + max(0, lines - 1) * element.get("line_height", 1.2))
    max_lines = element.get("max_lines")
    return dict(fits=widest <= width and required <= height and (max_lines is None or lines <= max_lines),
                lines=lines, measured_width=widest, measured_height=required)


def fit_candidate_text(raw, measure=None):
    """Shrink only font_size, respecting hierarchy; injected measurement may use renderer fonts."""
    result = copy.deepcopy(raw)
    measure = measure or estimated_fit
    texts = sorted((e for e in result["elements"] if e["type"] == "text"), key=lambda e: -e["font_size"])
    cap, previous = math.inf, None
    group_cap = cap
    for element in texts:
        initial = element["font_size"]
        if initial != previous:
            group_cap = cap
        previous = initial
        element["font_size"] = max(1.0, min(initial, group_cap))
        for _ in range(200):
            measured = measure(element)
            limit = element.get("max_lines")
            if (measured["fits"] and (limit is None or measured["lines"] <= limit)) or element["font_size"] <= 1:
                break
            element["font_size"] = max(1.0, element["font_size"] * .95)
        cap = min(cap, element["font_size"])
    return result
