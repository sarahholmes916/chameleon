"""Render every Step 4 candidate for the three fixtures (dev preview only).

Run from the repo root:  python scripts/preview_candidates.py

Writes output/preview/<difficulty>_<family>[_<mode>].png via
engine.main.render_to_png (real renderer when importable, labeled-box
fallback otherwise) and prints validity plus a per-element geometry/font
table per candidate. output/ is gitignored; leave this script uncommitted
unless its owner says otherwise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.candidate_generator import (  # noqa: E402
    banner_zones_candidates,
    baseline_candidate,
    hero_left_candidates,
    hero_right_candidates,
    horizontal_flow_candidates,
    vertical_stack_candidates,
)
from engine.candidate_validator import validate_candidate  # noqa: E402
from engine.internal_schema import Canvas, Design  # noqa: E402
from engine.layout_objective import score_deterministic  # noqa: E402
from engine.layout_units import required_group_modes  # noqa: E402
from engine.main import ASSETS_DIR, INPUT_DIR, render_to_png  # noqa: E402

PREVIEW_DIR = Path("output/preview")


def describe(candidate: dict) -> list[str]:
    lines = []
    for element in candidate["elements"]:
        extra = ""
        if element["type"] == "text":
            extra = f" fs={element['font_size']:.2f}"
        lines.append(
            f"    {element['id']}: ({element['x']:.1f},{element['y']:.1f},"
            f"{element['width']:.1f},{element['height']:.1f}){extra}"
        )
    return lines


def main() -> int:
    try:
        import renderer.converter.convert  # noqa: F401
    except ImportError:
        print("renderer: labeled-box fallback (renderer.converter.convert missing)")
    else:
        print("renderer: real (renderer.converter.convert)")
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    for task_file in sorted(INPUT_DIR.glob("*.json")):
        task = json.loads(task_file.read_text(encoding="utf-8"))
        source = Design.from_adobe(task["source"])
        target = Canvas.from_adobe(task["target_canvas"])
        modes = required_group_modes(source)
        jobs = [
            ("baseline", [("single", baseline_candidate(source, target))]),
            ("flow", list(zip([m for m in modes],
                              horizontal_flow_candidates(source, target)))),
            ("stack", list(zip([m for m in modes],
                               vertical_stack_candidates(source, target)))),
        ]
        for multi_name, multi_fn in (("hero_left", hero_left_candidates),
                                        ("hero_right", hero_right_candidates),
                                        ("zones", banner_zones_candidates)):
            multi_cands = multi_fn(source, target)
            per_mode = len(multi_cands) // len(modes)
            labels = []
            for mode in modes:
                # orders outer, small-variants inner; index keeps labels unique.
                labels += [f"{mode}_{index}" for index in range(per_mode)]
            jobs.append((multi_name, list(zip(labels, multi_cands))))
        for family, candidates in jobs:
            for mode, candidate in candidates:
                name = f"{task['difficulty']}_{family}"
                if len(candidates) > 1:
                    name += f"_{mode}"
                png_path = PREVIEW_DIR / f"{name}.png"
                render_to_png(candidate, ASSETS_DIR, png_path)
                report = validate_candidate(source, candidate, target)
                status = "valid" if report.is_valid else (
                    f"INVALID {report.violations}")
                if report.is_valid:
                    scored = score_deterministic(
                        source, [Design.from_adobe(candidate)], target)[0]
                    status += f" score={scored:.4f}"
                print(f"{png_path.as_posix()}: {status}")
                print("\n".join(describe(candidate)))
                count += 1
    print(f"wrote {count} previews to {PREVIEW_DIR.as_posix()}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
