"""Blind contact sheets for AI visual assessment of rendered candidates.

The assessor (an image-capable model reading the PNG sheet) sees only the source and numbered, shuffled candidates:
no generator, recipe, intended category or score. Tasks are chosen by a fixed stride over the sorted task list, so
the choice cannot depend on candidate quality. Assessments are stored in `visual_assessments.json` and used as the
`ai_visual_assessment` evaluation method; they are model judgments, never human labels.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from annotation.candidates import REPO_ROOT
from creative_pilot.sheets import contact_sheet

TIERS = {1: "strong", 2: "acceptable", 3: "weak", 4: "poor or unusable"}


def select_tasks(tasks: list[dict], stride: int = 13) -> list[dict]:
    ok = sorted((t for t in tasks if "error" not in t), key=lambda t: t["task_id"])
    return ok[::stride]


def prepare(pilot_root: Path, stride: int = 13, seed: int = 7) -> dict:
    tasks = [json.loads(l) for l in (pilot_root / "tasks.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    out = pilot_root / "visual_review"
    (out / "sheets").mkdir(parents=True, exist_ok=True)
    manifest = {}
    for line in (pilot_root / "candidate_manifest_all.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        manifest.setdefault(r["set_id"], []).append(r)
    key = {}
    for t in select_tasks(tasks, stride):
        rows = sorted(manifest[t["task_id"]], key=lambda r: r["candidate_id"])
        random.Random(f"{seed}:{t['task_id']}").shuffle(rows)
        w, h = rows[0]["target_canvas"]["width"], rows[0]["target_canvas"]["height"]
        if w / h > 3 and w > 1000:
            cell, cols = (540, 140), 3
        elif w > h:
            cell, cols = (540, 74), 3
        else:
            cell, cols = (96, 400), 12
        src = rows[0].get("source_preview_path")
        items = []
        if src:
            sp = Path(src) if Path(src).is_absolute() else REPO_ROOT / src
            items.append((sp, "SOURCE"))
        for i, r in enumerate(rows, 1):
            p = Path(r["preview_path"])
            items.append((p if p.is_absolute() else REPO_ROOT / p, f"#{i}"))
        contact_sheet(items, out / "sheets" / f"{t['task_id']}.png", cols=cols, cell=cell)
        key[t["task_id"]] = dict(target=f"{w}x{h}", numbering={str(i): r["candidate_id"] for i, r in enumerate(rows, 1)})
    (out / "key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    return dict(tasks=len(key), sheets=str(out / "sheets"), stride=stride)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-root", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=13)
    a = ap.parse_args(argv)
    print(json.dumps(prepare(a.pilot_root, a.stride)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
