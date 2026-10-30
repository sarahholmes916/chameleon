"""Build two portable, balanced old/S04 annotation assignments (no generated labels)."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import shutil

from annotation import candidates
from scripts.source_dataset import canonical

ROOT = Path(__file__).resolve().parents[1]


def read_pools(root):
    pools = defaultdict(dict)
    for path in sorted(root.rglob(candidates.MANIFEST_NAME)):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            target = row["target_canvas"]
            key = (row["source_design_id"], target["width"], target["height"])
            pools[key][row["candidate_id"]] = row
    return pools


def select_tasks(old_root, new_root, *, seed=42):
    """Retain every existing candidate and target, keeping generator pools separate."""
    tasks = []
    seen = set()
    for cohort, root in (("old", old_root), ("s04", new_root)):
        for key, pool in sorted(read_pools(root).items()):
            ids = sorted(pool)
            if seen.intersection(ids):
                raise ValueError("candidate IDs overlap between source/target pools")
            seen.update(ids)
            if not 2 <= len(ids) <= 16:
                raise ValueError(f"unsupported full pool size for {key}: {len(ids)}")
            rows = [pool[cid] for cid in ids]
            tasks.append(dict(task_id=hashlib.sha256(canonical([cohort, key])).hexdigest()[:16],
                              cohort=cohort, source_design_id=key[0],
                              target_canvas=dict(width=key[1], height=key[2]),
                              template_family_id=rows[0]["template_family_id"], candidate_ids=ids,
                              bracket_seed=seed, rows=rows))
    if not tasks:
        raise ValueError("no candidate tasks")
    return tasks


def _subset(tasks, total):
    reachable = {0: []}
    for i, task in enumerate(tasks):
        n = len(task["candidate_ids"])
        for count, indices in list(reachable.items()):
            if count + n <= total and count + n not in reachable:
                reachable[count + n] = indices + [i]
    return reachable.get(total)


def split_tasks(tasks, seed=42):
    """Split each cohort exactly in half, preserving whole sets wherever possible.

    If necessary split one set into two viable brackets; never duplicate or drop candidates.
    Odd cohort totals differ by one candidate, without dropping the remainder.
    """
    rng = random.Random(seed)
    assignments = [[], []]
    for cohort in sorted({t["cohort"] for t in tasks}):
        ordered = sorted((t for t in tasks if t["cohort"] == cohort), key=lambda t: t["task_id"])
        rng.shuffle(ordered)
        quota = sum(len(t["candidate_ids"]) for t in ordered) // 2
        chosen = _subset(ordered, quota)
        split = None
        if chosen is None:
            for i, task in enumerate(ordered):
                remainder = ordered[:i] + ordered[i + 1:]
                for n in range(2, len(task["candidate_ids"]) - 1):
                    chosen = _subset(remainder, quota - n)
                    if chosen is not None:
                        split = (task, n)
                        ordered = remainder
                        break
                if split:
                    break
        if chosen is None:
            raise ValueError(f"cannot halve {cohort} without singleton brackets")
        for i, task in enumerate(ordered):
            assignments[0 if i in chosen else 1].append(task)
        if split:
            task, n = split
            for who, ids in enumerate((task["candidate_ids"][:n], task["candidate_ids"][n:])):
                assignments[who].append({**task, "task_id": task["task_id"] + f"_part{who + 1}",
                                         "candidate_ids": ids,
                                         "rows": [r for r in task["rows"] if r["candidate_id"] in ids]})
    return assignments, set()


LAUNCHER = '''"""Run your portable assignment or export its saved judgments."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "_app"))
from annotation.assignment import prepare_assignment
if "--export" in sys.argv:
    from annotation import export, store
    db, annotator = prepare_assignment(ROOT)
    conn = store.connect(db)
    try:
        print(export.export_comparisons(conn, ROOT / "exports", stem=annotator + "_comparisons"))
    finally:
        conn.close()
else:
    from annotation.app import main
    raise SystemExit(main(["--assignment", str(ROOT)] + sys.argv[1:]))
'''


def write_package(folder, tasks, shared_ids, annotator, *, include_app=True):
    folder.mkdir(parents=True, exist_ok=False)
    records, files = [], {}
    for task in sorted(tasks, key=lambda t: t["task_id"]):
        for original in task["rows"]:
            row = json.loads(json.dumps(original))
            problems = candidates.check_record(row)
            if problems:
                raise ValueError(f"cannot package invalid candidate: {problems}")
            for field in ("source_path", "source_preview_path", "candidate_path", "preview_path"):
                if not row.get(field):
                    raise ValueError(f"missing {field}; original previews are required")
                source = candidates.resolve(row[field])
                digest = hashlib.sha256(source.read_bytes()).hexdigest()
                relative = f"files/{digest}{source.suffix.lower()}"
                dest = folder / relative
                dest.parent.mkdir(exist_ok=True)
                if not dest.exists():
                    shutil.copyfile(source, dest)
                files[relative] = digest
                row[field] = relative
            # Keep provenance, but remove old paths/scores from anything shown to annotators.
            row["provenance"] = {"original_set_id": original["set_id"],
                                 "original_provenance": original.get("provenance", {})}
            records.append(row)
    manifest = folder / candidates.MANIFEST_NAME
    manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    files[manifest.name] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    clean_tasks = [{**{k: v for k, v in t.items() if k != "rows"}, "shared": t["task_id"] in shared_ids} for t in tasks]
    config = dict(version="ranking_assignment_v1", annotator_id=annotator,
                  assignment_id=hashlib.sha256(canonical([annotator, clean_tasks, files])).hexdigest(),
                  tasks=clean_tasks, files=files)
    (folder / "assignment.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    if include_app:
        for directory in ("annotation", "engine", "scripts"):
            for src in (ROOT / directory).iterdir():
                if src.suffix in (".py", ".json"):
                    dest = folder / "_app" / directory / src.name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, dest)
        shutil.copyfile(ROOT / "chameleon_design_schema_v0.9.json", folder / "_app/chameleon_design_schema_v0.9.json")
        shutil.copyfile(ROOT / "requirements-annotation.txt", folder / "requirements.txt")
        (folder / "run.py").write_text(LAUNCHER, encoding="utf-8")
    (folder / "START_HERE.md").write_text(f'''# {annotator}'s ranking assignment

This is a standalone snapshot: no Git checkout, renderer, GPU, or asset regeneration is required.
Install Python 3.11 (recommended; code supports 3.10+), extract the ZIP completely, then open
PowerShell **inside this folder**, where run.py is located. First run:

```powershell
python -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.\\.venv\\Scripts\\python.exe run.py
```

Later runs only need the last command. Open http://127.0.0.1:7860 if the browser does not open.
If that port is busy, use `run.py --port 7861`. Keep the terminal open while labeling.

You have {len(tasks)} assigned sets with {len(records)} candidates across all existing targets.
Expect {sum(len(t["candidate_ids"]) - 1 for t in tasks)} bracket comparisons if you finish every set.
Each person gets half of the old pool and half of the latest full S04 pool.
Six-candidate sets use byes, with no fabricated judgments or omitted candidates.
One S04 set is divided into two three-candidate brackets to make the halves exactly equal.
The queue opens automatically, remembers progress, and moves to the next set. Do not pick/import
sources manually. Your annotator ID is `{annotator}` and your separate database is rankings.db here.
Original source previews appear beside both candidates; no separate source approval is required.

Choose A/B, tie, both bad, or skip honestly. Ties/both-bad/skips still require selecting who advances
in the bracket; this advancement is not exported as a preference winner. Reject a broken original
with 'Reject this source for all targets'. That excludes its labels from training.
Do not discuss individual judgments with the other annotator until both finish.
No candidates are shared between the two assignments.
No human labels are pre-filled; seeds and generator scores are not human judgments.

To send your results back, stop the UI with Ctrl+C, then run:

```powershell
.\\.venv\\Scripts\\python.exe run.py --export
```

Send the `exports` folder back (especially {annotator}_comparisons.jsonl). Keep rankings.db as your
backup. Exported IDs match the original immutable candidates; local filesystem paths need not match
the other computer. Do not send your .venv folder. These are annotation assignments, NOT train/test
splits: related source families must still remain together during later model evaluation.
''', encoding="utf-8")
    return dict(annotator=annotator, tasks=len(tasks), candidates=len(records),
                candidates_by_cohort=dict(Counter(t["cohort"] for t in tasks for _ in t["candidate_ids"])),
                bracket_comparisons=sum(len(t["candidate_ids"]) - 1 for t in tasks))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, default=Path("data/candidates/batch_pilot20"))
    parser.add_argument("--s04", type=Path, default=Path("data/evaluation/s04_refinement_v3"))
    parser.add_argument("--output", type=Path, default=Path("data/rankings"))
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; choose a fresh directory")
    tasks = select_tasks(args.old, args.s04)
    assignments, shared = split_tasks(tasks)
    summaries = []
    for annotator, assigned in zip(("sarah", "teammate"), assignments):
        folder = args.output / annotator
        summaries.append(write_package(folder, assigned, shared, annotator))
        shutil.make_archive(str(folder), "zip", root_dir=args.output, base_dir=annotator)
    (args.output / "split_summary.json").write_text(json.dumps(dict(assignments=summaries, shared_task_ids=sorted(shared)), indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
