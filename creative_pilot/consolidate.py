"""Merge per-task manifests and independently re-verify every candidate with the annotation importer's checks.

`annotation.candidates.check_record` re-hashes source and candidate JSON, re-runs the production
`validate_candidate` against the real source, checks root keys and the preview PNG size. Candidates that fail
are listed and excluded from the consolidated manifest (they never reach preference data).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from annotation.candidates import REPO_ROOT, check_record
from scripts import source_dataset as sd


def independent_text_fit(row: dict, renderer) -> int:
    """Count text elements that do not fit, measured with a FRESH renderer call (no cache, no generator state)."""
    cand = json.loads((REPO_ROOT / row["candidate_path"] if not Path(row["candidate_path"]).is_absolute()
                       else Path(row["candidate_path"])).read_text(encoding="utf-8"))
    return sum(not renderer.text_measurement(e)["fits"] for e in cand["elements"] if e["type"] == "text" and e["content"].strip())


def consolidate(pilot_root: Path) -> dict:
    renderer = sd.SourceRenderer()
    tasks = [json.loads(l) for l in (pilot_root / "tasks.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rows, failures, per_task, text_fit_fail = [], [], Counter(), Counter()
    s04_unfit = 0
    for t in tasks:
        if "error" in t:
            failures.append(dict(task_id=t["task_id"], problems=[t["error"]]))
            continue
        manifest = Path(t["manifest"])
        if not manifest.is_absolute():
            manifest = REPO_ROOT / manifest
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            problems = check_record(r)
            if problems:
                failures.append(dict(task_id=t["task_id"], candidate_id=r["candidate_id"], problems=problems))
                continue
            fit_failures = independent_text_fit(r, renderer)
            # S04 baselines are kept as produced (an unfittable 1 px floor is a real S04 outcome, recorded in diagnostics);
            # anything this pilot generated must fit, so a failure there is a generator bug and is reported, not hidden.
            if fit_failures and r["generator_id"] == "s04":
                s04_unfit += 1
            elif fit_failures:
                failures.append(dict(task_id=t["task_id"], candidate_id=r["candidate_id"], problems=[f"{fit_failures} text elements do not fit (independent renderer measurement)"],
                                     intended=r["provenance"]["intended_category"]))
                text_fit_fail[r["provenance"]["intended_category"]] += 1
                continue
            rows.append(r)
            per_task[t["task_id"]] += 1
    ids = [r["candidate_id"] for r in rows]
    dup = [k for k, n in Counter(ids).items() if n > 1]
    sd.write_jsonl(pilot_root / "candidate_manifest_all.jsonl", rows)
    report = dict(tasks=len(tasks), candidates_verified=len(rows), verification_failures=len(failures), text_fit_failures_by_intended=dict(text_fit_fail), s04_candidates_with_unfittable_text=s04_unfit,
                  duplicate_candidate_ids=len(dup), failures=failures[:50],
                  by_intended=dict(Counter(r["provenance"]["intended_category"] for r in rows)),
                  by_generator=dict(Counter(r["generator_id"] for r in rows)),
                  by_target=dict(Counter(f"{r['target_canvas']['width']}x{r['target_canvas']['height']}" for r in rows)),
                  tasks_with_fewer_than_8=sorted(t for t, n in per_task.items() if n < 8),
                  checks="annotation.candidates.check_record: sha256 re-hash, validate_candidate vs real source, root keys, PNG size")
    sd.write_json(pilot_root / "validation_report.json", report)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-root", type=Path, required=True)
    a = ap.parse_args(argv)
    report = consolidate(a.pilot_root)
    print(json.dumps(report, indent=1)[:2500])
    return 1 if report["verification_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
