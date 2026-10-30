"""Command line for the S05 tournament tooling.

  make-candidates  generate + render a dev candidate set          (needs requirements-sources.txt env)
  import           validate and import a candidate_manifest.jsonl (any generator, including S04)
  create           start a tournament from imported candidates
  export           write comparisons.jsonl/.csv and the candidate manifest
  status           print every tournament's progress
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from annotation import candidates, export, source_review, store
from annotation.candidates import REPO_ROOT


def _target(text: str) -> dict:
    width, _, height = text.lower().partition("x")
    return {"width": int(width), "height": int(height)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=store.DEFAULT_DB)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("make-candidates")
    make.add_argument("--source", type=Path, required=True)
    make.add_argument("--target", required=True, help="WIDTHxHEIGHT, e.g. 1600x400")
    make.add_argument("--seed", type=int, default=42)
    make.add_argument("--count", type=int, default=8)
    make.add_argument("--poor", type=int, default=2, help="how many deliberately weak (still valid) layouts")
    make.add_argument("--output", type=Path)
    imp = sub.add_parser("import")
    imp.add_argument("--manifest", type=Path, required=True)
    create = sub.add_parser("create")
    create.add_argument("--manifest", type=Path, required=True, help="imported manifest whose candidates form the pool")
    create.add_argument("--size", type=int, default=8, choices=(4, 8, 16))
    create.add_argument("--seed", type=int, default=0)
    create.add_argument("--annotator", default="local_annotator")
    create.add_argument("--allow-unapproved-source", action="store_true",
                        help="development override; the source's real review status is still recorded")
    exp = sub.add_parser("export")
    exp.add_argument("--tournament")
    exp.add_argument("--out", type=Path, default=Path("data/exports"))
    exp.add_argument("--include-voided", action="store_true")
    sub.add_parser("status")
    args = parser.parse_args(argv)
    db = args.db if args.db.is_absolute() else REPO_ROOT / args.db

    if args.command == "make-candidates":
        from annotation.dev_candidates import build_candidate_set
        from scripts.source_dataset import SourceRenderer
        result = build_candidate_set(args.source, _target(args.target), args.output, renderer=SourceRenderer(),
                                     seed=args.seed, count=args.count, poor=args.poor)
        print(json.dumps(dict(manifest=str(result.manifest), accepted=result.accepted, attempts=result.attempts,
                              rejected_attempts=result.rejected, validator_diagnostics=result.diagnostics), indent=2))
        return 0
    conn = store.connect(db)
    if args.command == "import":
        result = candidates.import_manifest(conn, candidates.resolve(args.manifest))
        print(f"imported {len(result['imported'])}, rejected {len(result['rejected'])}")
        for r in result["rejected"]:
            print(" rejected", r)
        return int(bool(result["rejected"]))
    if args.command == "create":
        rows = [json.loads(line) for line in candidates.resolve(args.manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
        status = source_review.review_status(candidates.resolve(rows[0]["source_path"]), rows[0]["source_design_id"])
        if status != "approved" and not args.allow_unapproved_source:
            print(f"Source review status is {status!r}; approve it first or pass --allow-unapproved-source.")
            return 2
        tid = store.create_tournament(conn, [r["candidate_id"] for r in rows], size=args.size, seed=args.seed,
                                      annotator_id=args.annotator, source_review_status=status)
        print(tid)
        return 0
    if args.command == "export":
        print(json.dumps(export.export_comparisons(conn, args.out, args.tournament, include_voided=args.include_voided), indent=2))
        return 0
    for t in store.list_tournaments(conn):
        state = store.load_state(conn, t["tournament_id"])
        done, total = store.progress(state)
        print(t["tournament_id"], f"{done}/{total}", "complete" if state.complete else "in progress")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
