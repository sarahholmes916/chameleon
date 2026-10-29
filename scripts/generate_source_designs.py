"""Generate, review and export original S05 source designs (never resize candidates)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.source_dataset import apply_review, export_handoff, generate_dataset, write_jsonl
from scripts.source_variation import load_profile


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate", help="Create a new dataset; existing outputs are protected")
    generate.add_argument("--output", type=Path, default=Path("data/sources/pilot20"))
    generate.add_argument("--count", type=int, default=20)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--profile", type=Path, help="Complete variation-profile JSON; defaults to scripts/source_variation.json")
    review = sub.add_parser("review", help="Apply a complete gallery review snapshot")
    review.add_argument("--dataset", type=Path, required=True)
    review.add_argument("--decisions", type=Path, required=True)
    handoff = sub.add_parser("handoff", help="Export approved sources with an explicit target")
    handoff.add_argument("--dataset", type=Path, required=True)
    handoff.add_argument("--target-width", type=int, required=True)
    handoff.add_argument("--target-height", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "generate":
        summary = generate_dataset(args.output, count=args.count, seed=args.seed,
                                   profile=load_profile(args.profile) if args.profile else None)
        print(json.dumps(summary, indent=2))
        print(f"Gallery: {(args.output / 'gallery.html').resolve()}")
        return int(summary["rendered"] != summary["requested"])
    if args.command == "review":
        count = apply_review(args.dataset, json.loads(args.decisions.read_text(encoding="utf-8")))
        print(f"Approved sources: {count}")
        return 0
    tasks = export_handoff(args.dataset, {"width": args.target_width, "height": args.target_height})
    path = args.dataset / f"handoff_{args.target_width}x{args.target_height}.jsonl"
    write_jsonl(path, tasks)
    print(f"Exported {len(tasks)} approved source tasks to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
