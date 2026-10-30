# S05 tournament annotation (development tool)

Local side-by-side tournament UI for collecting human layout preferences (Appendix E.1). It labels
**resized candidates of one source for one target**; it never edits designs, shows model scores, or trains
anything. No human labels have been collected yet.

**Framework decision.** [reference/E_annotation_training.md](reference/E_annotation_training.md) E.1 (part of the authoritative
spec) specifies a local **Gradio Blocks** app backed by SQLite. [CHAMELEON_DATA_GENERATION_GUIDE.md](CHAMELEON_DATA_GENERATION_GUIDE.md)
§7–8 says Streamlit. The spec takes precedence ([TECHNICAL_SPEC.md](TECHNICAL_SPEC.md) §0.2), so this tool uses Gradio
(`gradio>=5,<6`; Gradio 6 deprecations are already warned about in tests and will need a small update).

## Install and run

### Automatic annotation queue

Launch the UI and click **Start / resume all layouts** in the Tournament tab.
It recursively discovers and validates manifests under `data/candidates`, groups candidates
by exact source ID and target dimensions, and processes sources and their targets consecutively.
No manual import, source/target matching, or source approval is required for this workflow.
The source's actual review status remains recorded; annotation never auto-approves source designs.
Completed sets are not repeated. Unfinished tournaments resume from SQLite, including after restart.
The queue uses the largest supported bracket (4, 8, or 16) that fits the pool; larger pools
are sampled deterministically and pools below four are excluded. A completed set is not reopened
automatically when more candidates are later added; use the manual tournament controls for that.

Finish a bracket to advance automatically. Ties, both-bad and skipped comparisons remain
non-winning labels; existing manual advancement controls resolve their bracket positions.
The original source preview is always visible beside both candidates, with the same zoom control.
For a broken original, select
**Reject this source for all targets**. This persists a queue rejection, skips all its remaining
targets, and exports previous judgments with `source_rejected_in_queue=true` and
`usable_for_pairwise_loss=false`, retaining their original labels for audit.
Source review remains available as an optional separate workflow. Import failures are reported
on starting the queue; structural validation is still mandatory.

```powershell
python -m venv .venv/annotation
.\.venv\annotation\Scripts\python.exe -m pip install -r requirements-annotation.txt
# 1. Dev candidates (needs the renderer environment from requirements-sources.txt, e.g. .venv):
.\.venv\Scripts\python.exe -m annotation.cli make-candidates --source data/sources/pilot20_v3/sources/<file>.json --target 1600x400 --seed 42
# 2. Launch the UI (http://127.0.0.1:7860):
.\.venv\annotation\Scripts\python.exe -m annotation.app
```

In the UI: **New tournament** → import the printed `candidate_manifest.jsonl` → pick the candidate set → create.
Command-line equivalents: `annotation.cli import|create|export|status` (`--allow-unapproved-source` is the
development override). Data lives in `data/annotations.db` (override with `--db` or `CHAMELEON_ANNOTATION_DB`),
candidate sets in `data/candidates/<set_id>/`, exports in `data/exports/`. `/data/` is git-ignored.

## Behaviour

* **Bracket**: 4, 8 (default) or 16 entrants, seeded reproducibly from the seed and the candidate IDs; screen
  left/right is pseudo-random per match and stored. Winners advance immediately; the bracket, round/match
  label, progress and champion update after each decision. Zoom (Fit / actual pixels / 200%) applies to both
  sides; the original source is in an expandable panel.
* **Responses**: A wins, B wins, Tie, Both bad, Skip. Optional reason tags (no form required). Keyboard:
  ← A · → B · T · B · S · U (undo).
* **Non-decisive responses** are saved as the label and the match stays **unresolved**; nobody advances
  automatically. *Resolution mechanism:* the screen then offers "Advance A/B" (explicit, stored in a separate
  `resolutions` row, `method = manual_resolution`). The original label is never overwritten and the export marks
  `advancement_overrides_label`.
* **Undo** voids the newest decision (or any chosen match via *Redo this match*) together with every dependent
  later match. Nothing is deleted: voided rows stay in SQLite with a reason and are excluded from exports unless
  `--include-voided` / the checkbox is used.
* **Duplicates**: each presentation has a token (`match#revision`). Re-sent requests are idempotent no-ops, stale
  pages are refused, and a partial unique index allows one active judgment per match.
* **Extra comparisons** (History tab): never-compared pairs among eliminated candidates, and *repeats* of judged
  pairs with reversed screen order for annotator consistency. They never change the bracket.
* **Source review tab**: reuses `scripts/source_dataset.apply_review` (same `review.json` / `approved.jsonl` as the
  static gallery), keeps every other source's decision, never auto-approves. Sources, not candidates, are listed.
* **Guardrails**: tournaments require candidates of one source and target (also enforced in the database layer);
  sources that are not approved require the explicit override, and the status is recorded on the tournament.

## Candidate manifest contract (also how S04 plugs in)

Any generator writes `candidate_manifest.jsonl` (see `annotation/candidates.py`): `candidate_id`
(`sha256:` of canonical candidate JSON), `source_design_id`, `template_family_id`, `source_path`,
`source_preview_path`, `target_canvas`, `candidate_path`, `preview_path`, `generator_id`, `generator_version`,
`renderer_version`, `set_id`, `provenance`. `import` **re-validates every record** (source and candidate hashes,
`engine.candidate_validator.validate_candidate`, root keys, PNG size) and stores only valid ones. S04 therefore
needs no database or UI change: emit validated candidates plus previews in this format and import them.
Generator identity and `provenance` are never shown to annotators.

`annotation/dev_candidates.py` is a **diagnostic S05 generator, not S04**. It moves/resizes elements only
(through `layout_edit`), so immutable fields and image ratios hold; includes deliberately weak but valid layouts;
uses seeded, bounded attempts; and writes deliberately invalid cases to `diagnostics/` (shown in the *Validator
diagnostics* tab, never mixed into labels). Strategy names are provenance, **not quality labels**.

## Export (`preference.v1`)

`comparisons.jsonl` / `.csv` plus `*_candidates.jsonl` (paths, generator and renderer versions). One row per
active judgment: tournament/match IDs, round, position, source and template-family IDs, target size, candidate
A/B IDs (canonical order), screen-left/right IDs, `label`, `preferred_candidate_id`, `usable_for_pairwise_loss`
(A/B only; TIE is an optional weak signal per E.3; BOTH_BAD/SKIP carry no winner), reason tags, annotator,
timestamp, generator/renderer versions, `source_review_status`, and `split_group_id` (= template family, so all
siblings stay in one split). Advancement (`advancement_candidate_id`, `advancement_method`) is descriptive only,
**not** an extra preference label, and no score or "rounds survived" value is produced. Single elimination does
not rank all candidates; use match-level rows, and extra/repeat comparisons for more signal.

## Verification status

Automated: `tests/test_annotation.py` (run in the *Source generation / Renderer* CI job). Not verified:
hosted-CI results, rendering in a real browser (no screenshot tooling was available; wiring was exercised through
`gradio_client` against the running app), annotator-consistency statistics, and any visual quality of the demo
candidates beyond inspection of the contact sheets.
