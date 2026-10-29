"""S05 source-only validation, rendering, provenance and human review workflow."""

from __future__ import annotations

import copy
import hashlib
import html
import importlib.metadata
import json
import platform
import re
from pathlib import Path

from engine.internal_schema import Canvas, Design, SchemaError
from scripts.source_templates import ASSETS, FAMILIES, GENERATOR_VERSION, make_source_with_metadata
from scripts.source_variation import MIN_CONTRAST, load_profile, text_contrasts, validate_profile


ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT / "brand_kit/assets"
FONTS_DIR = ROOT / "brand_kit/fonts"
SCHEMA_PATH = ROOT / "chameleon_design_schema_v0.9.json"
POLICY_VERSION = "source_quality_v2"


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, records: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text("".join(canonical(row).decode("utf-8")+"\n" for row in records), encoding="utf-8")
    temporary.replace(path)


def read_records(root: Path) -> list[dict]:
    return [json.loads(line) for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]


def validate_source(source: dict, assets_dir: Path = ASSETS_DIR,
                    fonts_dir: Path = FONTS_DIR) -> dict:
    """Source quality gate, explicitly separate from target candidate validation."""
    from jsonschema import Draft202012Validator

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    issues = []

    def issue(code, category, message):
        issues.append(dict(code=code, rule_class=category, message=message))

    errors = sorted(Draft202012Validator(schema).iter_errors(source), key=lambda e: str(list(e.path)))
    for error in errors:
        issue("SCHEMA", "official", f"{list(error.path)}: {error.message}")
    if errors:
        return dict(schema_ok=False, accepted=False, issues=issues, policy_version=POLICY_VERSION)
    try:
        design = Design.from_adobe(source)
    except SchemaError as exc:
        issue("IDENTITY_OR_JSON", "official", str(exc))
        return dict(schema_ok=True, accepted=False, issues=issues, policy_version=POLICY_VERSION)
    allowed = set(schema["$defs"]["element"]["properties"])
    for e in source["elements"]:
        eid = e["id"]
        if set(e)-allowed:
            issue("GENERATOR_FIELDS", "internal", f"{eid}: extensions are not generated")
        for field in ("color", "fill_color", "stroke_color"):
            if e.get(field) is not None:
                match = re.fullmatch(r"rgba\((\d+),(\d+),(\d+),(0|1|0\.\d+)\)", e[field])
                if not match or any(int(match[i]) > 255 for i in (1, 2, 3)):
                    issue("SOURCE_COLOR", "internal", f"{eid}: use bounded rgba() for renderer fidelity")
        if e.get("rotation", 0) != 0:
            issue("ROTATION_UNSUPPORTED", "internal", f"{eid}: this source gate supports unrotated templates")
        if e["width"] <= 0 or e["height"] <= 0:
            issue("POSITIVE_SOURCE_BOX", "internal", f"{eid}: source templates require positive boxes")
        if (e["x"] < 0 or e["y"] < 0 or e["x"]+e["width"] > design.canvas.width+.001
                or e["y"]+e["height"] > design.canvas.height+.001):
            issue("SOURCE_BOUNDS", "internal", f"{eid}: generated source box exceeds canvas")
        if e["type"] == "image":
            if e["asset_id"] not in ASSETS or not (assets_dir / e["asset_id"]).is_file():
                issue("ASSET_UNAVAILABLE", "internal", f"{eid}: asset outside bundled allowlist or missing")
            if e.get("fit", "contain") not in ("contain", "cover"):
                issue("SOURCE_IMAGE_FIT", "internal", f"{eid}: avoid stretched source imagery")
        if e["type"] == "text":
            family = {"Solstice Sans": "SolsticeSans", "Solstice Serif": "SolsticeSerif"}.get(e["font_family"])
            suffix = "Bold" if e.get("font_weight") == "bold" else "Regular"
            if e.get("font_style") == "italic":
                suffix = "BoldItalic" if suffix == "Bold" else "Italic"
            if not family or not (fonts_dir / f"{family}-{suffix}.ttf").is_file():
                issue("FONT_UNAVAILABLE", "internal", f"{eid}: exact bundled font face missing")
            if not e["content"].strip() or e["font_size"] < 12:
                issue("SOURCE_TEXT", "internal", f"{eid}: empty text or size below pilot 12px floor")
            if e.get("style_runs") or e.get("letter_spacing", 0) != 0 or e.get("max_lines") is not None:
                issue("TEXT_MEASUREMENT_SCOPE", "internal", f"{eid}: unsupported source measurement styling")
            if e.get("line_height", 1.2) <= 0:
                issue("SOURCE_LINE_HEIGHT", "internal", f"{eid}: nonpositive line height")
    for group, members in design.groups.items():
        if len(members) < 2:
            issue("SINGLETON_GROUP", "internal", f"{group}: generated groups need related members")
    checks = text_contrasts(source)
    for check in checks:
        if check["ratio"] is not None and check["ratio"] < MIN_CONTRAST:
            issue("SOURCE_CONTRAST", "internal", f"{check['element_id']}: declared text/background contrast below {MIN_CONTRAST}:1")
    return dict(schema_ok=True, accepted=not issues, issues=issues, policy_version=POLICY_VERSION,
                text_contrasts=checks)


def near_identical(a: dict, b: dict) -> bool:
    """Conservative layout dedupe, ignoring recolors/copy/ID changes (not a quality score)."""
    aw, ah = a["canvas"]["width"], a["canvas"]["height"]
    bw, bh = b["canvas"]["width"], b["canvas"]["height"]
    if abs((aw/ah)/(bw/bh)-1) > .02 or len(a["elements"]) != len(b["elements"]):
        return False
    group_maps = ({}, {})
    for x, y in zip(a["elements"], b["elements"]):
        if any(x.get(k) != y.get(k) for k in ("type", "asset_id", "fit", "shape_type", "text_align")):
            return False
        groups = []
        for e, mapping in zip((x, y), group_maps):
            group = e.get("group_id")
            groups.append(None if group is None else mapping.setdefault(group, len(mapping)))
        if groups[0] != groups[1]:
            return False
        for key, da, db in (("x", aw, bw), ("y", ah, bh), ("width", aw, bw), ("height", ah, bh)):
            if abs(x[key]/da-y[key]/db) > .015:
                return False
    return True


class SourceRenderer:
    """Use the actual converter/Skia fonts; never fall back to box placeholders."""

    def __init__(self):
        from renderer.converter.brand_fonts import load_brand_font_manager
        self.font_manager = load_brand_font_manager(str(FONTS_DIR))
        distribution = importlib.metadata.distribution("cr-renderer")
        direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
        self.version = dict(
            package=distribution.version,
            commit=direct_url.get("vcs_info", {}).get("commit_id", "unknown"),
            converter_sha256=digest({p.name: file_hash(p) for p in sorted((ROOT / "renderer/converter").glob("*.py"))}),
            skia=importlib.metadata.version("skia-python"),
            pillow=importlib.metadata.version("Pillow"),
            python=platform.python_version(),
        )

    def text_measurement(self, element: dict) -> dict:
        import skia
        from renderer.converter.wrap import char_widths, wrap_text

        family, size = element["font_family"], element["font_size"]
        weight = element.get("font_weight", "regular")
        style = "bold" if weight == "bold" else "regular"
        if element.get("font_style") == "italic":
            style = "bolditalic" if weight == "bold" else "italic"
        content = element["content"]
        widths = char_widths(self.font_manager, family, weight, style, size, content)
        wrapped = wrap_text(content, widths, element["width"])
        font = skia.Font(skia.Typeface.MakeFromData(self.font_manager.lookup(family, weight, style)), int(size))
        metrics = font.getMetrics()
        lines = wrapped.split("\n")
        max_width = max((sum(font.getWidths(font.textToGlyphs(line))) for line in lines), default=0)
        height = (len(lines)-1)*size*element.get("line_height", 1.2)+metrics.fDescent-metrics.fAscent
        return dict(element_id=element["id"], lines=len(lines), measured_width=max_width,
                    measured_height=height, fits=max_width <= element["width"]-2 and height <= element["height"]-2)

    def fit_text(self, source: dict) -> tuple[dict, list[dict]]:
        fitted = copy.deepcopy(source)
        measurements = []
        for element in fitted["elements"]:
            if element["type"] != "text":
                continue
            initial = element["font_size"]
            while True:
                measured = self.text_measurement(element)
                if measured["fits"] or element["font_size"] <= 12:
                    break
                element["font_size"] = max(12, round(element["font_size"]-1, 2))
            measured["initial_font_size"] = initial
            measured["final_font_size"] = element["font_size"]
            measurements.append(measured)
        return fitted, measurements

    def render(self, source: dict, path: Path) -> list[str]:
        from renderer.converter.convert import render_design_to_png
        return render_design_to_png(copy.deepcopy(source), ASSETS_DIR, path, FONTS_DIR,
                                    max_size=max(source["canvas"].values()))


def verify_png(path: Path, source: dict) -> str:
    from PIL import Image
    with Image.open(path) as image:
        image.load()
        if image.format != "PNG" or image.size != (source["canvas"]["width"], source["canvas"]["height"]):
            raise ValueError("Renderer did not produce a full-resolution PNG of the source canvas")
        return hashlib.sha256(canonical(image.size)+image.convert("RGBA").tobytes()).hexdigest()


def generate_dataset(output: Path, *, count: int = 20, seed: int = 42,
                     renderer=None, factory=None, profile: dict | None = None) -> dict:
    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive integer")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new/empty output directory; existing datasets are never overwritten")
    profile = load_profile() if profile is None else validate_profile(profile)
    renderer = renderer if renderer is not None else SourceRenderer()
    for folder in ("sources", "previews", "quarantine"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    run = dict(schema_version="source_dataset_v1", generator_version=GENERATOR_VERSION,
               generator_sha256=digest({p.name: file_hash(p) for p in (Path(__file__), ROOT / "scripts/source_templates.py",
                   ROOT / "scripts/source_variation.py", ROOT / "scripts/source_copy.json", ROOT / "scripts/source_variation.json")}),
               variation_profile=profile, variation_profile_sha256=digest(profile),
               seed=seed, count=count, schema_sha256=file_hash(SCHEMA_PATH),
               assets={p.name: file_hash(p) for p in sorted(ASSETS_DIR.glob("*.png"))},
               fonts={p.name: file_hash(p) for p in sorted(FONTS_DIR.glob("*.ttf"))},
               renderer_version=renderer.version, validation_policy_version=POLICY_VERSION,
               dedupe_policy="exact canonical JSON; normalized box deltas <=0.015 and aspect delta <=2%; identical RGBA renders",
               split_policy="unassigned; partition by template_family_id before creating resize candidates")
    write_json(output / "run.json", run)
    records, accepted, pixels = [], [], {}
    families = list(FAMILIES)
    for index in range(count):
        family, source_seed = families[index % len(families)], seed+index
        variant = index//len(families)
        if factory is None:
            source, parameters = make_source_with_metadata(family, source_seed, variant, profile=profile)
        else:
            source, parameters = factory(family, source_seed, variant), None
        report = validate_source(source)
        measurements, warnings, error = [], [], None
        if report["accepted"]:
            try:
                source, measurements = renderer.fit_text(source)
                report = validate_source(source)
                if any(not measurement["fits"] for measurement in measurements):
                    report["issues"].append(dict(code="TEXT_OVERFLOW", rule_class="internal", message="Text does not fit at the pilot font floor"))
                    report["accepted"] = False
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        source_id = "sha256:"+digest(source)
        stem = f"{index:04d}_{family}_{source_id[-12:]}"
        elements = source.get("elements", [])
        elements = elements if isinstance(elements, list) else []
        record = dict(source_design_id=source_id, template_family_id=family,
                      source_family_id=family, content_category=FAMILIES[family],
                      source_provenance="original_synthetic", seed=source_seed,
                      variant=variant, generation_parameters=parameters, variation_profile_sha256=digest(profile),
                      generator_version=GENERATOR_VERSION, canvas=source.get("canvas"),
                      element_count=len(elements), assets=sorted({e["asset_id"] for e in elements if isinstance(e, dict) and isinstance(e.get("asset_id"), str)}),
                      fonts=sorted({e["font_family"] for e in elements if isinstance(e, dict) and isinstance(e.get("font_family"), str)}),
                      validation=report, renderer_version=renderer.version,
                      text_measurements=measurements, review_status="ineligible", split="unassigned",
                      status="validation_failed", renderer_warnings=warnings, renderer_error=error)
        duplicate = (next((sid for sid, prior in accepted if source_id == sid or near_identical(source, prior)), None)
                     if report["accepted"] else None)
        preview = output / "quarantine" / f"{stem}.png"
        if error:
            record["status"] = "renderer_failed"
        elif report["accepted"] and duplicate:
            record.update(status="duplicate", duplicate_of=duplicate)
        elif report["accepted"]:
            try:
                warnings = renderer.render(source, preview)
                record["renderer_warnings"] = warnings
                if warnings:
                    raise ValueError("Renderer warnings require quarantine")
                pixel_hash = verify_png(preview, source)
                if pixel_hash in pixels:
                    record.update(status="duplicate", duplicate_of=pixels[pixel_hash])
                else:
                    record.update(status="rendered", review_status="pending", pixel_sha256=pixel_hash)
                    pixels[pixel_hash] = source_id
            except Exception as exc:
                record.update(status="renderer_failed", renderer_error=f"{type(exc).__name__}: {exc}")
        if record["status"] == "rendered":
            accepted.append((source_id, source))
            dest = output / "previews" / preview.name
            preview.replace(dest)
            preview = dest
            json_path = output / "sources" / f"{stem}.json"
        else:
            json_path = output / "quarantine" / f"{stem}.json"
        write_json(json_path, source)
        record["source_path"] = json_path.relative_to(output).as_posix()
        record["preview_path"] = preview.relative_to(output).as_posix() if preview.exists() else None
        record["preview_sha256"] = file_hash(preview) if preview.exists() else None
        records.append(record)
        write_jsonl(output / "manifest.jsonl", records)
    summary = {status: sum(r["status"] == status for r in records)
               for status in ("rendered", "validation_failed", "renderer_failed", "duplicate")}
    summary.update(requested=count, approved=0, pending_review=summary["rendered"])
    write_json(output / "summary.json", summary)
    write_jsonl(output / "approved.jsonl", [])
    write_gallery(output, records)
    return summary


def write_gallery(root: Path, records: list[dict], decisions: dict | None = None) -> None:
    manifest_hash = file_hash(root / "manifest.jsonl")
    decisions = decisions or {}
    cards = []
    for row in records:
        sid = row["source_design_id"]
        canvas = row["canvas"] if isinstance(row["canvas"], dict) else {}
        decision = decisions.get(sid, {"status": "pending", "reason": ""})
        options = "".join(f'<option value="{s}"'+(' selected' if decision["status"] == s else '')+f'>{s}</option>'
                          for s in ("pending", "approved", "rejected"))
        preview = (f'<a href="{html.escape(row["preview_path"])}" target="_blank"><img loading="lazy" src="{html.escape(row["preview_path"])}" alt="{html.escape(row["template_family_id"])}"></a>'
                   if row["preview_path"] else '<p>No rendered preview</p>')
        review = (f'<select>{options}</select><input placeholder="Review reason (required for rejection)" value="{html.escape(decision["reason"], quote=True)}">'
                  if row["status"] == "rendered" else '<p>Quarantined: cannot approve</p>')
        cards.append(f'<article data-id="{sid}"><h2>{html.escape(row["template_family_id"])}</h2>{preview}'
                     f'<p>{html.escape(str(canvas.get("width", "?")))} × {html.escape(str(canvas.get("height", "?")))} · {row["element_count"]} elements · {row["status"]}</p>'
                     f'<a href="{row["source_path"]}">Adobe JSON</a><p><code>{sid[-12:]}</code></p>{review}'
                     f'<details><summary>Validation / renderer details</summary><pre>{html.escape(json.dumps(row, indent=2))}</pre></details></article>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Chameleon source review</title>
<style>body{font:16px system-ui;margin:24px;background:#eeeeea;color:#222}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:24px}article{background:white;padding:18px}h2{font-size:18px}img{width:100%;height:340px;object-fit:contain;background:#ddd}select,input{display:block;margin:12px 0;padding:8px;max-width:95%}pre{white-space:pre-wrap;overflow-wrap:anywhere}button{padding:12px;margin-bottom:20px}</style>
<h1>Original source designs — visual review</h1><p>Schema-valid does not mean visually good. Open each PNG at full resolution; check clipping, contrast, hierarchy, grouping and unwanted overlaps. Pending and rejected sources are excluded from approved.jsonl.</p>
<p>Selections are saved in this browser. Download decisions, then apply them with the review CLI to update the approved dataset. No automatic aesthetic labels are created.</p>
<button onclick="download()">Download review decisions</button><main>'''+"".join(cards)+'''</main><script>
const manifestHash = "'''+manifest_hash+'''";
const key = 'chameleon-review-' + manifestHash;
const cards = [...document.querySelectorAll('article')].filter(c => c.querySelector('select'));
function entries(){ return cards.map(c=>({source_design_id:c.dataset.id,status:c.querySelector('select').value,reason:c.querySelector('input').value})); }
try { const old = JSON.parse(localStorage.getItem(key)||'[]'); cards.forEach(c=>{const d=old.find(x=>x.source_design_id===c.dataset.id);if(d){c.querySelector('select').value=d.status;c.querySelector('input').value=d.reason;}}); } catch(e) {}
cards.forEach(c=>c.addEventListener('change',()=>{try{localStorage.setItem(key,JSON.stringify(entries()));}catch(e){}}));
function download(){const decisions=entries();if(decisions.some(d=>d.status==='rejected'&&!d.reason.trim())){alert('Give a reason for each rejection.');return;}const blob=new Blob([JSON.stringify({manifest_sha256:manifestHash,decisions},null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='review-decisions.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}
</script></html>'''
    (root / "gallery.html").write_text(page, encoding="utf-8")


def checked_source(root: Path, record: dict) -> dict:
    run = json.loads((root / "run.json").read_text(encoding="utf-8"))
    for directory, key in ((ASSETS_DIR, "assets"), (FONTS_DIR, "fonts")):
        if any(not (directory / name).is_file() or file_hash(directory / name) != expected
               for name, expected in run[key].items()):
            raise ValueError("Bundled assets/fonts changed since rendering")
    source = json.loads((root / record["source_path"]).read_text(encoding="utf-8"))
    if "sha256:"+digest(source) != record["source_design_id"]:
        raise ValueError("Source changed after generation")
    if not validate_source(source)["accepted"]:
        raise ValueError("Source no longer passes validation")
    if file_hash(root / record["preview_path"]) != record["preview_sha256"]:
        raise ValueError("Preview changed after generation")
    return source


def apply_review(root: Path, review: dict) -> int:
    if review.get("manifest_sha256") != file_hash(root / "manifest.jsonl"):
        raise ValueError("Review belongs to a different generation manifest")
    records = read_records(root)
    eligible = {r["source_design_id"]: r for r in records if r["status"] == "rendered"}
    decisions = {}
    for decision in review["decisions"]:
        sid = decision["source_design_id"]
        if sid not in eligible or sid in decisions:
            raise ValueError("Unknown, ineligible or repeated source ID in review")
        status, reason = decision["status"], decision.get("reason", "")
        if status not in ("pending", "approved", "rejected") or not isinstance(reason, str):
            raise ValueError("Invalid review status/reason")
        if status == "rejected" and not reason.strip():
            raise ValueError("Rejection requires a reason")
        if status == "approved":
            checked_source(root, eligible[sid])
        decisions[sid] = dict(status=status, reason=reason)
    approved = [dict(r, review_status="approved", review_reason=decisions[sid]["reason"])
                for sid, r in eligible.items() if decisions.get(sid, {}).get("status") == "approved"]
    # A review is a complete snapshot: omitted IDs return to pending, never implicit approval.
    write_json(root / "review.json", review)
    write_jsonl(root / "approved.jsonl", approved)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    summary.update(approved=len(approved), rejected=sum(d["status"] == "rejected" for d in decisions.values()))
    summary["pending_review"] = len(eligible)-summary["approved"]-summary["rejected"]
    write_json(root / "summary.json", summary)
    write_gallery(root, records, decisions)
    return len(approved)


def export_handoff(root: Path, target: dict) -> list[dict]:
    """Noninvasive S04 task envelope, only for explicitly approved sources."""
    Canvas.from_adobe(target)
    review_path = root / "review.json"
    if not review_path.exists():
        return []
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if review["manifest_sha256"] != file_hash(root / "manifest.jsonl"):
        raise ValueError("Review manifest mismatch")
    approved = {d["source_design_id"] for d in review["decisions"] if d["status"] == "approved"}
    tasks = []
    for record in read_records(root):
        if record["status"] == "rendered" and record["source_design_id"] in approved:
            tasks.append(dict(source=checked_source(root, record), target_canvas=copy.deepcopy(target),
                              source_design_id=record["source_design_id"], template_family_id=record["template_family_id"],
                              generation_manifest=dict(path="manifest.jsonl", sha256=file_hash(root / "manifest.jsonl"), record=record)))
    return tasks
