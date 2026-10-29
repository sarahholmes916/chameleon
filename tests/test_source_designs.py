"""Source-only data contracts; actual renderer integration runs in the sources CI job."""

import copy
import json
from pathlib import Path

import pytest
from PIL import Image
from jsonschema import Draft202012Validator

from engine.internal_schema import Design
from scripts import source_dataset as dataset
from scripts.generate_source_designs import main
from scripts.source_templates import ASSETS, FAMILIES, GENERATOR_VERSION, make_source, make_source_with_metadata
from scripts.source_variation import contrast, load_profile, sample_palette, text_contrasts, validate_profile


@pytest.fixture(scope="module")
def renderer():
    return dataset.SourceRenderer()


@pytest.fixture(scope="module")
def pilot(tmp_path_factory, renderer):
    root = tmp_path_factory.mktemp("source-pilot")
    summary = dataset.generate_dataset(root, renderer=renderer)
    return root, summary, dataset.read_records(root)


@pytest.mark.parametrize("family", FAMILIES)
def test_templates_follow_actual_schema_deterministically(family):
    schema = json.loads(dataset.SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    for seed in (1, 42, 991):
        for variant in range(3):
            source = make_source(family, seed, variant)
            assert source == make_source(family, seed, variant)
            validator.validate(source)
            assert Design.from_adobe(source).to_adobe() == source
            assert dataset.validate_source(source)["accepted"]
            assert all(set(e) <= set(schema["$defs"]["element"]["properties"]) for e in source["elements"])
    assert make_source(family, 42) != make_source(family, 991)


def test_template_coverage_and_different_compositions(pilot):
    root, summary, records = pilot
    assert summary == dict(rendered=20, validation_failed=0, renderer_failed=0,
                           duplicate=0, requested=20, approved=0, pending_review=20)
    assert len(FAMILIES) >= 10
    assert {r["content_category"] for r in records} == {
        "editorial", "product_advertisement", "event_flyer", "announcement", "promotion"}
    assert {r["template_family_id"] for r in records} == set(FAMILIES)
    assert len({r["source_design_id"] for r in records}) == 20
    assert len({r["element_count"] for r in records}) >= 5
    ratios = {r["canvas"]["width"]/r["canvas"]["height"] for r in records}
    assert min(ratios) < 1 < max(ratios) and 1 in ratios
    designs = [json.loads((root/r["source_path"]).read_text()) for r in records]
    assert {sum(e["type"] == "image" for e in d["elements"]) for d in designs} >= {0, 1, 3}
    for i, a in enumerate(designs):
        for b in designs[i+1:]:
            assert not dataset.near_identical(a, b)


def test_assets_and_groups_are_real_and_preserved(pilot):
    root, _, records = pilot
    for record in records:
        source = json.loads((root/record["source_path"]).read_text())
        design = Design.from_adobe(source)
        assert all(asset in ASSETS and (dataset.ASSETS_DIR/asset).is_file() for asset in record["assets"])
        assert "solstice_holiday_badge_2025.png" not in record["assets"]
        assert all(len(members) >= 2 for members in design.groups.values())
        assert len({e["id"] for e in source["elements"]}) == record["element_count"]
        assert design.to_adobe() == source
        assert record["source_family_id"] == record["template_family_id"]
        assert record["split"] == "unassigned"
    buttons = [e for e in make_source("split_product", 42)["elements"] if e.get("group_id") == "g_action"]
    outer, inner = buttons
    assert outer["x"] <= inner["x"] and outer["y"] <= inner["y"]
    assert inner["x"]+inner["width"] <= outer["x"]+outer["width"]
    assert inner["y"]+inner["height"] <= outer["y"]+outer["height"]
    assert outer["z_index"] < inner["z_index"]


def test_real_renderer_colors_text_and_dimensions(pilot):
    root, _, records = pilot
    for record in records:
        assert record["renderer_version"]["package"]
        assert record["renderer_version"]["converter_sha256"]
        assert not record["renderer_warnings"] and not record["renderer_error"]
        assert all(m["fits"] and m["final_font_size"] >= 12 for m in record["text_measurements"])
        source = json.loads((root/record["source_path"]).read_text())
        assert dataset.verify_png(root/record["preview_path"], source) == record["pixel_sha256"]
        if record["template_family_id"] != "image_panel":
            # Regression: hex shape fills silently became gray in the converter.
            color = source["elements"][0]["fill_color"]
            expected = tuple(int(v) for v in color[5:-1].split(",")[:3])
            with Image.open(root/record["preview_path"]) as image:
                assert image.convert("RGB").getpixel((image.width-2, 2)) == expected


def test_fitting_and_rendering_are_repeatable_and_do_not_mutate(renderer, tmp_path):
    original = make_source("information_cards", 42)
    snapshot = copy.deepcopy(original)
    a, measurements = renderer.fit_text(original)
    b, again = renderer.fit_text(original)
    assert (a, measurements) == (b, again)
    assert original == snapshot
    assert all(x["fits"] for x in measurements)
    assert [e.get("content") for e in a["elements"]] == [e.get("content") for e in original["elements"]]
    renderer.render(a, tmp_path/"a.png")
    renderer.render(b, tmp_path/"b.png")
    assert dataset.file_hash(tmp_path/"a.png") == dataset.file_hash(tmp_path/"b.png")


@pytest.mark.parametrize("mutation,code,category", [
    (lambda d: d["elements"][1].pop("content"), "SCHEMA", "official"),
    (lambda d: d["elements"][1].update(id=d["elements"][0]["id"]), "IDENTITY_OR_JSON", "official"),
    (lambda d: d["elements"][1].update(x=-2), "SOURCE_BOUNDS", "internal"),
    (lambda d: d["elements"][1].update(font_family="Unknown"), "FONT_UNAVAILABLE", "internal"),
    (lambda d: d["elements"][1].update(group_id="alone"), "SINGLETON_GROUP", "internal"),
    (lambda d: d["elements"][1].update(color="#FFFFFF"), "SOURCE_COLOR", "internal"),
    (lambda d: d["elements"][1].update(font_size=1), "SOURCE_TEXT", "internal"),
    (lambda d: d["elements"][1].update(rotation=10), "ROTATION_UNSUPPORTED", "internal"),
])
def test_source_rejections_are_classified(mutation, code, category):
    source = make_source("centered_editorial", 42)
    mutation(source)
    snapshot = copy.deepcopy(source)
    report = dataset.validate_source(source)
    assert not report["accepted"]
    assert any(i["code"] == code and i["rule_class"] == category for i in report["issues"])
    assert source == snapshot


@pytest.mark.parametrize("asset", ["invented.png", "solstice_holiday_badge_2025.png", "../fonts/SolsticeSans-Regular.ttf"])
def test_missing_and_unapproved_assets_rejected(asset):
    source = make_source("split_product", 42)
    next(e for e in source["elements"] if e["type"] == "image")["asset_id"] = asset
    assert any(i["code"] == "ASSET_UNAVAILABLE" for i in dataset.validate_source(source)["issues"])


def test_missing_exact_font_face_rejected(tmp_path):
    assert not dataset.validate_source(make_source("centered_editorial", 42), fonts_dir=tmp_path)["accepted"]


class FakeRenderer:
    """Only for error-path tests; actual PNG integration is tested above."""
    version = {"test_double": True}

    def fit_text(self, source):
        return copy.deepcopy(source), []

    def render(self, source, path):
        Image.new("RGB", tuple(source["canvas"].values()), "navy").save(path)
        return []


@pytest.mark.parametrize("failure", ["exception", "warning", "missing_png", "corrupt_png", "wrong_size"])
def test_renderer_failure_quarantined_and_next_source_attempted(tmp_path, failure):
    class BrokenFirst(FakeRenderer):
        calls = 0

        def render(self, source, path):
            self.calls += 1
            if self.calls > 1:
                return super().render(source, path)
            if failure == "exception":
                raise RuntimeError("renderer failed")
            if failure == "warning":
                super().render(source, path)
                return ["placeholder warning"]
            if failure == "corrupt_png":
                path.write_bytes(b"not png")
            if failure == "wrong_size":
                Image.new("RGB", (10, 10)).save(path)
            return []
    renderer = BrokenFirst()
    summary = dataset.generate_dataset(tmp_path, count=2, renderer=renderer)
    assert summary["renderer_failed"] == 1 and summary["rendered"] == 1
    records = dataset.read_records(tmp_path)
    assert records[0]["source_path"].startswith("quarantine/")
    assert records[0]["renderer_error"]
    assert not (tmp_path/"approved.jsonl").read_text()


def test_schema_rejection_never_calls_renderer(tmp_path):
    class NeverRenderer(FakeRenderer):
        def fit_text(self, source):
            raise AssertionError("must not fit invalid source")

    def invalid(*args):
        source = make_source("centered_editorial", 42)
        source["elements"][1].pop("font_family")
        return source
    result = dataset.generate_dataset(tmp_path, count=1, renderer=NeverRenderer(), factory=invalid)
    assert result["validation_failed"] == 1
    assert dataset.read_records(tmp_path)[0]["renderer_error"] is None


@pytest.mark.parametrize("source", [{"elements": []}, {"canvas": {}, "elements": [None]}, {"canvas": None, "elements": "bad"}])
def test_bad_root_schema_is_quarantined_with_a_reviewable_record(tmp_path, source):
    result = dataset.generate_dataset(tmp_path, count=1, renderer=FakeRenderer(), factory=lambda *args: source)
    assert result["validation_failed"] == 1
    assert (tmp_path/"gallery.html").is_file()
    assert dataset.read_records(tmp_path)[0]["source_path"].startswith("quarantine/")


def test_exact_near_and_pixel_duplicates_quarantined(tmp_path):
    base = make_source("centered_editorial", 42)
    copy_swap = copy.deepcopy(base)
    copy_swap["elements"][1]["content"] = "ANOTHER ISSUE"
    copy_swap["elements"][1]["x"] += 1
    assert dataset.digest(base) != dataset.digest(copy_swap)
    assert dataset.near_identical(base, copy_swap)
    result = dataset.generate_dataset(tmp_path, count=3, renderer=FakeRenderer(),
                                      factory=lambda f, s, v: base if s < 44 else copy_swap)
    assert result["rendered"] == 1 and result["duplicate"] == 2
    assert all(r["duplicate_of"] == dataset.read_records(tmp_path)[0]["source_design_id"]
               for r in dataset.read_records(tmp_path)[1:])
    changed = copy.deepcopy(base)
    changed["elements"][1]["x"] += 30
    assert not dataset.near_identical(base, changed)
    result = dataset.generate_dataset(tmp_path/"pixel", count=2, renderer=FakeRenderer(),
                                      factory=lambda f, s, v: base if s == 42 else changed)
    assert result["rendered"] == 1 and result["duplicate"] == 1


def review_for(root, records, status="approved"):
    return dict(manifest_sha256=dataset.file_hash(root/"manifest.jsonl"), decisions=[
        dict(source_design_id=r["source_design_id"], status=status, reason="manual test review") for r in records])


def test_review_excludes_pending_rejected_and_keeps_s04_handoff_exact(tmp_path):
    dataset.generate_dataset(tmp_path, count=2, renderer=FakeRenderer())
    records = dataset.read_records(tmp_path)
    target = {"width": 728, "height": 90}
    assert dataset.export_handoff(tmp_path, target) == []
    review = review_for(tmp_path, records)
    review["decisions"][1]["status"] = "rejected"
    assert dataset.apply_review(tmp_path, review) == 1
    tasks = dataset.export_handoff(tmp_path, target)
    assert len(tasks) == 1 and tasks[0]["target_canvas"] == target
    assert set(tasks[0]) == {"source", "target_canvas", "source_design_id", "template_family_id", "generation_manifest"}
    assert tasks[0]["source"] == json.loads((tmp_path/records[0]["source_path"]).read_text())
    assert set(tasks[0]["source"]) == {"canvas", "elements"}
    assert tasks[0]["generation_manifest"]["record"] == records[0]
    review["decisions"] = []
    assert dataset.apply_review(tmp_path, review) == 0  # omitted IDs become pending
    assert dataset.export_handoff(tmp_path, target) == []


@pytest.mark.parametrize("bad", ["wrong_manifest", "unknown", "duplicate", "reason", "status", "source_changed", "preview_changed"])
def test_invalid_review_fails_without_approving(tmp_path, bad):
    dataset.generate_dataset(tmp_path, count=1, renderer=FakeRenderer())
    records = dataset.read_records(tmp_path)
    review = review_for(tmp_path, records)
    if bad == "wrong_manifest":
        review["manifest_sha256"] = "wrong"
    elif bad == "unknown":
        review["decisions"][0]["source_design_id"] = "unknown"
    elif bad == "duplicate":
        review["decisions"] *= 2
    elif bad == "reason":
        review["decisions"][0].update(status="rejected", reason="")
    elif bad == "status":
        review["decisions"][0]["status"] = "good"
    elif bad == "source_changed":
        path = tmp_path/records[0]["source_path"]
        source = json.loads(path.read_text())
        source["elements"][1]["content"] += " changed"
        dataset.write_json(path, source)
    else:
        (tmp_path/records[0]["preview_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError):
        dataset.apply_review(tmp_path, review)
    assert not (tmp_path/"approved.jsonl").read_text()


def test_repeat_run_protects_existing_dataset(pilot):
    root, _, _ = pilot
    before = dataset.file_hash(root/"manifest.jsonl")
    with pytest.raises(FileExistsError):
        dataset.generate_dataset(root)
    assert dataset.file_hash(root/"manifest.jsonl") == before


def test_cli_reports_failed_run_and_validates_count(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset, "SourceRenderer", FakeRenderer)
    assert main(["generate", "--count", "1", "--output", str(tmp_path)]) == 0
    with pytest.raises(ValueError):
        dataset.generate_dataset(tmp_path/"bad", count=0)


def test_cli_returns_nonzero_and_disallows_approving_renderer_failures(tmp_path, monkeypatch):
    class Broken(FakeRenderer):
        def render(self, source, path):
            raise RuntimeError("broken renderer")
    monkeypatch.setattr(dataset, "SourceRenderer", Broken)
    assert main(["generate", "--count", "1", "--output", str(tmp_path)]) == 1
    with pytest.raises(ValueError, match="ineligible"):
        dataset.apply_review(tmp_path, review_for(tmp_path, dataset.read_records(tmp_path)))


def test_text_that_cannot_fit_is_quarantined_before_rendering(tmp_path):
    class Overflow(FakeRenderer):
        def fit_text(self, source):
            return source, [dict(element_id="e001", fits=False)]

        def render(self, source, path):
            raise AssertionError("must not render known overflowing text")
    summary = dataset.generate_dataset(tmp_path, count=1, renderer=Overflow())
    assert summary["validation_failed"] == 1
    assert dataset.read_records(tmp_path)[0]["validation"]["issues"][0]["code"] == "TEXT_OVERFLOW"


def test_python310_syntax():
    import ast
    for filename in ("source_templates.py", "source_variation.py", "source_dataset.py", "generate_source_designs.py"):
        ast.parse((dataset.ROOT/"scripts"/filename).read_text(encoding="utf-8"), feature_version=(3, 10))


def test_versioned_template_fixture():
    fixture = json.loads((Path(__file__).parent/f"fixtures/{GENERATOR_VERSION}.json").read_text())
    for record in fixture:
        assert dataset.digest(make_source(record["family"], record["seed"])) == record["sha256"]


def test_sampled_palettes_are_diverse_and_check_all_used_color_pairs():
    import random
    profile = load_profile()
    colors = set()
    for scheme in profile["schemes"]:
        for mode in profile["modes"]:
            for seed in range(40):
                palette = sample_palette(random.Random(seed), scheme, mode, profile)
                assert palette == sample_palette(random.Random(seed), scheme, mode, profile)
                bg, ink, accent, panel = palette
                assert min(contrast(ink, bg), contrast(ink, panel), contrast(accent, bg), contrast(accent, panel)) >= 4.5
                colors.add(palette)
    assert len(colors) > 300
    assert contrast("#FFFFFF", "#000000") == pytest.approx(21)
    assert contrast("#123456", "#123456") == 1


@pytest.mark.parametrize("family", FAMILIES)
def test_varied_sources_preserve_bounds_groups_and_known_contrast(family):
    content, canvases, palettes, counts, styles = set(), set(), set(), set(), set()
    for seed in range(30):
        source, params = make_source_with_metadata(family, seed)
        assert dataset.validate_source(source)["accepted"]
        for check in text_contrasts(source):
            assert check["ratio"] is not None, (family, seed, check)
            assert check["ratio"] >= 4.5
        assert source == make_source(family, seed)
        canvases.add(tuple(source["canvas"].values()))
        palettes.add(params["palette"])
        content.add(tuple(e.get("content") for e in source["elements"]))
        counts.add(len(source["elements"]))
        styles.add((params["heading_font"], params["body_font"], params["heading_weight"]))
    assert len(canvases) >= 25 and len(palettes) == 30 and len(content) > 10 and len(styles) >= 4
    if family in ("product_pedestal", "information_cards", "event_schedule", "offer_coupon", "announcement_bands", "image_panel"):
        assert len(counts) > 1


def test_profile_controls_and_metadata_reproduce_the_source(pilot):
    root, _, records = pilot
    run = json.loads((root/"run.json").read_text())
    assert run["variation_profile_sha256"] == dataset.digest(run["variation_profile"])
    for record in records:
        raw, params = make_source_with_metadata(record["template_family_id"], record["seed"], record["variant"],
                                                profile=run["variation_profile"])
        assert dataset.canonical(params) == dataset.canonical(record["generation_parameters"])
        assert record["variation_profile_sha256"] == run["variation_profile_sha256"]
        assert raw["canvas"] == record["canvas"]
        assert record["template_family_id"] == record["source_family_id"]
    profile = load_profile()
    profile.update(short_edge=[900, 900], schemes=["complementary"], modes=["dark"],
                   mirror_probability=1, heading_fonts=["Solstice Sans"], body_fonts=["Solstice Sans"],
                   card_count=[4, 4], body_alignments=["left"])
    source, params = make_source_with_metadata("information_cards", 1, profile=profile)
    assert params["mirrored"] and params["card_count"] == 4
    assert params["palette_mode"] == "dark" and params["palette_scheme"] == "complementary"
    assert min(source["canvas"].values()) == 900
    assert all(e["font_family"] == "Solstice Sans" for e in source["elements"] if e["type"] == "text")


@pytest.mark.parametrize("change", [
    {"unknown": 1}, {"short_edge": [0, 100]}, {"font_scale": [1.1, .9]},
    {"axis_jitter": [0, .5]}, {"card_count": [2.0, 4.0]}, {"modes": []},
    {"heading_fonts": ["invented"]}, {"mirror_probability": True}, {"version": "bad"},
    {"hue": [float("nan"), 1]},
])
def test_invalid_profiles_fail_explicitly(change):
    profile = load_profile()
    profile.update(change)
    with pytest.raises(ValueError):
        validate_profile(profile)


def test_impossible_palette_is_not_silently_replaced():
    import random
    profile = load_profile()
    profile.update(light_background_lightness=[.5, .5], dark_ink_lightness=[.5, .5], panel_lightness_offset=[0, 0])
    with pytest.raises(ValueError, match="contrasting palette"):
        sample_palette(random.Random(1), "monochromatic", "light", profile)


def test_source_contrast_checks_actual_panel_and_handles_unknown_background():
    source = make_source("split_product", 42)
    button = next(e for e in source["elements"] if e.get("group_id") == "g_action" and e["type"] == "shape")
    label = next(e for e in source["elements"] if e.get("group_id") == "g_action" and e["type"] == "text")
    label["color"] = button["fill_color"]
    report = dataset.validate_source(source)
    assert not report["accepted"] and any(i["code"] == "SOURCE_CONTRAST" for i in report["issues"])
    button.update(type="image", asset_id=ASSETS[0])
    button.pop("shape_type")
    button.pop("fill_color")
    assert next(c for c in text_contrasts(source) if c["element_id"] == label["id"])["ratio"] is None


def test_randomization_does_not_touch_global_random_state():
    import random
    state = random.getstate()
    make_source("offer_coupon", 52)
    assert random.getstate() == state


@pytest.mark.parametrize("mirrored", [False, True])
def test_mirroring_keeps_sequences_and_number_label_pairs_readable(mirrored):
    profile = load_profile()
    profile.update(mirror_probability=1 if mirrored else 0, feature_count=[3, 3], card_count=[3, 3], stat_count=[3, 3])
    for seed in range(8):
        for family in ("product_pedestal", "information_cards"):
            source, params = make_source_with_metadata(family, seed, profile=profile)
            assert params["mirrored"] is mirrored
            numbers = sorted((e for e in source["elements"] if e["type"] == "text" and e["content"] in ("01", "02", "03")),
                             key=lambda e: e["x"])
            assert [e["content"] for e in numbers] == ["01", "02", "03"]  # reads 01, 02, 03 left to right
        source, _ = make_source_with_metadata("announcement_bands", seed, profile=profile)
        by_group = {}
        for e in source["elements"]:
            if e.get("group_id", "").startswith("g") and e["type"] == "text":
                by_group.setdefault(e["group_id"], []).append(e)
        assert len(by_group) == 3
        for number, label in by_group.values():  # number is authored first; it must stay left of its label
            assert number["x"] < label["x"] and label["x"] - number["x"] < source["canvas"]["width"]/3
        assert len({round(n["x"]) for n, _ in by_group.values()}) == 3  # distinct slots, no overlap
