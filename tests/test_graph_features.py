"""S03 feature-contract tests using controlled layouts, never quality labels."""

import ast
from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

# Graph dependencies are optional for the lightweight S02/resize environment.
torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv

from engine.graph_features import (
    EDGE_DIMENSION_V4, EDGE_FEATURE_INDEX, EDGE_FEATURE_INDEX_V4, EDGE_FEATURE_NAMES,
    EDGE_FEATURE_NAMES_V4, FEATURE_CONTRACT_VERSION, FEATURE_VERSION_V4, NODE_DIMENSION_V4,
    NODE_FEATURE_INDEX, NODE_FEATURE_INDEX_V4, NODE_FEATURE_NAMES, NODE_FEATURE_NAMES_V4,
    build_candidate_graph, feature_dimensions,
)
from engine.internal_schema import ABSENT, Canvas, Design, SchemaError
from engine.text_fit import estimate_text_height
from engine.text_embeddings import (
    EmbeddingUnavailable, FrozenSentenceTransformerProvider, TextEmbeddingCache,
)


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = (
    ("chameleon_sample_level1_warmup.json", 4, 12),
    ("chameleon_sample_level2_core.json", 11, 110),
    ("chameleon_sample_level3_advanced.json", 17, 272),
)


def element(id="a", type="image", **changes):
    data = dict(id=id, type=type, x=0, y=0, width=10, height=10)
    if type == "text":
        data.update(content="Hello", font_family="Family", font_size=12, color="#fff")
    elif type == "shape":
        data.update(shape_type="rectangle", fill_color="#123456")
    else:
        data.update(asset_id="unavailable.png")
    data.update(changes)
    return data


def design(*elements, width=100, height=100):
    return Design.from_adobe({"canvas": {"width": width, "height": height}, "elements": list(elements)})


def candidate_for(source, target=None, updates=None, reverse=False):
    """Controlled fixture edits only, not a generator or a validated layout."""
    target = target or source.canvas
    data = source.to_adobe()
    data["canvas"] = target.to_adobe()
    for item in data["elements"]:
        item.update((updates or {}).get(item["id"], {}))
    if reverse:
        data["elements"].reverse()
    return Design.from_adobe(data)


def edges(graph):
    return list(map(tuple, graph.edge_index.t().tolist()))


def assert_graph_contract(graph, n, e):
    assert isinstance(graph, Data)
    assert graph.num_nodes == n
    assert graph.x_numeric.shape == (n, 57)
    assert graph.text_vec.shape == (n, 384)
    assert graph.text_mask.shape == (n, 1)
    assert graph.edge_index.shape == (2, e)
    assert graph.edge_attr.shape == (e, 31)
    assert graph.edge_index.dtype == torch.int64
    assert graph.edge_index.is_contiguous()
    for name in ("x_numeric", "text_vec", "text_mask", "edge_attr"):
        tensor = graph[name]
        assert tensor.dtype == torch.float32
        assert tensor.device.type == "cpu"
        assert torch.isfinite(tensor).all()
        assert not tensor.requires_grad
    assert graph.context_version == "graph_v3"
    assert graph.validate(raise_on_error=True)


def assert_connected(graph):
    pairs = edges(graph)
    seen = {0}
    while True:
        expanded = seen | {j for i, j in pairs if i in seen}
        if expanded == seen:
            break
        seen = expanded
    assert len(seen) == graph.num_nodes


def test_feature_names_lock_the_complete_order():
    expected_nodes = (
        "type_text type_image type_shape source_x source_y source_width source_height "
        "source_center_x source_center_y source_area source_log_aspect rotation_sin rotation_cos "
        "opacity has_z_index z_index_scaled font_size_log bold italic align_left align_center "
        "align_right char_count_log word_count_log has_max_lines max_lines_scaled fit_contain "
        "fit_cover fit_fill shape_rectangle shape_ellipse shape_line fill_r fill_g fill_b "
        "fill_parse_mask text_r text_g text_b text_parse_mask has_group group_size_fraction "
        "candidate_x candidate_y candidate_width candidate_height center_delta_x center_delta_y "
        "width_log_change height_log_change target_log_aspect canvas_width_log_change canvas_height_log_change "
        "candidate_font_size_log font_size_log_change fill_alpha text_alpha"
    ).split()
    expected_edges = (
        "source_dx source_dy source_gap_x source_gap_y source_iou source_i_contains_j "
        "source_j_contains_i source_align_left source_align_center_x source_align_right "
        "source_align_top source_align_center_y source_align_bottom source_log_area_ratio "
        "same_explicit_group z_known z_sign semantic_cos semantic_mask candidate_dx candidate_dy "
        "candidate_gap_x candidate_gap_y candidate_iou candidate_align_left candidate_align_center_x "
        "candidate_align_right candidate_align_top candidate_align_center_y candidate_align_bottom candidate_log_area_ratio"
    ).split()
    assert list(NODE_FEATURE_NAMES) == expected_nodes
    assert list(EDGE_FEATURE_NAMES) == expected_edges
    assert len(set(NODE_FEATURE_NAMES)) == 57
    assert len(set(EDGE_FEATURE_NAMES)) == 31
    assert dict(NODE_FEATURE_INDEX) == {name: i for i, name in enumerate(expected_nodes)}
    assert dict(EDGE_FEATURE_INDEX) == {name: i for i, name in enumerate(expected_edges)}
    assert FEATURE_CONTRACT_VERSION == "graph_v3"


@pytest.mark.parametrize("filename,n,e", SAMPLES)
def test_real_samples_have_exact_dense_graphs_without_mutation(filename, n, e):
    task = json.loads((ROOT / "input" / filename).read_text(encoding="utf-8"))
    original = deepcopy(task)
    source = Design.from_adobe(task["source"], target_canvas=task["target_canvas"])
    target = source.target_canvas
    updates = {el.id: {"x": i * 2, "y": i * 3, "width": el.width / 2, "height": el.height / 3}
               for i, el in enumerate(source.elements)}
    candidate = candidate_for(source, target, updates)
    before = candidate.to_adobe()
    graph = build_candidate_graph(source, candidate, target)
    repeated = build_candidate_graph(source, candidate, target)
    assert_graph_contract(graph, n, e)
    assert graph.element_ids == tuple(el.id for el in source.elements)
    assert edges(graph) == [(i, j) for i in range(n) for j in range(n) if i != j]
    assert graph.graph_topology == "dense_v1"
    assert graph.text_embedding_status == "none"
    assert graph.text_encoder_version == "none"
    assert torch.count_nonzero(graph.text_vec) == 0
    assert torch.count_nonzero(graph.edge_attr[:, 17:19]) == 0
    for name in ("x_numeric", "text_vec", "text_mask", "edge_index", "edge_attr"):
        assert torch.equal(graph[name], repeated[name])
    assert_connected(graph)
    assert task == original
    assert source.to_adobe() == task["source"]
    assert candidate.to_adobe() == before
    graph.x_numeric.fill_(99)
    graph.edge_attr.fill_(-99)
    assert source.to_adobe() == task["source"]
    assert candidate.to_adobe() == before


@pytest.mark.parametrize("count", [1, 4, 64])
def test_dense_threshold_and_single_node(count):
    source = design(*(element(id=str(i)) for i in range(count)))
    graph = build_candidate_graph(source, source, source.canvas)
    assert_graph_contract(graph, count, count * (count - 1))
    assert graph.graph_topology == "dense_v1"
    assert graph.text_embedding_status == "no_text"
    assert edges(graph) == [(i, j) for i in range(count) for j in range(count) if i != j]


def test_complete_text_node_vector_uses_source_style_and_target_geometry():
    source = design(
        element("z", "text", x=20, y=10, width=60, height=20, rotation=90,
                opacity=0.6, z_index=64, content="Hi 🌞\nworld", font_size=24,
                font_weight="bold", font_style="italic", text_align="center",
                max_lines=5, color="#aBc", group_id=""),
        element("a", "shape", group_id=""), width=200, height=100,
    )
    target = Canvas(width=400, height=200)
    candidate = candidate_for(source, target, {"z": {"x": 80, "y": 40, "width": 120, "height": 60}}, reverse=True)
    graph = build_candidate_graph(source, candidate, target)
    expected = [
        1, 0, 0, .1, .1, .3, .2, .25, .2, .06, math.log(61 / 21) / 4,
        1, 0, .6, 1, 1, math.log(25) / math.log(201), 1, 1, 0, 1, 0,
        math.log(11) / math.log(513), math.log(4) / math.log(257), 1, .25,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 170 / 255, 187 / 255, 204 / 255, 1,
        1, 1, .2, .2, .3, .3, .1, .15, math.log(121 / 61) / 4, math.log(61 / 21) / 4,
        math.log(2) / 4, math.log(2) / 4, math.log(2) / 4,
        math.log(25) / math.log(401), 0, 0, 1,
    ]
    torch.testing.assert_close(graph.x_numeric[0], torch.tensor(expected), atol=1e-7, rtol=1e-6)
    assert graph.element_ids == ("z", "a")


def test_font_only_candidate_edits_are_visible_and_joined_by_id():
    source = design(element("a", "text", font_size=60), element("b", "text", font_size=30),
                    element("c", "image"))
    target = Canvas(width=200, height=300)
    first = candidate_for(source, target)
    second = candidate_for(source, target, {"a": {"font_size": 12}}, reverse=True)
    before = (source.to_adobe(), first.to_adobe(), second.to_adobe())
    a = build_candidate_graph(source, first, target)
    b = build_candidate_graph(source, second, target)
    assert torch.equal(a.x_numeric[:, :53], b.x_numeric[:, :53])
    assert not torch.equal(a.x_numeric[0, 53:55], b.x_numeric[0, 53:55])
    torch.testing.assert_close(b.x_numeric[0, 53:55], torch.tensor([
        math.log1p(12) / math.log1p(300), math.log(12 / 60) / 4,
    ]))
    assert torch.equal(a.x_numeric[1:], b.x_numeric[1:])
    assert b.x_numeric[2, 53:55].tolist() == [0, 0]
    for name in ("text_vec", "text_mask", "edge_index", "edge_attr"):
        assert torch.equal(a[name], b[name])
    assert before == (source.to_adobe(), first.to_adobe(), second.to_adobe())


@pytest.mark.parametrize("original,proposed", [(1, 1), (12, 60), (60, 12), (1, 10**300), (10**300, 1)])
def test_font_features_handle_growth_shrinkage_minimum_and_extremes(original, proposed):
    source = design(element("a", "text", font_size=original), width=1, height=1)
    candidate = candidate_for(source, updates={"a": {"font_size": proposed}})
    graph = build_candidate_graph(source, candidate, source.canvas)
    assert torch.isfinite(graph.x_numeric).all()
    assert graph.x_numeric[0, 53].item() == pytest.approx(math.log1p(proposed) / math.log(2))
    assert graph.x_numeric[0, 54].item() == pytest.approx(
        max(-4, min(4, math.log(proposed) - math.log(original))) / 4)


def test_nontext_font_extensions_do_not_become_typography_features():
    source = design(element("a", "image", font_size=12))
    candidate = candidate_for(source, updates={"a": {"font_size": 60}})
    assert build_candidate_graph(source, candidate, source.canvas).x_numeric[0, 53:55].tolist() == [0, 0]


@pytest.mark.parametrize("kind,option,slot", [
    ("image", "contain", 26), ("image", "cover", 27), ("image", "fill", 28),
    ("shape", "rectangle", 29), ("shape", "ellipse", 30), ("shape", "line", 31),
])
def test_image_and_shape_features_are_type_specific(kind, option, slot):
    kwargs = {"fit" if kind == "image" else "shape_type": option}
    source = design(element(type=kind, content="ignored", font_size=50, font_weight="bold", **kwargs))
    row = build_candidate_graph(source, source, source.canvas).x_numeric[0]
    assert row[:3].tolist() == ([0, 1, 0] if kind == "image" else [0, 0, 1])
    assert torch.count_nonzero(row[16:26]) == 0
    assert row[slot] == 1
    assert row[26:32].sum() == 1
    assert torch.count_nonzero(row[36:40]) == 0
    if kind == "shape":
        torch.testing.assert_close(row[32:36], torch.tensor([18 / 255, 52 / 255, 86 / 255, 1]))
    else:
        assert torch.count_nonzero(row[32:36]) == 0


def test_optional_defaults_do_not_materialize_in_adobe_data():
    source = design(element(type="text"), element(id="b"))
    before = source.to_adobe()
    graph = build_candidate_graph(source, source, source.canvas)
    row = graph.x_numeric[0]
    assert row[11:16].tolist() == [0, 1, 1, 0, 0]
    assert row[17:22].tolist() == [0, 0, 1, 0, 0]
    assert row[24:26].tolist() == [0, 0]
    assert row[40:42].tolist() == [0, 0]
    assert graph.x_numeric[1, 26:29].tolist() == [1, 0, 0]
    assert source.elements[0].rotation is ABSENT
    assert source.to_adobe() == before


@pytest.mark.parametrize("value,expected", [(None, [0, 0]), (0, [1, 0]), (-2, [1, 0]), (20, [1, 1]), (21, [1, 1])])
def test_max_lines_presence_and_clipping(value, expected):
    source = design(element(type="text", max_lines=value))
    assert build_candidate_graph(source, source, source.canvas).x_numeric[0, 24:26].tolist() == expected


@pytest.mark.parametrize("angle,sine,cosine", [(0, 0, 1), (90, 1, 0), (-90, -1, 0), (180, 0, -1), (450, 1, 0)])
def test_rotation_in_degrees(angle, sine, cosine):
    source = design(element(rotation=angle))
    values = build_candidate_graph(source, source, source.canvas).x_numeric[0, 11:13]
    torch.testing.assert_close(values, torch.tensor([sine, cosine], dtype=torch.float32), atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize("literal,expected", [
    ("#abc", [170 / 255, 187 / 255, 204 / 255, 1]),
    ("#00FF80", [0, 1, 128 / 255, 1]),
    ("#00ff8000", [0, 1, 128 / 255, 1]),  # RGB/mask are independent of alpha, now encoded separately.
    ("rgba(1,2,3,1)", [1 / 255, 2 / 255, 3 / 255, 1]), ("red", [0, 0, 0, 0]),
    ("#abcd", [170 / 255, 187 / 255, 204 / 255, 1]), ("#gggggg", [0, 0, 0, 0]), (" #fff ", [1, 1, 1, 1]),
])
@pytest.mark.parametrize("kind,start,key", [("text", 36, "color"), ("shape", 32, "fill_color")])
def test_hex_colors_and_parse_masks(literal, expected, kind, start, key):
    source = design(element(type=kind, **{key: literal}))
    graph = build_candidate_graph(source, source, source.canvas)
    torch.testing.assert_close(graph.x_numeric[0, start:start + 4], torch.tensor(expected, dtype=torch.float32))
    assert source.to_adobe()["elements"][0][key] == literal


FIXTURE_COLORS = {
    "rgba(255,255,255,1)": (255, 255, 255, 1),
    "rgba(232,106,58,1)": (232, 106, 58, 1),
    "rgba(240,178,66,1)": (240, 178, 66, 1),
    "rgba(255,255,255,0.85)": (255, 255, 255, .85),
    "rgba(255,255,255,0.9)": (255, 255, 255, .9),
    "rgba(255,255,255,0.6)": (255, 255, 255, .6),
    "rgba(43,27,46,1)": (43, 27, 46, 1),
    "rgba(220,60,50,1)": (220, 60, 50, 1),
}


@pytest.mark.parametrize("filename,n,e", SAMPLES)
def test_actual_adobe_fixture_text_and_shape_colors(filename, n, e):
    task = json.loads((ROOT / "input" / filename).read_text(encoding="utf-8"))
    original = deepcopy(task)
    source = Design.from_adobe(task["source"])
    target = Canvas.from_adobe(task["target_canvas"])
    candidate = candidate_for(source, target, reverse=True)
    graph = build_candidate_graph(source, candidate, target)
    assert_graph_contract(graph, n, e)
    for i, item in enumerate(task["source"]["elements"]):
        if item["type"] == "image":
            assert torch.count_nonzero(graph.x_numeric[i, 32:40]) == 0
            assert graph.x_numeric[i, 55:57].tolist() == [0, 0]
            continue
        is_text = item["type"] == "text"
        start, alpha_column = (36, 56) if is_text else (32, 55)
        literal = item["color" if is_text else "fill_color"]
        red, green, blue, alpha = FIXTURE_COLORS[literal]
        torch.testing.assert_close(graph.x_numeric[i, start:start + 4], torch.tensor([
            red / 255, green / 255, blue / 255, 1,
        ]))
        assert graph.x_numeric[i, alpha_column].item() == pytest.approx(alpha)
        assert graph.x_numeric[i, 13].item() == pytest.approx(item.get("opacity", 1))
        # Exercise literals appearing only in a style run through the same text
        # parser; run-level pooling is not part of this feature contract.
        for run in item.get("style_runs", []):
            r, g, b, a = FIXTURE_COLORS[run["color"]]
            run_source = design(element(type="text", color=run["color"]))
            run_graph = build_candidate_graph(run_source, run_source, run_source.canvas)
            torch.testing.assert_close(run_graph.x_numeric[0, 36:40], torch.tensor([r / 255, g / 255, b / 255, 1]))
            assert run_graph.x_numeric[0, 56].item() == pytest.approx(a)
    assert source.to_adobe() == original["source"]
    assert task == original
    assert candidate.to_adobe() == candidate_for(source, target, reverse=True).to_adobe()


@pytest.mark.parametrize("kind,start,key,alpha_column", [("text", 36, "color", 56), ("shape", 32, "fill_color", 55)])
@pytest.mark.parametrize("literal,expected", [
    (" rgba(232, 106, 58, .6) ", [232 / 255, 106 / 255, 58 / 255, .6]),
    ("RGB(0, 127.5, 255)", [0, .5, 1, 1]),
    ("rgba(0,0,0,0)", [0, 0, 0, 0]),
    ("rgba(255,255,255,0)", [1, 1, 1, 0]),
    ("#00ff8080", [0, 1, 128 / 255, 128 / 255]),
    ("#abcd", [170 / 255, 187 / 255, 204 / 255, 221 / 255]),
    ("rgba(2.55e2,0,0,1e-1)", [1, 0, 0, .1]),
])
def test_supported_colors_preserve_alpha_separately_from_opacity(kind, start, key, alpha_column, literal, expected):
    source = design(element(type=kind, opacity=.4, **{key: literal}))
    before = source.to_adobe()
    row = build_candidate_graph(source, source, source.canvas).x_numeric[0]
    torch.testing.assert_close(row[start:start + 4], torch.tensor([*expected[:3], 1], dtype=torch.float32))
    assert row[alpha_column].item() == pytest.approx(expected[3])
    assert row[13].item() == pytest.approx(.4)
    assert source.to_adobe() == before


@pytest.mark.parametrize("literal", [
    "rgba(255,0,0)", "rgb(255,0,0,1)", "rgba(255,0,0,1)junk",
    "rgba(256,0,0,1)", "rgba(-1,0,0,1)", "rgba(0,0,0,1.1)",
    "rgba(0,0,0,-.1)", "rgba(nan,0,0,1)", "rgba(0,0,0,inf)",
    "rgba(1e999,0,0,1)", "rgba(0,0,0,1e999)", "rgba(1,,2,1)",
    "rgb(100%,0%,0%)", "rgb(255 0 0 / .5)", "red", "", "#gggggg",
])
@pytest.mark.parametrize("kind,start,key,alpha_column", [("text", 36, "color", 56), ("shape", 32, "fill_color", 55)])
def test_unsupported_colors_are_masked_without_rewriting_json(literal, kind, start, key, alpha_column):
    source = design(element(type=kind, **{key: literal}))
    row = build_candidate_graph(source, source, source.canvas).x_numeric[0]
    assert row[start:start + 4].tolist() == [0, 0, 0, 0]
    assert row[alpha_column] == 0
    assert torch.isfinite(row).all()
    assert source.to_adobe()["elements"][0][key] == literal


def test_feature_clipping_uses_unclipped_geometry_for_deltas_and_preserves_source():
    source = design(element(x=-1000, y=500, width=2000, height=0, z_index=-100))
    candidate = candidate_for(source, updates={"a": {"x": 1000, "y": -1000, "width": 0, "height": 2000}})
    row = build_candidate_graph(source, candidate, source.canvas).x_numeric[0]
    assert row[3:10].tolist() == [-4, 4, 4, 0, 0, 4, 0]
    assert row[15] == -1
    assert row[42:50].tolist() == [4, -4, 0, 4, 4, -4, -1, 1]
    assert source.elements[0].x == -1000
    # Both centers clip to 4, but their actual difference is still 1.
    source = design(element(x=1000, width=0))
    candidate = candidate_for(source, updates={"a": {"x": 1100}})
    assert build_candidate_graph(source, candidate, source.canvas).x_numeric[0, 46] == 1


def test_portrait_font_normalization_and_three_distinct_target_context_features():
    source = design(element(type="text", font_size=24), width=100, height=200)
    target = Canvas(width=200, height=100)
    candidate = candidate_for(source, target)
    row = build_candidate_graph(source, candidate, target).x_numeric[0]
    assert row[16].item() == pytest.approx(math.log(25) / math.log(201))
    # Aspect, width change, height change must use their own denominators.
    target = Canvas(width=50, height=800)
    row = build_candidate_graph(source, candidate_for(source, target), target).x_numeric[0]
    torch.testing.assert_close(row[50:53], torch.tensor([
        math.log(1 / 16) / 4, math.log(.5) / 4, math.log(4) / 4,
    ]))
    target = Canvas(width=10**100, height=1)
    row = build_candidate_graph(source, candidate_for(source, target), target).x_numeric[0]
    assert row[50:53].tolist() == [1, 1, -1]


def test_extreme_valid_numbers_zero_sizes_and_log_features_stay_finite():
    source = design(element("a", x=10**400, width=10**400, height=10**400, rotation=10**400),
                    element("b", x=-10**400, width=0, height=0))
    candidate = candidate_for(source, updates={"a": {"width": 0, "height": 0}})
    graph = build_candidate_graph(source, candidate, source.canvas)
    assert_graph_contract(graph, 2, 2)
    assert graph.x_numeric[0, 9] == 4
    assert graph.x_numeric[0, 48:50].tolist() == [-1, -1]
    assert graph.edge_attr[0, 0] == -4
    assert graph.edge_attr[0, 7] == 2
    assert graph.edge_attr[0, 13] == -1
    assert graph.edge_attr[0, 30] == 0


def test_group_features_distinguish_null_absent_and_empty_string():
    source = design(element("a", group_id=""), element("b", group_id=""),
                    element("c", group_id=None), element("d"), element("e", group_id="single"))
    graph = build_candidate_graph(source, source, source.canvas)
    torch.testing.assert_close(graph.x_numeric[:, 40:42], torch.tensor([[1, .4], [1, .4], [0, 0], [0, 0], [1, .2]]))
    for index, (i, j) in enumerate(edges(graph)):
        assert graph.edge_attr[index, 14] == int({i, j} == {0, 1})
    assert_connected(graph)


@pytest.mark.parametrize("fields", [{}, {"group_id": None}])
def test_two_ungrouped_elements_never_form_an_explicit_group(fields):
    source = design(element("a", **fields), element("b", **fields))
    graph = build_candidate_graph(source, source, source.canvas)
    assert graph.edge_attr[:, 14].tolist() == [0, 0]
    assert torch.count_nonzero(graph.x_numeric[:, 40:42]) == 0


def test_complete_edge_vectors_and_reverse_direction():
    source = design(
        element("a", x=10, y=10, width=20, height=20, group_id="", z_index=2),
        element("b", x=50, y=60, width=10, height=20, group_id="", z_index=-1),
    )
    target = Canvas(width=200, height=100)
    candidate = candidate_for(source, target, {
        "a": dict(x=20, y=0, width=40, height=20), "b": dict(x=40, y=10, width=40, height=20),
    })
    graph = build_candidate_graph(source, candidate, target)
    expected = [
        .35, .5, .2, .3, 0, 0, 0, .4, .35, .3, .5, .5, .5,
        math.log(201 / 401) / 6, 1, 1, -1, 0, 0,
        .1, .1, 0, 0, 1 / 7, .1, .1, .1, .1, .1, .1, 0,
    ]
    torch.testing.assert_close(graph.edge_attr[0], torch.tensor(expected), atol=1e-7, rtol=1e-6)
    reverse = list(expected)
    for index in (0, 1, 2, 3, 13, 16, 19, 20, 21, 22, 30):
        reverse[index] *= -1
    torch.testing.assert_close(graph.edge_attr[1], torch.tensor(reverse), atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize("box,expected_iou,contains", [
    ((0, 0, 20, 20), 1, [1, 1]),
    ((5, 5, 10, 10), .25, [1, 0]),
    ((20, 0, 10, 10), 0, [0, 0]),
    ((10, 10, 20, 20), 1 / 7, [0, 0]),
    ((5, 5, 0, 10), 0, [0, 0]),
])
def test_iou_containment_and_touching_boxes(box, expected_iou, contains):
    source = design(element("a", width=20, height=20),
                    element("b", **dict(zip(("x", "y", "width", "height"), box))))
    graph = build_candidate_graph(source, source, source.canvas)
    assert graph.edge_attr[0, 4].item() == pytest.approx(expected_iou)
    assert graph.edge_attr[0, 5:7].tolist() == contains
    assert graph.edge_attr[1, 5:7].tolist() == contains[::-1]


def test_zero_union_and_worked_spec_example():
    source = design(element("a", width=0, height=0), element("b", width=0, height=0))
    assert build_candidate_graph(source, source, source.canvas).edge_attr[:, 4:7].tolist() == [[0, 0, 0], [0, 0, 0]]
    source = design(element("H", x=100, y=80, width=300, height=60, group_id="unit"),
                    element("S", x=105, y=150, width=290, height=40, group_id="unit"), width=1000, height=500)
    row = build_candidate_graph(source, source, source.canvas).edge_attr[0]
    assert row[0].item() == 0
    assert row[1].item() == pytest.approx(.12)
    assert row[3].item() == pytest.approx(.02)
    assert row[7].item() == pytest.approx(.005)
    assert row[14].item() == 1


@pytest.mark.parametrize("first,second,expected", [({}, {}, [0, 0]), ({"z_index": 0}, {}, [0, 0]),
    ({}, {"z_index": 1}, [0, 0]), ({"z_index": 0}, {"z_index": 0}, [1, 0]),
    ({"z_index": -3}, {"z_index": 0}, [1, 1])])
def test_z_index_evidence_requires_both_explicit_values(first, second, expected):
    source = design(element("a", **first), element("b", **second))
    assert build_candidate_graph(source, source, source.canvas).edge_attr[0, 15:17].tolist() == expected


def test_candidate_matching_uses_exact_ids_and_rejects_mismatches():
    source = design(element("a"), element("b", x=50))
    candidate = candidate_for(source, updates={"a": {"x": 30}, "b": {"x": 70}}, reverse=True)
    graph = build_candidate_graph(source, candidate, source.canvas)
    torch.testing.assert_close(graph.x_numeric[:, 42], torch.tensor([.3, .7]))
    for invalid in (design(element("a")), design(element("a"), element("unexpected"))):
        with pytest.raises(ValueError, match="IDs mismatch"):
            build_candidate_graph(source, invalid, source.canvas)
    with pytest.raises(SchemaError, match="duplicate"):
        design(element("a"), element("a"))
    with pytest.raises(ValueError, match="changed type"):
        build_candidate_graph(source, design(element("a", "shape"), element("b")), source.canvas)
    with pytest.raises(ValueError, match="candidate.canvas"):
        build_candidate_graph(source, candidate, Canvas(width=200, height=100))
    shifted_source = replace(source, canvas=Canvas(width=200, height=100))
    with pytest.raises(ValueError, match="source.canvas"):
        build_candidate_graph(shifted_source, candidate, source.canvas)


def test_feature_extraction_does_not_call_the_json_serializer(monkeypatch):
    source = design(element("a"), element("b"))
    def forbidden(*args, **kwargs):
        raise AssertionError("feature extraction must use typed model fields")
    monkeypatch.setattr(Design, "to_adobe", forbidden)
    assert_graph_contract(build_candidate_graph(source, source, source.canvas), 2, 2)


def test_sparse_bridges_connect_eight_separate_neighbor_components():
    source = design(*(element(str(i), x=(i // 9) * 1000 + i % 9, width=0, height=0)
                      for i in range(72)), width=10000, height=100)
    graph = build_candidate_graph(source, source, source.canvas)
    expected = {(i, j) for i in range(72) for j in range(72) if i != j and i // 9 == j // 9}
    for i in range(8, 71, 9):
        expected.update(((i, i + 1), (i + 1, i)))
    assert_graph_contract(graph, 72, 590)
    assert graph.graph_topology == "sparse_v1"
    assert edges(graph) == sorted(expected)
    assert_connected(graph)
    target = Canvas(width=500, height=300)
    candidate = candidate_for(source, target, {e.id: {"x": 0, "y": 0} for e in source.elements}, reverse=True)
    assert torch.equal(graph.edge_index, build_candidate_graph(source, candidate, target).edge_index)


@pytest.mark.parametrize("grouped,neighbors", [(False, 8), (True, 16)])
def test_sparse_tie_breaking_and_group_neighbor_cap(grouped, neighbors):
    source = design(*(element(str(i), **({"group_id": ""} if grouped else {})) for i in range(65)))
    graph = build_candidate_graph(source, source, source.canvas)
    # All centers coincide: source indices determine the selected neighbors.
    expected = [(i, j) for i in range(65) for j in range(65) if i != j and min(i, j) < neighbors]
    assert edges(graph) == expected
    assert (63, 64) not in edges(graph)
    assert_connected(graph)


def test_sparse_explicit_groups_add_distant_relationships():
    source = design(*(element(str(i), x=i, group_id="" if i in (0, 64) else None)
                      for i in range(65)))
    graph = build_candidate_graph(source, source, source.canvas)
    pairs = edges(graph)
    assert (0, 64) in pairs and (64, 0) in pairs
    assert graph.edge_attr[pairs.index((0, 64)), 14] == 1
    assert_connected(graph)


def test_sparse_neighbors_use_normalized_centers_on_a_nonsquare_canvas():
    positions = [(0, 0)] + [(100, 0)] * 8 + [(0, 5)] * 9 + [(900, 9)] * 47
    source = design(*(element(str(i), x=x, y=y, width=0, height=0)
                      for i, (x, y) in enumerate(positions)), width=1000, height=10)
    graph = build_candidate_graph(source, source, source.canvas)
    # (100, 0) is .1 away in normalized space, (0, 5) is .5 away.
    # The 9th node is the single closest-component bridge, not a kNN choice.
    assert [j for i, j in edges(graph) if i == 0] == list(range(1, 10))
    assert_connected(graph)


class StubProvider:
    def __init__(self, values, version="test-encoder-v1"):
        self.encoder_version = version
        self.values = values
        self.calls = []

    def encode(self, text):
        self.calls.append(text)
        value = self.values.get(text)
        if isinstance(value, Exception):
            raise value
        return value


def vector(*values):
    return torch.tensor([*values, *([0.0] * (384 - len(values)))])


def test_text_embeddings_masks_cosine_cache_and_no_input_mutation():
    provider = StubProvider({"alpha": vector(1, 0), "beta": vector(.6, .8)})
    cache = TextEmbeddingCache(provider)
    source = design(element("a", "text", content="alpha"), element("b", "text", content="beta"),
                    element("c", "text", content="alpha"), element("d"))
    graph = build_candidate_graph(source, source, source.canvas, text_embeddings=cache)
    assert_graph_contract(graph, 4, 12)
    assert graph.text_mask[:, 0].tolist() == [1, 1, 1, 0]
    assert graph.text_embedding_status == "complete"
    assert graph.text_encoder_version == "test-encoder-v1"
    pairs = edges(graph)
    assert graph.edge_attr[pairs.index((0, 1)), 17].item() == pytest.approx(.6)
    assert graph.edge_attr[pairs.index((1, 0)), 17].item() == pytest.approx(.6)
    assert graph.edge_attr[pairs.index((0, 2)), 17:19].tolist() == [1, 1]
    assert graph.edge_attr[pairs.index((0, 3)), 17:19].tolist() == [0, 0]
    graph.text_vec[0].zero_()
    repeated = build_candidate_graph(source, source, source.canvas, text_embeddings=cache)
    assert repeated.text_vec[0, 0] == 1
    assert provider.calls == ["alpha", "beta"]


def test_absent_failed_and_zero_embeddings_are_explicitly_unavailable():
    provider = StubProvider({"yes": vector(1), "zero": vector(), "fail": EmbeddingUnavailable("offline")})
    cache = TextEmbeddingCache(provider)
    source = design(*(element(str(i), "text", content=text) for i, text in enumerate(("yes", "zero", "fail", "missing"))))
    graph = build_candidate_graph(source, source, source.canvas, text_embeddings=cache)
    assert graph.text_embedding_status == "partial"
    assert graph.text_mask[:, 0].tolist() == [1, 0, 0, 0]
    assert torch.count_nonzero(graph.edge_attr[:, 17:19]) == 0
    build_candidate_graph(source, source, source.canvas, text_embeddings=cache)
    assert provider.calls == ["yes", "zero", "fail", "missing"]
    cache.clear()
    cache.get("fail")
    assert provider.calls[-1] == "fail" and len(provider.calls) == 5


def test_embedding_cache_preserves_literal_text_versions_and_ownership():
    tensor = vector(1).requires_grad_()
    provider = StubProvider({"a": tensor, " a ": vector(0, 1), "é": vector(1), "e\u0301": vector(1)})
    cache = TextEmbeddingCache(provider)
    first = cache.get("a")
    first.zero_()
    assert not first.requires_grad
    with torch.no_grad():
        tensor.zero_()
    assert cache.get("a")[0] == 1
    for text in (" a ", "é", "e\u0301"):
        cache.get(text)
    assert provider.calls == ["a", " a ", "é", "e\u0301"]
    provider.encoder_version = "test-encoder-v2"
    assert cache.get("a") is None  # Re-encode under the new version; tensor is now zero.
    assert provider.calls[-1] == "a" and len(provider.calls) == 5
    provider.encoder_version = ""
    with pytest.raises(ValueError, match="encoder_version"):
        cache.get("a")


@pytest.mark.parametrize("bad", [torch.zeros(383), torch.zeros(1, 384), vector(float("nan")), vector(float("inf")), [1e300] * 384])
def test_malformed_embeddings_fail_clearly(bad):
    cache = TextEmbeddingCache(StubProvider({"text": bad}))
    with pytest.raises(ValueError, match="embedding"):
        cache.get("text")


def test_large_finite_embeddings_have_stable_cosine():
    cache = TextEmbeddingCache(StubProvider({"a": vector(1e30), "b": vector(-1e30)}))
    source = design(element("a", "text", content="a"), element("b", "text", content="b"))
    graph = build_candidate_graph(source, source, source.canvas, text_embeddings=cache)
    assert graph.edge_attr[:, 17].tolist() == [-1, -1]
    assert_graph_contract(graph, 2, 2)


def test_encoder_version_cannot_change_mid_graph():
    class ChangingProvider(StubProvider):
        def encode(self, text):
            self.encoder_version = "new-version"
            return vector(1)

    source = design(element(type="text"))
    with pytest.raises(ValueError, match="changed during graph"):
        build_candidate_graph(source, source, source.canvas,
                              text_embeddings=TextEmbeddingCache(ChangingProvider({})))


def test_frozen_sentence_transformer_adapter_without_a_model_download():
    class FakeSentenceTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.calls = []

        def get_sentence_embedding_dimension(self):
            return 384

        def encode(self, texts, **kwargs):
            assert not self.training
            assert not torch.is_grad_enabled()
            self.calls.append((texts, kwargs))
            return vector(1).unsqueeze(0)

    model = FakeSentenceTransformer()
    provider = FrozenSentenceTransformerProvider(model, encoder_version="local-minilm@revision")
    assert not model.weight.requires_grad
    assert not model.training
    result = TextEmbeddingCache(provider).get("literal text")
    assert result[0] == 1
    assert model.calls == [(["literal text"], {
        "convert_to_tensor": True, "show_progress_bar": False, "normalize_embeddings": False,
    })]


def test_frozen_adapter_rejects_wrong_dimensions_and_exposes_inference_failure():
    class FailingModel(torch.nn.Module):
        def get_sentence_embedding_dimension(self):
            return 384

        def encode(self, *args, **kwargs):
            raise RuntimeError("encoder unavailable")

    model = FailingModel()
    provider = FrozenSentenceTransformerProvider(model, encoder_version="test-v1")
    assert TextEmbeddingCache(provider).get("text") is None
    model.get_sentence_embedding_dimension = lambda: 128
    with pytest.raises(ValueError, match="dimension"):
        FrozenSentenceTransformerProvider(model, encoder_version="wrong-dimension")


def test_pyg_batching_and_edge_aware_layer_forward_backward_on_cpu():
    graphs = []
    for count in (1, 4, 17):
        source = design(*(element(str(i), x=i * 5) for i in range(count)))
        graphs.append(build_candidate_graph(source, source, source.canvas))
    batch = Batch.from_data_list(graphs)
    assert batch.num_nodes == 22
    assert batch.ptr.tolist() == [0, 1, 5, 22]
    assert batch.x_numeric.shape == (22, 57)
    assert batch.edge_index.shape == (2, 284)
    assert torch.equal(batch.edge_index[:, :12], graphs[1].edge_index + 1)
    assert torch.equal(batch.edge_index[:, 12:], graphs[2].edge_index + 5)
    for restored, original in zip(batch.to_data_list(), graphs):
        assert restored.element_ids == original.element_ids
        assert torch.equal(restored.edge_index, original.edge_index)
        assert torch.equal(restored.x_numeric, original.x_numeric)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        layer = GATv2Conv(57, 4, heads=2, edge_dim=31, add_self_loops=True, fill_value=0.0).cpu()
        output = layer(batch.x_numeric, batch.edge_index, batch.edge_attr)
        assert output.shape == (22, 8)
        assert torch.isfinite(output).all()
        output.square().mean().backward()
        gradients = [parameter.grad for parameter in layer.parameters()]
        assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)


def test_core_resize_imports_without_ml_and_new_modules_use_python310_syntax(tmp_path):
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from engine.resize import resize; from engine.internal_schema import Design; "
        "assert not any(name.split('.')[0] in {'torch', 'torch_geometric'} for name in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-S", "-c", code, str(ROOT)], cwd=tmp_path,
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    for path in ("engine/graph_features.py", "engine/text_embeddings.py", "tests/test_graph_features.py"):
        ast.parse((ROOT / path).read_text(encoding="utf-8"), feature_version=(3, 10))


# --- graph_v4: typography features appended to graph_v3 (docs/reference/B_graph_features.md B.4.3) ---

V4 = "graph_v4"


def graph_v4(source, candidate, target=None):
    return build_candidate_graph(source, candidate, target or source.canvas, feature_version=V4)


def text_fit_ratio(candidate_changes, **source_changes):
    """The graph_v4 text_fit_height_ratio of a single text element after candidate edits."""
    source = design(element("t", "text", **source_changes), width=1000, height=1000)
    candidate = candidate_for(source, updates={"t": candidate_changes})
    return graph_v4(source, candidate).x_numeric[0, NODE_FEATURE_INDEX_V4["text_fit_height_ratio"]].item()


def edge_value(graph, i, j, name):
    column = EDGE_FEATURE_INDEX_V4[name]
    pairs = edges(graph)
    return graph.edge_attr[pairs.index((i, j)), column].item()


@pytest.mark.parametrize("name,nodes,edge_count", SAMPLES)
def test_graph_v4_keeps_every_graph_v3_column_and_appends_two(name, nodes, edge_count):
    task = json.loads((ROOT / "input" / name).read_text(encoding="utf-8"))
    source = Design.from_adobe(task["source"], target_canvas=task["target_canvas"])
    target = Canvas.from_adobe(task["target_canvas"])
    candidate = candidate_for(source, target)
    before = (source.to_adobe(), candidate.to_adobe())
    v3 = build_candidate_graph(source, candidate, target)
    v4 = graph_v4(source, candidate, target)
    assert v3.context_version == "graph_v3" and v4.context_version == V4
    assert v4.x_numeric.shape == (nodes, 58) and v4.edge_attr.shape == (edge_count, 32)
    assert torch.equal(v4.x_numeric[:, :57], v3.x_numeric)
    assert torch.equal(v4.edge_attr[:, :31], v3.edge_attr)
    for field in ("text_vec", "text_mask", "edge_index"):
        assert torch.equal(v4[field], v3[field])
    assert v4.element_ids == v3.element_ids and v4.graph_topology == v3.graph_topology
    assert torch.isfinite(v4.x_numeric).all() and torch.isfinite(v4.edge_attr).all()
    assert before == (source.to_adobe(), candidate.to_adobe())  # inputs are not mutated
    assert torch.equal(v4.x_numeric, graph_v4(source, candidate, target).x_numeric)  # deterministic


def test_graph_v4_names_dimensions_and_default_version():
    assert FEATURE_VERSION_V4 == V4 and FEATURE_CONTRACT_VERSION == "graph_v3"
    assert len(set(NODE_FEATURE_NAMES_V4)) == NODE_DIMENSION_V4 == 58
    assert len(set(EDGE_FEATURE_NAMES_V4)) == EDGE_DIMENSION_V4 == 32
    assert NODE_FEATURE_NAMES_V4[:57] == NODE_FEATURE_NAMES and EDGE_FEATURE_NAMES_V4[:31] == EDGE_FEATURE_NAMES
    assert NODE_FEATURE_INDEX_V4["text_fit_height_ratio"] == 57
    assert EDGE_FEATURE_INDEX_V4["size_order_preserved"] == 31
    assert feature_dimensions("graph_v3") == (57, 31) and feature_dimensions(V4) == (58, 32)
    # The default is still graph_v3 and unchanged.
    source = design(element("a", "text"), element("b", "text"))
    default = build_candidate_graph(source, source, source.canvas)
    assert default.context_version == "graph_v3"
    assert default.x_numeric.shape == (2, 57) and default.edge_attr.shape == (2, 31)


@pytest.mark.parametrize("version", ["graph_v2", "graph_v5", "", None, ["graph_v4"]])
def test_unsupported_feature_version_is_rejected(version):
    source = design(element("a", "text"))
    with pytest.raises(ValueError, match="unsupported feature version"):
        build_candidate_graph(source, source, source.canvas, feature_version=version)


@pytest.mark.parametrize("source_changes,candidate_changes,expected", [
    # 9 chars * 0.5 * 10 / 45 = 1.0 -> 1 line; height 1 * 10 * 1.2 = 12; 12 / 30 / 4
    (dict(content="abcd efgh", width=45, height=30, font_size=10), {}, 12 / 30 / 4),
    # paragraphs "aaaa", "", "bb" are 1 line each -> 3 * 10 * 1.2 = 36; 36 / 36 / 4
    (dict(content="aaaa\n\nbb", width=20, height=36, font_size=10), {}, 36 / 36 / 4),
    # 40 chars * 0.5 * 10 / 20 = 10 lines -> 10 * 10 * 1.2 = 120; 120 / 100 / 4
    (dict(content="x" * 40, width=20, height=100, font_size=10), {}, 120 / 100 / 4),
    # an explicit line_height of 2 -> 1 * 10 * 2 = 20; 20 / 40 / 4
    (dict(content="abc", width=100, height=40, font_size=10, line_height=2), {}, 20 / 40 / 4),
    # candidate edits are what is measured: size 20 -> 1 line of 20 * 1.2 = 24; 24 / 48 / 4
    (dict(content="abc", width=100, height=48, font_size=10), dict(font_size=20), 24 / 48 / 4),
    # overflowing text is capped at 4 before the /4 scaling
    (dict(content="x" * 400, width=10, height=10, font_size=10), {}, 1.0),
])
def test_text_fit_height_ratio_matches_hand_computed_values(source_changes, candidate_changes, expected):
    assert text_fit_ratio(candidate_changes, **source_changes) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("changes", [dict(width=0), dict(height=0), dict(width=0, height=0)])
def test_text_fit_height_ratio_is_zero_for_a_degenerate_candidate_box(changes):
    assert text_fit_ratio(dict(changes), content="Hello", width=50, height=20, font_size=10) == 0.0


def test_text_fit_height_ratio_is_zero_off_text_and_ignores_font_extensions():
    source = design(element("i", "image", font_size=99, content="ignored text"),
                    element("s", "shape", font_size=99, content="ignored text"), width=1000, height=1000)
    graph = graph_v4(source, source)
    assert graph.x_numeric[:, 57].tolist() == [0.0, 0.0]


def test_text_fit_separates_overflowing_from_fitting_sizes_in_identical_boxes():
    # The font-size-only case that motivated graph_v4: same box, different size.
    ratios = [text_fit_ratio(dict(font_size=size), content="Shop the Summer Line", width=200, height=60, font_size=12)
              for size in (8, 24, 80)]
    assert ratios[0] < ratios[1] < ratios[2] and ratios[2] == pytest.approx(1.0)
    assert 0 < ratios[1] < 1


def test_font_only_candidate_edits_change_exactly_the_typography_columns_in_graph_v4():
    # Boxes large enough that neither size saturates the x4 overflow cap of the fit feature.
    source = design(element("a", "text", font_size=60, width=200, height=100),
                    element("b", "text", font_size=30, width=200, height=100),
                    element("c", "image"))
    first = candidate_for(source)
    second = candidate_for(source, updates={"a": {"font_size": 12}})
    a, b = graph_v4(source, first), graph_v4(source, second)
    differing = (a.x_numeric != b.x_numeric).any(dim=0).nonzero().flatten().tolist()
    assert differing == [53, 54, 57]
    assert torch.equal(a.x_numeric[1:], b.x_numeric[1:])
    edge_differing = (a.edge_attr != b.edge_attr).any(dim=0).nonzero().flatten().tolist()
    assert edge_differing == [31]  # only the text-text size-order edge feature changes
    for name in ("text_vec", "text_mask", "edge_index"):
        assert torch.equal(a[name], b[name])


def test_size_order_preserved_is_plus_minus_or_zero_for_text_pairs_only():
    source = design(element("a", "text", font_size=60), element("b", "text", font_size=30),
                    element("c", "text", font_size=30), element("d", "image"))
    name = "size_order_preserved"
    reversed_a = graph_v4(source, candidate_for(source, updates={"a": {"font_size": 12}}))
    preserved = graph_v4(source, candidate_for(source, updates={"a": {"font_size": 90}}))
    collapsed = graph_v4(source, candidate_for(source, updates={"a": {"font_size": 30}}))
    # a is larger than b and c in the source; node indices a=0, b=1, c=2, d=3
    for i, j in ((0, 1), (1, 0), (0, 2), (2, 0)):
        assert edge_value(reversed_a, i, j, name) == -1.0   # candidate order reversed
        assert edge_value(preserved, i, j, name) == 1.0     # candidate order kept
        assert edge_value(collapsed, i, j, name) == 0.0     # candidate sizes tied
    for graph in (reversed_a, preserved, collapsed):
        assert edge_value(graph, 1, 2, name) == 0.0 and edge_value(graph, 2, 1, name) == 0.0  # source tie
        for other in (0, 1, 2):
            assert edge_value(graph, other, 3, name) == 0.0 and edge_value(graph, 3, other, name) == 0.0
        pairs = edges(graph)
        column = EDGE_FEATURE_INDEX_V4[name]
        for k, (i, j) in enumerate(pairs):  # the feature is symmetric
            assert graph.edge_attr[k, column] == graph.edge_attr[pairs.index((j, i)), column]


def test_graph_v4_estimator_is_exact_and_safe_at_extreme_inputs():
    assert estimate_text_height("x", 0, 10, 1.2) is None and estimate_text_height("x", -5, 10, 1.2) is None
    assert float(estimate_text_height("abcd", 40, 10, 1)) == 10.0  # 1 line * 10 * 1
    assert estimate_text_height("", 10, 10, 1) == 10  # an empty paragraph is one line
    for original, proposed in ((1, 10**300), (10**300, 1), (10**300, 10**300)):
        source = design(element("a", "text", font_size=original, width=1, height=1), width=1, height=1)
        candidate = candidate_for(source, updates={"a": {"font_size": proposed}})
        graph = graph_v4(source, candidate)
        assert torch.isfinite(graph.x_numeric).all() and 0 <= graph.x_numeric[0, 57] <= 1


def test_graph_v4_sparse_topology_has_the_extra_edge_column():
    source = design(*[element(f"t{i}", "text", x=i, y=i % 7, font_size=10 + i) for i in range(70)],
                    width=200, height=200)
    graph = graph_v4(source, source)
    assert graph.graph_topology == "sparse_v1"
    assert graph.edge_attr.shape == (graph.edge_index.shape[1], 32)
    assert graph.x_numeric.shape == (70, 58) and torch.isfinite(graph.edge_attr).all()


def test_text_fit_module_is_stdlib_only_and_python310_syntax(tmp_path):
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); import engine.text_fit; "
        "assert not any(name.split('.')[0] in {'torch', 'torch_geometric'} for name in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-S", "-c", code, str(ROOT)], cwd=tmp_path,
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    ast.parse((ROOT / "engine/text_fit.py").read_text(encoding="utf-8"), feature_version=(3, 10))
