"""Independently stretched shapes must stay representable by the unchanged S03 graph builder.

ML suite: needs torch and torch_geometric, so CI runs it in the Graph and ML / CPU job.
"""
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from engine.candidate_pool import build_candidate_pool
from engine.graph_features import (
    FEATURE_VERSION_V3, FEATURE_VERSION_V4, build_candidate_graph, feature_dimensions,
)
from engine.internal_schema import Canvas, Design
from engine.shape_stretch import VARIANT_PROPORTIONAL


@pytest.mark.parametrize("version", [FEATURE_VERSION_V3, FEATURE_VERSION_V4])
@pytest.mark.parametrize("size", [(728, 90), (90, 728)])
def test_stretched_candidates_build_finite_graphs_of_the_contract_shape(version, size):
    task = json.loads(Path("input/chameleon_sample_level2_core.json").read_text(encoding="utf-8"))
    source = Design.from_adobe(task["source"])
    target = Canvas(width=size[0], height=size[1])
    pool = build_candidate_pool(source, target)
    stretched = [r for r in pool.candidates if r.variant != VARIANT_PROPORTIONAL]
    assert stretched, "the sample has buttons, so label_fit variants must exist"
    node_dim, edge_dim = feature_dimensions(version)
    count = len(source.elements)
    before = source.to_adobe()
    for record in stretched:
        graph = build_candidate_graph(source, record.design, target, feature_version=version)
        assert tuple(graph.x_numeric.shape) == (count, node_dim)
        assert tuple(graph.edge_index.shape) == (2, count * (count - 1))
        assert tuple(graph.edge_attr.shape) == (count * (count - 1), edge_dim)
        assert torch.isfinite(graph.x_numeric).all() and torch.isfinite(graph.edge_attr).all()
        assert graph.element_ids == tuple(e.id for e in source.elements)
    assert source.to_adobe() == before
