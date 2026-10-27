"""S03 graph_v3: deterministic source/candidate graphs, not layout scores.

Feature order follows docs/S03-S06_layout_engine_technical_spec.md sections 3–4.
No Adobe data is changed, and no candidate is certified as contest-compliant.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
import re
from types import MappingProxyType

import torch
from torch_geometric.data import Data

from engine.internal_schema import ABSENT, Canvas, Design, Element
from engine.text_embeddings import TEXT_DIMENSION, TextEmbeddingCache


FEATURE_CONTRACT_VERSION = "graph_v3"
NODE_FEATURE_NAMES = (
    "type_text", "type_image", "type_shape",
    "source_x", "source_y", "source_width", "source_height",
    "source_center_x", "source_center_y", "source_area", "source_log_aspect",
    "rotation_sin", "rotation_cos", "opacity", "has_z_index", "z_index_scaled",
    "font_size_log", "bold", "italic", "align_left", "align_center", "align_right",
    "char_count_log", "word_count_log", "has_max_lines", "max_lines_scaled",
    "fit_contain", "fit_cover", "fit_fill",
    "shape_rectangle", "shape_ellipse", "shape_line",
    "fill_r", "fill_g", "fill_b", "fill_parse_mask",
    "text_r", "text_g", "text_b", "text_parse_mask",
    "has_group", "group_size_fraction",
    "candidate_x", "candidate_y", "candidate_width", "candidate_height",
    "center_delta_x", "center_delta_y", "width_log_change", "height_log_change",
    "target_log_aspect", "canvas_width_log_change", "canvas_height_log_change",
    "candidate_font_size_log", "font_size_log_change",
    "fill_alpha", "text_alpha",
)
EDGE_FEATURE_NAMES = (
    "source_dx", "source_dy", "source_gap_x", "source_gap_y", "source_iou",
    "source_i_contains_j", "source_j_contains_i",
    "source_align_left", "source_align_center_x", "source_align_right",
    "source_align_top", "source_align_center_y", "source_align_bottom",
    "source_log_area_ratio", "same_explicit_group", "z_known", "z_sign",
    "semantic_cos", "semantic_mask",
    "candidate_dx", "candidate_dy", "candidate_gap_x", "candidate_gap_y", "candidate_iou",
    "candidate_align_left", "candidate_align_center_x", "candidate_align_right",
    "candidate_align_top", "candidate_align_center_y", "candidate_align_bottom",
    "candidate_log_area_ratio",
)
NODE_FEATURE_INDEX = MappingProxyType({name: i for i, name in enumerate(NODE_FEATURE_NAMES)})
EDGE_FEATURE_INDEX = MappingProxyType({name: i for i, name in enumerate(EDGE_FEATURE_NAMES)})
NODE_DIMENSION = 57
EDGE_DIMENSION = 31


def _clip(value: Fraction | float, low: float, high: float) -> float:
    return float(min(high, max(low, value)))


def _log(value: Fraction) -> float:
    # Avoid float conversion/overflow of large finite source dimensions.
    return math.log(value.numerator) - math.log(value.denominator)


def _log1p(value: int | float) -> float:
    return _log(Fraction(value) + 1)


def _log_ratio(numerator: Fraction, denominator: Fraction, limit: int) -> float:
    return _clip(_log(numerator) - _log(denominator), -limit, limit) / limit


@dataclass(frozen=True)
class _Box:
    # Rational intermediates avoid overflow/underflow before specified clipping.
    # This is only derived geometry, not a second element/design data model.
    x: Fraction
    y: Fraction
    width: Fraction
    height: Fraction

    @classmethod
    def of(cls, element: Element) -> _Box:
        return cls(*(Fraction(value) for value in (
            element.x, element.y, element.width, element.height,
        )))

    @property
    def right(self) -> Fraction:
        return self.x + self.width

    @property
    def bottom(self) -> Fraction:
        return self.y + self.height

    @property
    def center(self) -> tuple[Fraction, Fraction]:
        return self.x + self.width / 2, self.y + self.height / 2

    @property
    def area(self) -> Fraction:
        return self.width * self.height

    def normalized(self, canvas: Canvas) -> _Box:
        w, h = Fraction(canvas.width), Fraction(canvas.height)
        return _Box(self.x / w, self.y / h, self.width / w, self.height / h)


def _color_rgba(value: str) -> tuple[float, float, float, float, float]:
    """Parse supported literals to RGBA + availability, without editing JSON.

    Supports hex and comma-separated numeric rgb()/rgba(), including the Adobe
    fixtures. Unsupported/malformed/out-of-range literals have zero availability.
    This is a deliberately bounded parser, not a full browser CSS implementation.
    """
    literal = value.strip()
    if re.fullmatch(r"#(?:[0-9a-fA-F]{3,4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})", literal):
        digits = literal[1:]
        if len(digits) in (3, 4):
            digits = "".join(char * 2 for char in digits)
        rgb = tuple(int(digits[i:i + 2], 16) / 255 for i in (0, 2, 4))
        alpha = int(digits[6:8], 16) / 255 if len(digits) == 8 else 1.0
        return (*rgb, alpha, 1.0)
    match = re.fullmatch(r"(rgb|rgba)\(\s*(.*?)\s*\)", literal, flags=re.IGNORECASE)
    if match:
        parts = [part.strip() for part in match[2].split(",")]
        count = 4 if match[1].lower() == "rgba" else 3
        number = r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
        if len(parts) == count and all(re.fullmatch(number, part) for part in parts):
            values = [float(part) for part in parts]
            alpha = values[3] if count == 4 else 1.0
            if (all(math.isfinite(v) and 0 <= v <= 255 for v in values[:3])
                    and math.isfinite(alpha) and 0 <= alpha <= 1):
                return (*(v / 255 for v in values[:3]), alpha, 1.0)
    return 0.0, 0.0, 0.0, 0.0, 0.0


def _node_features(
    element: Element, proposed: Element, source_box: _Box, candidate_box: _Box,
    source_canvas: Canvas, target: Canvas, group_size: int, count: int,
) -> list[float]:
    src = source_box.normalized(source_canvas)
    cand = candidate_box.normalized(target)
    # Reduce periodic angles before conversion, including arbitrarily large ints.
    angle = math.radians(float(element.effective_value("rotation") % 360))
    row = [float(element.type == kind) for kind in ("text", "image", "shape")]
    row += [_clip(src.x, -4, 4), _clip(src.y, -4, 4),
            _clip(src.width, 0, 4), _clip(src.height, 0, 4)]
    row += [_clip(value, -4, 4) for value in src.center]
    row += [_clip(src.area, 0, 4), _log_ratio(source_box.width + 1, source_box.height + 1, 4)]
    row += [math.sin(angle), math.cos(angle), float(element.effective_value("opacity"))]
    row += [float(element.z_index is not ABSENT),
            _clip(Fraction(element.z_index) / 32, -1, 1) if element.z_index is not ABSENT else 0.0]
    text = [0.0] * 10
    if element.type == "text":
        text = [
            _log1p(element.font_size) / _log1p(max(source_canvas.width, source_canvas.height)),
            float(element.effective_value("font_weight") == "bold"),
            float(element.effective_value("font_style") == "italic"),
            *[float(element.effective_value("text_align") == align) for align in ("left", "center", "right")],
            math.log1p(len(element.content)) / math.log1p(512),
            math.log1p(len(element.content.split())) / math.log1p(256),
            float(element.max_lines is not ABSENT and element.max_lines is not None),
            _clip(Fraction(element.max_lines) / 20, 0, 1)
            if element.max_lines is not ABSENT and element.max_lines is not None else 0.0,
        ]
    row += text
    row += [float(element.type == "image" and element.effective_value("fit") == fit)
            for fit in ("contain", "cover", "fill")]
    row += [float(element.type == "shape" and element.shape_type == shape)
            for shape in ("rectangle", "ellipse", "line")]
    fill = _color_rgba(element.fill_color) if element.type == "shape" else (0.0,) * 5
    text_color = _color_rgba(element.color) if element.type == "text" else (0.0,) * 5
    row += [*fill[:3], fill[4]]
    row += [*text_color[:3], text_color[4]]
    row += [float(isinstance(element.group_id, str)), group_size / count]
    row += [_clip(cand.x, -4, 4), _clip(cand.y, -4, 4),
            _clip(cand.width, 0, 4), _clip(cand.height, 0, 4)]
    row += [_clip(b - a, -4, 4) for a, b in zip(src.center, cand.center)]
    row += [_log_ratio(candidate_box.width + 1, source_box.width + 1, 4),
            _log_ratio(candidate_box.height + 1, source_box.height + 1, 4)]
    row += [_log_ratio(Fraction(target.width), Fraction(target.height), 4),
            _log_ratio(Fraction(target.width), Fraction(source_canvas.width), 4),
            _log_ratio(Fraction(target.height), Fraction(source_canvas.height), 4)]
    row += [
        _log1p(proposed.font_size) / _log1p(max(target.width, target.height)),
        _log_ratio(Fraction(proposed.font_size), Fraction(element.font_size), 4),
    ] if element.type == "text" else [0.0, 0.0]
    row += [fill[3], text_color[3]]
    return row


def _gap(a_min: Fraction, a_max: Fraction, b_min: Fraction, b_max: Fraction) -> Fraction:
    if b_min >= a_max:
        return b_min - a_max
    if b_max <= a_min:
        return b_max - a_min
    return Fraction(0)


def _pair_geometry(a: _Box, b: _Box, canvas: Canvas) -> list[float]:
    width, height = Fraction(canvas.width), Fraction(canvas.height)
    acx, acy = a.center
    bcx, bcy = b.center
    intersection = max(0, min(a.right, b.right) - max(a.x, b.x)) * max(
        0, min(a.bottom, b.bottom) - max(a.y, b.y),
    )
    union = a.area + b.area - intersection
    positive = a.area > 0 and b.area > 0
    return [
        _clip((bcx - acx) / width, -4, 4), _clip((bcy - acy) / height, -4, 4),
        _clip(_gap(a.x, a.right, b.x, b.right) / width, -4, 4),
        _clip(_gap(a.y, a.bottom, b.y, b.bottom) / height, -4, 4),
        float(intersection / union) if union > 0 else 0.0,
        float(positive and a.x <= b.x and a.y <= b.y and a.right >= b.right and a.bottom >= b.bottom),
        float(positive and b.x <= a.x and b.y <= a.y and b.right >= a.right and b.bottom >= a.bottom),
        *[_clip(abs(v) / width, 0, 2) for v in (b.x - a.x, bcx - acx, b.right - a.right)],
        *[_clip(abs(v) / height, 0, 2) for v in (b.y - a.y, bcy - acy, b.bottom - a.bottom)],
        _log_ratio(b.area + 1, a.area + 1, 6),
    ]


def _edge_pairs(elements: tuple[Element, ...], boxes: list[_Box], canvas: Canvas) -> list[tuple[int, int]]:
    count = len(elements)
    if count <= 64:
        return [(i, j) for i in range(count) for j in range(count) if i != j]
    centers = [box.normalized(canvas).center for box in boxes]
    distances = {}
    for i in range(count):
        for j in range(i + 1, count):
            distances[i, j] = sum((a - b) ** 2 for a, b in zip(centers[i], centers[j]))

    def distance(i: int, j: int) -> Fraction:
        return distances[min(i, j), max(i, j)]

    edges: set[tuple[int, int]] = set()

    def connect(i: int, j: int) -> None:
        edges.add((i, j))
        edges.add((j, i))

    for i, element in enumerate(elements):
        neighbors = sorted((j for j in range(count) if j != i), key=lambda j: (distance(i, j), j))
        for j in neighbors[:8]:
            connect(i, j)
        if isinstance(element.group_id, str):
            group_neighbors = [j for j in neighbors if elements[j].group_id == element.group_id]
            for j in group_neighbors[:16]:
                connect(i, j)

    # Kruskal over cross-component pairs implements repeated closest bridges.
    parents = list(range(count))

    def find(i: int) -> int:
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def merge(i: int, j: int) -> bool:
        a, b = find(i), find(j)
        if a == b:
            return False
        parents[b] = a
        return True

    components = count
    for i, j in sorted(edges):
        components -= int(merge(i, j))
    for i, j in sorted(distances, key=lambda pair: (distances[pair], *pair)):
        if components == 1:
            break
        if merge(i, j):
            connect(i, j)
            components -= 1
    return sorted(edges)


def _match_candidate(source: Design, candidate: Design, target: Canvas) -> tuple[Element, ...]:
    if source.canvas != source.source_canvas:
        raise ValueError("source.canvas must describe the original source_canvas coordinates")
    if candidate.canvas != target:
        raise ValueError("candidate.canvas must match target")
    source_ids = [element.id for element in source.elements]
    candidate_ids = [element.id for element in candidate.elements]
    if len(set(source_ids)) != len(source_ids) or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("source and candidate must have unique element IDs")
    missing = set(source_ids) - set(candidate_ids)
    unexpected = set(candidate_ids) - set(source_ids)
    if missing or unexpected:
        raise ValueError(f"candidate IDs mismatch: missing={sorted(missing)!r}, unexpected={sorted(unexpected)!r}")
    by_id = {element.id: element for element in candidate.elements}
    matched = tuple(by_id[element.id] for element in source.elements)
    for original, proposed in zip(source.elements, matched):
        if original.type != proposed.type:
            raise ValueError(f"candidate element {original.id!r} changed type")
    return matched


@torch.no_grad()
def build_candidate_graph(
    source: Design,
    candidate: Design,
    target: Canvas,
    *,
    text_embeddings: TextEmbeddingCache | None = None,
) -> Data:
    """Build CPU tensors in source order, matching candidate elements by ID.

    Identity/type/canvas compatibility is checked, not full stage-one policy.
    Candidate array order may differ; output node order always follows source.
    Optional embeddings describe literal SOURCE text and never inspect assets.
    """
    matched = _match_candidate(source, candidate, target)
    count = len(source.elements)
    src_boxes = [_Box.of(element) for element in source.elements]
    cand_boxes = [_Box.of(element) for element in matched]
    pairs = _edge_pairs(source.elements, src_boxes, source.source_canvas)
    nodes = [
        _node_features(element, proposed, a, b, source.source_canvas, target,
                       len(source.groups[element.group_id]) if isinstance(element.group_id, str) else 0,
                       count)
        for element, proposed, a, b in zip(source.elements, matched, src_boxes, cand_boxes)
    ]
    vectors = torch.zeros((count, TEXT_DIMENSION), dtype=torch.float32, device="cpu")
    masks = torch.zeros((count, 1), dtype=torch.float32, device="cpu")
    encoder_version = text_embeddings.encoder_version if text_embeddings is not None else "none"
    text_count = 0
    for index, element in enumerate(source.elements):
        if element.type != "text":
            continue
        text_count += 1
        vector = text_embeddings.get(element.content) if text_embeddings is not None else None
        if vector is not None:
            vectors[index] = vector
            masks[index, 0] = 1
    if text_embeddings is not None and text_embeddings.encoder_version != encoder_version:
        raise ValueError("embedding encoder_version changed during graph construction")
    directions = vectors.double()
    norms = torch.linalg.vector_norm(directions, dim=1, keepdim=True)
    directions = directions / torch.where(norms > 0, norms, torch.ones_like(norms))
    edges = []
    for i, j in pairs:
        a, b = source.elements[i], source.elements[j]
        group = isinstance(a.group_id, str) and a.group_id == b.group_id
        z_known = a.z_index is not ABSENT and b.z_index is not ABSENT
        z_sign = int(b.z_index > a.z_index) - int(b.z_index < a.z_index) if z_known else 0
        semantic = bool(masks[i, 0].item() and masks[j, 0].item())
        cosine = _clip(torch.dot(directions[i], directions[j]).item(), -1, 1) if semantic else 0.0
        src = _pair_geometry(src_boxes[i], src_boxes[j], source.source_canvas)
        cand = _pair_geometry(cand_boxes[i], cand_boxes[j], target)
        edges.append(src + [float(group), float(z_known), float(z_sign), cosine, float(semantic)]
                     + cand[:5] + cand[7:])
    available = int(masks.sum().item())
    status = ("no_text" if text_count == 0 else "none" if available == 0
              else "complete" if available == text_count else "partial")
    graph = Data(
        num_nodes=count,
        x_numeric=torch.tensor(nodes, dtype=torch.float32, device="cpu"),
        text_vec=vectors,
        text_mask=masks,
        edge_index=torch.tensor(pairs, dtype=torch.int64, device="cpu").reshape(-1, 2).t().contiguous(),
        edge_attr=torch.tensor(edges, dtype=torch.float32, device="cpu").reshape(-1, EDGE_DIMENSION),
        element_ids=tuple(element.id for element in source.elements),
        context_version=FEATURE_CONTRACT_VERSION,
        graph_topology="dense_v1" if count <= 64 else "sparse_v1",
        text_encoder_version=encoder_version,
        text_embedding_status=status,
    )
    for name in ("x_numeric", "text_vec", "text_mask", "edge_attr"):
        if not torch.isfinite(graph[name]).all().item():
            raise ValueError(f"{name} contains values outside finite float32 range")
    return graph
