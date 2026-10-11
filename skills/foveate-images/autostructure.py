"""Optional, deterministic image structure discovery; geometry stays source-global."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from itertools import pairwise
from time import perf_counter
from typing import Any, cast

from PIL import Image

DETECTOR_VERSION = "1.0.1"
POLICY_VERSION = "autostructure-v1.0"
DEFAULTS: dict[str, dict[str, float]] = {
    "hough": {
        "angle_tolerance_degrees": 3.0,
        "min_segment_length": 0.10,
        "max_line_gap": 0.02,
    },
    "gutters": {"min_gutter_width": 0.005},
    "panels": {"size_tolerance": 0.20, "alignment_tolerance": 0.02},
    "selection": {"min_score": 0.70, "min_margin": 0.08},
    "probe": {"max_side": 1024.0},
}


def validate_config(config: dict[str, Any] | None) -> dict[str, dict[str, float]]:
    """Reject unknown knobs and nonfinite values before accessing dependencies."""
    result = {group: dict(values) for group, values in DEFAULTS.items()}
    for group, values in (config or {}).items():
        if group in {"force_all_detectors", "collect_timings"}:
            if not isinstance(values, bool):
                raise ValueError(f"{group} must be a boolean")
            continue
        if group not in result or not isinstance(values, dict):
            raise ValueError(f"Unknown detector configuration group: {group}")
        for name, value in cast(dict[str, Any], values).items():
            if name not in result[group]:
                raise ValueError(f"Unknown detector parameter: {group}.{name}")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(
                    f"Detector parameter {group}.{name} must be finite numeric"
                )
            if name == "max_side":
                valid = 128 <= value <= 4096 and int(value) == value
            elif name == "angle_tolerance_degrees":
                valid = 0 < value <= 10
            elif name in {"min_score", "min_margin", "max_line_gap"}:
                valid = 0 <= value <= 1
            else:
                valid = 0 < value <= 1
            if not valid:
                raise ValueError(
                    f"Detector parameter {group}.{name} is outside its allowed range"
                )
            result[group][name] = float(value)
    return result


def _runs(values: Any) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(values):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(values)))
    return runs


def _iou(a: list[int], b: list[int]) -> float:
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / max(1, area_a + area_b - intersection)


def _foreground(rgb: Any, cv: Any, np: Any) -> Any:
    # Border color handles both light and dark page backgrounds. Canny retains faint rules.
    border = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]))
    quantized = (border // 16).astype(np.int32)
    packed = quantized[:, 0] * 256 + quantized[:, 1] * 16 + quantized[:, 2]
    mode = int(np.bincount(packed).argmax())
    chosen = border[packed == mode]
    background = np.median(chosen, axis=0)
    delta = np.max(np.abs(rgb.astype(np.float32) - background), axis=2)
    mask = (delta > 18).astype(np.uint8)
    gray = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
    edges = cv.Canny(gray, 18, 65)
    mask[edges > 0] = 1
    return mask


def _rectangles(rgb: Any, cv: Any, np: Any) -> list[list[int]]:
    height, width = rgb.shape[:2]
    gray = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
    edges = cv.Canny(gray, 15, 60)
    edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    boxes: list[list[int]] = []
    for contour in contours:
        x, y, w, h = map(int, cv.boundingRect(contour))
        if w < width * 0.07 or h < height * 0.035 or w * h > width * height * 0.80:
            continue
        if w * h < width * height * 0.003 or w / h > 8 or h / w > 8:
            continue
        area = abs(float(cv.contourArea(contour)))
        if area < w * h * 0.70:
            continue
        box = [x, y, x + w, y + h]
        if any(_iou(box, old) > 0.85 for old in boxes):
            continue
        boxes.append(box)
    boxes = [
        box
        for box in boxes
        if not any(
            other != box
            and other[0] <= box[0]
            and other[1] <= box[1]
            and other[2] >= box[2]
            and other[3] >= box[3]
            and _iou(box, other) > 0.65
            for other in boxes
        )
    ]
    return sorted(boxes, key=lambda box: (-(box[2] - box[0]) * (box[3] - box[1]), box))[
        :256
    ]


def _cluster(values: list[float], tolerance: float) -> list[float]:
    clusters: list[list[float]] = []
    for value in sorted(values):
        if clusters and value - sum(clusters[-1]) / len(clusters[-1]) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [sum(group) / len(group) for group in clusters]


def _panel_groups(
    boxes: list[list[int]], width: int, height: int, config: dict[str, dict[str, float]]
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    size_tolerance = config["panels"]["size_tolerance"]
    alignment = config["panels"]["alignment_tolerance"]
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for seed in boxes:
        sw, sh = seed[2] - seed[0], seed[3] - seed[1]
        similar = [
            box
            for box in boxes
            if abs((box[2] - box[0]) / sw - 1) <= size_tolerance
            and abs((box[3] - box[1]) / sh - 1) <= size_tolerance
        ]
        if len(similar) < 3 or sw / sh > 4.5 or sh / sw > 4.5:
            # Repeated header strips alone do not establish the panels below them.
            continue
        xs = _cluster(
            [(box[0] + box[2]) / 2 for box in similar], max(2, width * alignment)
        )
        ys = _cluster(
            [(box[1] + box[3]) / 2 for box in similar], max(2, height * alignment)
        )
        if len(xs) * len(ys) > len(similar) * 1.6 or len(xs) * len(ys) > 128:
            continue
        assigned: dict[tuple[int, int], list[int]] = {}
        for box in similar:
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            column = min(range(len(xs)), key=lambda index: abs(xs[index] - cx))
            row = min(range(len(ys)), key=lambda index: abs(ys[index] - cy))
            key = (row, column)
            old = assigned.get(key)
            if old is None or (box[2] - box[0]) * (box[3] - box[1]) > (
                old[2] - old[0]
            ) * (old[3] - old[1]):
                assigned[key] = box
        selected = [assigned[key] for key in sorted(assigned)]
        median_aspect = sorted((b[2] - b[0]) / max(1, b[3] - b[1]) for b in selected)[
            len(selected) // 2
        ]
        if median_aspect > 4.5 or median_aspect < 1 / 4.5:
            continue
        signature = tuple(tuple(box) for box in selected)
        if signature in seen or len(selected) < 3:
            continue
        seen.add(signature)
        # Distinct lattice cells must have nonoverlapping visible extents.
        if any(
            _iou(a, b) > 0.02
            for index, a in enumerate(selected)
            for b in selected[index + 1 :]
        ):
            continue
        gaps: list[float] = []
        for axis, centers in ((0, xs), (1, ys)):
            typical = sw if axis == 0 else sh
            gaps.extend(
                (b - a - typical) / max(1, typical) for a, b in pairwise(centers)
            )
        if gaps and min(gaps) < -0.15:
            continue
        completeness = len(selected) / (len(xs) * len(ys))
        groups.append(
            {"boxes": selected, "xs": xs, "ys": ys, "fit": 0.85 + 0.15 * completeness}
        )
    return sorted(
        groups,
        key=lambda group: (
            -sum((b[2] - b[0]) * (b[3] - b[1]) for b in group["boxes"]),
            -len(group["boxes"]),
        ),
    )[:8]


def _line_probe(
    rgb: Any, cv: Any, np: Any, settings: dict[str, dict[str, float]]
) -> dict[str, int]:
    gray = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
    edges = np.maximum(cv.Canny(gray, 12, 45), cv.Canny(gray, 28, 90))
    height, width = gray.shape
    measurements: dict[str, int] = {}
    kernels = (
        (
            "horizontal",
            (max(5, round(width * settings["hough"]["min_segment_length"])), 1),
        ),
        (
            "vertical",
            (1, max(5, round(height * settings["hough"]["min_segment_length"]))),
        ),
    )
    for name, kernel in kernels:
        directional = cv.morphologyEx(
            edges, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, kernel)
        )
        count, _, _, _ = cv.connectedComponentsWithStats(
            (directional > 0).astype(np.uint8), 8
        )
        measurements[name] = max(0, int(count) - 1)
    return measurements


def _segments(
    rgb: Any, cv: Any, np: Any, config: dict[str, dict[str, float]]
) -> tuple[list[list[int]], list[list[int]]]:
    height, width = rgb.shape[:2]
    gray = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
    edges = cv.Canny(gray, 18, 65)
    settings = config["hough"]
    horizontal: list[list[int]] = []
    vertical: list[list[int]] = []
    for fraction in (
        settings["min_segment_length"],
        max(0.035, settings["min_segment_length"] / 2),
    ):
        lines = cv.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=max(12, int(min(width, height) * fraction * 0.30)),
            minLineLength=max(8, int(min(width, height) * fraction)),
            maxLineGap=max(1, int(min(width, height) * settings["max_line_gap"])),
        )
        if lines is None:
            continue
        for item in lines:
            x1, y1, x2, y2 = map(int, np.asarray(item).reshape(-1))
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            angle = math.degrees(math.atan2(dy, dx))
            if angle <= settings["angle_tolerance_degrees"] and dx >= width * fraction:
                horizontal.append(
                    [
                        min(x1, x2),
                        round((y1 + y2) / 2),
                        max(x1, x2),
                        round((y1 + y2) / 2),
                    ]
                )
            elif (
                90 - angle <= settings["angle_tolerance_degrees"]
                and dy >= height * fraction
            ):
                vertical.append(
                    [
                        round((x1 + x2) / 2),
                        min(y1, y2),
                        round((x1 + x2) / 2),
                        max(y1, y2),
                    ]
                )

    def merge(items: list[list[int]], axis: int) -> list[list[int]]:
        merged: list[list[int]] = []
        fixed = 1 - axis
        for segment in sorted(items, key=lambda item: (item[fixed], item[axis])):
            for previous in merged:
                if (
                    abs(previous[fixed] - segment[fixed]) <= 3
                    and segment[axis]
                    <= previous[axis + 2]
                    + max(3, int(min(width, height) * settings["max_line_gap"]))
                    and previous[axis] <= segment[axis + 2] + 3
                ):
                    previous[axis] = min(previous[axis], segment[axis])
                    previous[axis + 2] = max(previous[axis + 2], segment[axis + 2])
                    break
            else:
                merged.append(list(segment))
        return merged[:1000]

    return merge(horizontal, 0), merge(vertical, 1)


def _border_support(
    box: list[int], horizontal: list[list[int]], vertical: list[list[int]]
) -> float:
    x1, y1, x2, y2 = box
    tolerance = max(3, round(min(x2 - x1, y2 - y1) * 0.025))
    supports: list[float] = []
    for y in (y1, y2):
        supports.append(
            max(
                (
                    max(0, min(x2, s[2]) - max(x1, s[0])) / max(1, x2 - x1)
                    for s in horizontal
                    if abs(s[1] - y) <= tolerance
                ),
                default=0.0,
            )
        )
    for x in (x1, x2):
        supports.append(
            max(
                (
                    max(0, min(y2, s[3]) - max(y1, s[1])) / max(1, y2 - y1)
                    for s in vertical
                    if abs(s[0] - x) <= tolerance
                ),
                default=0.0,
            )
        )
    return sum(supports) / 4


def _grid(bounds: list[int], xs: list[int], ys: list[int]) -> dict[str, Any]:
    return {"type": "grid", "bounds_px": list(bounds), "x_edges": xs, "y_edges": ys}


def _grid_tree(
    partitions: dict[str, Any],
    anchor: str,
    bounds: list[int],
    xs: list[int],
    ys: list[int],
    depth: int = 0,
) -> bool:
    rows, columns = len(ys) - 1, len(xs) - 1
    if rows * columns <= 16:
        partitions[anchor] = _grid(bounds, xs, ys)
        return True
    if depth >= 2 or rows > 16:
        return False
    partitions[anchor] = _grid(bounds, [xs[0], xs[-1]], ys)
    for row in range(rows):
        child = f"{anchor}/A{row + 1}"
        if not _grid_tree(
            partitions,
            child,
            [xs[0], ys[row], xs[-1], ys[row + 1]],
            xs,
            [ys[row], ys[row + 1]],
            depth + 1,
        ):
            return False
    return True


def _group_geometry(
    group: dict[str, Any], view: list[int], scale_x: float, scale_y: float
) -> tuple[dict[str, Any], list[list[int]]] | None:
    boxes = [
        [
            view[0] + round(b[0] * scale_x),
            view[1] + round(b[1] * scale_y),
            view[0] + round(b[2] * scale_x),
            view[1] + round(b[3] * scale_y),
        ]
        for b in group["boxes"]
    ]
    centers_x = [view[0] + round(x * scale_x) for x in group["xs"]]
    centers_y = [view[1] + round(y * scale_y) for y in group["ys"]]
    top, bottom = min(b[1] for b in boxes), max(b[3] for b in boxes)
    median_w = sorted(b[2] - b[0] for b in boxes)[len(boxes) // 2]
    median_h = sorted(b[3] - b[1] for b in boxes)[len(boxes) // 2]
    touching = (
        len(centers_x) > 1
        and len(centers_y) > 1
        and all(
            abs(b - a - median_w) <= max(5, scale_x * 3) for a, b in pairwise(centers_x)
        )
        and all(
            abs(b - a - median_h) <= max(5, scale_y * 3) for a, b in pairwise(centers_y)
        )
    )
    left = min(b[0] for b in boxes) if touching else view[0]
    right = max(b[2] for b in boxes) if touching else view[2]
    x_edges = [left] + [(a + b) // 2 for a, b in pairwise(centers_x)] + [right]
    y_edges = [top] + [(a + b) // 2 for a, b in pairwise(centers_y)] + [bottom]
    if any(a >= b for edges in (x_edges, y_edges) for a, b in pairwise(edges)):
        return None
    partitions: dict[str, Any] = {}
    children: list[dict[str, Any]] = []
    strips: list[list[int]] = []
    if top > view[1]:
        strips.append([view[0], view[1], view[2], top])
    if left > view[0]:
        strips.append([view[0], top, left, bottom])
    body_index = len(strips)
    body = [left, top, right, bottom]
    strips.append(body)
    if right < view[2]:
        strips.append([right, top, view[2], bottom])
    if bottom < view[3]:
        strips.append([view[0], bottom, view[2], view[3]])
    if len(strips) == 1:
        anchor = "ROOT"
    else:
        children = [
            {"segment": f"R{index + 1}", "core_bounds_px": bounds}
            for index, bounds in enumerate(strips)
        ]
        partitions["ROOT"] = {
            "type": "regions",
            "bounds_px": list(view),
            "children": children,
        }
        anchor = f"ROOT/R{body_index + 1}"
    if not _grid_tree(partitions, anchor, body, x_edges, y_edges):
        return None
    return partitions, boxes


def _boundary_support(
    partitions: dict[str, Any],
    mask: Any,
    view: list[int],
    np: Any,
    edge_mask: Any = None,
) -> tuple[float, float]:
    supports: list[float] = []
    clipping: list[float] = []
    height, width = mask.shape

    def strip(bounds: list[int], axis: int, edge: int) -> None:
        x1, y1, x2, y2 = bounds
        x1, x2 = max(0, x1 - view[0]), min(width, x2 - view[0])
        y1, y2 = max(0, y1 - view[1]), min(height, y2 - view[1])
        local = edge - view[axis]
        if axis == 0:
            values = mask[y1:y2, max(0, local - 1) : min(width, local + 2)]
        else:
            values = mask[max(0, local - 1) : min(height, local + 2), x1:x2]
        if values.size:
            occupied = float(np.mean(values))
            line_coverage = 0.0
            if edge_mask is not None:
                if axis == 0:
                    edge_values = edge_mask[
                        y1:y2, max(0, local - 2) : min(width, local + 3)
                    ]
                    line_coverage = (
                        float(np.mean(np.any(edge_values, axis=1)))
                        if edge_values.size
                        else 0.0
                    )
                else:
                    edge_values = edge_mask[
                        max(0, local - 2) : min(height, local + 3), x1:x2
                    ]
                    line_coverage = (
                        float(np.mean(np.any(edge_values, axis=0)))
                        if edge_values.size
                        else 0.0
                    )
            supports.append(max(1 - occupied, line_coverage))
            clipping.append(occupied * (1 - line_coverage))

    for partition in partitions.values():
        bounds = partition["bounds_px"]
        if partition["type"] == "grid":
            for x in partition["x_edges"][1:-1]:
                strip(bounds, 0, x)
            for y in partition["y_edges"][1:-1]:
                strip(bounds, 1, y)
        else:
            edges: set[tuple[int, int]] = set()
            for child in partition["children"]:
                b = child["core_bounds_px"]
                for axis, edge in ((0, b[0]), (0, b[2]), (1, b[1]), (1, b[3])):
                    if bounds[axis] < edge < bounds[axis + 2]:
                        edges.add((axis, edge))
            for axis, edge in sorted(edges):
                strip(bounds, axis, edge)
    return sum(supports) / len(supports) if supports else 0.5, 1 - sum(clipping) / len(
        clipping
    ) if clipping else 0.5


def candidate_id(source_id: str, view: list[int], partitions: dict[str, Any]) -> str:
    """Bind a replayable candidate to its source, view, and canonical geometry."""
    canonical = json.dumps(
        {"source_id": source_id, "view": view, "partitions": partitions},
        sort_keys=True,
        separators=(",", ":"),
    )
    return "structure-" + hashlib.sha256(canonical.encode()).hexdigest()[:20]


def _candidate(
    method: str,
    partitions: dict[str, Any],
    evidence: list[list[int]],
    fit: float,
    boundary: float,
    integrity: float,
    reason: str,
    source_id: str,
    view: list[int],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    leaves = (
        sum(
            (len(p["x_edges"]) - 1) * (len(p["y_edges"]) - 1)
            if p["type"] == "grid"
            else len(p["children"])
            for p in partitions.values()
        )
        - len(partitions)
        + 1
    )
    useful = min(1.0, max(0.0, (leaves - 1) / 3))
    if method == "gutters":
        useful *= 0.60
    area = sum((b[2] - b[0]) * (b[3] - b[1]) for b in evidence)
    view_area = (view[2] - view[0]) * (view[3] - view[1])
    if (extra or {}).get("panel_count"):
        useful *= min(1.0, area / max(1, view_area * 0.50))
    elif method == "hough":
        useful *= min(1.0, area / max(1, view_area * 0.15))
    penalty = max(0, leaves - 16) * 0.002 + max(0, len(partitions) - 4) * 0.005
    features: dict[str, Any] = {
        "boundary_support": round(boundary, 6),
        "structure_fit": round(fit, 6),
        "content_integrity": round(integrity, 6),
        "navigation_usefulness": round(useful, 6),
        "complexity_penalty": round(penalty, 6),
        "leaf_count": leaves,
        "source_resolution_verified": True,
    }
    features.update(extra or {})
    score = max(
        0,
        min(
            1, 0.30 * boundary + 0.25 * fit + 0.25 * integrity + 0.20 * useful - penalty
        ),
    )
    return {
        "id": candidate_id(source_id, view, partitions),
        "method": method,
        "methods": [method],
        "score": round(score, 6),
        "raw_score": round(fit, 6),
        "features": features,
        "partitions": partitions,
        "evidence_bounds_px": evidence,
        "reason": reason,
        "rejection_reason": None,
    }


def _gutter_candidate(
    mask: Any,
    view: list[int],
    config: dict[str, dict[str, float]],
    np: Any,
    source_id: str,
) -> dict[str, Any] | None:
    partitions: dict[str, Any] = {}
    evidence: list[list[int]] = []
    quality: list[float] = []
    height, width = mask.shape
    count, _, stats, _ = importlib.import_module("cv2").connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )
    nonrule_area = sum(
        int(stats[i, 4])
        for i in range(1, count)
        if not (
            int(stats[i, 2]) > 10 * max(1, int(stats[i, 3]))
            and int(stats[i, 3]) < max(4, height * 0.02)
        )
        and not (
            int(stats[i, 3]) > 10 * max(1, int(stats[i, 2]))
            and int(stats[i, 2]) < max(4, width * 0.02)
        )
    )
    if nonrule_area < float(mask.sum()) * 0.10:
        return None
    component_heights = [
        int(stats[i, 3])
        for i in range(1, count)
        if int(stats[i, 4]) >= 4 and int(stats[i, 3]) < height * 0.12
    ]
    component_height = float(np.median(component_heights)) if component_heights else 3.0

    def add_evidence(bounds: list[int], local: Any) -> None:
        ys, xs = np.nonzero(local)
        if len(xs):
            evidence.append(
                [
                    bounds[0] + int(xs.min()),
                    bounds[1] + int(ys.min()),
                    bounds[0] + int(xs.max()) + 1,
                    bounds[1] + int(ys.max()) + 1,
                ]
            )

    def recurse(bounds: list[int], anchor: str, depth: int) -> None:
        x1, y1, x2, y2 = bounds
        local = mask[y1 - view[1] : y2 - view[1], x1 - view[0] : x2 - view[0]]
        if depth >= 3 or min(local.shape) < 12:
            add_evidence(bounds, local)
            return
        total = float(local.sum())
        if total < 16:
            add_evidence(bounds, local)
            return
        options: list[tuple[float, int, int, float]] = []
        for axis in (0, 1):
            profile = np.mean(local, axis=0 if axis == 0 else 1)
            length = len(profile)
            # Ordinary text leading is smaller than estimated glyph/component height.
            min_width = max(
                3,
                round(length * config["gutters"]["min_gutter_width"]),
                round(component_height * (1.8 if axis == 1 else 1.2)),
            )
            low = profile <= min(0.025, max(0.003, float(np.percentile(profile, 20))))
            for a, b in _runs(low):
                if b - a < min_width or a == 0 or b == length:
                    continue
                before = float(np.sum(local[:, :a] if axis == 0 else local[:a, :]))
                after = float(np.sum(local[:, b:] if axis == 0 else local[b:, :]))
                balance = min(before, after) / total
                if balance < 0.06:
                    continue
                gap = (b - a) / length
                occupied = float(np.mean(profile[a:b]))
                score = min(1, gap * 8) + balance * 1.2 + (1 - occupied) * 0.15
                options.append((score, axis, (a + b) // 2, 1 - occupied))
        if not options:
            add_evidence(bounds, local)
            return
        score, axis, _cut, _support = max(
            options, key=lambda item: (item[0], -item[1], -item[2])
        )
        if score < 0.30:
            add_evidence(bounds, local)
            return
        supported = sorted(
            (option for option in options if option[1] == axis and option[0] >= 0.30),
            key=lambda option: option[2],
        )
        cuts = (
            [bounds[axis]]
            + [option[2] + bounds[axis] for option in supported[:15]]
            + [bounds[axis + 2]]
        )
        children: list[dict[str, Any]] = []
        for index, (a, b) in enumerate(pairwise(cuts)):
            child = list(bounds)
            child[axis], child[axis + 2] = a, b
            children.append({"segment": f"R{index + 1}", "core_bounds_px": child})
        partitions[anchor] = {
            "type": "regions",
            "bounds_px": list(bounds),
            "children": children,
        }
        quality.extend(option[3] for option in supported[:15])
        for child in children:
            recurse(child["core_bounds_px"], anchor + "/" + child["segment"], depth + 1)

    recurse(view, "ROOT", 0)
    if not partitions:
        return None
    boundary, integrity = _boundary_support(partitions, mask, view, np)
    # A purely binary text-column split earns less structure evidence than repeated blocks.
    fit = min(0.65, 0.55 + len(partitions) * 0.015)
    return _candidate(
        "gutters",
        partitions,
        evidence,
        fit,
        boundary,
        integrity,
        f"{len(partitions)} recursive splits at broad, low-occupancy gutters",
        source_id,
        view,
        {"component_height_px": component_height, "gutter_count": len(quality)},
    )


def _line_grid(
    horizontal: list[list[int]],
    vertical: list[list[int]],
    view: list[int],
    sx: float,
    sy: float,
    mask: Any,
    np: Any,
    source_id: str,
) -> dict[str, Any] | None:
    # Require each family to support the same bounded rectangle; never extend unrelated lines globally.
    if len(horizontal) < 3 or len(vertical) < 3:
        return None
    best: tuple[float, list[int], list[int], list[int]] | None = None
    for h in horizontal:
        matching = [
            s for s in horizontal if abs(s[0] - h[0]) <= 5 and abs(s[2] - h[2]) <= 5
        ]
        if len(matching) < 3:
            continue
        ys = [round(value) for value in _cluster([float(s[1]) for s in matching], 3)]
        top, bottom = min(ys), max(ys)
        vs = [
            s
            for s in vertical
            if s[0] >= h[0] - 5
            and s[0] <= h[2] + 5
            and s[1] <= top + 5
            and s[3] >= bottom - 5
        ]
        xs = [round(value) for value in _cluster([float(s[0]) for s in vs], 3)]
        if len(xs) < 3 or len(ys) < 3:
            continue
        box = [min(xs), top, max(xs), bottom]
        support = _border_support(box, horizontal, vertical)
        area = (box[2] - box[0]) * (box[3] - box[1])
        if support < 0.7:
            continue
        value = area * support
        if best is None or value > best[0]:
            best = (value, box, xs, ys)
    if best is None:
        return None
    _, box, xs, ys = best
    bounds = [
        view[0] + round(box[0] * sx),
        view[1] + round(box[1] * sy),
        view[0] + round(box[2] * sx),
        view[1] + round(box[3] * sy),
    ]
    # Restrict frame margins by retaining them as exact rectangular residual strips.
    partitions: dict[str, Any] = {}
    children: list[dict[str, Any]] = []
    residual: list[list[int]] = []
    l, t, r, b = bounds
    if t > view[1]:
        residual.append([view[0], view[1], view[2], t])
    if l > view[0]:
        residual.append([view[0], t, l, b])
    body_index = len(residual)
    residual.append(bounds)
    if r < view[2]:
        residual.append([r, t, view[2], b])
    if b < view[3]:
        residual.append([view[0], b, view[2], view[3]])
    if len(residual) == 1:
        anchor = "ROOT"
    else:
        children = [
            {"segment": f"R{i + 1}", "core_bounds_px": bound}
            for i, bound in enumerate(residual)
        ]
        partitions["ROOT"] = {
            "type": "regions",
            "bounds_px": view,
            "children": children,
        }
        anchor = f"ROOT/R{body_index + 1}"
    source_x = [view[0] + round(x * sx) for x in xs]
    source_y = [view[1] + round(y * sy) for y in ys]
    if not _grid_tree(partitions, anchor, bounds, source_x, source_y):
        return None
    cells = [
        [source_x[c], source_y[row], source_x[c + 1], source_y[row + 1]]
        for row in range(len(source_y) - 1)
        for c in range(len(source_x) - 1)
    ]
    source_supports: list[float] = []
    for axis, edges in ((0, source_x), (1, source_y)):
        for coordinate in edges:
            local = coordinate - view[axis]
            if axis == 0:
                strip = mask[
                    max(0, t - view[1]) : min(mask.shape[0], b - view[1]),
                    max(0, local - 3) : min(mask.shape[1], local + 4),
                ]
                coverage = float(np.mean(np.any(strip, axis=1))) if strip.size else 0.0
            else:
                strip = mask[
                    max(0, local - 3) : min(mask.shape[0], local + 4),
                    max(0, l - view[0]) : min(mask.shape[1], r - view[0]),
                ]
                coverage = float(np.mean(np.any(strip, axis=0))) if strip.size else 0.0
            source_supports.append(coverage)
    source_support = sum(source_supports) / len(source_supports)
    if source_support < 0.70:
        return None
    # Observed rules are expected foreground at ownership cuts; source edge coverage verifies each cut.
    return _candidate(
        "hough",
        partitions,
        cells,
        0.94,
        source_support,
        source_support,
        f"{len(xs)} vertical and {len(ys)} horizontal supported local grid boundaries",
        source_id,
        view,
        {
            "observed_x_boundaries": source_x,
            "observed_y_boundaries": source_y,
            "inferred_boundaries": [],
        },
    )


def _valid_geometry(partitions: dict[str, Any], view: list[int]) -> bool:
    if "ROOT" not in partitions or partitions["ROOT"]["bounds_px"] != view:
        return False
    owned: dict[str, list[int]] = {"ROOT": view}
    for anchor in sorted(partitions, key=lambda key: (key.count("/"), key)):
        p = partitions[anchor]
        bounds = p["bounds_px"]
        if anchor.count("/") > 3 or owned.get(anchor) != bounds:
            return False
        children: list[tuple[str, list[int]]] = []
        if p["type"] == "grid":
            xs, ys = p["x_edges"], p["y_edges"]
            if len(xs) < 2 or len(ys) < 2 or (len(xs) - 1) * (len(ys) - 1) > 16:
                return False
            if (
                xs[0] != bounds[0]
                or xs[-1] != bounds[2]
                or ys[0] != bounds[1]
                or ys[-1] != bounds[3]
            ):
                return False
            if any(a >= b for edges in (xs, ys) for a, b in pairwise(edges)):
                return False
            children = [
                (f"{chr(65 + c)}{r + 1}", [xs[c], ys[r], xs[c + 1], ys[r + 1]])
                for r in range(len(ys) - 1)
                for c in range(len(xs) - 1)
            ]
        elif p["type"] == "regions":
            children = [
                (child["segment"], child["core_bounds_px"]) for child in p["children"]
            ]
            if not 1 <= len(children) <= 16 or len(
                {segment for segment, _ in children}
            ) != len(children):
                return False
        else:
            return False
        boxes = [b for _, b in children]
        if any(
            not (
                bounds[0] <= b[0] < b[2] <= bounds[2]
                and bounds[1] <= b[1] < b[3] <= bounds[3]
            )
            for b in boxes
        ):
            return False
        if any(
            min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])
            for i, a in enumerate(boxes)
            for b in boxes[i + 1 :]
        ):
            return False
        if sum((b[2] - b[0]) * (b[3] - b[1]) for b in boxes) != (
            bounds[2] - bounds[0]
        ) * (bounds[3] - bounds[1]):
            return False
        for segment, b in children:
            owned[anchor + "/" + segment] = b
    return True


def _equivalent(first: dict[str, Any], second: dict[str, Any]) -> bool:
    if first["partitions"] == second["partitions"]:
        return True
    # Parent topology must agree; corresponding evidence boxes alone are insufficient.
    if sorted(first["partitions"]) != sorted(second["partitions"]):
        return False
    for anchor, p in first["partitions"].items():
        q = second["partitions"][anchor]
        if p["type"] != q["type"]:
            return False
        if p["type"] == "grid":
            if len(p["x_edges"]) != len(q["x_edges"]) or len(p["y_edges"]) != len(
                q["y_edges"]
            ):
                return False
        elif len(p["children"]) != len(q["children"]):
            return False

        def cores(partition: dict[str, Any]) -> list[list[int]]:
            if partition["type"] == "regions":
                return [child["core_bounds_px"] for child in partition["children"]]
            xs, ys = partition["x_edges"], partition["y_edges"]
            return [
                [xs[c], ys[r], xs[c + 1], ys[r + 1]]
                for r in range(len(ys) - 1)
                for c in range(len(xs) - 1)
            ]

        core_scores = [_iou(a, b) for a, b in zip(cores(p), cores(q), strict=True)]
        if not core_scores or sum(core_scores) / len(core_scores) < 0.90:
            return False
    a, b = first["evidence_bounds_px"], second["evidence_bounds_px"]
    if len(a) != len(b) or not a:
        return False
    pairs = sorted((_iou(x, y), i, j) for i, x in enumerate(a) for j, y in enumerate(b))
    used_a: set[int] = set()
    used_b: set[int] = set()
    scores: list[float] = []
    for score, i, j in reversed(pairs):
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            scores.append(score)
    return bool(scores) and sum(scores) / len(scores) >= 0.9


def discover(
    source: Image.Image,
    view: list[int],
    source_id: str,
    mode: str = "auto",
    detectors: tuple[str, ...] = ("hough", "gutters", "panels"),
    config: dict[str, Any] | None = None,
    min_score: float | None = None,
) -> dict[str, Any]:
    """Return complete proposals and selection diagnostics, without changing the raster."""
    started = perf_counter()
    settings = validate_config(config)
    if mode not in {"auto", "hough", "gutters", "panels"}:
        raise ValueError(f"Unsupported discovery mode: {mode}")
    if (
        not detectors
        or len(set(detectors)) != len(detectors)
        or any(d not in {"hough", "gutters", "panels"} for d in detectors)
    ):
        raise ValueError(
            "detectors must contain distinct hough, gutters, or panels names"
        )
    if (
        len(view) != 4
        or any(isinstance(v, bool) or type(v) is not int for v in view)
        or not (
            0 <= view[0] < view[2] <= source.width
            and 0 <= view[1] < view[3] <= source.height
        )
    ):
        raise ValueError("Discovery view must be a nonempty source-global rectangle")
    if min_score is not None:
        if (
            isinstance(min_score, bool)
            or type(min_score) not in (float, int)
            or not math.isfinite(min_score)
            or not 0 <= min_score <= 1
        ):
            raise ValueError("min_score must be between zero and one")
        settings["selection"]["min_score"] = float(min_score)
    report: dict[str, Any] = {
        "version": 1,
        "source_id": source_id,
        "view": list(view),
        "view_bounds_px": list(view),
        "mode": mode,
        "enabled_detectors": list(detectors),
        "parameter_overrides": config or {},
        "parameters": settings,
        "min_score_override": min_score,
        "detector_version": DETECTOR_VERSION,
        "policy_version": POLICY_VERSION,
        "backend_versions": {},
        "probes": {},
        "detector_runs": [],
        "candidates": [],
        "selected_candidate_id": None,
        "fallback_reason": None,
    }
    backend_started = perf_counter()
    try:
        cv = importlib.import_module("cv2")
        np = importlib.import_module("numpy")
    except ImportError:
        guidance = "Automatic structure discovery requires OpenCV and NumPy; install the detection extra (pip install 'foveate-images[detection]')."
        if mode != "auto":
            raise ValueError(guidance) from None
        report["fallback_reason"] = guidance
        report["detector_runs"] = [
            {"method": d, "status": "unavailable", "reason": guidance}
            for d in detectors
        ]
        return report
    backend_ms = (perf_counter() - backend_started) * 1000
    preprocess_started = perf_counter()
    report["backend_versions"] = {
        "opencv": str(cv.__version__),
        "numpy": str(np.__version__),
        "pillow": str(importlib.import_module("PIL").__version__),
    }
    # OpenCV's RNG is fixed; proposal sorting makes sequential runs independent of execution order.
    cv.setRNGSeed(0)
    crop = source.crop((view[0], view[1], view[2], view[3]))
    has_alpha = (
        any(band.upper() == "A" for band in crop.getbands())
        or "transparency" in crop.info
    )
    if has_alpha:
        # Detect visible pixels on a white canvas; hidden RGB must not become structure.
        # Only the analysis raster is composited. Original source and raw crops retain alpha.
        crop = Image.alpha_composite(
            Image.new("RGBA", crop.size, "white"), crop.convert("RGBA")
        ).convert("RGB")
    else:
        crop = crop.convert("RGB")
    report["analysis_rendering"] = {
        "alpha_background_rgb": [255, 255, 255] if has_alpha else None
    }
    max_side = int(settings["probe"]["max_side"])
    probe = crop.copy()
    probe.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    rgb = np.asarray(probe)
    full = np.asarray(crop)
    sx, sy = crop.width / probe.width, crop.height / probe.height
    mask = _foreground(rgb, cv, np)
    full_mask = _foreground(full, cv, np)
    source_edge = cv.Canny(cv.cvtColor(full, cv.COLOR_RGB2GRAY), 15, 60) > 0
    preprocess_ms = (perf_counter() - preprocess_started) * 1000
    probe_started = perf_counter()
    boxes = _rectangles(rgb, cv, np)
    groups = _panel_groups(boxes, probe.width, probe.height, settings)
    horizontal, vertical = _segments(rgb, cv, np, settings)
    gutter_bands: dict[str, list[list[int]]] = {"horizontal": [], "vertical": []}
    for axis, name in ((0, "vertical"), (1, "horizontal")):
        profile = np.mean(mask, axis=0 if axis == 0 else 1)
        length = len(profile)
        for a, b in _runs(profile <= 0.025):
            if (
                b - a >= max(3, round(length * settings["gutters"]["min_gutter_width"]))
                and a > 0
                and b < length
                and float(profile[:a].sum()) > 1
                and float(profile[b:].sum()) > 1
            ):
                gutter_bands[name].append([a, b])
    morphology_probe = _line_probe(rgb, cv, np, settings)
    line_probe = (len(horizontal) >= 2 and len(vertical) >= 2) or (
        morphology_probe["horizontal"] >= 2 and morphology_probe["vertical"] >= 2
    )
    gutter_probe = bool(gutter_bands["horizontal"] or gutter_bands["vertical"])
    panel_probe = bool(groups)
    report["probes"] = {
        "size_px": [probe.width, probe.height],
        "source_transform": {
            "scale_x": sx,
            "scale_y": sy,
            "offset_x": view[0],
            "offset_y": view[1],
        },
        "directional_morphology_segments": morphology_probe,
        "horizontal_segments": len(horizontal),
        "vertical_segments": len(vertical),
        "hough_supported": line_probe,
        "gutter_bands": gutter_bands,
        "gutters_supported": gutter_probe,
        "rectangles": len(boxes),
        "repeated_groups": len(groups),
        "panels_supported": panel_probe,
        "foreground_fraction": round(float(np.mean(mask)), 6),
    }
    # Thin lines may vanish under reduction. Retry when another structured-content probe fires.
    if sx > 1 and (gutter_probe or panel_probe) and not line_probe:
        horizontal, vertical = _segments(full, cv, np, settings)
        if len(horizontal) >= 2 and len(vertical) >= 2:
            report["probes"]["hough_source_retry"] = True
            line_probe = True
            # Convert source segments back to probe coordinates for common proposal arithmetic.
            horizontal = [
                [round(s[0] / sx), round(s[1] / sy), round(s[2] / sx), round(s[3] / sy)]
                for s in horizontal
            ]
            vertical = [
                [round(s[0] / sx), round(s[1] / sy), round(s[2] / sx), round(s[3] / sy)]
                for s in vertical
            ]
    probe_ms = (perf_counter() - probe_started) * 1000
    detector_times: dict[str, float] = {}
    enabled = detectors if mode == "auto" else (mode,)
    gates = {"hough": line_probe, "gutters": gutter_probe, "panels": panel_probe}
    proposals: list[dict[str, Any]] = []
    for detector in enabled:
        if (
            mode == "auto"
            and not gates[detector]
            and not (config or {}).get("force_all_detectors", False)
        ):
            report["detector_runs"].append(
                {
                    "method": detector,
                    "status": "skipped",
                    "reason": "probe found insufficient supporting evidence",
                }
            )
            continue
        detector_started = perf_counter()
        before = len(proposals)
        if detector in {"panels", "hough"}:
            for group in groups:
                line_support = sum(
                    _border_support(b, horizontal, vertical) for b in group["boxes"]
                ) / len(group["boxes"])
                if detector == "hough" and line_support < 0.50:
                    continue
                geometry = _group_geometry(group, view, sx, sy)
                if geometry is None:
                    continue
                partitions, evidence = geometry
                boundary, integrity = _boundary_support(
                    partitions, full_mask, view, np, source_edge
                )
                # Verify original-resolution panel outlines rather than accepting only rescaled contours.
                edge = source_edge
                verified: list[float] = []
                for b in evidence:
                    x1, y1, x2, y2 = (
                        b[0] - view[0],
                        b[1] - view[1],
                        b[2] - view[0],
                        b[3] - view[1],
                    )
                    radius = max(2, round(min(x2 - x1, y2 - y1) * 0.015))
                    htop = edge[
                        max(0, y1 - radius) : min(crop.height, y1 + radius + 1), x1:x2
                    ]
                    hbottom = edge[
                        max(0, y2 - radius) : min(crop.height, y2 + radius + 1), x1:x2
                    ]
                    vleft = edge[
                        y1:y2, max(0, x1 - radius) : min(crop.width, x1 + radius + 1)
                    ]
                    vright = edge[
                        y1:y2, max(0, x2 - radius) : min(crop.width, x2 + radius + 1)
                    ]
                    strengths = [
                        float(np.mean(np.any(a, axis=0 if i < 2 else 1)))
                        if a.size
                        else 0
                        for i, a in enumerate((htop, hbottom, vleft, vright))
                    ]
                    verified.append(sum(strengths) / 4)
                source_support = sum(verified) / len(verified)
                if source_support < 0.40:
                    continue
                extra = {
                    "panel_count": len(evidence),
                    "rows": len(group["ys"]),
                    "columns": len(group["xs"]),
                    "source_border_support": round(source_support, 6),
                    "segment_support": round(line_support, 6),
                    "inferred_boundaries": "gutter midpoints; page residual strips",
                }
                proposals.append(
                    _candidate(
                        detector,
                        partitions,
                        evidence,
                        group["fit"],
                        boundary,
                        integrity,
                        f"{len(evidence)} similarly sized panels aligned in {len(group['ys'])} rows and {len(group['xs'])} columns",
                        source_id,
                        view,
                        extra,
                    )
                )
            if detector == "hough":
                grid = _line_grid(
                    horizontal, vertical, view, sx, sy, source_edge, np, source_id
                )
                if grid is not None:
                    proposals.append(grid)
        else:
            gutter = _gutter_candidate(full_mask, view, settings, np, source_id)
            if gutter is not None:
                proposals.append(gutter)
        detector_times[detector] = round((perf_counter() - detector_started) * 1000, 3)
        report["detector_runs"].append(
            {
                "method": detector,
                "status": "ran",
                "reason": "explicit mode"
                if mode != "auto"
                else "probe found supporting evidence",
                "candidate_count": len(proposals) - before,
            }
        )
    selection_started = perf_counter()
    rejected: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []
    for proposal in proposals:
        if _valid_geometry(proposal["partitions"], view):
            valid.append(proposal)
        else:
            proposal["rejection_reason"] = (
                "Invalid ownership partition or navigation limits"
            )
            rejected.append(proposal)
    proposals = valid
    report["rejected_candidates"] = rejected
    # Prefer candidates that explain the complete coarse repeated-panel extent.
    panel_proposals = [p for p in proposals if p["features"].get("panel_count", 0) >= 3]
    if panel_proposals:
        max_area = max(
            sum((b[2] - b[0]) * (b[3] - b[1]) for b in p["evidence_bounds_px"])
            for p in panel_proposals
        )
        for p in panel_proposals:
            area = sum((b[2] - b[0]) * (b[3] - b[1]) for b in p["evidence_bounds_px"])
            ratio = min(1.0, area / max(1, max_area))
            before = p["features"]["navigation_usefulness"]
            after = before * ratio * ratio
            p["features"]["navigation_usefulness"] = round(after, 6)
            p["features"]["coarse_extent_ratio"] = round(ratio, 6)
            p["score"] = round(max(0, p["score"] - 0.20 * (before - after)), 6)
    # Coarse panel proposals dominate their internal grids as first-pass navigation.
    coarse = [p for p in proposals if p["features"].get("panel_count", 0) >= 3]
    for proposal in proposals:
        if (
            proposal["method"] == "hough"
            and "panel_count" not in proposal["features"]
            and coarse
        ):
            evidence = proposal["evidence_bounds_px"]
            for parent in coarse:
                area = sum((b[2] - b[0]) * (b[3] - b[1]) for b in evidence)
                parent_area = sum(
                    (b[2] - b[0]) * (b[3] - b[1]) for b in parent["evidence_bounds_px"]
                )
                if area < parent_area * 0.5:
                    proposal["score"] = round(proposal["score"] * 0.70, 6)
                    proposal["reason"] += (
                        "; local detail is less useful than the supported coarse panel layout"
                    )
                    break
    merged: list[dict[str, Any]] = []
    for proposal in sorted(
        proposals, key=lambda p: (-p["score"], p["id"], p["method"])
    ):
        equivalent = next((old for old in merged if _equivalent(old, proposal)), None)
        if equivalent is None:
            proposal["supporting_evidence"] = {
                proposal["method"]: {
                    "features": proposal["features"],
                    "raw_score": proposal["raw_score"],
                    "reason": proposal["reason"],
                }
            }
            merged.append(proposal)
        else:
            equivalent["methods"] = sorted(
                set(equivalent["methods"] + [proposal["method"]])
            )
            equivalent["supporting_evidence"][proposal["method"]] = {
                "features": proposal["features"],
                "raw_score": proposal["raw_score"],
                "reason": proposal["reason"],
            }
    merged.sort(key=lambda p: (-p["score"], p["id"]))
    report["candidates"] = merged
    threshold = settings["selection"]["min_score"]
    for candidate in merged:
        if candidate["features"]["navigation_usefulness"] < 0.20:
            candidate["rejection_reason"] = (
                "Structure covers too little substantial content for useful first-pass navigation"
            )
        elif candidate["score"] < threshold:
            candidate["rejection_reason"] = (
                f"score {candidate['score']:.3f} below acceptance threshold {threshold:.3f}"
            )
    if not merged:
        report["fallback_reason"] = "No supported, valid structure was found"
    elif merged[0]["rejection_reason"]:
        report["fallback_reason"] = merged[0]["rejection_reason"]
    elif (
        mode == "auto"
        and len(merged) > 1
        and merged[0]["score"] - merged[1]["score"]
        < settings["selection"]["min_margin"]
    ):
        report["fallback_reason"] = (
            "Competing structures are ambiguous: score lead is below the selection margin"
        )
    else:
        report["selected_candidate_id"] = merged[0]["id"]
    report["summary"] = {
        "alternatives": [
            {
                "id": p["id"],
                "methods": p["methods"],
                "score": p["score"],
                "reason": p["reason"],
            }
            for p in merged[:3]
        ],
        "grid_option": True,
        "selection_reason": merged[0]["reason"]
        if report["selected_candidate_id"]
        else report["fallback_reason"],
    }
    timings_ms = {
        "backend": round(backend_ms, 3),
        "preprocessing": round(preprocess_ms, 3),
        "probes": round(probe_ms, 3),
        "detectors": detector_times,
        "selection": round((perf_counter() - selection_started) * 1000, 3),
        "total": round((perf_counter() - started) * 1000, 3),
    }
    if (config or {}).get("collect_timings", False):
        report["timings_ms"] = timings_ms
    if mode != "auto" and report["selected_candidate_id"] is None:
        raise ValueError(f"{mode} discovery failed: {report['fallback_reason']}")
    return report
