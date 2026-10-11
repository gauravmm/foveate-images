# /// script
# requires-python = ">=3.10"
# dependencies = ["Pillow>=10.2"]
# ///
"""Deterministic, string-addressed image crops for vision agents."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import sys
from collections.abc import Iterable
from copy import deepcopy
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

from PIL import Image, ImageDraw, ImageFont, ImageOps

GEOMETRY_VERSION = 1
STRUCTURE_GEOMETRY_VERSION = 2
MAX_TILES = 16
MAX_DEPTH = 4
FONT_PATHS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
)


def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for path in FONT_PATHS:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _validate_grid(rows: int, columns: int) -> None:
    if (
        type(rows) is not int
        or type(columns) is not int
        or rows < 1
        or columns < 1
        or rows * columns > MAX_TILES
    ):
        raise ValueError(f"grid must contain 1..{MAX_TILES} tiles")


def parse_bounds(value: str) -> tuple[int, int, int, int]:
    try:
        bounds = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "bounds must be left,top,right,bottom"
        ) from exc
    if len(bounds) != 4 or bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
        raise argparse.ArgumentTypeError("bounds must be left,top,right,bottom")
    return bounds  # type: ignore[return-value]


def _normalized_image(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        return ImageOps.exif_transpose(opened).copy()


def _source_id(image: Image.Image, *, legacy_palette: bool = False) -> str:
    digest = hashlib.sha256()
    digest.update(image.mode.encode())
    digest.update(f"{image.width}x{image.height}".encode())
    digest.update(image.tobytes())
    if image.mode == "P" and not legacy_palette:
        digest.update(image.convert("RGBA").tobytes())
    return f"sha256:{digest.hexdigest()}"


def _map_id(
    source_id: str,
    view_bounds: list[int],
    grids: dict[str, Any],
    geometry_version: int = GEOMETRY_VERSION,
) -> str:
    value = {
        "geometry_version": geometry_version,
        "source_id": source_id,
        "view_bounds": view_bounds,
        "grids" if geometry_version == 1 else "partitions": grids,
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def grid_tiles(
    bounds: Iterable[int], rows: int, columns: int, parent: str = "ROOT"
) -> list[dict[str, Any]]:
    """Return half-open, source-global bounds for a row-major grid."""
    _validate_grid(rows, columns)
    left, top, right, bottom = bounds
    width, height = right - left, bottom - top
    if width < columns or height < rows:
        raise ValueError("grid cells must have positive width and height")
    tiles: list[dict[str, Any]] = []
    for row in range(rows):
        y0 = top + math.floor(row * height / rows)
        y1 = top + math.floor((row + 1) * height / rows)
        for column in range(columns):
            x0 = left + math.floor(column * width / columns)
            x1 = left + math.floor((column + 1) * width / columns)
            segment = f"{chr(65 + column)}{row + 1}"
            tiles.append(
                {
                    "anchor": f"{parent}/{segment}",
                    "parent": parent,
                    "core_bounds_px": [x0, y0, x1, y1],
                }
            )
    return tiles


def _display_rgb(image: Image.Image) -> Image.Image:
    if (
        any(band.upper() == "A" for band in image.getbands())
        or "transparency" in image.info
    ):
        return Image.alpha_composite(
            Image.new("RGBA", image.size, "white"), image.convert("RGBA")
        ).convert("RGB")
    return image.convert("RGB")


def _grid_annotation(
    image: Image.Image,
    bounds: list[int],
    rows: int,
    columns: int,
    breadcrumb: str,
    origin: tuple[int, int],
) -> Image.Image:
    """Draw labels in gutters; source pixels are only crossed by grid lines."""
    rgb = _display_rgb(image)
    cell_w, cell_h = rgb.width / columns, rgb.height / rows
    label_size = max(18, min(42, int(min(cell_w, cell_h) * 0.15)))
    gutter_left = max(48, label_size + 22)
    gutter_top = max(68, label_size * 2)
    canvas = Image.new(
        "RGB", (rgb.width + gutter_left, rgb.height + gutter_top), "white"
    )
    canvas.paste(rgb, (gutter_left, gutter_top))
    draw = ImageDraw.Draw(canvas)
    label_font = _font(label_size)
    crumb_font = _font(max(16, min(24, label_size)))
    draw.text((10, 8), f"ANCHOR {breadcrumb}", fill="black", font=crumb_font)

    source_left, source_top, source_right, source_bottom = bounds
    source_width, source_height = source_right - source_left, source_bottom - source_top
    local_x, local_y = origin
    grid_left = gutter_left + source_left - local_x
    grid_top = gutter_top + source_top - local_y
    grid_right = gutter_left + source_right - local_x - 1
    grid_bottom = gutter_top + source_bottom - local_y - 1
    for column in range(columns + 1):
        source_x = source_left + math.floor(column * source_width / columns)
        x = min(grid_right, gutter_left + source_x - local_x)
        draw.line((x, grid_top, x, grid_bottom), fill="black", width=2)
        if column < columns:
            next_x = source_left + math.floor((column + 1) * source_width / columns)
            center = gutter_left + ((source_x + next_x) / 2) - local_x
            text = chr(65 + column)
            box = draw.textbbox((0, 0), text, font=label_font)
            draw.text(
                (center - (box[2] - box[0]) / 2, gutter_top - label_size - 8),
                text,
                fill="black",
                font=label_font,
            )

    for row in range(rows + 1):
        source_y = source_top + math.floor(row * source_height / rows)
        y = min(grid_bottom, gutter_top + source_y - local_y)
        draw.line((grid_left, y, grid_right, y), fill="black", width=2)
        if row < rows:
            next_y = source_top + math.floor((row + 1) * source_height / rows)
            center = gutter_top + ((source_y + next_y) / 2) - local_y
            text = str(row + 1)
            box = draw.textbbox((0, 0), text, font=label_font)
            draw.text(
                (gutter_left - (box[2] - box[0]) - 12, center - (box[3] - box[1]) / 2),
                text,
                fill="black",
                font=label_font,
            )
    return canvas


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(label)  # noqa: TRY004
    return cast(dict[str, Any], value)


def _array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(label)  # noqa: TRY004
    return cast(list[Any], value)


def _bounds(value: Any, size: list[int]) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(cast(list[Any], value)) != 4:
        raise ValueError("bounds must contain four integers")
    result = list(cast(list[int], value))
    if any(type(v) is not int for v in result):
        raise ValueError("bounds must contain four integers")
    left, top, right, bottom = result
    if not (0 <= left < right <= size[0] and 0 <= top < bottom <= size[1]):
        raise ValueError("bounds are outside source or have zero area")
    return result


def _grid_partition(bounds: list[int], rows: int, columns: int) -> dict[str, Any]:
    grid_tiles(bounds, rows, columns)
    left, top, right, bottom = bounds
    return {
        "type": "grid",
        "bounds_px": bounds,
        "x_edges": [left + i * (right - left) // columns for i in range(columns + 1)],
        "y_edges": [top + i * (bottom - top) // rows for i in range(rows + 1)],
    }


def _partition_children(partition: dict[str, Any], parent: str) -> list[dict[str, Any]]:
    if partition.get("type") == "grid":
        x = cast(list[int], _array(partition.get("x_edges"), "grid requires x_edges"))
        y = cast(list[int], _array(partition.get("y_edges"), "grid requires y_edges"))
        _validate_grid(len(y) - 1, len(x) - 1)
        bounds: list[int] = partition["bounds_px"]
        for edges, start, end in [(x, bounds[0], bounds[2]), (y, bounds[1], bounds[3])]:
            if (
                any(type(v) is not int for v in edges)
                or edges[0] != start
                or edges[-1] != end
            ):
                raise ValueError("grid edges must be integers spanning the parent")
            if any(a >= b for a, b in pairwise(edges)):
                raise ValueError("grid edges must increase strictly")
        return [
            {
                "anchor": f"{parent}/{chr(65 + col)}{row + 1}",
                "parent": parent,
                "core_bounds_px": [x[col], y[row], x[col + 1], y[row + 1]],
            }
            for row in range(len(y) - 1)
            for col in range(len(x) - 1)
        ]
    if partition.get("type") != "regions":
        raise ValueError("unsupported partition type")
    children = _array(partition.get("children"), "region requires children")
    if not 1 <= len(children) <= MAX_TILES:
        raise ValueError("region partition must contain 1..16 children")
    entries: list[dict[str, Any]] = []
    segments: set[str] = set()
    for child in children:
        if not isinstance(child, dict):
            raise ValueError("invalid region child")  # noqa: TRY004
        child = cast(dict[str, Any], child)
        segment = child.get("segment")
        if not isinstance(segment, str) or not re.fullmatch(
            r"R(?:[1-9]|1[0-6])", segment
        ):
            raise ValueError("invalid region segment")
        if segment in segments:
            raise ValueError("duplicate region segment")
        segments.add(segment)
        entries.append(
            {
                "anchor": f"{parent}/{segment}",
                "parent": parent,
                "core_bounds_px": child.get("core_bounds_px"),
            }
        )
    return entries


def _validate_partitions(
    partitions: dict[str, Any], view: list[int], size: list[int]
) -> dict[str, Any]:
    _bounds(view, size)
    if "ROOT" not in partitions:
        raise ValueError("partitions require ROOT")
    anchors: dict[str, Any] = {}
    for parent, partition in sorted(
        partitions.items(), key=lambda item: (item[0].count("/"), item[0])
    ):
        if not isinstance(partition, dict):
            raise ValueError("invalid partition")  # noqa: TRY004
        partition = cast(dict[str, Any], partition)
        if parent != "ROOT" and parent not in anchors:
            raise ValueError(f"invalid partition parent: {parent}")
        if parent.count("/") >= MAX_DEPTH:
            raise ValueError(f"maximum anchor depth is {MAX_DEPTH}")
        bounds = _bounds(partition.get("bounds_px"), size)
        expected = view if parent == "ROOT" else anchors[parent]["core_bounds_px"]
        if bounds != expected:
            raise ValueError("partition bounds do not match parent core")
        children = _partition_children(partition, parent)
        covered = 0
        prior: list[list[int]] = []
        for entry in children:
            core = _bounds(entry["core_bounds_px"], size)
            if (
                core[0] < bounds[0]
                or core[1] < bounds[1]
                or core[2] > bounds[2]
                or core[3] > bounds[3]
            ):
                raise ValueError("child is outside parent")
            if any(
                max(core[0], other[0]) < min(core[2], other[2])
                and max(core[1], other[1]) < min(core[3], other[3])
                for other in prior
            ):
                raise ValueError("child cores overlap")
            covered += (core[2] - core[0]) * (core[3] - core[1])
            prior.append(core)
            anchors[entry["anchor"]] = entry
        if covered != (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]):
            raise ValueError("child cores do not cover parent")
    return anchors


def _geometry(manifest: dict[str, Any]) -> dict[str, Any]:
    return manifest["grids" if manifest["geometry_version"] == 1 else "partitions"]


def _as_partitions(manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest["geometry_version"] == 2:
        return deepcopy(_geometry(manifest))
    return {
        parent: _grid_partition(grid["bounds_px"], grid["rows"], grid["columns"])
        for parent, grid in manifest["grids"].items()
    }


def _partition_annotation(
    image: Image.Image,
    children: list[dict[str, Any]],
    breadcrumb: str,
    origin: tuple[int, int],
) -> Image.Image:
    gutter = 44
    canvas = Image.new("RGB", (image.width, image.height + gutter), "white")
    canvas.paste(_display_rgb(image), (0, gutter))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), f"ANCHOR {breadcrumb}", fill="black", font=_font(18))
    for child in children:
        left, top, right, bottom = child["core_bounds_px"]
        box = (
            left - origin[0],
            top - origin[1] + gutter,
            right - origin[0] - 1,
            bottom - origin[1] + gutter - 1,
        )
        draw.rectangle(box, outline="#222222", width=2)
        label = child["anchor"].removeprefix(breadcrumb + "/")
        font = _font(14)
        text_box = draw.textbbox((box[0] + 4, box[1] + 3), label, font=font)
        draw.rectangle(text_box, fill="white")
        draw.text((box[0] + 4, box[1] + 3), label, fill="black", font=font)
    return canvas


def _discovery_module() -> Any:
    directory = str(Path(__file__).resolve().parent)
    if directory not in sys.path:
        sys.path.insert(0, directory)
    return importlib.import_module("autostructure")


def _load_config(config: dict[str, Any] | Path | None) -> dict[str, Any] | None:
    if isinstance(config, Path):
        data = json.loads(config.read_text())
        if not isinstance(data, dict):
            raise ValueError("detector config must be a JSON object")  # noqa: TRY004
        return cast(dict[str, Any], data)
    return config


def _offered_candidates(
    report: dict[str, Any],
    source_id: str,
    view: list[int],
    size: list[int],
    rows: int | None,
    columns: int | None,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    module = _discovery_module()
    fallback = {
        "ROOT": _grid_partition(
            view,
            rows if rows is not None else min(3, view[3] - view[1]),
            columns if columns is not None else min(3, view[2] - view[0]),
        )
    }
    grid = {
        "id": module.candidate_id(source_id, view, fallback),
        "method": "grid",
        "methods": ["grid"],
        "score": 0.0,
        "features": {},
        "partitions": fallback,
        "evidence_bounds_px": [],
        "reason": "uniform grid option",
    }
    valid: list[dict[str, Any]] = []
    offered: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for value in _array(report.get("candidates"), "report requires candidates"):
        option = _object(value, "invalid candidate")
        try:
            geometry = _object(
                option.get("partitions"), "candidate requires partitions"
            )
            anchors = _validate_partitions(geometry, view, size)
            if option.get("id") != module.candidate_id(source_id, view, geometry):
                raise ValueError("candidate identity does not match geometry")
        except (ValueError, KeyError, TypeError) as exc:
            if option.get("id") == report.get("selected_candidate_id"):
                raise ValueError("selected candidate is invalid") from exc
            report.setdefault("rejected_candidates", []).append(
                {"id": option.get("id"), "reason": str(exc)}
            )
            continue
        valid.append(option)
        if option.get("method") != "grid" and len(offered) < 3:
            offered.append((option, anchors))
    existing_grid = next((c for c in valid if c["id"] == grid["id"]), None)
    if existing_grid is None:
        valid.append(grid)
    else:
        grid = existing_grid
        offered = [(c, a) for c, a in offered if c["id"] != grid["id"]]
    offered.append((grid, _validate_partitions(fallback, view, size)))
    report["candidates"] = valid
    report["offered_candidate_ids"] = [option["id"] for option, _ in offered]
    return offered


def _navigation_entries(anchors: dict[str, Any]) -> list[dict[str, Any]]:
    parents = {entry["parent"] for entry in anchors.values()}
    return [entry for anchor, entry in anchors.items() if anchor not in parents]


def create_map(
    image_path: Path,
    output_dir: Path,
    rows: int | None = None,
    columns: int | None = None,
    region: tuple[int, int, int, int] | None = None,
    *,
    mode: str | None = None,
    detectors: tuple[str, ...] | None = None,
    detector_config: dict[str, Any] | Path | None = None,
    min_score: float | None = None,
    candidate: str | None = None,
    discovery: Path | None = None,
) -> Path:
    selected_mode = mode or (
        "grid" if rows is not None or columns is not None else "auto"
    )
    if selected_mode not in ("auto", "grid", "hough", "gutters", "panels"):
        raise ValueError("unknown layout mode")
    if selected_mode not in ("auto", "grid") and (
        rows is not None or columns is not None
    ):
        raise ValueError("grid dimensions cannot configure an explicit detector")
    if (candidate is None) != (discovery is None):
        raise ValueError("candidate and discovery must be supplied together")
    if candidate is not None and any(
        value is not None
        for value in [mode, rows, columns, detectors, detector_config, min_score]
    ):
        raise ValueError(
            "candidate replay cannot combine with detector or grid options"
        )
    if selected_mode == "grid" and any(
        value is not None for value in [detectors, detector_config, min_score]
    ):
        raise ValueError("grid mode does not accept detector options")
    if detectors is not None and selected_mode != "auto":
        raise ValueError("detectors option requires auto mode")
    source = _normalized_image(image_path)
    try:
        size = [source.width, source.height]
        view = _bounds(region or [0, 0, *size], size)
        source_id = _source_id(source)
        report: dict[str, Any] | None = None
        version = 1 if selected_mode == "grid" and candidate is None else 2
        if version == 1:
            grid_tiles(
                view,
                rows if rows is not None else 3,
                columns if columns is not None else 3,
            )
            geometry: dict[str, Any] = {
                "ROOT": {
                    "bounds_px": view,
                    "rows": rows if rows is not None else 3,
                    "columns": columns if columns is not None else 3,
                }
            }
            anchors = {
                entry["anchor"]: entry
                for entry in grid_tiles(
                    view, geometry["ROOT"]["rows"], geometry["ROOT"]["columns"]
                )
            }
        else:
            module = _discovery_module()
            if candidate is not None:
                assert discovery is not None
                report = _object(
                    json.loads(discovery.read_text()),
                    "discovery report must be an object",
                )
                if (
                    report.get("source_id") != source_id
                    or report.get("view_bounds_px") != view
                ):
                    raise ValueError("discovery source or view does not match image")
                matches = [
                    c for c in report.get("candidates", []) if c.get("id") == candidate
                ]
                if len(matches) != 1:
                    raise ValueError("candidate does not exist uniquely in report")
                chosen = matches[0]
                if (
                    module.candidate_id(source_id, view, chosen["partitions"])
                    != candidate
                ):
                    raise ValueError("candidate identity does not match geometry")
                geometry = deepcopy(chosen["partitions"])
                report["selected_candidate_id"] = candidate
                report["selection_reason"] = "explicit candidate replay"
            else:
                report = cast(
                    dict[str, Any],
                    module.discover(
                        source,
                        view,
                        source_id,
                        mode=selected_mode,
                        detectors=detectors
                        if detectors is not None
                        else ("hough", "gutters", "panels"),
                        config=_load_config(detector_config),
                        min_score=min_score,
                    ),
                )
                selected = report.get("selected_candidate_id")
                chosen = next(
                    (c for c in report["candidates"] if c["id"] == selected), None
                )
                geometry = (
                    deepcopy(chosen["partitions"])
                    if chosen is not None
                    else {
                        "ROOT": _grid_partition(
                            view,
                            rows if rows is not None else min(3, view[3] - view[1]),
                            columns
                            if columns is not None
                            else min(3, view[2] - view[0]),
                        )
                    }
                )
            anchors = _validate_partitions(geometry, view, size)
        offered = (
            _offered_candidates(report, source_id, view, size, rows, columns)
            if report is not None
            else []
        )
        if any(
            (output_dir / name).exists() for name in ("manifest.json", "source.png")
        ):
            raise ValueError(
                "map output already exists; choose a fresh output directory"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        source.save(output_dir / "source.png", format="PNG")
        raw = source.crop((view[0], view[1], view[2], view[3]))
        raw.save(output_dir / "overview_raw.png", format="PNG")
        if version == 1:
            nav = _grid_annotation(
                raw,
                view,
                geometry["ROOT"]["rows"],
                geometry["ROOT"]["columns"],
                "ROOT",
                (view[0], view[1]),
            )
        else:
            nav = _partition_annotation(
                raw, _navigation_entries(anchors), "ROOT", (view[0], view[1])
            )
        nav.save(output_dir / "overview_annotated.png", format="PNG")
        manifest: dict[str, Any] = {
            "geometry_version": version,
            "source_id": source_id,
            "map_id": _map_id(source_id, view, geometry, version),
            "source_path": "source.png",
            "source_dimensions": size,
            "view_bounds_px": view,
            "grids" if version == 1 else "partitions": geometry,
            "anchors": anchors,
            "artifacts": {
                "overview_raw": "overview_raw.png",
                "overview_annotated": "overview_annotated.png",
            },
        }
        if report is not None:
            report["map_id"] = manifest["map_id"]
            report["requested_mode"] = "candidate" if candidate else selected_mode
            for index, (option, option_anchors) in enumerate(offered):
                path = f"alternative_{index + 1}_annotated.png"
                _partition_annotation(
                    raw, _navigation_entries(option_anchors), "ROOT", (view[0], view[1])
                ).save(output_dir / path)
                option["artifacts"] = {"annotated": path, "raw": "overview_raw.png"}
            (output_dir / "structures.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            manifest["artifacts"]["structures"] = "structures.json"
        path = output_dir / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        return path
    finally:
        source.close()


def _expanded_bounds(
    bounds: list[int], margin: float, size: list[int]
) -> tuple[int, int, int, int]:
    left, top, right, bottom = bounds
    dx, dy = (right - left) * margin, (bottom - top) * margin
    return (
        max(0, math.floor(left - dx)),
        max(0, math.floor(top - dy)),
        min(size[0], math.ceil(right + dx)),
        min(size[1], math.ceil(bottom + dy)),
    )


def _load_manifest(manifest_path: Path) -> tuple[dict[str, Any], Image.Image, Path]:
    manifest: dict[str, Any] = json.loads(manifest_path.read_text())
    required = {
        "geometry_version",
        "source_id",
        "map_id",
        "source_path",
        "source_dimensions",
        "view_bounds_px",
        "anchors",
    }
    if not required <= manifest.keys():
        raise ValueError("manifest is missing required fields")
    version = manifest["geometry_version"]
    if type(version) is not int or version not in (1, 2):
        raise ValueError("unsupported geometry version")
    size = cast(
        list[int], _array(manifest["source_dimensions"], "invalid source dimensions")
    )
    if len(size) != 2 or any(type(v) is not int or v < 1 for v in size):
        raise ValueError("invalid source dimensions")
    view = _bounds(manifest["view_bounds_px"], size)
    geometry = _object(
        manifest.get("grids" if version == 1 else "partitions"),
        "manifest is missing geometry",
    )
    if _map_id(manifest["source_id"], view, geometry, version) != manifest["map_id"]:
        raise ValueError("map_id does not match manifest")
    if version == 2:
        expected = _validate_partitions(geometry, view, size)
    else:
        expected: dict[str, Any] = {}
        if "ROOT" not in geometry:
            raise ValueError("root grid is missing")
        for parent, grid in sorted(
            geometry.items(), key=lambda item: (item[0].count("/"), item[0])
        ):
            if not isinstance(grid, dict) or parent.count("/") >= MAX_DEPTH:
                raise ValueError("invalid grid or excessive depth")
            grid = cast(dict[str, Any], grid)
            bounds = _bounds(grid.get("bounds_px"), size)
            parent_core = (
                view
                if parent == "ROOT"
                else expected.get(parent, {}).get("core_bounds_px")
            )
            if bounds != parent_core:
                raise ValueError("invalid grid parent bounds")
            children = grid_tiles(bounds, grid["rows"], grid["columns"], parent)
            expected.update({child["anchor"]: child for child in children})
    if manifest["anchors"] != expected:
        raise ValueError("anchor geometry does not match partitions")
    source_path = Path(manifest["source_path"])
    if not source_path.is_absolute():
        source_path = manifest_path.parent / source_path
    with Image.open(source_path) as opened:
        source = opened.copy()
    source_matches = _source_id(source) == manifest["source_id"]
    if version == 1 and source.mode == "P" and not source_matches:
        source_matches = (
            _source_id(source, legacy_palette=True) == manifest["source_id"]
        )
    if [source.width, source.height] != size or not source_matches:
        source.close()
        raise ValueError("source image does not match manifest")
    return manifest, source, source_path.resolve()


def focus_map(
    manifest_path: Path,
    output_dir: Path,
    requested: list[str] | None = None,
    margin: float = 0.08,
    child_rows: int | None = None,
    child_columns: int | None = None,
    *,
    mode: str | None = None,
    detectors: tuple[str, ...] | None = None,
    detector_config: dict[str, Any] | Path | None = None,
    min_score: float | None = None,
) -> Path:
    if not math.isfinite(margin) or not 0 <= margin <= 0.5:
        raise ValueError("margin must be finite and in [0, 0.5]")
    if (child_rows is None) != (child_columns is None):
        raise ValueError("child rows and columns must be supplied together")
    if child_rows is not None:
        _validate_grid(child_rows, child_columns or 0)
    if mode is not None and (
        mode not in ("auto", "hough", "gutters", "panels") or child_rows is not None
    ):
        raise ValueError(
            "structural refinement cannot combine with child grid dimensions"
        )
    if mode is None and any(
        v is not None for v in [detectors, detector_config, min_score]
    ):
        raise ValueError("detector options require a refinement mode")
    if mode != "auto" and detectors is not None:
        raise ValueError("detectors option requires auto mode")
    manifest, source, source_path = _load_manifest(manifest_path)
    try:
        anchors = manifest["anchors"]
        selected = grid_tiles_top(manifest) if requested is None else requested
        if (
            not selected
            or len(selected) > MAX_TILES
            or len(selected) != len(set(selected))
        ):
            raise ValueError("request 1..16 unique anchors")
        if any(a not in anchors for a in selected):
            raise ValueError("unknown anchor")
        refining = child_rows is not None or mode is not None
        version = 2 if mode is not None else manifest["geometry_version"]
        next_source_id = _source_id(source) if version == 2 else manifest["source_id"]
        geometry = (
            _as_partitions(manifest) if version == 2 else deepcopy(manifest["grids"])
        )
        reports: dict[str, Any] = {}
        offers: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
        for anchor in selected:
            if not refining:
                continue
            if anchor.count("/") >= MAX_DEPTH:
                raise ValueError(f"maximum anchor depth is {MAX_DEPTH}")
            core = anchors[anchor]["core_bounds_px"]
            if mode is not None:
                module = _discovery_module()
                report = cast(
                    dict[str, Any],
                    module.discover(
                        source,
                        core,
                        next_source_id,
                        mode=mode,
                        detectors=detectors
                        if detectors is not None
                        else ("hough", "gutters", "panels"),
                        config=_load_config(detector_config),
                        min_score=min_score,
                    ),
                )
                remaining_depth = MAX_DEPTH - anchor.count("/")
                eligible: list[dict[str, Any]] = []
                for proposal in report["candidates"]:
                    local_depth = max(
                        parent.count("/") + 1 for parent in proposal["partitions"]
                    )
                    if local_depth > remaining_depth:
                        proposal["rejection_reason"] = (
                            "Structure exceeds the remaining anchor depth budget"
                        )
                        report.setdefault("rejected_candidates", []).append(proposal)
                    else:
                        eligible.append(proposal)
                if len(eligible) != len(report["candidates"]):
                    report["candidates"] = eligible
                    report["selected_candidate_id"] = None
                    report["fallback_reason"] = (
                        "No accepted structure fits the remaining anchor depth budget"
                    )
                    if eligible and not eligible[0].get("rejection_reason"):
                        ambiguous = (
                            mode == "auto"
                            and len(eligible) > 1
                            and eligible[0]["score"] - eligible[1]["score"]
                            < report["parameters"]["selection"]["min_margin"]
                        )
                        if not ambiguous:
                            report["selected_candidate_id"] = eligible[0]["id"]
                            report["fallback_reason"] = None
                    report["summary"] = {
                        "alternatives": [
                            {
                                "id": p["id"],
                                "methods": p["methods"],
                                "score": p["score"],
                                "reason": p["reason"],
                            }
                            for p in eligible[:3]
                        ],
                        "grid_option": True,
                        "selection_reason": eligible[0]["reason"]
                        if report["selected_candidate_id"]
                        else report["fallback_reason"],
                    }
                    if mode != "auto" and report["selected_candidate_id"] is None:
                        raise ValueError(report["fallback_reason"])
                chosen = next(
                    (
                        c
                        for c in report["candidates"]
                        if c["id"] == report.get("selected_candidate_id")
                    ),
                    None,
                )
                added = (
                    chosen["partitions"]
                    if chosen is not None
                    else {
                        "ROOT": _grid_partition(
                            core, min(3, core[3] - core[1]), min(3, core[2] - core[0])
                        )
                    }
                )
                offers[anchor] = _offered_candidates(
                    report,
                    next_source_id,
                    core,
                    manifest["source_dimensions"],
                    None,
                    None,
                )
                reports[anchor] = report
                replacements = {
                    anchor + parent.removeprefix("ROOT"): deepcopy(partition)
                    for parent, partition in added.items()
                }
            else:
                assert child_rows is not None and child_columns is not None
                replacements = {
                    anchor: _grid_partition(core, child_rows, child_columns)
                    if version == 2
                    else {
                        "bounds_px": core,
                        "rows": child_rows,
                        "columns": child_columns,
                    }
                }
            for parent, partition in replacements.items():
                if parent in geometry and geometry[parent] != partition:
                    raise ValueError(
                        f"anchor {parent} already has different child geometry"
                    )
                geometry[parent] = partition
        if version == 2:
            next_anchors = _validate_partitions(
                geometry, manifest["view_bounds_px"], manifest["source_dimensions"]
            )
        else:
            next_anchors = dict(anchors)
            for parent, grid in geometry.items():
                next_anchors.update(
                    {
                        child["anchor"]: child
                        for child in grid_tiles(
                            grid["bounds_px"], grid["rows"], grid["columns"], parent
                        )
                    }
                )
        planned = ["focus_manifest.json"]
        if refining:
            planned.append("successor_manifest.json")
        for anchor in selected:
            slug = anchor.replace("/", "__")
            planned.extend([f"{slug}_raw.png", f"{slug}_annotated.png"])
            if anchor in reports:
                planned.append(f"{slug}_structures.json")
                planned.extend(
                    f"{slug}_alternative_{i + 1}_annotated.png"
                    for i in range(len(offers[anchor]))
                )
        protected = {manifest_path.resolve(), source_path.resolve()}
        if any((output_dir / name).resolve() in protected for name in planned):
            raise ValueError("output would overwrite the input manifest or source")
        if any((output_dir / name).exists() for name in planned):
            raise ValueError(
                "focus output already exists; choose a fresh output directory"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        artifacts: list[dict[str, Any]] = []
        for anchor in selected:
            core = anchors[anchor]["core_bounds_px"]
            bounds = _expanded_bounds(core, margin, manifest["source_dimensions"])
            crop = source.crop(bounds)
            slug = anchor.replace("/", "__")
            raw_path, nav_path = f"{slug}_raw.png", f"{slug}_annotated.png"
            crop.save(output_dir / raw_path, format="PNG")
            children = [a for a in next_anchors.values() if a["parent"] == anchor]
            nav = _partition_annotation(crop, children, anchor, (bounds[0], bounds[1]))
            nav.save(output_dir / nav_path, format="PNG")
            if anchor in reports:
                reports[anchor]["artifacts"] = {"raw": raw_path, "annotated": nav_path}
                reports[anchor]["focus_anchor"] = anchor
                for index, (option, option_anchors) in enumerate(offers[anchor]):
                    entries = [
                        dict(
                            entry, anchor=anchor + entry["anchor"].removeprefix("ROOT")
                        )
                        for entry in _navigation_entries(option_anchors)
                    ]
                    preview = f"{slug}_alternative_{index + 1}_annotated.png"
                    _partition_annotation(
                        crop, entries, anchor, (bounds[0], bounds[1])
                    ).save(output_dir / preview)
                    option["artifacts"] = {"annotated": preview, "raw": raw_path}
            artifacts.append(
                {
                    "anchor": anchor,
                    "core_bounds_px": core,
                    "crop_bounds_px": list(bounds),
                    "dimensions_px": [crop.width, crop.height],
                    "source_area_fraction": round(
                        crop.width * crop.height / (source.width * source.height), 8
                    ),
                    "raw_crop": raw_path,
                    "annotated_crop": nav_path,
                }
            )
        result: dict[str, Any] = {
            "input_map_id": manifest["map_id"],
            "coverage": "complete"
            if set(selected) == set(grid_tiles_top(manifest))
            else "partial",
            "margin": margin,
            "crops": artifacts,
        }
        if refining:
            successor = dict(manifest)
            successor.pop("grids", None)
            successor.pop("partitions", None)
            successor["geometry_version"] = version
            successor["source_id"] = next_source_id
            successor["grids" if version == 1 else "partitions"] = geometry
            successor["anchors"] = next_anchors
            successor["map_id"] = _map_id(
                next_source_id, manifest["view_bounds_px"], geometry, version
            )
            successor["source_path"] = os.path.relpath(
                source_path, output_dir.resolve()
            )
            successor["artifacts"] = cast(
                dict[str, Any], {"focus": "focus_manifest.json"}
            )
            if reports:
                for anchor, report in reports.items():
                    path = anchor.replace("/", "__") + "_structures.json"
                    (output_dir / path).write_text(json.dumps(report, indent=2) + "\n")
                successor["artifacts"]["discovery_reports"] = {
                    a: a.replace("/", "__") + "_structures.json" for a in reports
                }
            path = output_dir / "successor_manifest.json"
            path.write_text(json.dumps(successor, indent=2) + "\n")
            result.update(
                successor_map_id=successor["map_id"], successor_manifest=path.name
            )
        path = output_dir / "focus_manifest.json"
        path.write_text(json.dumps(result, indent=2) + "\n")
        return path
    finally:
        source.close()


def grid_tiles_top(manifest: dict[str, Any]) -> list[str]:
    return [
        anchor
        for anchor, entry in manifest["anchors"].items()
        if entry["parent"] == "ROOT"
    ]


def _detector_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mode", choices=["auto", "grid", "hough", "gutters", "panels"]
    )
    parser.add_argument("--detectors", type=lambda s: tuple(s.split(",")))
    parser.add_argument("--detector-config", type=Path)
    parser.add_argument("--min-score", type=float)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    map_parser = subparsers.add_parser(
        "map", help="discover structure or make an explicit grid"
    )
    map_parser.add_argument("image", type=Path)
    map_parser.add_argument("--output", type=Path, required=True)
    map_parser.add_argument("--rows", type=int)
    map_parser.add_argument("--columns", type=int)
    map_parser.add_argument("--region", type=parse_bounds)
    map_parser.add_argument("--candidate")
    map_parser.add_argument("--discovery", type=Path)
    _detector_flags(map_parser)
    focus_parser = subparsers.add_parser(
        "focus", help="materialize selected crops and optionally refine"
    )
    focus_parser.add_argument("manifest", type=Path)
    focus_parser.add_argument("--output", type=Path, required=True)
    focus_parser.add_argument("--anchor", action="append", dest="anchors")
    focus_parser.add_argument("--all", action="store_true")
    focus_parser.add_argument("--margin", type=float, default=0.08)
    focus_parser.add_argument("--rows", type=int, dest="child_rows")
    focus_parser.add_argument("--columns", type=int, dest="child_columns")
    _detector_flags(focus_parser)
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    try:
        options = {
            "mode": args.mode,
            "detectors": args.detectors,
            "detector_config": args.detector_config,
            "min_score": args.min_score,
        }
        if args.command == "map":
            path = create_map(
                args.image,
                args.output,
                args.rows,
                args.columns,
                args.region,
                candidate=args.candidate,
                discovery=args.discovery,
                **options,
            )
        else:
            if args.all == bool(args.anchors):
                raise ValueError("supply either --all or one or more --anchor options")
            path = focus_map(
                args.manifest,
                args.output,
                None if args.all else args.anchors,
                args.margin,
                args.child_rows,
                args.child_columns,
                **options,
            )
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print(path)


if __name__ == "__main__":
    main()
