# /// script
# requires-python = ">=3.10"
# dependencies = ["Pillow>=10.2"]
# ///
"""Deterministic, string-addressed image crops for vision agents."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

GEOMETRY_VERSION = 1
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
    if rows < 1 or columns < 1 or rows * columns > MAX_TILES:
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


def _source_id(image: Image.Image) -> str:
    digest = hashlib.sha256()
    digest.update(image.mode.encode())
    digest.update(f"{image.width}x{image.height}".encode())
    digest.update(image.tobytes())
    return f"sha256:{digest.hexdigest()}"


def _map_id(source_id: str, view_bounds: list[int], grids: dict[str, Any]) -> str:
    value = {
        "geometry_version": GEOMETRY_VERSION,
        "source_id": source_id,
        "view_bounds": view_bounds,
        "grids": grids,
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


def _grid_annotation(
    image: Image.Image,
    bounds: list[int],
    rows: int,
    columns: int,
    breadcrumb: str,
    origin: tuple[int, int],
) -> Image.Image:
    """Draw labels in gutters; source pixels are only crossed by grid lines."""
    rgb = image.convert("RGB")
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


def create_map(
    image_path: Path,
    output_dir: Path,
    rows: int = 3,
    columns: int = 3,
    region: tuple[int, int, int, int] | None = None,
) -> Path:
    _validate_grid(rows, columns)
    source = _normalized_image(image_path)
    full = (0, 0, source.width, source.height)
    region = region or full
    if (
        region[0] < 0
        or region[1] < 0
        or region[2] > source.width
        or region[3] > source.height
        or region[0] >= region[2]
        or region[1] >= region[3]
    ):
        raise ValueError(f"region {region} is outside source bounds {full}")
    grid_tiles(region, rows, columns)

    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = output_dir / "source.png"
    raw_path = output_dir / "overview_raw.png"
    annotated_path = output_dir / "overview_annotated.png"
    source.save(source_path, format="PNG")
    raw = source.crop(region)
    raw.save(raw_path, format="PNG")
    annotated = _grid_annotation(
        raw, list(region), rows, columns, "ROOT", (region[0], region[1])
    )
    annotated.save(annotated_path, format="PNG")

    source_id = _source_id(source)
    grids = {"ROOT": {"bounds_px": list(region), "rows": rows, "columns": columns}}
    anchors = {tile["anchor"]: tile for tile in grid_tiles(region, rows, columns)}
    manifest = {
        "geometry_version": GEOMETRY_VERSION,
        "source_id": source_id,
        "map_id": _map_id(source_id, list(region), grids),
        "source_path": source_path.name,
        "source_dimensions": [source.width, source.height],
        "view_bounds_px": list(region),
        "grids": grids,
        "anchors": anchors,
        "artifacts": {
            "overview_raw": raw_path.name,
            "overview_annotated": annotated_path.name,
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


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
    manifest = json.loads(manifest_path.read_text())
    required = {
        "geometry_version",
        "source_id",
        "map_id",
        "source_path",
        "source_dimensions",
        "view_bounds_px",
        "grids",
        "anchors",
    }
    missing = required - manifest.keys()
    if missing:
        raise ValueError(f"manifest is missing: {', '.join(sorted(missing))}")
    if manifest["geometry_version"] != GEOMETRY_VERSION:
        raise ValueError("unsupported geometry version")
    source_path = Path(manifest["source_path"])
    if not source_path.is_absolute():
        source_path = manifest_path.parent / source_path
    source = Image.open(source_path)
    if [source.width, source.height] != manifest["source_dimensions"] or _source_id(
        source
    ) != manifest["source_id"]:
        source.close()
        raise ValueError("source image does not match manifest")
    expected_id = _map_id(
        manifest["source_id"], manifest["view_bounds_px"], manifest["grids"]
    )
    if expected_id != manifest["map_id"]:
        source.close()
        raise ValueError("map_id does not match manifest")

    expected: dict[str, Any] = {}
    for parent, grid in sorted(
        manifest["grids"].items(), key=lambda item: item[0].count("/")
    ):
        if parent == "ROOT":
            if grid["bounds_px"] != manifest["view_bounds_px"]:
                source.close()
                raise ValueError("root grid bounds do not match view")
        elif (
            parent not in expected
            or grid["bounds_px"] != expected[parent]["core_bounds_px"]
        ):
            source.close()
            raise ValueError(f"invalid grid parent: {parent}")
        children = grid_tiles(grid["bounds_px"], grid["rows"], grid["columns"], parent)
        expected.update({child["anchor"]: child for child in children})
    if manifest["anchors"] != expected:
        source.close()
        raise ValueError("anchor geometry does not match grids")
    return manifest, source, source_path.resolve()


def focus_map(
    manifest_path: Path,
    output_dir: Path,
    requested: list[str] | None = None,
    margin: float = 0.08,
    child_rows: int | None = None,
    child_columns: int | None = None,
) -> Path:
    if not math.isfinite(margin) or not 0 <= margin <= 0.5:
        raise ValueError("margin must be finite and in [0, 0.5]")
    if (child_rows is None) != (child_columns is None):
        raise ValueError("child rows and columns must be supplied together")
    if child_rows is not None:
        _validate_grid(child_rows, child_columns or 0)

    manifest, source, source_path = _load_manifest(manifest_path)
    anchors = manifest["anchors"]
    if requested is None:
        requested = grid_tiles_top(manifest)
    if (
        not requested
        or len(requested) > MAX_TILES
        or len(requested) != len(set(requested))
    ):
        raise ValueError(f"request 1..{MAX_TILES} unique anchors")
    unknown = [anchor for anchor in requested if anchor not in anchors]
    if unknown:
        raise ValueError(f"unknown anchors: {', '.join(unknown)}")

    replacement: dict[str, int] | None = None
    if child_rows is not None and child_columns is not None:
        replacement = {"rows": child_rows, "columns": child_columns}
        for anchor in requested:
            existing = manifest["grids"].get(anchor)
            if existing and (
                existing["rows"] != child_rows or existing["columns"] != child_columns
            ):
                source.close()
                raise ValueError(f"anchor {anchor} already has a different child grid")
    output_dir.mkdir(parents=True, exist_ok=True)
    grids = dict(manifest["grids"])
    next_anchors = dict(anchors)
    artifacts: list[dict[str, Any]] = []
    for anchor in requested:
        core = anchors[anchor]["core_bounds_px"]
        crop_bounds = _expanded_bounds(core, margin, manifest["source_dimensions"])
        crop = source.crop(crop_bounds)
        slug = anchor.replace("/", "__")
        raw_path = output_dir / f"{slug}_raw.png"
        nav_path = output_dir / f"{slug}_annotated.png"
        crop.save(raw_path, format="PNG")

        if child_rows is not None and child_columns is not None:
            assert replacement is not None
            if anchor.count("/") >= MAX_DEPTH:
                raise ValueError(f"maximum anchor depth is {MAX_DEPTH}")
            grids[anchor] = {"bounds_px": core, **replacement}
            children = grid_tiles(core, child_rows, child_columns, anchor)
            next_anchors.update({child["anchor"]: child for child in children})
            nav = _grid_annotation(
                crop,
                core,
                child_rows,
                child_columns,
                anchor,
                (crop_bounds[0], crop_bounds[1]),
            )
        else:
            gutter = 44
            nav = Image.new("RGB", (crop.width, crop.height + gutter), "white")
            nav.paste(crop.convert("RGB"), (0, gutter))
            ImageDraw.Draw(nav).text(
                (10, 10), f"ANCHOR {anchor}", fill="black", font=_font(20)
            )
        nav.save(nav_path, format="PNG")
        artifacts.append(
            {
                "anchor": anchor,
                "core_bounds_px": core,
                "crop_bounds_px": crop_bounds,
                "dimensions_px": [crop.width, crop.height],
                "source_area_fraction": round(
                    (
                        (crop_bounds[2] - crop_bounds[0])
                        * (crop_bounds[3] - crop_bounds[1])
                    )
                    / (
                        manifest["source_dimensions"][0]
                        * manifest["source_dimensions"][1]
                    ),
                    8,
                ),
                "raw_crop": raw_path.name,
                "annotated_crop": nav_path.name,
            }
        )
    source.close()

    result: dict[str, Any] = {
        "input_map_id": manifest["map_id"],
        "coverage": "complete"
        if set(requested) == set(grid_tiles_top(manifest))
        else "partial",
        "margin": margin,
        "crops": artifacts,
    }
    if child_rows is not None:
        successor = dict(manifest)
        successor["grids"] = grids
        successor["anchors"] = next_anchors
        successor["map_id"] = _map_id(
            manifest["source_id"], manifest["view_bounds_px"], grids
        )
        successor["source_path"] = os.path.relpath(source_path, output_dir.resolve())
        successor_path = output_dir / "successor_manifest.json"
        successor_path.write_text(json.dumps(successor, indent=2) + "\n")
        result["successor_map_id"] = successor["map_id"]
        result["successor_manifest"] = successor_path.name
    result_path = output_dir / "focus_manifest.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    return result_path


def grid_tiles_top(manifest: dict[str, Any]) -> list[str]:
    return [
        anchor
        for anchor, entry in manifest["anchors"].items()
        if entry["parent"] == "ROOT"
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    map_parser = subparsers.add_parser("map", help="make a labeled overview")
    map_parser.add_argument("image", type=Path)
    map_parser.add_argument("--output", type=Path, required=True)
    map_parser.add_argument("--rows", type=int, default=3)
    map_parser.add_argument("--columns", type=int, default=3)
    map_parser.add_argument("--region", type=parse_bounds)

    focus_parser = subparsers.add_parser(
        "focus", help="materialize selected anchor crops"
    )
    focus_parser.add_argument("manifest", type=Path)
    focus_parser.add_argument("--output", type=Path, required=True)
    focus_parser.add_argument("--anchor", action="append", dest="anchors")
    focus_parser.add_argument("--all", action="store_true")
    focus_parser.add_argument("--margin", type=float, default=0.08)
    focus_parser.add_argument("--rows", type=int, dest="child_rows")
    focus_parser.add_argument("--columns", type=int, dest="child_columns")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "map":
        path = create_map(args.image, args.output, args.rows, args.columns, args.region)
    else:
        if args.all and args.anchors:
            raise SystemExit("use either --all or --anchor, not both")
        if not args.all and not args.anchors:
            raise SystemExit("supply --all or at least one --anchor")
        path = focus_map(
            args.manifest,
            args.output,
            None if args.all else args.anchors,
            args.margin,
            args.child_rows,
            args.child_columns,
        )
    print(path)


if __name__ == "__main__":
    main()
