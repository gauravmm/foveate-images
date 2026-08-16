#!/usr/bin/env python3
"""Generate synthetic calendars and run each through the foveate pipeline."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType

from PIL import Image, ImageDraw, ImageFont

from foveate_tools.calendar import generate_set


def _foveate() -> ModuleType:
    path = Path(__file__).parents[1] / "skills/foveate-images/foveate.py"
    spec = importlib.util.spec_from_file_location("foveate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _preview(items: list[tuple[str, Path]], output: Path) -> None:
    width, tile_h = 420, 350
    canvas = Image.new("RGB", (width * len(items), tile_h), "#dddddd")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, (label, path) in enumerate(items):
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            image.thumbnail((width - 20, tile_h - 45))
            x = index * width + (width - image.width) // 2
            y = 35 + (tile_h - 35 - image.height) // 2
            canvas.paste(image, (x, y))
            draw.text((index * width + 10, 10), label, fill="black", font=font)
    canvas.save(output, format="PNG")


def run_demo(output_dir: Path, seed: int = 20260816) -> Path:
    foveate = _foveate()
    generated = output_dir / "generated"
    pipeline = output_dir / "pipeline"
    truth_paths = generate_set(generated, seed)
    summary: list[dict[str, object]] = []
    for truth_path in truth_paths:
        truth = json.loads(truth_path.read_text())
        slug = truth_path.name.removesuffix(".truth.json")
        image_path = generated / truth["image"]
        run_dir = pipeline / slug
        map_dir, focus_dir = run_dir / "map", run_dir / "focus"
        layout = truth["layout"]
        manifest_path = foveate.create_map(
            image_path,
            map_dir,
            layout["rows"],
            layout["columns"],
            tuple(layout["body_bounds_px"]),
        )
        focus_path = foveate.focus_map(manifest_path, focus_dir, margin=0.08)
        focus = json.loads(focus_path.read_text())
        first = focus["crops"][0]
        preview_path = run_dir / "pipeline_preview.png"
        _preview(
            [
                ("1 source", image_path),
                ("2 anchor map", map_dir / "overview_annotated.png"),
                (f"3 raw {first['anchor']}", focus_dir / first["raw_crop"]),
                (f"4 labeled {first['anchor']}", focus_dir / first["annotated_crop"]),
            ],
            preview_path,
        )
        summary.append(
            {
                "calendar": slug,
                "truth": str(truth_path),
                "map_manifest": str(manifest_path),
                "focus_manifest": str(focus_path),
                "preview": str(preview_path),
                "crop_count": len(focus["crops"]),
            }
        )
    summary_path = output_dir / "demo_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/synthetic-calendars"))
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()
    print(run_demo(args.output, args.seed))


if __name__ == "__main__":
    main()
