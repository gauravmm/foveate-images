"""Independent integration and adversarial checks for automatic structure maps.

Run with ``python -m unittest -v test_autostructure``. These tests compare raw
pixels and independently reconstruct legacy identities rather than trusting
the implementation's validation routines as their oracle.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from functools import partial
from itertools import pairwise
from pathlib import Path
from typing import Any
from unittest.mock import patch

from PIL import Image, ImageDraw, ImageOps

SCRIPT = Path(__file__).parent / "skills/foveate-images/foveate.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("foveate_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
foveate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(foveate)
autostructure = importlib.import_module("autostructure")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def pixel_signature(image: Image.Image) -> tuple[Any, ...]:
    """Rendered color and alpha, including palette contents, are evidence."""
    return image.size, image.convert("RGBA").tobytes()


def ruled_image(size: tuple[int, int] = (400, 300)) -> Image.Image:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    width, height = size
    for x in (20, width // 3, width * 2 // 3, width - 20):
        draw.line((x, 20, x, height - 20), fill="black", width=3)
    for y in (20, height // 3, height * 2 // 3, height - 20):
        draw.line((20, y, width - 20, y), fill="black", width=3)
    for row in range(3):
        for col in range(3):
            draw.text(
                (40 + col * width // 3, 40 + row * height // 3), "Item", fill="black"
            )
    return image


class MapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "input.png"
        image = Image.new("RGB", (101, 83))
        image.putdata(
            [
                ((x * 13) % 256, (y * 17) % 256, (x * 3 + y * 7) % 256)
                for y in range(83)
                for x in range(101)
            ]
        )
        image.save(self.source)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def create(self, name: str = "map", **kwargs: Any) -> Path:
        return foveate.create_map(self.source, self.root / name, **kwargs)

    def assert_partition(self, manifest: dict[str, Any]) -> None:
        """Check exact ownership independently at each immediate parent."""
        parents = {entry["parent"] for entry in manifest["anchors"].values()}
        for parent in parents:
            bounds = (
                manifest["view_bounds_px"]
                if parent == "ROOT"
                else manifest["anchors"][parent]["core_bounds_px"]
            )
            left, top, right, bottom = bounds
            children = [
                entry
                for entry in manifest["anchors"].values()
                if entry["parent"] == parent
            ]
            self.assertGreaterEqual(len(children), 1)
            self.assertLessEqual(len(children), 16)
            area = 0
            for index, child in enumerate(children):
                x0, y0, x1, y1 = child["core_bounds_px"]
                self.assertTrue(left <= x0 < x1 <= right)
                self.assertTrue(top <= y0 < y1 <= bottom)
                self.assertLessEqual(child["anchor"].count("/"), 4)
                area += (x1 - x0) * (y1 - y0)
                for other in children[index + 1 :]:
                    a, b, c, d = other["core_bounds_px"]
                    self.assertFalse(
                        max(x0, a) < min(x1, c) and max(y0, b) < min(y1, d)
                    )
            self.assertEqual(area, (right - left) * (bottom - top))

    def assert_rejected_without_artifacts(self, call: Any, output: Path) -> None:
        with self.assertRaises((ValueError, RuntimeError)):
            call()
        self.assertFalse(output.exists() and any(output.iterdir()))

    def test_explicit_grid_preserves_old_geometry_and_source_coordinates(self) -> None:
        path = self.create(rows=2, columns=3, region=(5, 7, 96, 80))
        manifest = read_json(path)
        self.assertEqual(
            manifest["anchors"]["ROOT/A1"]["core_bounds_px"], [5, 7, 35, 43]
        )
        self.assert_partition(manifest)
        with (
            Image.open(self.source) as original,
            Image.open(path.parent / "overview_raw.png") as raw,
        ):
            self.assertEqual(
                pixel_signature(raw), pixel_signature(original.crop((5, 7, 96, 80)))
            )

    def test_map_identity_is_geometry_sensitive_and_path_independent(self) -> None:
        first = read_json(self.create("one", rows=2, columns=3))
        second = read_json(self.create("two", rows=2, columns=3))
        third = read_json(self.create("three", rows=3, columns=2))
        self.assertEqual(first["map_id"], second["map_id"])
        self.assertEqual(first["anchors"], second["anchors"])
        self.assertNotEqual(first["map_id"], third["map_id"])

    def test_all_focus_is_exact_top_level_coverage_with_raw_fidelity(self) -> None:
        path = self.create(rows=2, columns=3, region=(5, 7, 96, 80))
        output = self.root / "focus"
        result = read_json(foveate.focus_map(path, output, margin=0))
        self.assertEqual(result["coverage"], "complete")
        self.assertEqual(len(result["crops"]), 6)
        with Image.open(self.source) as original:
            for crop in result["crops"]:
                self.assertEqual(crop["crop_bounds_px"], crop["core_bounds_px"])
                with Image.open(output / crop["raw_crop"]) as raw:
                    expected = original.crop(crop["core_bounds_px"])
                    self.assertEqual(pixel_signature(raw), pixel_signature(expected))

    def test_focus_margins_clamp_to_source_not_mapped_view(self) -> None:
        path = self.create(rows=1, columns=1, region=(10, 10, 90, 70))
        result = read_json(foveate.focus_map(path, self.root / "focus", margin=0.5))
        self.assertEqual(result["crops"][0]["crop_bounds_px"], [0, 0, 101, 83])

    def test_successor_is_immutable_and_all_remains_top_only(self) -> None:
        path = self.create(rows=2, columns=2)
        before = path.read_bytes()
        output = self.root / "refine"
        result = read_json(
            foveate.focus_map(path, output, ["ROOT/A1"], child_rows=2, child_columns=2)
        )
        successor_path = output / result["successor_manifest"]
        successor = read_json(successor_path)
        self.assertEqual(path.read_bytes(), before)
        self.assertNotEqual(read_json(path)["map_id"], successor["map_id"])
        self.assert_partition(successor)
        all_result = read_json(foveate.focus_map(successor_path, self.root / "all"))
        self.assertEqual(len(all_result["crops"]), 4)
        self.assertTrue(
            all(crop["anchor"].count("/") == 1 for crop in all_result["crops"])
        )

    def test_focus_cannot_overwrite_its_input_successor_manifest(self) -> None:
        path = self.create(rows=2, columns=2)
        output = self.root / "refine"
        first = read_json(
            foveate.focus_map(path, output, ["ROOT/A1"], child_rows=2, child_columns=2)
        )
        successor_path = output / first["successor_manifest"]
        before = {artifact.name: artifact.read_bytes() for artifact in output.iterdir()}
        with self.assertRaises(ValueError):
            foveate.focus_map(
                successor_path, output, ["ROOT/B1"], child_rows=2, child_columns=2
            )
        self.assertEqual(
            before,
            {artifact.name: artifact.read_bytes() for artifact in output.iterdir()},
        )

    def test_map_cannot_overwrite_existing_map_and_source_artifacts(self) -> None:
        path = self.create(rows=2, columns=2)
        before = {
            artifact.name: artifact.read_bytes() for artifact in path.parent.iterdir()
        }
        with self.assertRaises(ValueError):
            foveate.create_map(self.source, path.parent, rows=3, columns=3)
        self.assertEqual(
            before,
            {
                artifact.name: artifact.read_bytes()
                for artifact in path.parent.iterdir()
            },
        )

    def test_focus_cannot_overwrite_existing_result_artifacts(self) -> None:
        path = self.create(rows=2, columns=2)
        output = self.root / "existing-focus"
        foveate.focus_map(path, output, ["ROOT/A1"], margin=0)
        before = {artifact.name: artifact.read_bytes() for artifact in output.iterdir()}
        with self.assertRaises(ValueError):
            foveate.focus_map(path, output, ["ROOT/A1"], margin=0.5)
        self.assertEqual(
            before,
            {artifact.name: artifact.read_bytes() for artifact in output.iterdir()},
        )

    def test_auto_refinement_respects_remaining_depth_budget(self) -> None:
        ruled_image().save(self.source)
        path = self.create(rows=1, columns=1)
        anchor = "ROOT/A1"
        for depth in (1, 2):
            output = self.root / f"prep-depth{depth}"
            result = read_json(
                foveate.focus_map(path, output, [anchor], child_rows=1, child_columns=1)
            )
            path = output / result["successor_manifest"]
            anchor += "/A1"
        output = self.root / "last-level-auto"
        result = read_json(
            foveate.focus_map(
                path,
                output,
                [anchor],
                mode="auto",
                detectors=("hough",),
                detector_config={"selection": {"min_margin": 0}},
            )
        )
        successor = read_json(output / result["successor_manifest"])
        self.assert_partition(successor)
        self.assertTrue(
            any(entry["parent"] == anchor for entry in successor["anchors"].values())
        )

    def test_refinement_order_does_not_change_map_identity(self) -> None:
        path = self.create(rows=2, columns=2)
        ids: list[str] = []
        for index, anchors in enumerate(
            (["ROOT/A1", "ROOT/B1"], ["ROOT/B1", "ROOT/A1"])
        ):
            output = self.root / f"refine{index}"
            result = read_json(
                foveate.focus_map(path, output, anchors, child_rows=2, child_columns=1)
            )
            ids.append(read_json(output / result["successor_manifest"])["map_id"])
        self.assertEqual(ids[0], ids[1])

    def test_conflicting_refinement_is_rejected_before_output(self) -> None:
        path = self.create(rows=2, columns=2)
        first = self.root / "first"
        result = read_json(
            foveate.focus_map(path, first, ["ROOT/A1"], child_rows=2, child_columns=2)
        )
        output = self.root / "conflict"
        self.assert_rejected_without_artifacts(
            lambda: foveate.focus_map(
                first / result["successor_manifest"],
                output,
                ["ROOT/A1"],
                child_rows=1,
                child_columns=2,
            ),
            output,
        )

    def test_depth_limit_is_atomic(self) -> None:
        path = self.create(rows=1, columns=1)
        anchor = "ROOT/A1"
        for depth in range(1, 4):
            output = self.root / f"depth{depth}"
            result = read_json(
                foveate.focus_map(path, output, [anchor], child_rows=1, child_columns=1)
            )
            path = output / result["successor_manifest"]
            anchor += "/A1"
        output = self.root / "too-deep"
        self.assert_rejected_without_artifacts(
            lambda: foveate.focus_map(
                path, output, [anchor], child_rows=1, child_columns=1
            ),
            output,
        )

    def test_batch_invalid_geometry_does_not_write_first_crop(self) -> None:
        path = self.create(rows=1, columns=16)
        output = self.root / "too-small"
        self.assert_rejected_without_artifacts(
            lambda: foveate.focus_map(
                path, output, ["ROOT/P1", "ROOT/A1"], child_rows=1, child_columns=7
            ),
            output,
        )

    def test_invalid_source_regions_are_atomic(self) -> None:
        for index, region in enumerate(
            (
                (-1, 0, 50, 50),
                (0, 0, 102, 83),
                (0, 0, 101, 84),
                (4, 4, 4, 10),
                (10, 10, 5, 20),
            )
        ):
            with self.subTest(region=region):
                output = self.root / f"invalid{index}"
                self.assert_rejected_without_artifacts(
                    partial(foveate.create_map, self.source, output, 2, 2, region),
                    output,
                )

    def test_invalid_grids_are_atomic(self) -> None:
        for index, grid in enumerate(((0, 1), (1, 0), (-1, 2), (5, 4), (1000, 1))):
            with self.subTest(grid=grid):
                output = self.root / f"invalid{index}"
                self.assert_rejected_without_artifacts(
                    partial(foveate.create_map, self.source, output, *grid),
                    output,
                )

    def test_invalid_focus_requests_are_atomic(self) -> None:
        path = self.create(rows=2, columns=2)
        for index, requested in enumerate(
            (
                [],
                ["ROOT/A1", "ROOT/A1"],
                ["ROOT/Z99"],
                ["ROOT"],
                [f"ROOT/A{i}" for i in range(17)],
            )
        ):
            output = self.root / f"invalid{index}"
            self.assert_rejected_without_artifacts(
                partial(foveate.focus_map, path, output, requested),
                output,
            )

    def test_nonfinite_or_out_of_range_margins_are_rejected(self) -> None:
        path = self.create(rows=2, columns=2)
        for index, margin in enumerate((math.nan, math.inf, -0.1, 0.51)):
            output = self.root / f"margin{index}"
            self.assert_rejected_without_artifacts(
                lambda margin=margin, output=output: foveate.focus_map(
                    path, output, margin=margin
                ),
                output,
            )

    def test_manifest_corruption_is_rejected_without_output(self) -> None:
        path = self.create(rows=2, columns=2)
        original = read_json(path)
        corruptions: list[dict[str, Any]] = [
            {"geometry_version": 999},
            {"map_id": "sha256:invalid"},
            {"source_dimensions": [102, 83]},
            {"view_bounds_px": [-1, 0, 101, 83]},
            {"anchors": {}},
            {"source_id": "sha256:invalid"},
        ]
        for index, change in enumerate(corruptions):
            with self.subTest(change=change):
                damaged = original | change
                damaged_path = path.parent / f"damaged{index}.json"
                damaged_path.write_text(json.dumps(damaged))
                output = self.root / f"corrupt{index}"
                self.assert_rejected_without_artifacts(
                    lambda damaged_path=damaged_path, output=output: foveate.focus_map(
                        damaged_path, output
                    ),
                    output,
                )

    def test_source_pixel_mutation_is_rejected(self) -> None:
        path = self.create(rows=2, columns=2)
        source_path = path.parent / read_json(path)["source_path"]
        with Image.open(source_path) as opened:
            image = opened.copy()
        image.putpixel((0, 0), (255, 255, 255))
        image.save(source_path)
        output = self.root / "mutated"
        self.assert_rejected_without_artifacts(
            lambda path=path, output=output: foveate.focus_map(path, output), output
        )

    def test_palette_color_mutation_is_part_of_source_identity(self) -> None:
        image = Image.new("P", (101, 83), 1)
        palette = [0] * 768
        palette[3:6] = [30, 80, 140]
        image.putpalette(palette)
        image.save(self.source)
        path = self.create(rows=2, columns=2)
        source_path = path.parent / read_json(path)["source_path"]
        with Image.open(source_path) as opened:
            changed = opened.copy()
        changed_palette = changed.getpalette()
        assert changed_palette is not None
        changed_palette[3:6] = [190, 20, 10]
        changed.putpalette(changed_palette)
        changed.save(source_path)
        output = self.root / "changed-palette"
        self.assert_rejected_without_artifacts(
            lambda path=path, output=output: foveate.focus_map(path, output), output
        )

    def test_palette_transparency_and_rgba_crops_preserve_evidence(self) -> None:
        for mode in ("P", "RGBA"):
            with self.subTest(mode=mode):
                if mode == "P":
                    image = Image.new("P", (101, 83), 1)
                    image.putpalette([0, 0, 0, 40, 90, 120] + [0] * 762)
                    image.info["transparency"] = 1
                else:
                    image = Image.new("RGBA", (101, 83), (40, 90, 120, 71))
                image.save(self.source)
                path = self.create(f"map-{mode}", rows=2, columns=2)
                output = self.root / f"focus-{mode}"
                result = read_json(foveate.focus_map(path, output, margin=0))
                for crop in result["crops"]:
                    with Image.open(output / crop["raw_crop"]) as raw:
                        self.assertEqual(
                            pixel_signature(raw),
                            pixel_signature(image.crop(crop["core_bounds_px"])),
                        )

    def test_annotations_honor_transparency_while_raw_crops_keep_hidden_pixels(
        self,
    ) -> None:
        hidden = ruled_image().convert("RGBA")
        hidden.putalpha(0)
        for mode in ("grid", "auto"):
            options: dict[str, Any] = {"mode": mode}
            if mode == "grid":
                options.update(rows=2, columns=2)
            hidden.save(self.source)
            hidden_map = self.create(f"hidden-{mode}", **options)
            Image.new("RGB", hidden.size, "white").save(self.source)
            blank_map = self.create(f"blank-{mode}", **options)
            with (
                Image.open(hidden_map.parent / "overview_annotated.png") as annotated,
                Image.open(blank_map.parent / "overview_annotated.png") as blank,
            ):
                self.assertEqual(pixel_signature(annotated), pixel_signature(blank))
            with Image.open(hidden_map.parent / "overview_raw.png") as raw:
                self.assertEqual(pixel_signature(raw), pixel_signature(hidden))
            hidden_output = self.root / f"hidden-focus-{mode}"
            blank_output = self.root / f"blank-focus-{mode}"
            hidden_focus = read_json(
                foveate.focus_map(hidden_map, hidden_output, ["ROOT/A1"], margin=0)
            )
            blank_focus = read_json(
                foveate.focus_map(blank_map, blank_output, ["ROOT/A1"], margin=0)
            )
            hidden_crop = hidden_focus["crops"][0]
            blank_crop = blank_focus["crops"][0]
            with (
                Image.open(hidden_output / hidden_crop["annotated_crop"]) as annotated,
                Image.open(blank_output / blank_crop["annotated_crop"]) as blank,
            ):
                self.assertEqual(pixel_signature(annotated), pixel_signature(blank))
            with Image.open(hidden_output / hidden_crop["raw_crop"]) as raw:
                self.assertEqual(
                    pixel_signature(raw),
                    pixel_signature(hidden.crop(hidden_crop["core_bounds_px"])),
                )

    def test_exif_orientation_coordinates_use_normalized_source(self) -> None:
        with Image.open(self.source) as image:
            exif = image.getexif()
            exif[274] = 6
            image.save(self.root / "oriented.jpg", exif=exif)
        self.source = self.root / "oriented.jpg"
        path = self.create(rows=2, columns=2)
        manifest = read_json(path)
        self.assertEqual(manifest["source_dimensions"], [83, 101])
        with Image.open(self.source) as opened:
            expected = ImageOps.exif_transpose(opened).copy()
        with Image.open(path.parent / manifest["source_path"]) as actual:
            self.assertEqual(pixel_signature(actual), pixel_signature(expected))

    def legacy_manifest(self) -> Path:
        with Image.open(self.source) as image:
            digest = hashlib.sha256()
            digest.update(image.mode.encode())
            digest.update(f"{image.width}x{image.height}".encode())
            digest.update(image.tobytes())
        source_id = f"sha256:{digest.hexdigest()}"
        grids = {"ROOT": {"bounds_px": [0, 0, 101, 83], "rows": 1, "columns": 2}}
        geometry = {
            "geometry_version": 1,
            "source_id": source_id,
            "view_bounds": [0, 0, 101, 83],
            "grids": grids,
        }
        map_id = (
            "sha256:"
            + hashlib.sha256(
                json.dumps(geometry, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        )
        anchors = {
            "ROOT/A1": {
                "anchor": "ROOT/A1",
                "parent": "ROOT",
                "core_bounds_px": [0, 0, 50, 83],
            },
            "ROOT/B1": {
                "anchor": "ROOT/B1",
                "parent": "ROOT",
                "core_bounds_px": [50, 0, 101, 83],
            },
        }
        manifest = {
            "geometry_version": 1,
            "source_id": source_id,
            "map_id": map_id,
            "source_path": str(self.source),
            "source_dimensions": [101, 83],
            "view_bounds_px": [0, 0, 101, 83],
            "grids": grids,
            "anchors": anchors,
        }
        path = self.root / "legacy.json"
        path.write_text(json.dumps(manifest))
        return path

    def test_independently_constructed_v1_manifest_remains_loadable(self) -> None:
        path = self.legacy_manifest()
        result = read_json(
            foveate.focus_map(path, self.root / "legacy-focus", margin=0)
        )
        self.assertEqual(result["coverage"], "complete")
        self.assertEqual(len(result["crops"]), 2)

    def test_old_v1_palette_source_identity_remains_loadable_and_upgrades(self) -> None:
        image = Image.new("P", (101, 83), 1)
        image.putpalette([0, 0, 0, 40, 90, 120] + [0] * 762)
        image.save(self.source)
        path = self.legacy_manifest()
        result = read_json(
            foveate.focus_map(path, self.root / "legacy-palette", margin=0)
        )
        self.assertEqual(len(result["crops"]), 2)
        output = self.root / "upgraded-palette"
        refined = read_json(foveate.focus_map(path, output, ["ROOT/A1"], mode="auto"))
        successor = output / refined["successor_manifest"]
        after = read_json(foveate.focus_map(successor, self.root / "after-upgrade"))
        self.assertEqual(after["coverage"], "complete")

    def test_default_auto_writes_discovery_and_preserves_partition(self) -> None:
        Image.new("RGB", (101, 83), "white").save(self.source)
        path = self.create()
        self.assertTrue((path.parent / "structures.json").exists())
        self.assert_partition(read_json(path))

    def custom_manifest(self, partitions: dict[str, Any], name: str = "custom") -> Path:
        """Build v2 geometry independently, with a correctly recomputed ID."""
        baseline = self.create(name, rows=1, columns=1)
        manifest = read_json(baseline)
        geometry = {
            "geometry_version": 2,
            "source_id": manifest["source_id"],
            "view_bounds": manifest["view_bounds_px"],
            "partitions": partitions,
        }
        manifest["geometry_version"] = 2
        manifest["partitions"] = partitions
        manifest.pop("grids", None)
        manifest["map_id"] = (
            "sha256:"
            + hashlib.sha256(
                json.dumps(geometry, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        )
        anchors: dict[str, Any] = {}
        for parent, partition in partitions.items():
            if partition["type"] == "regions":
                children = partition["children"]
            else:
                children: list[dict[str, Any]] = []
                xs, ys = partition["x_edges"], partition["y_edges"]
                for row, (top, bottom) in enumerate(pairwise(ys), 1):
                    for col, (left, right) in enumerate(pairwise(xs)):
                        children.append(
                            {
                                "segment": f"{chr(65 + col)}{row}",
                                "core_bounds_px": [left, top, right, bottom],
                            }
                        )
            for child in children:
                anchor = parent + "/" + child["segment"]
                anchors[anchor] = {
                    "anchor": anchor,
                    "parent": parent,
                    "core_bounds_px": child["core_bounds_px"],
                }
        manifest["anchors"] = anchors
        baseline.write_text(json.dumps(manifest))
        return baseline

    def test_unequal_grid_and_region_partitions_have_exact_ownership(self) -> None:
        configurations = [
            {
                "ROOT": {
                    "type": "grid",
                    "bounds_px": [0, 0, 101, 83],
                    "x_edges": [0, 17, 88, 101],
                    "y_edges": [0, 9, 83],
                }
            },
            {
                "ROOT": {
                    "type": "regions",
                    "bounds_px": [0, 0, 101, 83],
                    "children": [
                        {"segment": "R1", "core_bounds_px": [0, 0, 25, 83]},
                        {"segment": "R2", "core_bounds_px": [25, 0, 101, 40]},
                        {"segment": "R3", "core_bounds_px": [25, 40, 101, 83]},
                    ],
                }
            },
        ]
        for index, partitions in enumerate(configurations):
            path = self.custom_manifest(partitions, f"valid{index}")
            self.assert_partition(read_json(path))
            output = self.root / f"custom-focus{index}"
            result = read_json(foveate.focus_map(path, output, margin=0))
            self.assertEqual(result["coverage"], "complete")
            with Image.open(self.source) as original:
                for crop in result["crops"]:
                    with Image.open(output / crop["raw_crop"]) as raw:
                        self.assertEqual(
                            pixel_signature(raw),
                            pixel_signature(original.crop(crop["core_bounds_px"])),
                        )

    def test_signed_but_invalid_partition_geometry_is_rejected(self) -> None:
        region_children = [
            [("R1", [0, 0, 60, 83]), ("R2", [50, 0, 101, 83])],  # overlap
            [("R1", [0, 0, 40, 83]), ("R2", [50, 0, 101, 83])],  # hole
            [("R1", [-1, 0, 50, 83]), ("R2", [50, 0, 101, 83])],
            [("R1", [0, 0, 0, 83]), ("R2", [0, 0, 101, 83])],
            [("R1", [0, 0, 50.5, 83]), ("R2", [50.5, 0, 101, 83])],
            [("R1", [False, 0, 50, 83]), ("R2", [50, 0, 101, 83])],
            [("../escape", [0, 0, 101, 83])],
            [("R1", [0, 0, 50, 83]), ("R1", [50, 0, 101, 83])],
        ]
        for index, children in enumerate(region_children):
            with self.subTest(children=children):
                partitions = {
                    "ROOT": {
                        "type": "regions",
                        "bounds_px": [0, 0, 101, 83],
                        "children": [
                            {"segment": segment, "core_bounds_px": bounds}
                            for segment, bounds in children
                        ],
                    }
                }
                path = self.custom_manifest(partitions, f"invalid-region{index}")
                output = self.root / f"invalid-region-focus{index}"
                self.assert_rejected_without_artifacts(
                    lambda path=path, output=output: foveate.focus_map(path, output),
                    output,
                )

    def test_signed_but_invalid_grid_edges_are_rejected(self) -> None:
        edge_lists = (
            [0, 50, 50, 101],
            [0, 60, 40, 101],
            [1, 50, 101],
            [0, 50, 102],
            [0, 50.5, 101],
            [False, 50, 101],
        )
        for index, xs in enumerate(edge_lists):
            with self.subTest(edges=xs):
                partitions = {
                    "ROOT": {
                        "type": "grid",
                        "bounds_px": [0, 0, 101, 83],
                        "x_edges": xs,
                        "y_edges": [0, 83],
                    }
                }
                path = self.custom_manifest(partitions, f"invalid-grid{index}")
                output = self.root / f"invalid-grid-focus{index}"
                self.assert_rejected_without_artifacts(
                    lambda path=path, output=output: foveate.focus_map(path, output),
                    output,
                )

    def test_oversized_partition_is_rejected_even_with_valid_id(self) -> None:
        partitions = {
            "ROOT": {
                "type": "grid",
                "bounds_px": [0, 0, 101, 83],
                "x_edges": list(range(17)) + [101],
                "y_edges": [0, 83],
            }
        }
        path = self.custom_manifest(partitions)
        output = self.root / "oversized-focus"
        self.assert_rejected_without_artifacts(
            lambda path=path, output=output: foveate.focus_map(path, output), output
        )

    def test_explicit_detector_failure_does_not_silently_use_grid(self) -> None:
        Image.new("RGB", (101, 83), "white").save(self.source)
        for mode in ("hough", "gutters", "panels"):
            output = self.root / mode
            self.assert_rejected_without_artifacts(
                lambda mode=mode, output=output: foveate.create_map(
                    self.source, output, mode=mode
                ),
                output,
            )

    def test_detector_and_grid_options_cannot_conflict(self) -> None:
        for mode in ("hough", "gutters", "panels"):
            output = self.root / mode
            self.assert_rejected_without_artifacts(
                lambda mode=mode, output=output: foveate.create_map(
                    self.source, output, rows=2, columns=2, mode=mode
                ),
                output,
            )

    def test_auto_dimensions_configure_only_fallback_grid(self) -> None:
        Image.new("RGB", (101, 83), "white").save(self.source)
        path = self.create(mode="auto", rows=2, columns=4)
        manifest = read_json(path)
        self.assertEqual(len(foveate.grid_tiles_top(manifest)), 8)
        self.assert_partition(manifest)
        self.assertIsNone(
            read_json(path.parent / "structures.json")["selected_candidate_id"]
        )

    def test_candidate_replay_can_select_below_threshold_geometry(self) -> None:
        ruled_image().save(self.source)
        original = self.create(mode="auto", min_score=1)
        discovery = original.parent / "structures.json"
        report = read_json(discovery)
        self.assertIsNone(report["selected_candidate_id"])
        candidate = next(
            candidate
            for candidate in report["candidates"]
            if candidate["method"] != "grid"
        )
        path = self.create("replayed", candidate=candidate["id"], discovery=discovery)
        replayed = read_json(path)
        self.assertEqual(replayed["partitions"], candidate["partitions"])
        self.assert_partition(replayed)
        for candidate in read_json(path.parent / "structures.json")["candidates"]:
            if "artifacts" in candidate:
                self.assertTrue((path.parent / candidate["artifacts"]["raw"]).exists())
                self.assertTrue(
                    (path.parent / candidate["artifacts"]["annotated"]).exists()
                )

    def test_candidate_replay_rejects_changed_source_view_geometry_and_unknown_id(
        self,
    ) -> None:
        Image.new("RGB", (101, 83), "white").save(self.source)
        original = self.create()
        discovery = original.parent / "structures.json"
        report = read_json(discovery)
        candidate = report["candidates"][0]
        scenarios: list[tuple[dict[str, Any], str, dict[str, Any]]] = []
        wrong_source = json.loads(json.dumps(report))
        wrong_source["source_id"] = "incorrect-source"
        scenarios.append((wrong_source, candidate["id"], {}))
        scenarios.append((report, candidate["id"], {"region": (1, 1, 100, 82)}))
        tampered = json.loads(json.dumps(report))
        tampered["candidates"][0]["partitions"]["ROOT"]["x_edges"][1] += 1
        scenarios.append((tampered, candidate["id"], {}))
        scenarios.append((report, "nonexistent", {}))
        duplicate = json.loads(json.dumps(report))
        duplicate["candidates"].append(duplicate["candidates"][0])
        scenarios.append((duplicate, candidate["id"], {}))
        for index, (data, candidate_id, kwargs) in enumerate(scenarios):
            with self.subTest(index=index):
                report_path = self.root / f"replay-report{index}.json"
                report_path.write_text(json.dumps(data))
                output = self.root / f"invalid-replay{index}"
                self.assert_rejected_without_artifacts(
                    partial(
                        foveate.create_map,
                        self.source,
                        output,
                        candidate=candidate_id,
                        discovery=report_path,
                        **kwargs,
                    ),
                    output,
                )

    def test_candidate_replay_conflicting_options_are_rejected(self) -> None:
        Image.new("RGB", (101, 83), "white").save(self.source)
        path = self.create()
        discovery = path.parent / "structures.json"
        candidate_id = read_json(discovery)["candidates"][0]["id"]
        options = (
            {"mode": "auto"},
            {"rows": 2},
            {"columns": 2},
            {"min_score": 0.8},
            {"detectors": ("hough",)},
            {"detector_config": {"selection": {"min_score": 0.8}}},
        )
        for index, kwargs in enumerate(options):
            output = self.root / f"conflict-replay{index}"
            self.assert_rejected_without_artifacts(
                lambda output=output, kwargs=kwargs: foveate.create_map(
                    self.source,
                    output,
                    candidate=candidate_id,
                    discovery=discovery,
                    **kwargs,
                ),
                output,
            )

    def test_structural_focus_converts_legacy_geometry_to_immutable_v2(self) -> None:
        ruled_image().save(self.source)
        path = self.create(rows=1, columns=1)
        before = path.read_bytes()
        output = self.root / "structural-focus"
        result = read_json(
            foveate.focus_map(path, output, ["ROOT/A1"], mode="hough", margin=0)
        )
        successor_path = output / result["successor_manifest"]
        successor = read_json(successor_path)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(successor["geometry_version"], 2)
        self.assertIn("ROOT/A1", successor["partitions"])
        self.assert_partition(successor)
        all_result = read_json(
            foveate.focus_map(successor_path, self.root / "structural-all")
        )
        self.assertEqual(len(all_result["crops"]), 1)
        self.assertEqual(all_result["crops"][0]["anchor"], "ROOT/A1")
        for report_name in successor["artifacts"]["discovery_reports"].values():
            report = read_json(output / report_name)
            self.assertEqual(report["view_bounds_px"], [0, 0, 400, 300])
            offered = {candidate["id"]: candidate for candidate in report["candidates"]}
            for candidate_id in report["offered_candidate_ids"]:
                artifacts = offered[candidate_id]["artifacts"]
                self.assertTrue((output / artifacts["raw"]).exists())
                self.assertTrue((output / artifacts["annotated"]).exists())

    def test_tiny_auto_views_still_have_positive_exact_ownership(self) -> None:
        for size in ((1, 1), (1, 2), (2, 1), (2, 2)):
            Image.new("RGB", size, "white").save(self.source)
            path = self.create(f"tiny-{size[0]}x{size[1]}")
            self.assert_partition(read_json(path))

    def test_invalid_detector_config_files_leave_no_artifacts(self) -> None:
        configurations: tuple[Any, ...] = (
            [1, 2],
            {"hough": {"unknown": 1}},
            {"selection": {"min_score": -1}},
        )
        for index, config in enumerate(configurations):
            config_path = self.root / f"config{index}.json"
            config_path.write_text(json.dumps(config))
            output = self.root / f"config-output{index}"
            self.assert_rejected_without_artifacts(
                partial(
                    foveate.create_map, self.source, output, detector_config=config_path
                ),
                output,
            )

    def test_structural_refinement_configuration_conflicts_are_atomic(self) -> None:
        path = self.create(rows=2, columns=2)
        configurations = (
            {"mode": "auto", "child_rows": 2, "child_columns": 2},
            {"min_score": 0.8},
            {"mode": "grid"},
            {"mode": "hough", "detectors": ("hough",)},
        )
        for index, kwargs in enumerate(configurations):
            output = self.root / f"invalid-refine{index}"
            self.assert_rejected_without_artifacts(
                lambda output=output, kwargs=kwargs: foveate.focus_map(
                    path, output, **kwargs
                ),
                output,
            )

    def test_cli_map_focus_and_invalid_options(self) -> None:
        output = self.root / "cli"
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "map",
                str(self.source),
                "--output",
                str(output),
                "--mode",
                "grid",
                "--rows",
                "2",
                "--columns",
                "3",
            ],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "focus",
                str(output / "manifest.json"),
                "--output",
                str(self.root / "cli-focus"),
                "--all",
            ],
            check=True,
            capture_output=True,
        )
        result = read_json(self.root / "cli-focus/focus_manifest.json")
        self.assertEqual(len(result["crops"]), 6)
        failed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "map",
                str(self.source),
                "--output",
                str(self.root / "bad-cli"),
                "--mode",
                "unknown",
            ],
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(failed.returncode, 0)
        self.assertFalse((self.root / "bad-cli").exists())


class DiscoveryTests(unittest.TestCase):
    """Use image-only fixtures; the detector never receives reference bounds."""

    def discover(self, image: Image.Image, **kwargs: Any) -> dict[str, Any]:
        view = kwargs.pop("view", [0, 0, image.width, image.height])
        return autostructure.discover(image, view, "test-source-id", **kwargs)

    def test_missing_backend_auto_reports_fallback_and_explicit_fails(self) -> None:
        original_import = autostructure.importlib.import_module

        def without_cv(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "cv2":
                raise ImportError("deliberately unavailable")
            return original_import(name, *args, **kwargs)

        image = Image.new("RGB", (100, 80), "white")
        with patch.object(
            autostructure.importlib, "import_module", side_effect=without_cv
        ):
            report = self.discover(image)
            self.assertIsNone(report["selected_candidate_id"])
            self.assertIn("install", report["fallback_reason"].lower())
            self.assertTrue(
                all(run["status"] == "unavailable" for run in report["detector_runs"])
            )
            for mode in ("hough", "gutters", "panels"):
                with self.subTest(mode=mode), self.assertRaises(ValueError):
                    self.discover(image, mode=mode)

    def test_invalid_configuration_is_rejected_before_backend_import(self) -> None:
        configurations: tuple[dict[str, Any], ...] = (
            {"unknown": {}},
            {"hough": {"made_up": 1}},
            {"gutters": {"min_gutter_width": 0}},
            {"hough": {"max_line_gap": math.inf}},
            {"panels": {"size_tolerance": True}},
            {"selection": {"min_score": 1.1}},
            {"probe": {"max_side": 1024.5}},
        )
        with patch.object(autostructure.importlib, "import_module") as backend:
            for configuration in configurations:
                with self.subTest(config=configuration), self.assertRaises(ValueError):
                    self.discover(Image.new("RGB", (100, 80)), config=configuration)
            backend.assert_not_called()

    def test_invalid_view_and_score_are_rejected(self) -> None:
        image = Image.new("RGB", (100, 80))
        for view in (
            [-1, 0, 100, 80],
            [0, 0, 101, 80],
            [0, 0, 100, 81],
            [0, 0, 0, 80],
            [False, 0, 100, 80],
            [0, 0, 100.0, 80],
        ):
            with self.subTest(view=view), self.assertRaises(ValueError):
                self.discover(image, view=view)
        for score in (math.nan, math.inf, -0.1, 1.1, True):
            with self.subTest(score=score), self.assertRaises(ValueError):
                self.discover(image, min_score=score)

    def test_blank_and_parallel_rules_do_not_claim_structure(self) -> None:
        images = [
            Image.new("RGB", (400, 300), color)
            for color in ("white", "black", "#586074")
        ]
        rules = Image.new("RGB", (400, 300), "white")
        draw = ImageDraw.Draw(rules)
        for y in range(30, 280, 30):
            draw.line((20, y, 380, y), fill="black", width=2)
        images.append(rules)
        for index, image in enumerate(images):
            with self.subTest(index=index):
                report = self.discover(image)
                self.assertIsNone(report["selected_candidate_id"])
                self.assertIsNotNone(report["fallback_reason"])

    def test_text_line_spacing_is_not_automatically_a_panel_layout(self) -> None:
        image = Image.new("RGB", (600, 400), "white")
        draw = ImageDraw.Draw(image)
        for index in range(16):
            draw.text(
                (30, 20 + index * 22),
                "An ordinary paragraph with words of varying length.",
                fill="black",
            )
        report = self.discover(image)
        self.assertIsNone(report["selected_candidate_id"])

    def test_fully_transparent_pixels_cannot_supply_structural_evidence(self) -> None:
        rgba = ruled_image().convert("RGBA")
        rgba.putalpha(0)
        palette = ruled_image().convert("P", palette=Image.Palette.ADAPTIVE)
        palette.info["transparency"] = bytes([0] * 256)
        for image in (rgba, palette):
            with self.subTest(mode=image.mode):
                before = image.tobytes(), image.getpalette(), dict(image.info)
                report = self.discover(image)
                self.assertIsNone(report["selected_candidate_id"])
                self.assertEqual(len(report["candidates"]), 0)
                self.assertEqual(
                    before, (image.tobytes(), image.getpalette(), image.info)
                )

    def test_partial_alpha_visible_grid_is_detectable_without_source_changes(
        self,
    ) -> None:
        image = ruled_image().convert("RGBA")
        image.putalpha(128)
        before = image.tobytes()
        report = self.discover(image, mode="hough")
        self.assertIsNotNone(report["selected_candidate_id"])
        self.assertEqual(before, image.tobytes())

    def test_ruled_table_forced_hough_is_deterministic_and_unmodified(self) -> None:
        image = ruled_image()
        before = pixel_signature(image)
        first = self.discover(image, mode="hough")
        second = self.discover(image, mode="hough")
        self.assertIsNotNone(first["selected_candidate_id"])
        self.assertEqual(first, second)
        self.assertEqual(before, pixel_signature(image))
        candidate = next(
            candidate
            for candidate in first["candidates"]
            if candidate["id"] == first["selected_candidate_id"]
        )
        self.assertGreaterEqual(len(candidate["evidence_bounds_px"]), 4)

    def test_source_coordinates_at_large_scale_and_nonzero_roi(self) -> None:
        image = Image.new("RGB", (2200, 1700), "#efefef")
        image.paste(ruled_image((1800, 1300)), (173, 211))
        view = [173, 211, 1973, 1511]
        report = self.discover(image, view=view, mode="hough")
        transform = report["probes"]["source_transform"]
        self.assertGreater(transform["scale_x"], 1)
        self.assertEqual(transform["offset_x"], 173)
        self.assertEqual(transform["offset_y"], 211)
        for candidate in report["candidates"]:
            for left, top, right, bottom in candidate["evidence_bounds_px"]:
                self.assertTrue(view[0] <= left < right <= view[2])
                self.assertTrue(view[1] <= top < bottom <= view[3])
            self.assertEqual(candidate["partitions"]["ROOT"]["bounds_px"], view)

    def test_restricting_detectors_is_reported_and_respected(self) -> None:
        image = ruled_image()
        report = self.discover(image, detectors=("hough",))
        self.assertEqual(report["enabled_detectors"], ["hough"])
        self.assertTrue(
            all(run["method"] == "hough" for run in report["detector_runs"])
        )
        self.assertTrue(
            all(candidate["methods"] == ["hough"] for candidate in report["candidates"])
        )
        for detectors in ((), ("unknown",), ("hough", "hough")):
            with self.subTest(detectors=detectors), self.assertRaises(ValueError):
                self.discover(image, detectors=detectors)


if __name__ == "__main__":
    unittest.main()
