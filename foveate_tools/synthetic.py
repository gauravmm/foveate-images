"""Truth-bearing deterministic images for structure benchmarks, never detection."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from foveate_tools.calendar import FONT_PATHS, PRESETS, generate_calendar

Box = tuple[int, int, int, int]


def _font(
    size: int, bold: bool = False
) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for name in FONT_PATHS:
        path = Path(name if bold else name.replace("-Bold", ""))
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


@dataclass(frozen=True)
class Workload:
    name: str
    family: str
    split: str
    seed: int
    image_path: Path
    boxes: tuple[Box, ...]
    negative: bool
    unit: str

    def metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "family": self.family,
            "split": self.split,
            "seed": self.seed,
            "image": str(self.image_path),
            "truth_boxes": self.boxes,
            "negative": self.negative,
            "unit": self.unit,
        }


COMMON_FAMILIES = (
    "ruled_table",
    "borderless_panels",
    "repeated_cards",
    "uneven_gutters",
    "missing_panels",
    "text_columns",
    "dark_panels",
    "faint_rules",
    "broken_rules",
    "noisy_panels",
    "small_skew",
    "high_resolution",
    "texture_negative",
    "irregular_negative",
)
HELDOUT_FAMILIES = (
    "mixed_dashboard",
    "contact_sheet",
    "mixed_panel_sizes",
    "line_art_negative",
    "single_frame_negative",
)


def _text_block(
    draw: ImageDraw.ImageDraw, box: Box, rng: random.Random, ink: str, scale: int
) -> None:
    x0, y0, x1, y1 = box
    font = _font(12 * scale)
    phrases = (
        "Inspection record",
        "North sector sample",
        "Quarterly update",
        "Observed details",
    )
    for y in range(y0 + 37 * scale, y1 - 10 * scale, 23 * scale):
        phrase = rng.choice(phrases)
        draw.text((x0 + 10 * scale, y), phrase, font=font, fill=ink)
        if y + 15 * scale < y1 - 10 * scale:
            width = rng.randint(max(1, (x1 - x0) // 3), max(1, x1 - x0 - 22 * scale))
            draw.line(
                (
                    x0 + 10 * scale,
                    y + 17 * scale,
                    x0 + 10 * scale + width,
                    y + 17 * scale,
                ),
                fill=ink,
                width=scale,
            )


def _synthetic(family: str, seed: int) -> tuple[Image.Image, list[Box], bool, str]:
    rng = random.Random(seed)
    scale = 3 if family == "high_resolution" else 1
    width, height = 760 * scale, 600 * scale
    dark = family == "dark_panels"
    bg, ink = ("#20242c", "#d8e0e8") if dark else ("#f7f5ef", "#354354")
    image = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(image)
    boxes: list[Box] = []
    negative = family.endswith("negative")
    unit = "panel"
    if negative:
        if family == "texture_negative":
            # Smooth, multiscale random texture: synthetic photographic surrogate.
            small = Image.new("RGB", (76, 60))
            small.putdata(
                [
                    (
                        rng.randrange(70, 180),
                        rng.randrange(80, 170),
                        rng.randrange(50, 160),
                    )
                    for _ in range(76 * 60)
                ]
            )
            image = small.resize((width, height), Image.Resampling.BICUBIC).filter(
                ImageFilter.GaussianBlur(7)
            )
        elif family == "irregular_negative":
            for _ in range(26):
                x, y = rng.randrange(width), rng.randrange(height)
                radius = rng.randrange(12, 110)
                draw.ellipse(
                    (x - radius, y - radius, x + radius, y + radius),
                    fill=tuple(rng.randrange(50, 220) for _ in range(3)),
                )
        elif family == "single_frame_negative":
            draw.rectangle((25, 25, width - 25, height - 25), outline=ink, width=4)
            draw.text((90, 110), "One framed illustration", font=_font(26), fill=ink)
            for _ in range(15):
                x, y = rng.randrange(80, width - 80), rng.randrange(170, height - 60)
                draw.ellipse((x - 17, y - 17, x + 17, y + 17), fill="#82a5bc")
        else:
            # Accidental structure without populated repeated content; a strict rejection challenge.
            for _ in range(18):
                x, y = rng.randrange(35, width - 100), rng.randrange(35, height - 100)
                if rng.random() < 0.5:
                    draw.rectangle(
                        (x, y, x + rng.randrange(25, 150), y + rng.randrange(20, 120)),
                        outline=ink,
                        width=2,
                    )
                else:
                    draw.line((x, y, rng.randrange(width), y), fill=ink, width=2)
        return image, boxes, True, "none"

    draw.text(
        (30 * scale, 16 * scale),
        "STRUCTURE INSPECTION / SUMMARY",
        font=_font(19 * scale, True),
        fill=ink,
    )
    if family in {"ruled_table", "faint_rules", "broken_rules"}:
        unit = "table cell"
        cols, rows = rng.choice((3, 4)), rng.choice((3, 4))
        xs = [round(38 + i * 682 / cols) for i in range(cols + 1)]
        ys = [round(78 + i * 458 / rows) for i in range(rows + 1)]
        line_ink = "#deddd8" if family == "faint_rules" else ink
        for x in xs:
            if family == "broken_rules":
                for y in range(78, 536, 33):
                    draw.line((x, y, x, min(y + 25, 536)), fill=line_ink, width=2)
            else:
                draw.line((x, 78, x, 536), fill=line_ink, width=2)
        for y in ys:
            if family == "broken_rules":
                for x in range(38, 720, 37):
                    draw.line((x, y, min(x + 29, 720), y), fill=line_ink, width=2)
            else:
                draw.line((38, y, 720, y), fill=line_ink, width=2)
        for row in range(rows):
            for col in range(cols):
                box = (xs[col], ys[row], xs[col + 1], ys[row + 1])
                boxes.append(box)
                _text_block(draw, box, rng, ink, scale)
        return image, boxes, False, unit

    if family == "text_columns":
        unit = "text column"
        for col in range(3):
            box = (35 + col * 245, 82, 35 + col * 245 + 195, 528)
            boxes.append(box)
            draw.text(
                (box[0], box[1]), f"COLUMN {col + 1}", font=_font(17, True), fill=ink
            )
            _text_block(draw, box, rng, ink, scale)
        return image, boxes, False, unit

    cols, rows = (3, 2)
    gap_x, gap_y = rng.randrange(22, 37) * scale, rng.randrange(25, 39) * scale
    panel_w = (width - 70 * scale - 2 * gap_x) // cols
    panel_h = (height - 125 * scale - gap_y) // rows
    for row in range(rows):
        for col in range(cols):
            if family == "missing_panels" and row == 1 and col == 1:
                continue
            x = 35 * scale + col * (panel_w + gap_x)
            y = 78 * scale + row * (panel_h + gap_y)
            w, h = panel_w, panel_h
            if family == "uneven_gutters":
                x += (col * col * 7 - col * 10) * scale
                w -= (col * 7) * scale
            if family in {"mixed_dashboard", "mixed_panel_sizes"}:
                if col == 1 and row == 0:
                    continue
                if col == 0 and row == 0:
                    w = 2 * panel_w + gap_x
                if family == "mixed_panel_sizes" and row == 1 and col == 2:
                    h -= 55
            box = (x, y, x + w, y + h)
            boxes.append(box)
            fill = "#323d49" if dark else rng.choice(("#ffffff", "#eef3f5", "#f4eee5"))
            border = (
                None
                if family == "borderless_panels"
                else ("#96a8ba" if dark else "#64768a")
            )
            if family != "borderless_panels":
                draw.rounded_rectangle(
                    box, radius=7 * scale, fill=fill, outline=border, width=2 * scale
                )
            if family == "contact_sheet":
                # Each tile has independently generated colored picture-like content.
                draw.rectangle(
                    (x + 5, y + 5, x + w - 5, y + h - 30),
                    fill=rng.choice(("#81a38c", "#8498b8", "#c2ab76")),
                )
                for _ in range(9):
                    px, py = (
                        rng.randrange(x + 12, x + w - 20),
                        rng.randrange(y + 12, y + h - 48),
                    )
                    draw.ellipse(
                        (px - 12, py - 12, px + 12, py + 12),
                        fill=rng.choice(("#a9bf99", "#576e88", "#decba2")),
                    )
                draw.text(
                    (x + 10, y + h - 25),
                    f"Photograph {row * cols + col + 1}",
                    font=_font(13),
                    fill=ink,
                )
            else:
                if family != "borderless_panels":
                    draw.rectangle(
                        (
                            x + 3 * scale,
                            y + 3 * scale,
                            x + w - 3 * scale,
                            y + 30 * scale,
                        ),
                        fill="#426b85" if dark else "#d5e3e9",
                    )
                draw.text(
                    (x + 10 * scale, y + 8 * scale),
                    f"PANEL {row * cols + col + 1}",
                    font=_font(15 * scale, True),
                    fill=ink,
                )
                _text_block(draw, box, rng, ink, scale)
    if family == "noisy_panels":
        pixels = image.load()
        assert pixels is not None
        for _ in range(width * height // 35):
            pixels[rng.randrange(width), rng.randrange(height)] = tuple(
                rng.randrange(256) for _ in range(3)
            )
    if family == "small_skew":
        angle = rng.choice((-2.2, 1.8))
        image = image.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=bg)
        # Truth rotates corners around the same canvas centre, then encloses them.
        import math

        a = math.radians(-angle)
        transformed: list[Box] = []
        for box in boxes:
            points = [
                (x - width / 2, y - height / 2)
                for x in (box[0], box[2])
                for y in (box[1], box[3])
            ]
            xs = [x * math.cos(a) - y * math.sin(a) + width / 2 for x, y in points]
            ys = [x * math.sin(a) + y * math.cos(a) + height / 2 for x, y in points]
            transformed.append(
                (
                    max(0, math.floor(min(xs))),
                    max(0, math.floor(min(ys))),
                    min(width, math.ceil(max(xs))),
                    min(height, math.ceil(max(ys))),
                )
            )
        boxes = transformed
    return image, boxes, False, unit


def generate_workloads(output: Path, repeats: int = 2) -> list[Workload]:
    """Freeze a corpus manifest before detection; no detector imports are allowed."""
    output.mkdir(parents=True, exist_ok=True)
    cases: list[Workload] = []
    for split, offset in (("development", 17011), ("heldout-seed", 92083)):
        for index in range(repeats):
            for family_index, family in enumerate(COMMON_FAMILIES):
                seed = offset + index * 1000 + family_index * 17
                name = f"{split}-{family}-{index}"
                image, boxes, negative, unit = _synthetic(family, seed)
                path = output / f"{name}.png"
                image.save(path)
                cases.append(
                    Workload(
                        name, family, split, seed, path, tuple(boxes), negative, unit
                    )
                )
        for preset_index, preset in enumerate(PRESETS):
            config = dict(preset, slug=f"{split}-{preset['slug']}")
            seed = offset + preset_index * 71
            truth_path = generate_calendar(config, output, seed)
            truth = json.loads(truth_path.read_text())
            left, top, _, _ = truth["layout"]["body_bounds_px"]
            pw, ph = config["panel_size"]
            boxes: list[Box] = []
            for month in range(12):
                row, col = divmod(month, config["columns"])
                x, y = left + col * (pw + 18), top + row * (ph + 18)
                boxes.append((x, y, x + pw, y + ph))
            truth["panel_bounds_px"] = boxes
            truth_path.write_text(json.dumps(truth, indent=2) + "\n")
            cases.append(
                Workload(
                    config["slug"],
                    "calendar",
                    split,
                    seed,
                    output / truth["image"],
                    tuple(boxes),
                    False,
                    "month panel",
                )
            )
    for index in range(repeats):
        for family_index, family in enumerate(HELDOUT_FAMILIES):
            seed = 581039 + index * 1000 + family_index * 17
            name = f"heldout-template-{family}-{index}"
            image, boxes, negative, unit = _synthetic(family, seed)
            path = output / f"{name}.png"
            image.save(path)
            cases.append(
                Workload(
                    name,
                    family,
                    "heldout-template",
                    seed,
                    path,
                    tuple(boxes),
                    negative,
                    unit,
                )
            )
    (output / "corpus.json").write_text(
        json.dumps([case.metadata() for case in cases], indent=2) + "\n"
    )
    return cases
