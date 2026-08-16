#!/usr/bin/env python3
"""Generate deterministic calendar images with exact highlight truth."""

from __future__ import annotations

import argparse
import calendar
import json
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

FONT_PATHS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
)
WEEKDAY_INITIALS = ("M", "T", "W", "T", "F", "S", "S")

PRESETS: list[dict[str, Any]] = [
    {
        "slug": "portrait_pastel_2024",
        "year": 2024,
        "columns": 3,
        "rows": 4,
        "firstweekday": calendar.MONDAY,
        "panel_size": (300, 250),
        "palette": {"rose": "#ef8f9c", "gold": "#f6d66f"},
        "page_bg": "#f5f1e8",
        "panel_bg": "#fffdf8",
        "ink": "#272522",
        "header_bg": "#dce8df",
    },
    {
        "slug": "landscape_cool_2025",
        "year": 2025,
        "columns": 4,
        "rows": 3,
        "firstweekday": calendar.SUNDAY,
        "panel_size": (270, 255),
        "palette": {"blue": "#75b9e7", "green": "#8dd39e"},
        "page_bg": "#edf3f8",
        "panel_bg": "#ffffff",
        "ink": "#17324d",
        "header_bg": "#c8ddec",
    },
    {
        "slug": "classic_warm_2026",
        "year": 2026,
        "columns": 3,
        "rows": 4,
        "firstweekday": calendar.MONDAY,
        "panel_size": (286, 238),
        "palette": {"red": "#ef7870", "yellow": "#f5cf58"},
        "page_bg": "#ffffff",
        "panel_bg": "#ffffff",
        "ink": "#222222",
        "header_bg": "#eeeeee",
    },
    {
        "slug": "tall_vivid_2027",
        "year": 2027,
        "columns": 2,
        "rows": 6,
        "firstweekday": calendar.SUNDAY,
        "panel_size": (330, 225),
        "palette": {"violet": "#b495e8", "orange": "#f3a45f", "mint": "#70c9b2"},
        "page_bg": "#25232d",
        "panel_bg": "#fbf9ff",
        "ink": "#24202b",
        "header_bg": "#ded5ef",
    },
    {
        "slug": "wide_bright_2028",
        "year": 2028,
        "columns": 6,
        "rows": 2,
        "firstweekday": calendar.MONDAY,
        "panel_size": (230, 242),
        "palette": {"cyan": "#66cfdd", "magenta": "#ea82bd"},
        "page_bg": "#fff8e7",
        "panel_bg": "#fffefa",
        "ink": "#293134",
        "header_bg": "#f0dfb4",
    },
]


def _font(
    size: int, bold: bool = False
) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    paths = (
        FONT_PATHS if bold else tuple(path.replace("-Bold", "") for path in FONT_PATHS)
    )
    for path in paths:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _centered(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    text: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    fill: str,
) -> None:
    text_box = draw.textbbox((0, 0), text, font=font)
    width, height = text_box[2] - text_box[0], text_box[3] - text_box[1]
    x = box[0] + (box[2] - box[0] - width) / 2
    y = box[1] + (box[3] - box[1] - height) / 2 - text_box[1]
    draw.text((x, y), text, font=font, fill=fill)


def _highlights(
    year: int, palette: dict[str, str], rng: random.Random
) -> dict[tuple[int, int], str]:
    choices: dict[tuple[int, int], str] = {}
    names = list(palette)
    for month in range(1, 13):
        days = calendar.monthrange(year, month)[1]
        count = rng.randint(2, 7)
        for index, day in enumerate(sorted(rng.sample(range(1, days + 1), count))):
            choices[(month, day)] = names[
                (index + month + rng.randrange(len(names))) % len(names)
            ]
    return choices


def generate_calendar(config: dict[str, Any], output_dir: Path, seed: int) -> Path:
    """Write one image and its JSON truth sidecar; return the truth path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    panel_w, panel_h = config["panel_size"]
    columns, rows = config["columns"], config["rows"]
    gap, pad, title_h, legend_h, footer_h = 18, 32, 82, 54, 34
    slot_w, slot_h = panel_w + gap, panel_h + gap
    body = [
        pad,
        title_h + legend_h,
        pad + columns * slot_w,
        title_h + legend_h + rows * slot_h,
    ]
    width, height = body[2] + pad, body[3] + footer_h
    image = Image.new("RGB", (width, height), config["page_bg"])
    draw = ImageDraw.Draw(image)
    title_fill = "#f8f8f8" if config["page_bg"] == "#25232d" else config["ink"]
    _centered(
        draw,
        (0, 10, width, title_h),
        f"{config['year']} YEAR PLANNER",
        _font(32, True),
        title_fill,
    )

    legend_font = _font(18, True)
    legend_width = sum(
        44 + draw.textlength(name.upper(), font=legend_font) + 28
        for name in config["palette"]
    )
    legend_x = (width - legend_width) / 2
    for name, color in config["palette"].items():
        draw.rounded_rectangle(
            (legend_x, title_h + 10, legend_x + 34, title_h + 38),
            radius=6,
            fill=color,
            outline=config["ink"],
            width=1,
        )
        draw.text(
            (legend_x + 44, title_h + 14),
            name.upper(),
            font=legend_font,
            fill=title_fill,
        )
        legend_x += 44 + draw.textlength(name.upper(), font=legend_font) + 28

    highlights = _highlights(config["year"], config["palette"], rng)
    month_font = _font(max(18, panel_w // 14), True)
    weekday_font = _font(max(12, panel_w // 23), True)
    day_font = _font(max(13, panel_w // 21))
    first_weekday = int(config["firstweekday"])
    cal = calendar.Calendar(first_weekday)
    weekday_order = [(first_weekday + index) % 7 for index in range(7)]
    anchor_months: dict[str, str] = {}

    for month in range(1, 13):
        row, column = divmod(month - 1, columns)
        x, y = body[0] + column * slot_w, body[1] + row * slot_h
        panel = (x, y, x + panel_w, y + panel_h)
        draw.rounded_rectangle(
            panel, radius=10, fill=config["panel_bg"], outline="#817d75", width=2
        )
        header_h, weekday_h = 34, 25
        draw.rounded_rectangle(
            (x, y, x + panel_w, y + header_h), radius=10, fill=config["header_bg"]
        )
        draw.rectangle(
            (x, y + header_h - 10, x + panel_w, y + header_h), fill=config["header_bg"]
        )
        _centered(
            draw,
            (x, y, x + panel_w, y + header_h),
            calendar.month_name[month].upper(),
            month_font,
            config["ink"],
        )

        cell_w = panel_w / 7
        grid_top = y + header_h + weekday_h
        cell_h = (panel_h - header_h - weekday_h) / 6
        for index, weekday in enumerate(weekday_order):
            _centered(
                draw,
                (x + index * cell_w, y + header_h, x + (index + 1) * cell_w, grid_top),
                WEEKDAY_INITIALS[weekday],
                weekday_font,
                config["ink"],
            )
        for index in range(1, 7):
            line_y = grid_top + index * cell_h
            draw.line((x, line_y, x + panel_w, line_y), fill="#d5d2cb", width=1)
        for index in range(1, 7):
            line_x = x + index * cell_w
            draw.line((line_x, grid_top, line_x, y + panel_h), fill="#d5d2cb", width=1)

        weeks = cal.monthdayscalendar(config["year"], month)
        weeks += [[0] * 7 for _ in range(6 - len(weeks))]
        for week_index, week in enumerate(weeks[:6]):
            for day_index, day in enumerate(week):
                if not day:
                    continue
                cell = (
                    x + day_index * cell_w,
                    grid_top + week_index * cell_h,
                    x + (day_index + 1) * cell_w,
                    grid_top + (week_index + 1) * cell_h,
                )
                color_name = highlights.get((month, day))
                if color_name:
                    inset = max(2, int(min(cell_w, cell_h) * 0.09))
                    draw.rounded_rectangle(
                        (
                            cell[0] + inset,
                            cell[1] + inset,
                            cell[2] - inset,
                            cell[3] - inset,
                        ),
                        radius=5,
                        fill=config["palette"][color_name],
                    )
                _centered(draw, cell, str(day), day_font, config["ink"])
        anchor_months[f"ROOT/{chr(65 + column)}{row + 1}"] = calendar.month_name[month]

    _centered(
        draw,
        (0, body[3], width, height),
        f"Week starts {calendar.day_name[config['firstweekday']]}",
        _font(15),
        title_fill,
    )
    image_path = output_dir / f"{config['slug']}.png"
    truth_path = output_dir / f"{config['slug']}.truth.json"
    image.save(image_path, format="PNG")
    truth: dict[str, Any] = {
        "image": image_path.name,
        "seed": seed,
        "year": config["year"],
        "layout": {
            "rows": rows,
            "columns": columns,
            "first_weekday": calendar.day_name[config["firstweekday"]],
            "body_bounds_px": body,
        },
        "palette": config["palette"],
        "anchor_months": anchor_months,
        "highlights": [
            {
                "date": f"{config['year']}-{month:02d}-{day:02d}",
                "color": color,
                "hex": config["palette"][color],
            }
            for (month, day), color in sorted(highlights.items())
        ],
    }
    truth_path.write_text(json.dumps(truth, indent=2) + "\n")
    return truth_path


def _contact_sheet(truth_paths: list[Path], output_path: Path) -> None:
    thumb_w, thumb_h, label_h = 520, 400, 36
    canvas = Image.new("RGB", (thumb_w * 2, (thumb_h + label_h) * 3), "#d8d8d8")
    draw = ImageDraw.Draw(canvas)
    for index, truth_path in enumerate(truth_paths):
        truth = json.loads(truth_path.read_text())
        with Image.open(truth_path.parent / truth["image"]) as image:
            image.thumbnail((thumb_w - 16, thumb_h - 16))
            column, row = index % 2, index // 2
            x = column * thumb_w + (thumb_w - image.width) // 2
            y = row * (thumb_h + label_h) + label_h + (thumb_h - image.height) // 2
            canvas.paste(image.convert("RGB"), (x, y))
            draw.text(
                (column * thumb_w + 12, row * (thumb_h + label_h) + 8),
                truth_path.stem.removesuffix(".truth"),
                font=_font(18, True),
                fill="black",
            )
    canvas.save(output_path, format="PNG")


def generate_set(output_dir: Path, seed: int = 20260816) -> list[Path]:
    truth_paths = [
        generate_calendar(config, output_dir, seed + index)
        for index, config in enumerate(PRESETS)
    ]
    _contact_sheet(truth_paths, output_dir / "contact_sheet.png")
    return truth_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("runs/synthetic-calendars/generated")
    )
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()
    paths = generate_set(args.output, args.seed)
    print(f"generated {len(paths)} calendars in {args.output}")


if __name__ == "__main__":
    main()
