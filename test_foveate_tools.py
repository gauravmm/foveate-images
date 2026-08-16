"""Project self-check."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

from foveate_tools.calendar import PRESETS, generate_calendar
from scripts.score_calendar_results import score


def _run_foveate(*args: str) -> None:
    script = Path(__file__).parent / "skills/foveate-images/foveate.py"
    subprocess.run(
        [sys.executable, str(script), *args], check=True, capture_output=True, text=True
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        image_path = root / "source.png"
        Image.new("RGB", (101, 83), "white").save(image_path)
        map_dir = root / "map"
        focus_dir = root / "focus"
        _run_foveate(
            "map",
            str(image_path),
            "--output",
            str(map_dir),
            "--rows",
            "2",
            "--columns",
            "3",
            "--region",
            "5,7,96,80",
        )
        manifest: dict[str, Any] = json.loads((map_dir / "manifest.json").read_text())
        assert manifest["anchors"]["ROOT/A1"]["core_bounds_px"] == [5, 7, 35, 43]

        _run_foveate(
            "focus",
            str(map_dir / "manifest.json"),
            "--output",
            str(focus_dir),
            "--all",
        )
        focus: dict[str, Any] = json.loads(
            (focus_dir / "focus_manifest.json").read_text()
        )
        assert focus["coverage"] == "complete"
        assert len(focus["crops"]) == 6

        truth_path = generate_calendar(PRESETS[0], root / "calendar", 7)
        truth: dict[str, Any] = json.loads(truth_path.read_text())
        assert len(truth["anchor_months"]) == 12
        assert len(truth["highlights"]) > 24

    result = score(
        {"highlights": [{"date": "2026-01-01", "color": "red"}]},
        {"highlights": [{"date": "2026-01-01", "color": "yellow"}]},
    )
    assert result["date"]["true_positive"] == 1
    assert result["date_color"]["true_positive"] == 0
    print("ok")


if __name__ == "__main__":
    main()
