---
name: foveate-images
description: Navigate large or dense images through deterministic string-addressed crops using the bundled foveate script. Use for exhaustive visual extraction, small text, calendars, tables, dashboards, screenshots, charts, or any image task where the agent needs to zoom into regions without repeatedly estimating pixel coordinates.
---

# Foveate Images

Use the bundled `foveate.py` script to replace model-written crop coordinates with labeled regions such as `ROOT/B2`. Treat labels as navigation aids, not semantic claims.

Keep `foveate.py` and `autostructure.py` in the same directory as this file. For automatic discovery, use `uv run --with opencv-python-headless --with numpy /absolute/path/to/foveate.py`. Pillow is declared in the script metadata. These dependencies run in an isolated environment. With Python, install Pillow, NumPy, and OpenCV in an isolated environment first. Grid and existing crop operations require only Pillow. Automatic mode falls back to the grid if detection dependencies are unavailable and records the reason.

Use fresh output directories for maps and focus batches; existing artifacts cannot be overwritten. Transparent pixels are composited onto white for analysis and annotated navigation. Raw evidence retains its original alpha and pixels.

## Workflow

1. Inspect the source image and the user's requested output.
2. Create an automatic map in a task-local directory:

   ```bash
   uv run --with opencv-python-headless --with numpy \
     /absolute/path/to/foveate.py map INPUT --output WORK/map
   ```

   Use `--region LEFT,TOP,RIGHT,BOTTOM` only when the task explicitly targets a source region. Otherwise, keep headers and legends available.
3. Read `structures.json` for the selected structure, fallback reason, and alternative candidate IDs. Use the default selection when its regions match the task. Use `--mode hough`, `--mode gutters`, or `--mode panels` for a specific detector. Use `--mode grid --rows ROWS --columns COLUMNS` for an explicit grid. An explicit detector reports failure instead of silently using a grid.

   To choose an alternative, create a separate map:

   ```bash
   uv run /absolute/path/to/foveate.py map INPUT --output WORK/alternative \
     --discovery WORK/map/structures.json --candidate CANDIDATE_ID
   ```

4. Inspect `overview_raw.png` and `overview_annotated.png`. Use the annotated image to bind anchors and the raw image as evidence.
5. Focus the required anchors:

   ```bash
   uv run /absolute/path/to/foveate.py focus WORK/map/manifest.json \
     --output WORK/focus --anchor ROOT/B2 --margin 0.08
   ```

   For exhaustive tasks, cover every top-level region in one call:

   ```bash
   uv run /absolute/path/to/foveate.py focus WORK/map/manifest.json \
     --output WORK/focus --all --margin 0.08
   ```

   A detected map can already contain descendants. `--all` covers only immediate children of `ROOT`. When a parent remains unreadable, inspect its descendants from the manifest. Batch at most 16 anchors per call. Every relevant leaf core must receive readable inspection before an exhaustive answer.

6. Inspect each `*_raw.png`. Use its paired `*_annotated.png` only when the breadcrumb or location is unclear.
7. If evidence remains unreadable, refine only that anchor and use the successor manifest. `--mode auto` discovers local structure. Manual grid dimensions remain available:

   ```bash
   uv run /absolute/path/to/foveate.py focus WORK/map/manifest.json \
     --output WORK/refine --anchor ROOT/B2 --rows 2 --columns 2
   uv run /absolute/path/to/foveate.py focus WORK/refine/successor_manifest.json \
     --output WORK/detail --anchor ROOT/B2/A1
   ```

8. Return the user's requested answer without a navigation transcript.

## Exhaustive and color-sensitive tasks

- Visit every top-level core tile. An overview can hide small targets.
- Treat overlap as context. Assign repeated content to the tile whose core owns it.
- Confirm the month, row header, panel title, or other context before you record local evidence.
- Read colors from raw images. Grid marks and gutters in annotated copies can bias interpretation.
- Use a source legend's printed color names as the output vocabulary.
- Keep a scratch list for each anchor. Merge the lists and check completeness before you answer.

## Guardrails

- Never infer that an anchor contains a target because it was requested.
- Never answer an exhaustive task after you inspect only promising regions.
- Prefer one coarse map plus selected refinement over repeated speculative coordinate crops.
- Keep rows times columns at or below 16 and refinement depth at or below 4.
- Treat detector scores as heuristic rankings. They do not prove that a region contains the requested evidence.
