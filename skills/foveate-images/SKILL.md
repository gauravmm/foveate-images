---
name: foveate-images
description: Navigate large or dense images through deterministic string-addressed crops using the bundled foveate script. Use for exhaustive visual extraction, small text, calendars, tables, dashboards, screenshots, charts, or any image task where the agent needs to zoom into regions without repeatedly estimating pixel coordinates.
---

# Foveate Images

Use the bundled `foveate.py` script to replace model-written crop coordinates with labeled regions such as `ROOT/B2`. Treat labels as navigation aids, not semantic claims.

Locate `foveate.py` in the same directory as this file. If `uv` is available, prefer `uv run /absolute/path/to/foveate.py`. Its PEP 723 metadata installs Pillow in an isolated environment. Otherwise, install Pillow in an isolated environment and use `python3 /absolute/path/to/foveate.py`.

## Workflow

1. Inspect the source image and the user's requested output.
2. Choose a coarse grid. Match obvious repeated panels when easy. Otherwise, start with `3 x 3`. Do not search for a perfect grid.
3. Create a map in a task-local directory:

   ```bash
   uv run /absolute/path/to/foveate.py map INPUT \
     --output WORK/map --rows ROWS --columns COLUMNS
   ```

   Use `--region LEFT,TOP,RIGHT,BOTTOM` only when a clear content body excludes headers or legends that would misalign the grid.
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

6. Inspect each `*_raw.png`. Use its paired `*_annotated.png` only when the breadcrumb or location is unclear.
7. If evidence remains unreadable, refine only that anchor and use the successor manifest:

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
