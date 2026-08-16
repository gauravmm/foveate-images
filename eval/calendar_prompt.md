# Calendar extraction task

Extract every colored date cell from the attached calendar image.

Work autonomously and use the loaded `$foveate-images` skill. You must invoke the `foveate` CLI at least once, inspect enough raw crops to cover the complete calendar, and use the legend's printed color names. Do not inspect files outside the current working directory except for invoking installed executables. There are no hidden hints in the filesystem.

Write `answer.json` in the current directory before finishing. It must be valid JSON with this shape:

``` json
{
  "strategy": {
    "rows": 3,
    "columns": 3,
    "region": null,
    "notes": "brief factual description"
  },
  "anchors_visited": ["ROOT/A1"],
  "highlights": [
    {"date": "2026-01-01", "color": "red", "anchor": "ROOT/A1"}
  ]
}
```

Use ISO dates, lowercase legend color names, and one entry per colored cell. Do not include uncolored dates. Check every top-level region and remove duplicates caused by crop margins. After saving the file, respond only with a short completion statement.
