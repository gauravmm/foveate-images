# Foveated vision

Status: implemented prototype

## Goal and verdict

Vision agents can often identify an interesting area but serialize its pixel coordinates unreliably. `foveate` presents a deterministic labeled grid, lets the agent select labels such as `ROOT/B2`, and returns high-resolution crops. The tool owns the coordinate arithmetic and preserves the label-to-pixel map.

String anchors do **not** replace spatial grounding. The model must still read the label, bind it to a region, and decide that the region matters. The testable claims are separate:

1. foveated crops can improve fine-detail perception; and
2. selecting a visible region ID can be more reliable than emitting a coordinate for some models and tasks.

[Set-of-Mark Prompting](https://arxiv.org/html/2310.11441), [Grid-Augmented Vision](https://arxiv.org/abs/2411.18270), and [PIVOT](https://deepmind.google/research/publications/72495/) support selection from visible candidates. They do not establish a general strings-over- coordinates result: candidate quality and annotation are confounds, and some models cannot reliably speak marks. [OSWorld](https://arxiv.org/html/2404.07972) reports a small SoM regression in dense GUIs, while [Qwen2.5-VL](https://qwenlm.github.io/blog/qwen2.5-vl/) demonstrates strong native coordinate output. [V\*](https://arxiv.org/html/2312.14135), [ZoomEye](https://arxiv.org/html/2411.16044), and [Zoom-Refine](https://arxiv.org/html/2506.01663) support foveation independently of string anchors.

## Behavior

Two operations are sufficient:

``` text
map(image, rows=4, columns=3)
  -> raw overview, labeled overview, manifest with ROOT/A1 .. ROOT/C4

focus(map, anchors=[ROOT/B2], rows=3, columns=3, margin=0.08)
  -> raw and labeled ROOT/B2 crops
  -> successor map with ROOT/B2/A1 .. ROOT/B2/C3
```

Columns are `A`, `B`, ... from left to right and rows are `1`, `2`, ... from top to bottom. Refinement appends a segment to the canonical path. The agent chooses the grid dimensions, margin, anchors, and whether to refine again.

The implementation is a Pillow-only CLI:

``` bash
foveate map IMAGE --output DIR [--rows 3 --columns 3] \
  [--region LEFT,TOP,RIGHT,BOTTOM]

foveate focus MANIFEST --output DIR \
  (--all | --anchor ROOT/A1 [--anchor ROOT/B2 ...]) \
  [--margin 0.08] [--rows 3 --columns 3]
```

`--all` means every top-level anchor, including after the map has descendants. Supplying child rows and columns writes a standalone immutable successor manifest that can be passed to another `focus` call. Re-adding the same child grid is idempotent; changing an existing child grid is rejected.

## Outputs and manifest

`map` writes:

- `source.png`: the orientation-normalized decoded raster;
- `overview_raw.png`: the annotation-free mapped region;
- `overview_annotated.png`: the same region with grid and axis labels; and
- `manifest.json`: source identity, map identity, grids, and anchor bounds.

`focus` writes one `*_raw.png` and one `*_annotated.png` per anchor plus `focus_manifest.json`. A refining call also writes `successor_manifest.json`.

A crop entry has the following shape:

``` json
{
  "anchor": "ROOT/B2/C3",
  "core_bounds_px": [426, 320, 604, 506],
  "crop_bounds_px": [412, 306, 618, 520],
  "dimensions_px": [206, 214],
  "source_area_fraction": 0.031,
  "raw_crop": "ROOT__B2__C3_raw.png",
  "annotated_crop": "ROOT__B2__C3_annotated.png"
}
```

`source_id` hashes the normalized raster. `map_id` hashes a canonical serialization of `source_id`, `geometry_version`, the mapped view, and the complete grid tree. Different grids over the same source therefore have different IDs; adding the same grids in a different order produces the same ID. Loading a manifest revalidates its ID, source hash, dimensions, parent geometry, and anchors.

## Geometry and image rules

All bounds are half-open, source-global coordinates in the EXIF-normalized decoded raster. For width `W` and `C` columns, vertical boundaries are `floor(i * W / C)` for `i = 0..C`; rows use the same rule. Core tiles cover their parent exactly once.

The displayed crop expands a core horizontally by `margin * core_width` and vertically by `margin * core_height`. Left/top are floored, right/bottom are ceiled, and all edges are clamped to the source. A child grid always subdivides the parent's **unexpanded core**, so margins provide context without changing ownership or descendant geometry.

Raw crops are annotation-free, unresampled, lossless PNGs of decoded samples. Navigation copies put the breadcrumb in an added gutter; grid marks never alter raw evidence. This separation is required for color-sensitive tasks.

Limits:

- `rows >= 1`, `columns >= 1`, and `rows * columns <= 16`;
- each cell has positive pixel width and height;
- finite `margin` in `[0, 0.5]`;
- at most 16 unique anchors per focus call; and
- maximum anchor depth 4.

Generated paths are uppercase and non-zero-padded: `ROOT(/[A-P](?:[1-9]|1[0-6])){1,4}`. A requested path must exist under its parent in the supplied map.

## Limitations

- Ordered labels such as `A1` encode position even if the model ignores the glyph. Shuffled opaque labels are required to test actual label binding.
- Grid cuts can split content. Margins expose boundary evidence, but the core bounds remain the sole owner for deduplication.
- A crop can lose global semantic context. Breadcrumbs preserve provenance, not meaning.
- An annotated overview can cover small evidence before the right crop is selected. Always provide its raw counterpart to the model.
- For exhaustive tasks, selection must cover every top-level anchor. Finding one target is not evidence that unvisited regions are empty.
- A syntactically valid anchor can still point to the visually wrong region. Manifest validation cannot detect model misbinding.

The original `sched.jpeg` also demonstrates an upstream limitation: its logo and legends mean a whole-image `4 x 3` grid does not align with the months. The calendar body must first be isolated, currently by `--region` or an external layout step.

v0 intentionally excludes OCR, saliency, semantic proposals, automatic grid selection, and learned search policies.

## Evaluation plan

### 1. Geometry and fidelity

Require zero failures for exact core coverage, clamped margins, deterministic nested bounds and map IDs, refinement-order independence, immutable predecessor maps, EXIF normalization, manifest validation, and annotation-free raw crops.

### 2. Isolate causes under matched budgets

Use the same model, pixels, post-clamp crop area, margin policy, call count, image tokens, and generated-token allowance where a contrast claims causality.

| Condition | What it measures |
| ---------------------------------------------------- | -------------------------------------- |
| Raw full image -\> final answer | Direct baseline |
| Oracle same-area crop -\> answer | Upper bound for useful foveation |
| Raw image -\> coordinate point -\> fixed crop | Coordinate-crop pipeline |
| Grid lines -\> row/column -\> same crop | Discretization |
| Labeled grid -\> row/column -\> same crop | Cost/effect of label pixels |
| Identical labeled grid -\> region ID -\> same crop | ID serialization/binding |
| Ordered IDs vs shuffled opaque IDs | Position convention vs glyph binding |
| Matched recursive coordinate and ID variants | Multi-step behavior |

Include an equal-token full-image re-view control whenever foveation receives more inference compute.

### 3. Annotation and context ablations

Compare raw-only, navigation-only, and paired inputs; lines only versus axis labels versus per-cell badges; labels over versus away from targets; `0%`, `8%`, and `20%` margins; and tasks whose answer depends on a header outside the selected tile.

### 4. Representative tasks

Use documents, charts, screenshots, photographs, and calendars. Include single-target and exhaustive cases with targets at cell centers, edges, and corners and with clean, dense, text-heavy, and color-sensitive backgrounds.

For calendars, score the exact set of `(date, color)` pairs. Report target inclusion, crop IoU and area, wrong/malformed anchors, exact-set precision, recall and F1, calls, image tokens, latency, cost, and boundary-stratified accuracy.

Pre-register separate gates:

- **foveation:** coordinate-crop pipeline versus direct full-image answer; and
- **string anchoring:** region ID versus labeled-grid row/column with identical input pixels.

An initial materiality threshold is a paired 95% confidence interval excluding zero and at least `+5` percentage points in final-task accuracy on two intended minimal models, with a one-sided non-inferiority interval excluding more than a `5` point regression on boundary and text-heavy strata. If improvement comes only from larger crops, extra calls, or extra image tokens, reject the relevant claim.
