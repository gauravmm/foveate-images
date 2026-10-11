# Automatic image structure discovery

Status: proposed. Version 1 covers Hough lines, gutters, and repeated panels.

## Goal

The first map of an image should discover useful structure before it creates a uniform grid. A calendar should offer month panels when the image supports that layout. A table should offer its visible cells. A page without clear structure should retain the general grid.

Discovery produces region geometry. The vision agent still interprets the regions and decides what evidence to inspect. A detected rectangle does not establish that it contains a month or any other semantic object.

This proposal extends [FOVEATE.md](FOVEATE.md). It preserves source-global coordinates, stable anchors, immutable maps, and lossless raw crops.

## Version 1 scope

Implement three detectors:

1. **Hough:** find local axis-aligned grids and bordered rectangles from line segments.
2. **Gutters:** split content along broad bands of whitespace or background.
3. **Panels:** find repeated rectangular regions from contours, background regions, and alignment.

Use inexpensive image probes to decide which detectors to run. Rank their proposals with a shared policy. Select a clear winner automatically and expose useful alternatives.

OCR-assisted naming and learned layout models remain future work. Version 1 does not infer month names, dates, or reading order from language. It does not download models or call external services.

## First-pass behavior

`map` defaults to automatic discovery when the caller supplies no layout mode or grid dimensions.

```bash
foveate map IMAGE --output DIR
```

The first pass:

1. Normalize EXIF orientation and establish the source identity.
2. Analyze the full image, or the caller's explicit `--region`.
3. Compute cheap probes for line evidence, gutters, and repeated rectangles.
4. Run the detectors whose probes indicate useful evidence.
5. Validate and rank candidate structures.
6. Select the best supported structure, or select the fallback grid.
7. Write the selected map and a report of alternative structures.

Selection does not require an interactive confirmation. The agent receives the selected overview, the raw overview, and the alternatives report. It can use the selected map immediately or request a specific alternative.

For a calendar, an expected result is a page partition with a header, a calendar body, and a footer. The body contains a detected arrangement of month panels. Local date grids can become deeper proposals.

Detection must not assume twelve panels or a particular calendar layout. The current fixtures include several row and column counts. Titles and legends remain available as visual context.

## Modes and caller control

| Mode | Behavior |
| --- | --- |
| `auto` | Probe the image, run suitable detectors, and select a structure or the fallback grid. |
| `grid` | Create the existing uniform grid without discovery. |
| `hough` | Run the Hough detector even if its probe would skip it. |
| `gutters` | Run the gutter detector even if its probe would skip it. |
| `panels` | Run the repeated-panel detector even if its probe would skip it. |

Explicit detector modes bypass detector gating and cross-detector selection. They still require valid geometry and sufficient evidence. If no candidate qualifies, return a diagnostic error. Do not silently replace an explicit detector with a grid.

Examples:

```bash
foveate map IMAGE --output DIR --mode auto
foveate map IMAGE --output DIR --mode hough
foveate map IMAGE --output DIR --mode gutters
foveate map IMAGE --output DIR --mode panels
foveate map IMAGE --output DIR --mode grid --rows 4 --columns 3
```

Preserve existing explicit grid calls. Supplying `--rows` or `--columns` without `--mode` selects `grid`. With `--mode auto`, those options configure the fallback grid only. With an explicit detector mode, reject those options rather than treating them as detected geometry.

Additional controls:

- `--detectors hough,gutters,panels` restricts the detector set in automatic mode. Its default includes all three.
- `--candidate ID` selects an alternative from a previous discovery report for the same source and view. It bypasses ranking and ambiguity thresholds, but never geometry validation.
- `--min-score VALUE` overrides the acceptance threshold. Store the override in the report.
- `--region LEFT,TOP,RIGHT,BOTTOM` restricts discovery to an explicit source region.
- `--detector-config FILE` supplies named detector parameters and selection thresholds. Reject unknown keys and invalid values.

The candidate option requires a report input, such as `--discovery DIR/structures.json`. Reject combinations with detector or grid options that would change its geometry.

Expose parameters for angle tolerance, minimum segment length, maximum line gap, minimum gutter width, panel size tolerance, and alignment tolerance. Express distance parameters as fractions of the analyzed region or estimated feature size where possible.

## Detector deployment heuristics

These are initial engineering heuristics. Freeze their values under a policy version after evaluation. A probe score is evidence for running a detector, not a probability that its result is correct.

Analyze a reduced image with a maximum side of 1024 pixels for cheap probes. Preserve the source-to-probe transform. Verify proposed boundaries against the source image before assigning final bounds. Thin lines that disappear at probe scale need a higher-resolution retry when other probes indicate structured content.

| Evidence in the image | Detector deployment | Expected use |
| --- | --- | --- |
| Several long horizontal and vertical segments | Run Hough first. Run panels if closed or repeated boxes also appear. | Ruled tables, bordered calendars, and spreadsheets. |
| Broad low-occupancy bands between substantial content blocks | Run gutters first. Run panels if blocks have similar sizes or backgrounds. | Borderless calendars, document columns, and dashboard sections. |
| Several similar rectangles or colored header strips aligned in rows or columns | Run panels first. Use Hough or gutters to confirm their boundaries. | Month panels, cards, and contact sheets. |
| Strong evidence for more than one method | Run each supported detector and compare or combine proposals. | Mixed pages with headers, panels, and internal grids. |
| Weak line, gutter, and repetition evidence | Select the fallback grid. | Photographs and irregular compositions. |

Detector order is an optimization. It must not make the result depend on whichever detector finishes first. A strong first result can limit later refinement, but every probe-indicated method gets an opportunity to propose a competing coarse layout.

### Hough probe and detector

Use directional morphology on threshold or edge masks to estimate horizontal and vertical line evidence. Try multiple contrast thresholds so faint rules and dark backgrounds remain detectable.

Initial deployment rule: find at least two supported segments in each orientation. Require segment lengths of at least 10% of the corresponding view dimension. An angle tolerance of three degrees is an initial setting. These values need tuning for small panels and nested grids.

When only one orientation is present, use the evidence to support gutter or panel proposals. Do not claim a two-dimensional grid from parallel lines alone.

The full detector:

1. Detect edges and run a probabilistic Hough transform.
2. Retain nearly horizontal or vertical segments.
3. Merge nearby collinear segments and duplicate edges from thick borders.
4. Preserve each segment's supported extent.
5. Find local rectangles and grid intersections with compatible extents.
6. Group rectangles by alignment, dimensions, and spacing.
7. Verify the proposed boundaries against source-resolution pixels.

Do not extend every line across the whole image. Separate month panels contain independent local grids. Global extension would create false cells in gutters and headers.

Allow small gaps and rounded corners. Keep observed and inferred boundaries separate in the report. Penalize proposals that depend heavily on unsupported boundaries.

Use multiple segment-length scales to distinguish panel borders from internal date rules. Prefer a coarse grouping of repeated panels on the first pass. Retain supported internal grids as descendant proposals.

Reference primitives: [OpenCV Hough transforms](https://docs.opencv.org/4.13.0/d9/db0/tutorial_hough_lines.html) and [directional morphology](https://docs.opencv.org/4.13.0/dd/dd7/tutorial_morph_lines_detection.html).

### Gutter probe and detector

Estimate a foreground mask from local contrast and background appearance. Background need not be white. Compute horizontal and vertical occupancy profiles.

Initial deployment rule: find a low-occupancy band at least 0.5% of the relevant view dimension wide. Require substantial foreground content on both sides. Reject cuts that mainly separate an empty outer margin.

The full detector recursively splits regions at supported gutters. Recompute profiles within each child region. A local gutter need not cross the whole page.

Rank cuts by width, persistence across the region, foreground separation, and the amount of content on each side. Penalize cuts through connected content or estimated text lines. Avoid treating ordinary line spacing as a panel boundary.

Use an estimated text or component height to raise the minimum gap where text dominates. Cap recursion at the navigation depth limit. Preserve headers and footers as regions when they contain content.

Gutters can identify irregular blocks as well as regular grids. Report a grid only when the resulting blocks support consistent row and column alignment.

### Repeated-panel probe and detector

Extract coarse contours and connected background regions. Look for similarly sized rectangular bounds. Colored header strips can support a panel proposal, but a strip alone does not establish the panel's full extent.

Initial deployment rule: find at least three plausible rectangles with widths and heights within 20% of their group medians. Require row or column alignment within 2% of the view dimension. These are starting tolerances, not acceptance guarantees.

The full detector groups candidates using:

- Width, height, and aspect-ratio similarity.
- Shared row and column alignment.
- Consistent gaps between neighboring panels.
- Similar backgrounds or header appearance.
- Border or gutter support around each proposed panel.
- Evidence of substantial content inside each panel.

Reject duplicate and nested detections of the same panel. Separate page-sized frames, small legend swatches, and large panels by scale. Do not require all panels to be identical or every lattice position to be occupied.

A regular cluster can propose a calendar body without reading its headings. A missing panel remains missing evidence. Do not invent it solely to complete a rectangle or reach twelve items.

Reference primitives: [OpenCV contours and connected components](https://docs.opencv.org/doc/doxygen/html/d3/dc0/group__imgproc__shape.html).

## Candidate validation and selection

Each detector returns proposed region bounds, parent relationships, boundary evidence, and method-specific diagnostics. Keep method-specific raw scores separate from the shared selection score.

First reject candidates with invalid bounds, conflicting ownership, excessive depth, or unsupported geometry. Then score candidates using common features:

- **Boundary support:** source pixels support cuts through edges or low-occupancy bands.
- **Structure fit:** proposed regions match the claimed alignment, repetition, or recursive separation pattern.
- **Content integrity:** cuts preserve coherent content and avoid clipping it.
- **Navigation usefulness:** regions expose substantial content at a useful scale.
- **Complexity penalty:** extra regions and inferred boundaries need additional evidence.

An initial weighted score is `0.30 * boundary_support + 0.25 * structure_fit + 0.25 * content_integrity + 0.20 * navigation_usefulness - complexity_penalty`. Normalize the positive features to `[0, 1]`. Clamp the final score to `[0, 1]`. Define feature measurements in the implementation before tuning weights.

Foreground coverage alone is insufficient. A single page-sized box covers all content but offers little navigation value. Tiny boxes around every glyph also make a poor first map.

Merge equivalent proposals before comparing scores. Use corresponding region bounds and parent relationships to establish equivalence. A mean region IoU of at least 0.9 is an initial geometric tolerance, subject to a consistent region assignment. Retain each detector's supporting evidence on the merged proposal.

Initial automatic-selection thresholds:

- Accept the best valid candidate when its score is at least 0.70.
- Require a lead of at least 0.08 over the next non-equivalent candidate.
- If evidence is weak or ambiguous, select the fallback grid and retain alternatives in the report.
- Apply a deterministic geometry-based tie-break only to equivalent candidates.

Include at most three ranked structural alternatives and the grid option in the agent-facing summary. The full report can retain additional diagnostics. Explain selection with concrete evidence, such as repeated panel alignment or broad gutters.

These scores are heuristic rankings. Do not describe them as calibrated confidence probabilities.

## Geometry and manifest changes

The current manifest derives all regions from equal subdivisions. Detected layouts need explicit boundaries and page regions with unequal dimensions.

Introduce a new geometry version with two partition types:

- **Grid partition:** explicit source-global x and y boundary arrays. Equal subdivisions remain a supported special case.
- **Region partition:** explicit rectangular child cores with stable identifiers.

Child cores must cover their parent exactly once. Detection proposals may contain gaps or overlapping boxes, but the accepted navigation partition must resolve them.

Use gutter midpoints as ownership boundaries where panels support a regular arrangement. Keep visible panel bounds as separate evidence metadata. For irregular arrangements, subtract accepted cores from the parent and partition the remaining area into deterministic rectangular residual regions.

Residual regions remain navigable even when they appear blank. Group them under container regions when necessary. If a valid partition cannot satisfy the region and depth limits, reject that candidate and use a coarser candidate or the grid.

Preserve the maximum of 16 immediate children per parent, 16 crops per focus call, and anchor depth four. A local date grid can exceed 16 cells. Such a grid needs intermediate row groups or a coarser navigation partition.

Use existing `A1`-style segments for aligned grids. Add deterministic segments such as `R1` for region partitions. Update anchor validation for the new geometry version. Semantic labels remain optional metadata and never determine anchor identity.

Persist accepted geometry before the agent navigates it. Hash the complete partition tree and source identity into `map_id`. Hash source, view, and canonical proposed geometry into each candidate ID. Detector metadata must not change an anchor's pixel bounds.

Load legacy manifests with the current geometry rules. Reject unsupported versions explicitly. Keep raw crops unresampled and annotation-free. Use reduced images only for detection.

## Reports, alternatives, and refinement

`map` keeps the existing raw and annotated overviews and manifest. It additionally writes `structures.json` with:

- Source identity and analyzed view bounds.
- Requested mode, enabled detectors, and parameter overrides.
- Detector, backend, and selection-policy versions.
- Probe measurements and detector run or skip reasons.
- Candidate IDs, bounds, hierarchy, feature scores, and rejection reasons.
- The selected candidate or fallback reason.
- Artifact paths for the selected view and retained alternatives.

Offer alternatives through their IDs and annotated previews. Pair every preview with access to its raw image. The report is a proposal artifact, not a replacement for the validated map manifest.

Repeated runs with identical source pixels, configuration, and backend versions must reproduce the selected geometry. Sort proposals before assigning IDs. A cache must include source identity, view, detector parameters, backend versions, and policy version in its key.

Ordinary `focus` calls use the accepted map without rerunning detection. An explicit `focus --mode auto` or detector mode may discover child structure within the selected core. It writes an immutable successor manifest. Reject conflicting replacement geometry under an already refined anchor.

Manual child `--rows` and `--columns` retain uniform refinement. Reject combinations with automatic or detector-based refinement. Each structural refinement uses the same validation, selection, and reporting rules as the first map.

`--all` continues to mean every immediate child of `ROOT`. Completeness means the crops cover the mapped view. It does not prove that the agent inspected every nested region or interpreted every item correctly.

## Implementation sequence

1. Add probes and candidate reports without changing existing explicit grid behavior.
2. Implement Hough and gutter candidates with shared validation and scoring.
3. Add repeated-panel grouping and cross-detector proposal merging.
4. Add explicit partitions, residual ownership, and versioned manifest loading.
5. Enable first-pass automatic selection and alternative selection.
6. Add explicit structural refinement through successor manifests.

Keep detection separate from crop arithmetic. Share source normalization and coordinate conversion. OpenCV is a proposed optional detection dependency. Plain grid and legacy focus operations retain the Pillow-only path.

The distributed skill must describe the automatic default, alternative selection, and explicit modes. It must keep its requirements for raw evidence, exhaustive coverage, and overlap ownership.

If detection dependencies are unavailable, automatic mode selects the grid and reports the unavailable capability. Explicit detector modes return installation guidance. Do not install dependencies silently during a map operation.

## Evaluation and acceptance

Use the five existing calendars as smoke tests. Detection receives images only. Ground-truth sidecars are available to the scorer, never the detectors. Extend fixture truth with panel bounds for geometric evaluation.

Add examples with borderless panels, faint rules, broken borders, uneven gutters, missing panels, mixed panel sizes, dark backgrounds, image noise, and small skew. Add documents, tables, dashboards, contact sheets, and photographs. Include negative cases with accidental line or rectangle patterns.

Measure:

- Panel and grid detection precision and recall.
- Region IoU and clipped-content rates.
- Correct hierarchy and separation of panels from internal grids.
- Automatic-selection accuracy, ambiguity, and fallback rates.
- Probe false negatives compared with running all three detectors.
- End-to-end extraction accuracy under matched crop and inference budgets.
- Detection latency, memory, and dependency cost.

Compare uniform grids, each forced detector, gated automatic selection, and automatic selection with every detector enabled. This separates detector quality from gating quality.

Require zero failures for exact partition coverage, source-coordinate mapping, raw crop fidelity, deterministic IDs, immutable predecessors, and legacy manifest loading. Test alternative selection and explicit-mode failure behavior.

Tune thresholds on a development set and freeze them before held-out evaluation. Report results separately for synthetic calendars and other image classes. A successful calendar demonstration alone does not establish general image-structure discovery.

## Future work

### Extended Hough transform for camera perspective

Add a `hough-projective` mode for photographed planar grids. Extend the Hough detector to find two families of lines whose intersections form a grid under a camera transformation. Lines can converge toward vanishing points instead of remaining horizontal, vertical, or parallel in the source image.

Require more grid-point evidence than the axis-aligned detector. An initial acceptance rule is at least nine observed intersections, supported by at least three distinct lines in each family. Points must span the proposed grid rather than cluster in one corner. Do not count intersections obtained only by extending unsupported segments across gaps.

Fit a planar homography from observed intersections to proposed grid coordinates. Use a robust fit to reject outliers. Score the proposal using inlier count, spatial coverage, reprojection error, and consistency of the rectified grid. Reject unstable transforms and proposals supported only by an outer quadrilateral. The nine-point rule is a starting heuristic that needs evaluation.

Deploy this detector when line probes show a repeated grid with consistent convergence or substantial rotation. A weak axis-aligned result alone is insufficient evidence for a camera transformation. A homography handles a planar surface. Curved pages and lens distortion need separate treatment.

When the pipeline selects this mode:

1. Replace the axis-aligned Hough branch for the mapped view. The selected pipeline uses exactly one Hough mode.
2. Deskew and rectify the mapped image with the accepted homography.
3. Create navigation regions from the rectified grid.
4. Run any subsequent gutter or panel analysis on the rectified image.

Automatic discovery may compare both Hough modes as alternative candidates. Once it selects `hough-projective`, exclude axis-aligned Hough from that pipeline, including refinement in the same rectified plane. Do not chain rectification into another axis-aligned Hough pass. An explicit `--mode hough-projective` selects this branch directly and retains the explicit-mode failure behavior.

Preserve the original normalized raster as raw evidence. Write a separate rectified raster and a validity mask. Choose an output canvas that retains the transformed mapped view, including its headers and legends. Mark rectified images and crops as derived, resampled artifacts. Saving them as PNG does not make the camera correction lossless.

The manifest needs explicit source and rectified coordinate frames. Store the homography, its inverse, output dimensions, interpolation policy, and derived-raster identity. Rectified anchors must retain their inverse-mapped source polygons and source bounding boxes. Exact ownership follows the polygons, since their enclosing raw crop rectangles can overlap. Include the transform and rectified geometry in map identity.

Evaluate rotated and perspective-warped fixtures with known transforms, plus real camera images. Measure grid recovery, transform error, clipped content, and downstream extraction accuracy. Include sparse grids, false intersections, curved surfaces, and lens distortion as rejection cases.

Reference: [OpenCV homography estimation and perspective correction](https://docs.opencv.org/4.5.2/d9/dab/tutorial_homography.html).
