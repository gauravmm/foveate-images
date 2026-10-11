"""Benchmark image-only discovery and write an untracked standalone HTML report.

Run: uv run --extra detection python scripts/benchmark_autostructure.py
The truth-bearing generator and scorer are deliberately outside detector code.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import importlib.util
import io
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, cast

from PIL import Image, ImageDraw

from foveate_tools.synthetic import Workload, generate_workloads

MODES = ("auto", "grid", "hough", "gutters", "panels", "auto-ungated")


def iou(a: list[int] | tuple[int, ...], b: list[int] | tuple[int, ...]) -> float:
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union else 0.0


def maximum_weight_assignment(weights: list[list[float]]) -> list[tuple[int, int]]:
    """Maximum-weight bipartite assignment, padded square Hungarian algorithm."""
    rows, cols = len(weights), len(weights[0]) if weights else 0
    size = max(rows, cols)
    if not size:
        return []
    u, v, p, way = (
        [0.0] * (size + 1),
        [0.0] * (size + 1),
        [0] * (size + 1),
        [0] * (size + 1),
    )
    for row in range(1, size + 1):
        p[0] = row
        j0 = 0
        minv, used = [float("inf")] * (size + 1), [False] * (size + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], float("inf"), 0
            for col in range(1, size + 1):
                if not used[col]:
                    weight = (
                        weights[i0 - 1][col - 1] if i0 <= rows and col <= cols else 0.0
                    )
                    cur = -weight - u[i0] - v[col]
                    if cur < minv[col]:
                        minv[col], way[col] = cur, j0
                    if minv[col] < delta:
                        delta, j1 = minv[col], col
            for col in range(size + 1):
                if used[col]:
                    u[p[col]] += delta
                    v[col] -= delta
                else:
                    minv[col] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    return [
        (p[col] - 1, col - 1)
        for col in range(1, size + 1)
        if 0 < p[col] <= rows and col <= cols
    ]


def score_boxes(
    predicted: list[list[int]], truth: tuple[tuple[int, int, int, int], ...]
) -> dict[str, Any]:
    overlaps = [[iou(pred, actual) for actual in truth] for pred in predicted]
    bonus = max(len(predicted), len(truth)) + 1
    weights = [
        [bonus + value if value >= 0.5 else 0.0 for value in row] for row in overlaps
    ]
    matches = [
        (p, t) for p, t in maximum_weight_assignment(weights) if overlaps[p][t] >= 0.5
    ]
    tp, fp, fn = len(matches), len(predicted) - len(matches), len(truth) - len(matches)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    clipped = 0
    for p, t in matches:
        pred, actual = predicted[p], truth[t]
        intersection = max(0, min(pred[2], actual[2]) - max(pred[0], actual[0])) * max(
            0, min(pred[3], actual[3]) - max(pred[1], actual[1])
        )
        actual_area = (actual[2] - actual[0]) * (actual[3] - actual[1])
        clipped += intersection / actual_area < 0.98
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "mean_iou": statistics.mean(overlaps[p][t] for p, t in matches)
        if matches
        else None,
        "matched_geometry_clipped_fraction": clipped / tp if tp else None,
        "matches": matches,
    }


def _grid(width: int, height: int) -> list[list[int]]:
    return [
        [c * width // 3, r * height // 3, (c + 1) * width // 3, (r + 1) * height // 3]
        for r in range(3)
        for c in range(3)
    ]


def _candidate_boxes(candidate: dict[str, Any]) -> list[list[int]]:
    return candidate.get("evidence_bounds_px", [])


def _run(
    case: Workload, mode: str, discover: Any, image: Image.Image
) -> dict[str, Any]:
    record: dict[str, Any] = dict(case.metadata(), mode=mode)
    start = time.perf_counter()
    selected: dict[str, Any] | None = None
    candidates: list[dict[str, Any]] = []
    try:
        if mode == "grid":
            report: dict[str, Any] = {
                "selected_candidate_id": None,
                "fallback_reason": "explicit 3x3 baseline",
                "candidates": [],
            }
        else:
            report = discover(
                image,
                [0, 0, image.width, image.height],
                "sha256:"
                + hashlib.sha256(
                    image.mode.encode()
                    + f"{image.width}x{image.height}".encode()
                    + image.tobytes()
                ).hexdigest(),
                mode="auto" if mode == "auto-ungated" else mode,
                config={"force_all_detectors": True}
                if mode == "auto-ungated"
                else None,
            )
        candidates = report.get("candidates", [])
        selected = next(
            (
                candidate
                for candidate in candidates
                if candidate["id"] == report.get("selected_candidate_id")
            ),
            None,
        )
        record["status"] = (
            "selected" if selected else "grid" if mode == "grid" else "fallback"
        )
        record["fallback_reason"] = report.get("fallback_reason")
        record["probes"] = report.get("probes")
        record["detector_runs"] = report.get("detector_runs", report.get("detectors"))
        record["selected_score"] = selected.get("score") if selected else None
        record["selected_method"] = (
            selected.get("method", selected.get("methods")) if selected else None
        )
        record["selected_candidate_id"] = report.get("selected_candidate_id")
        record["candidate_count"] = len(candidates)
        record["candidate_ids"] = [candidate["id"] for candidate in candidates]
        record["candidates"] = [
            {
                key: candidate.get(key)
                for key in (
                    "id",
                    "method",
                    "methods",
                    "score",
                    "features",
                    "evidence_bounds_px",
                    "rejection_reasons",
                )
            }
            for candidate in candidates
        ]
    except ValueError as exc:
        record.update(status="rejected", error=str(exc), candidate_count=0)
    except Exception as exc:  # noqa: BLE001 - benchmark records unexpected failures
        record.update(
            status="error", error=f"{type(exc).__name__}: {exc}", candidate_count=0
        )
    record["latency_ms"] = (time.perf_counter() - start) * 1000
    evidence = _candidate_boxes(selected) if selected else []
    predictions = (
        evidence
        if selected
        else _grid(image.width, image.height)
        if mode in {"grid", "auto", "auto-ungated"}
        else []
    )
    record["predicted_boxes"] = predictions
    record["evidence_boxes"] = evidence
    record["metrics"] = score_boxes(predictions, case.boxes)
    record["evidence_metrics"] = score_boxes(evidence, case.boxes)
    record["oracle_f1"] = max(
        (score_boxes(_candidate_boxes(c), case.boxes)["f1"] for c in candidates),
        default=0.0,
    )
    record["image_size"] = list(image.size)
    return record


def _percentile(values: list[float], fraction: float) -> float:
    return (
        sorted(values)[min(len(values) - 1, int((len(values) - 1) * fraction))]
        if values
        else 0.0
    )


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [r for r in rows if not r["negative"]]
    negatives = [r for r in rows if r["negative"]]
    tp = sum(r["metrics"]["tp"] for r in positives)
    fp = sum(r["metrics"]["fp"] for r in positives)
    fn = sum(r["metrics"]["fn"] for r in positives)
    precision, recall = (
        tp / (tp + fp) if tp + fp else 0.0,
        tp / (tp + fn) if tp + fn else 0.0,
    )
    values = [r["latency_ms"] for r in rows]
    return {
        "n": len(rows),
        "positive_n": len(positives),
        "negative_n": len(negatives),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "macro_f1": statistics.mean(r["metrics"]["f1"] for r in positives)
        if positives
        else 0.0,
        "negative_structure_acceptance": sum(
            r["status"] == "selected" for r in negatives
        )
        / len(negatives)
        if negatives
        else None,
        "selected": sum(r["status"] == "selected" for r in rows),
        "fallback": sum(r["status"] == "fallback" for r in rows),
        "rejected": sum(r["status"] == "rejected" for r in rows),
        "errors": sum(r["status"] == "error" for r in rows),
        "median_ms": statistics.median(values) if values else 0.0,
        "p95_ms": _percentile(values, 0.95),
    }


def _sample(case: Workload, row: dict[str, Any]) -> str:
    with Image.open(case.image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    line_width = max(2, image.width // 300)
    for box in case.boxes:
        draw.rectangle(box, outline="#16a34a", width=line_width)
    for box in row["predicted_boxes"]:
        draw.rectangle(box, outline="#f97316", width=line_width)
    image.thumbnail((660, 460))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _integration(cases: list[Workload], output: Path) -> list[dict[str, Any]]:
    output.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="run-", dir=output))
    path = Path(__file__).resolve().parents[1] / "skills/foveate-images/foveate.py"
    spec = importlib.util.spec_from_file_location("foveate_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checks: list[dict[str, Any]] = []
    selected = [
        next(c for c in cases if c.family == family)
        for family in (
            "calendar",
            "ruled_table",
            "borderless_panels",
            "missing_panels",
            "text_columns",
            "dark_panels",
            "high_resolution",
            "irregular_negative",
        )
    ]
    for case in selected:
        item: dict[str, Any] = {"case": case.name}
        start = time.perf_counter()
        try:
            run_dir = output / case.name
            manifest_path = module.create_map(
                case.image_path, run_dir / "map", mode="auto"
            )
            item["manifest"] = str(manifest_path)
            before = manifest_path.read_bytes()
            manifest = json.loads(before)
            repeat_path = module.create_map(
                case.image_path, run_dir / "repeat", mode="auto"
            )
            repeat = json.loads(repeat_path.read_text())
            item["deterministic_map_id"] = manifest["map_id"] == repeat["map_id"]
            focus_path = module.focus_map(manifest_path, run_dir / "focus", margin=0)
            focus = json.loads(focus_path.read_text())
            item["immutable_predecessor"] = manifest_path.read_bytes() == before
            fidelity = True
            source = module._normalized_image(case.image_path)
            for crop in focus["crops"]:
                bounds = crop.get(
                    "crop_bounds_px", crop.get("bounds_px", crop.get("core_bounds_px"))
                )
                if bounds is None:
                    raise RuntimeError(f"crop schema missing bounds: {list(crop)}")
                with Image.open(focus_path.parent / crop["raw_crop"]) as raw:
                    expected = source.crop(tuple(bounds))
                    fidelity &= (
                        raw.mode == expected.mode
                        and raw.size == expected.size
                        and raw.tobytes() == expected.tobytes()
                    )
            item["raw_crop_fidelity"] = fidelity
            item["root_crop_count"] = len(focus["crops"])
            # Exact coverage and disjoint ownership use pixel occupancy on root cores.
            # Area and pairwise disjointness establish exact rectangular coverage.
            boxes = [crop["core_bounds_px"] for crop in focus["crops"]]
            view = manifest.get("view_bounds_px", [0, 0, *source.size])
            area = sum((b[2] - b[0]) * (b[3] - b[1]) for b in boxes)
            disjoint = all(
                iou(a, b) == 0 for i, a in enumerate(boxes) for b in boxes[i + 1 :]
            )
            contained = all(
                view[0] <= b[0] < b[2] <= view[2] and view[1] <= b[1] < b[3] <= view[3]
                for b in boxes
            )
            item["exact_root_coverage"] = (
                disjoint
                and contained
                and area == (view[2] - view[0]) * (view[3] - view[1])
            )
            item["status"] = (
                "pass"
                if all(
                    item[key]
                    for key in (
                        "deterministic_map_id",
                        "immutable_predecessor",
                        "raw_crop_fidelity",
                        "exact_root_coverage",
                    )
                )
                else "fail"
            )
        except Exception as exc:  # noqa: BLE001 - benchmark records unexpected failures
            item.update(status="error", error=f"{type(exc).__name__}: {exc}")
        item["latency_ms"] = (time.perf_counter() - start) * 1000
        checks.append(item)
    return checks


def _memory_worker(image_path: Path, mode: str) -> None:
    """Fresh-process high-water RSS includes imports, decode, hash, and native work."""
    import resource

    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    source_id = (
        "sha256:"
        + hashlib.sha256(
            image.mode.encode()
            + f"{image.width}x{image.height}".encode()
            + image.tobytes()
        ).hexdigest()
    )
    status = "grid"
    error: str | None = None
    try:
        if mode == "grid":
            _grid(*image.size)
        else:
            detector_path = (
                Path(__file__).resolve().parents[1]
                / "skills/foveate-images/autostructure.py"
            )
            spec = importlib.util.spec_from_file_location(
                "memory_detector", detector_path
            )
            assert spec is not None and spec.loader is not None
            detector = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(detector)
            report = detector.discover(image, [0, 0, *image.size], source_id, mode=mode)
            status = "selected" if report.get("selected_candidate_id") else "fallback"
    except ValueError as exc:
        status, error = "rejected", str(exc)
    print(
        json.dumps(
            {
                "peak_process_rss_mib": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss
                / 1024,
                "status": status,
                "error": error,
            }
        )
    )


def _memory_measurements(cases: list[Workload]) -> list[dict[str, Any]]:
    """A small bootstrap prevents vfork inheriting the benchmark parent's RSS peak."""
    measurements: list[dict[str, Any]] = []
    for family in ("calendar", "high_resolution", "texture_negative"):
        case = next(case for case in cases if case.family == family)
        for mode in MODES[:-1]:
            start = time.perf_counter()
            child = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import subprocess,sys; sys.exit(subprocess.call(sys.argv[1:]))",
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--memory-image",
                    str(case.image_path),
                    "--memory-mode",
                    mode,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            measurement: dict[str, Any] = {
                "case": case.name,
                "family": family,
                "mode": mode,
                "process_wall_ms": (time.perf_counter() - start) * 1000,
            }
            if child.returncode:
                measurement.update(status="error", error=child.stderr.strip())
            else:
                measurement.update(json.loads(child.stdout))
            measurements.append(measurement)
    return measurements


def _vision_section(vision: dict[str, Any] | None) -> str:
    if not vision:
        return ""
    cases: list[dict[str, Any]] = vision.get("cases", [])
    body: list[str] = []
    has_format_diagnostic = any("metrics_format_tolerant" in case for case in cases)
    format_header = (
        "<th>Post-hoc envelope tolerant pair F1</th>" if has_format_diagnostic else ""
    )
    format_exact_header = (
        "<th>Post-hoc tolerant exact sets</th>" if has_format_diagnostic else ""
    )
    for case in cases:
        metrics: dict[str, Any] = case.get("metrics") or {}
        pair: dict[str, Any] = metrics.get("date_color", {})
        date: dict[str, Any] = metrics.get("date", {})
        format_metrics: dict[str, Any] = case.get("metrics_format_tolerant", metrics)
        format_pair: dict[str, Any] = format_metrics.get("date_color", {})
        format_cell = (
            f"<td>{format_pair.get('f1', 0):.1%}</td>" if has_format_diagnostic else ""
        )
        errors: Any = case.get("errors", [])
        error_count = (
            len(cast(list[Any], errors))
            if isinstance(errors, list)
            else int(errors)
            if isinstance(errors, (int, float))
            else int(bool(errors))
        )
        parse: Any = case.get("parse_errors", 0)
        parse_count = len(cast(list[Any], parse)) if isinstance(parse, list) else parse
        body.append(
            f"<tr><td>{html.escape(str(case.get('case', '')))}</td><td>{html.escape(str(case.get('mode', '')))}</td><td>{date.get('f1', 0):.1%}</td><td>{pair.get('precision', 0):.1%}</td><td>{pair.get('recall', 0):.1%}</td><td>{pair.get('f1', 0):.1%}</td>{format_cell}<td>{'yes' if metrics.get('exact_set') else 'no'}</td><td>{case.get('calls', 0)}</td><td>{case.get('image_count', 0)} / {case.get('submitted_pixels', 0) / 1000000:.1f}</td><td>{case.get('input_tokens', 0):,} / {case.get('output_tokens', 0):,}</td><td>{case.get('elapsed_seconds', 0):.1f}</td><td>{parse_count} / {error_count}</td></tr>"
        )
    summary_rows: list[str] = []
    for mode in ("full", "grid", "auto"):
        selected = [case for case in cases if case.get("mode") == mode]
        truths = sum(case.get("metrics", {}).get("truth_count", 0) for case in selected)
        predictions = sum(
            case.get("metrics", {}).get("prediction_count", 0) for case in selected
        )
        true_positive = sum(
            case.get("metrics", {}).get("date_color", {}).get("true_positive", 0)
            for case in selected
        )
        precision = true_positive / predictions if predictions else 0.0
        recall = true_positive / truths if truths else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        exact = sum(bool(case.get("metrics", {}).get("exact_set")) for case in selected)
        format_scores: list[dict[str, Any]] = [
            case.get("metrics_format_tolerant", case.get("metrics", {}))
            for case in selected
        ]
        format_tp = sum(
            score.get("date_color", {}).get("true_positive", 0)
            for score in format_scores
        )
        format_truths = sum(score.get("truth_count", 0) for score in format_scores)
        format_predictions = sum(
            score.get("prediction_count", 0) for score in format_scores
        )
        format_f1 = (
            2 * format_tp / (format_truths + format_predictions)
            if format_truths + format_predictions
            else 0.0
        )
        format_exact = sum(bool(score.get("exact_set")) for score in format_scores)
        format_summary_cell = (
            f"<td>{format_f1:.1%}</td>" if has_format_diagnostic else ""
        )
        format_exact_cell = (
            f"<td>{format_exact}/{len(selected)}</td>" if has_format_diagnostic else ""
        )
        calls = sum(case.get("calls", 0) for case in selected)
        input_tokens = sum(case.get("input_tokens", 0) for case in selected)
        output_tokens = sum(case.get("output_tokens", 0) for case in selected)
        summary_rows.append(
            f"<tr><td>{mode}</td><td>{len(selected)}</td><td>{precision:.1%}</td><td>{recall:.1%}</td><td>{f1:.1%}</td>{format_summary_cell}<td>{exact}/{len(selected)}</td>{format_exact_cell}<td>{calls}</td><td>{input_tokens:,} / {output_tokens:,}</td></tr>"
        )
    summary_table = (
        f'<div class="scroll"><table><thead><tr><th>Input mode</th><th>Calendars</th><th>Pair micro precision</th><th>Pair micro recall</th><th>Pair micro F1</th>{format_header}<th>Exact sets</th>{format_exact_header}<th>Total calls</th><th>Total input / output tokens</th></tr></thead><tbody>'
        + "".join(summary_rows)
        + "</tbody></table></div>"
    )
    protocol = vision.get("protocol", {})
    description = html.escape(
        str(
            protocol.get(
                "description",
                "Calendar highlight extraction using full images or navigation crops.",
            )
        )
    )
    note = html.escape(str(protocol.get("note", "")))
    format_note = (
        "<p><b>Secondary, post-hoc format diagnostic.</b> "
        + html.escape(
            str(
                protocol.get(
                    "posthoc_format_note",
                    "Accept valid array envelopes without changing their contents; primary requested-schema scores remain unchanged.",
                )
            )
        )
        + "</p>"
        if has_format_diagnostic
        else ""
    )
    return f'<h2>Exploratory calendar extraction on Virili</h2><p>Model <code>{html.escape(str(vision.get("model", "unknown")))}</code> served by <code>{html.escape(str(vision.get("endpoint", "Virili")))}</code>. {description} {note}</p><p>This is a separate small inference trial, not the geometric benchmark or a matched-budget comparison. Full-image, grid, and automatic navigation use different numbers of requests, images, and tokens. Server input constraint: {html.escape(str(protocol.get("server_image_limit", "2 images per request")))}. Crop requests pair source context with an individual raw crop. All actual call counts, submitted images and megapixels, and tokens appear below. Per-case seconds sum request latencies; whole-trial parallel wall time was {vision.get("wall_seconds", 0):.1f} seconds. Temperature {protocol.get("temperature", "unknown")}; output cap {protocol.get("max_tokens", "unknown")} tokens per request; seed {protocol.get("seed", "unknown")}.</p><p>Scores compare extracted highlighted dates and date\u2013color pairs against synthetic truth using the existing calendar scorer. Exact set requires every expected pair and no extras. Date-only F1 can remain high despite color mistakes. Parse and transport errors are retained; failed answers are not silently omitted. Request-budget differences, repeated context, and only five calendar templates prevent a causal claim that automatic structure improves extraction efficiency or general accuracy. Micro scores sum date\u2013color pair counts across calendars; the scorer deduplicates repeated predictions within each calendar.</p>{format_note}{summary_table}<div class="scroll"><table><thead><tr><th>Calendar</th><th>Input mode</th><th>Date F1</th><th>Pair precision</th><th>Pair recall</th><th>Pair F1</th>{format_header}<th>Exact set</th><th>Calls</th><th>Images / megapixels</th><th>Input / output tokens</th><th>Summed request sec</th><th>Parse / API errors</th></tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _html_report(data: dict[str, Any], cases: list[Workload], target: Path) -> None:
    rows = data["results"]
    tables: list[str] = []
    for split in ("development", "heldout-seed", "heldout-template"):
        body: list[str] = []
        for mode in MODES:
            a = _aggregate(
                [r for r in rows if r["split"] == split and r["mode"] == mode]
            )
            neg = (
                "\u2014"
                if a["negative_structure_acceptance"] is None
                else f"{a['negative_structure_acceptance']:.0%}"
            )
            body.append(
                f"<tr><td>{mode}</td><td>{a['n']}</td><td>{a['precision']:.1%}</td><td>{a['recall']:.1%}</td><td>{a['f1']:.1%}</td><td>{a['macro_f1']:.1%}</td><td>{neg}</td><td>{a['selected']}/{a['fallback']}/{a['rejected']}/{a['errors']}</td><td>{a['median_ms']:.1f}</td><td>{a['p95_ms']:.1f}</td></tr>"
            )
        tables.append(
            f'<h2>{split}</h2><div class="scroll"><table><thead><tr><th>Mode</th><th>N</th><th>Precision</th><th>Recall</th><th>Micro F1</th><th>Macro F1</th><th>Neg. accepted</th><th>Select/fallback/reject/error</th><th>Median ms</th><th>p95 ms</th></tr></thead><tbody>{"".join(body)}</tbody></table></div>'
        )
    families = sorted({r["family"] for r in rows})
    bars: list[str] = []
    for family in families:
        a = _aggregate(
            [
                r
                for r in rows
                if r["family"] == family
                and r["mode"] == "auto"
                and r["split"] != "development"
            ]
        )
        label = (
            f"{a['f1']:.0%}"
            if a["positive_n"]
            else f"{a['negative_structure_acceptance']:.0%} accepted"
        )
        fraction = (
            a["f1"] if a["positive_n"] else a["negative_structure_acceptance"] or 0
        )
        bars.append(
            f'<div class="barrow"><span>{family}</span><div class="bar"><i style="width:{fraction * 100:.1f}%"></i></div><b>{label}</b></div>'
        )
    previews: list[str] = []
    for family in families:
        case = next(c for c in cases if c.family == family and c.split != "development")
        row = next(r for r in rows if r["name"] == case.name and r["mode"] == "auto")
        previews.append(
            f'<figure><img loading="lazy" src="{_sample(case, row)}" alt="{html.escape(case.name)}"><figcaption><b>{html.escape(family)}</b> \xb7 {row["status"]} \xb7 F1 {row["metrics"]["f1"]:.0%}<br><small>{html.escape(case.name)}</small></figcaption></figure>'
        )
    integration = "".join(
        f"<tr><td>{html.escape(c['case'])}</td><td>{html.escape(c['status'])}</td><td>{html.escape(json.dumps({k: v for k, v in c.items() if k not in {'case', 'status'}}))}</td></tr>"
        for c in data["integration"]
    )
    memory_rows = "".join(
        f"<tr><td>{html.escape(item['family'])}</td><td>{html.escape(item['mode'])}</td><td>{html.escape(item['status'])}</td><td>{item.get('peak_process_rss_mib', 0):.1f}</td><td>{item['process_wall_ms']:.1f}</td></tr>"
        for item in data.get("memory", [])
    )
    encoded = json.dumps(rows).replace("</", "<\\/")
    summary = data["summary"]
    vision_section = _vision_section(data.get("vision"))
    measurement_scope = (
        "These are geometric detection results. A separate exploratory extraction trial appears below."
        if vision_section
        else "These are geometric detection results. No OCR, language model extraction, or calendar-date accuracy was measured."
    )
    doc = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Autostructure synthetic evaluation</title><style>
:root{{font-family:system-ui,sans-serif;color:#203044;background:#f3f5f7}}body{{max-width:1240px;margin:auto;padding:36px}}h1{{font-size:38px;letter-spacing:-1px}}h2{{margin-top:42px}}p{{max-width:1000px;line-height:1.65}}.muted,small{{color:#59697b}}table{{border-collapse:collapse;width:100%;background:white;font-size:13px}}th,td{{padding:11px;text-align:left;border-bottom:1px solid #dce3ea}}th{{background:#e6edf3;white-space:nowrap}}.scroll{{overflow:auto}}.bars{{max-width:1000px}}.barrow{{display:grid;grid-template-columns:190px 1fr 120px;gap:12px;align-items:center;margin:10px 0;font-size:14px}}.bar{{height:14px;background:#dce4ec}}.bar i{{display:block;height:100%;background:#426a8c}}.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px}}figure{{margin:0;background:white;border:1px solid #dce3ea}}figure img{{width:100%;height:300px;object-fit:contain}}figcaption{{padding:15px;font-size:14px}}select,input{{padding:9px;margin:5px;border:1px solid #b7c5d3;border-radius:4px}}code{{background:#e7edf2;padding:2px 5px;overflow-wrap:anywhere}}.callout{{padding:18px 24px;background:#e4edf4;border-left:4px solid #426a8c}}#details td{{max-width:250px;overflow-wrap:anywhere}}@media(max-width:650px){{body{{padding:16px}}h1{{font-size:28px}}.barrow{{grid-template-columns:140px 1fr 80px}}}}
</style><body><p class="muted">Foveate images \xb7 synthetic evaluation \xb7 {html.escape(data["generated_at"])}</p><h1>Automatic structure discovery</h1><p class="callout">{len(cases)} deterministic image workloads \xb7 {len(rows)} mode comparisons \xb7 {sum(c["status"] == "pass" for c in data["integration"])}/{len(data["integration"])} sampled pipeline invariant checks passed. {measurement_scope}</p>
<p>Version 1 compares axis-aligned Hough, recursive gutters, and repeated-panel detection with a uniform 3\xd73 navigation baseline. Auto abstention uses that same baseline for geometric scoring; forced detector rejection contributes zero predictions. Structure acceptance on negative cases is reported separately. The perspective-aware Hough extension remains future work.</p>
<h2>What the experiment establishes</h2><p>Automatic mode selected structures on {summary["auto_selected"]}/{summary["auto_total"]} workloads and fell back on {summary["auto_fallback"]}; {summary["auto_errors"]} unexpected errors occurred. On held-out positive cases it reached micro F1 {summary["heldout_auto_f1"]:.1%}, compared with {summary["heldout_grid_f1"]:.1%} for the uniform grid. Auto accepted {summary["negative_auto_selected"]}/{summary["negative_auto_total"]} synthetic negative scenes. The ungated comparison revealed {summary["gating_candidate_misses"]} cases where its best candidate F1 exceeded gated auto by more than 0.05; this uses an oracle scorer for diagnosis, not a selectable production policy.</p>
<h2>Protocol and scoring</h2><p class="muted">{html.escape(data["protocol"].get("evaluation_note", "Initial frozen-policy evaluation"))}. Detector source SHA-256: <code>{html.escape(data["protocol"]["detector_code_sha256"])}</code>.</p><p>Development seeds start at 17011; held-out seeds at 92083. Held-out templates use independent compositions and seeds starting at 581039 (mixed dashboards, contact sheets, mixed-size panels, accidental line art, and single framed illustrations). Template holdout shares low-level drawing primitives and is not an independent real-world dataset. The five existing calendar presets are generated in both seed splits. A corpus manifest is written before any detector runs. Detectors receive only the PIL image, full-image bounds, a source hash, and mode/configuration; truth sidecars and region labels are read only by the scorer.</p>
<p>Intended units differ by workload: month panels, table cells, panels, or text columns. Predictions use the selected candidate\u2019s visible <code>evidence_bounds_px</code>, not its larger ownership cells. One-to-one maximum-cardinality matching at IoU \u2265 0.50 uses a Hungarian assignment, with total IoU breaking cardinality ties. Precision and recall aggregate matched, spurious, and missed boxes; micro F1 weights each box, while macro F1 weights each image. Mean IoU is defined only for matched pairs. \u201cClipped\u201d means a matched predicted rectangle retains less than 98% of the truth rectangle\u2019s area; this is a geometric proxy, not measured glyph or semantic content clipping. At small skew, truth is the enclosing rectangle of transformed panel corners.</p>
<p>Negatives have no intended repeated navigation units. Smooth noise is a photograph-like surrogate, not a real photograph; accidental line art intentionally tests rejection despite geometric shapes. Such labels encode navigation intent and cannot establish semantic understanding. Inspection of the accepted single-frame negatives also found duplicate thick-border edges generating four-pixel sliver cells, a geometric weakness in stroke merging. Accidental line art produced structural selections too. These failures remain disclosed; no thresholds were tuned after held-out evaluation. Forced detectors may validly expose accidental geometry while still failing this strict intent test.</p>
<p>All thresholds remain implementation defaults; this harness changes no detector policy. Held-out results were not used for benchmark-driven tuning. Warm latency includes source-byte hashing and detector work, but excludes image decode, fixture generation, map/crop materialization, and report rendering. One untimed warm-up precedes measurements; cold detector module import took {data["environment"]["cold_import_ms"]:.1f} ms; the first automatic call including deferred backend imports took {data["environment"]["cold_first_call_ms"]:.1f} ms. The grid baseline performs no discovery, so its proposal latency measures only fixed geometry setup. The process is sequential on one machine with no repeated timing trials, so speed figures are descriptive. Environment: Python {html.escape(data["environment"]["python"])}, OpenCV {html.escape(data["environment"].get("opencv", "unknown"))}, Pillow {html.escape(data["environment"]["pillow"])}, {html.escape(data["environment"]["platform"])}.</p>
{"".join(tables)}
<h2>Held-out automatic performance by family</h2><p>Positive bars show micro F1; negative bars show structure acceptance (lower is preferable). Results should be judged by family: strong performance on bordered panels does not establish general layout discovery.</p><div class="bars">{"".join(bars)}</div>
<h2>Native memory and process overhead</h2><p>Each row below runs in a separate fresh worker process launched by a small clean Python bootstrap, avoiding Linux vfork inheritance of the benchmark parent\u2019s high-water mark. Linux <code>resource.getrusage(RUSAGE_SELF).ru_maxrss</code> measures the process peak resident set in MiB, including Python and benchmark imports, image decoding, source hashing, optional NumPy/OpenCV imports, and native detector allocations. These are process high-water marks for one run, not live per-image allocation or incremental detector memory. The Pillow-only grid process supplies a reference; differences describe aggregate backend and detection overhead. Wall time includes interpreter startup and every worker step, so it is distinct from the warm proposal latency above. No <code>tracemalloc</code> estimate is substituted for native memory.</p><div class="scroll"><table><thead><tr><th>Representative family</th><th>Mode</th><th>Outcome</th><th>Peak process RSS MiB</th><th>Whole process ms</th></tr></thead><tbody>{memory_rows}</tbody></table></div>
<h2>Pipeline correctness samples</h2><p>Eight workload families exercise map creation, a second identical map, and root focus with zero context margin. Checks cover map identity, immutable predecessor bytes, exact pixel-for-pixel raw crop fidelity, and disjoint complete root ownership. Ownership bounds extend across gutters to preserve every source pixel; their completeness is distinct from the evidence-box accuracy scored above. Unit and adversarial tests separately cover deeper refinement, alternative selection, legacy loading, invalid configuration, and dependency failure.</p><div class="scroll"><table><tr><th>Case</th><th>Status</th><th>Checks</th></tr>{integration}</table></div>
<h2>Visual inspection</h2><p>Green marks synthetic truth bounds; orange marks selected evidence or the fallback 3\xd73 grid. Overlaid boxes are diagnostic previews, never evidence used by detection.</p><div class="gallery">{"".join(previews)}</div>
<h2>Explore every result</h2><div><select id="split"><option value="">Every split</option>{"".join(f"<option>{s}</option>" for s in ("development", "heldout-seed", "heldout-template"))}</select><select id="mode"><option value="">Every mode</option>{"".join(f"<option>{m}</option>" for m in MODES)}</select><select id="family"><option value="">Every family</option>{"".join(f"<option>{f}</option>" for f in families)}</select><input id="search" placeholder="Filter name or rejection reason"><span id="count"></span></div><div class="scroll"><table id="details"><thead><tr><th>Case</th><th>Mode</th><th>Status</th><th>P/R/F1</th><th>Mean IoU</th><th>Clip fraction</th><th>ms</th><th>Reason</th></tr></thead><tbody></tbody></table></div>
{vision_section}
<h2>Reproduction and limits</h2><p>Run <code>uv run --extra detection python scripts/benchmark_autostructure.py --output reports --repeats 2</code> from the repository root. This writes <code>reports/autostructure.html</code>, <code>reports/results.json</code>, fixture images, a frozen corpus manifest, and sampled pipeline outputs. Reports are gitignored. Reproduce the optional local-model supplement with <code>uv run --extra detection python scripts/evaluate_calendar_vision.py --output reports</code>, then merge it into the HTML with <code>uv run --extra detection python scripts/benchmark_autostructure.py --output reports --render-only</code>. Raw JSON includes every prediction, matching result, candidate ID, probe measurement, and rejection reason. This HTML embeds its previews and all row data, needs no external assets or server, and can be opened offline.</p><p>This corpus is synthetic, small, and hand-composed. Seed holdout measures variation within a known template; template holdout tests limited transfer. It does not establish real-camera performance, confidence calibration, memory scaling beyond the representative process measurements, or downstream inference-budget improvement. Semantic extraction is addressed only by the separate small exploratory trial when present. Large-resolution fixtures test a 2280\xd71800 raster, not unrestricted image sizes. Calendar hierarchy quality is inspected indirectly through coarse box recovery and navigation correctness; complete internal-date-cell recall is not scored. Future evaluations should add independently sourced photographs and documents and matched crop/inference budgets.</p>
<script type="application/json" id="data">{encoded}</script><script>
const rows=JSON.parse(document.getElementById('data').textContent);const fields=['split','mode','family','search'];function esc(s){{return String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));}}function pct(n){{return n==null?'\u2014':(n*100).toFixed(0)+'%';}}function render(){{const values=Object.fromEntries(fields.map(f=>[f,document.getElementById(f).value]));const filtered=rows.filter(r=>(!values.split||r.split===values.split)&&(!values.mode||r.mode===values.mode)&&(!values.family||r.family===values.family)&&(!values.search||(r.name+' '+(r.error||r.fallback_reason||'')).toLowerCase().includes(values.search.toLowerCase())));document.getElementById('count').textContent=filtered.length+' results';document.querySelector('#details tbody').innerHTML=filtered.map(r=>`<tr><td>${{esc(r.name)}}</td><td>${{esc(r.mode)}}</td><td>${{esc(r.status)}}</td><td>${{pct(r.metrics.precision)}} / ${{pct(r.metrics.recall)}} / ${{pct(r.metrics.f1)}}</td><td>${{pct(r.metrics.mean_iou)}}</td><td>${{pct(r.metrics.matched_geometry_clipped_fraction)}}</td><td>${{r.latency_ms.toFixed(1)}}</td><td>${{esc(r.error||r.fallback_reason||r.selected_method||'')}}</td></tr>`).join('');}}fields.forEach(f=>{{const element=document.getElementById(f);element.addEventListener('input',render);element.addEventListener('change',render);}});render();
</script></body></html>"""
    target.write_text(doc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--retest-note",
        default="Initial frozen-policy evaluation",
        help="Record an independent correctness-retest explanation.",
    )
    parser.add_argument(
        "--refresh-memory",
        action="store_true",
        help="Refresh isolated RSS measurements while retaining detector results.",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Rebuild HTML from existing measurements without rerunning detectors.",
    )
    parser.add_argument("--memory-image", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--memory-mode", choices=MODES[:-1], default="auto", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--development-only",
        action="store_true",
        help="Do not evaluate held-out images while tuning.",
    )
    args = parser.parse_args()
    if args.memory_image is not None:
        _memory_worker(args.memory_image, args.memory_mode)
        return
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    cases = generate_workloads(args.output / "fixtures", args.repeats)
    if args.render_only or args.refresh_memory:
        data = json.loads((args.output / "results.json").read_text())
        if args.refresh_memory:
            frozen_path = (
                Path(__file__).resolve().parents[1]
                / "skills/foveate-images/autostructure.py"
            )
            if (
                hashlib.sha256(frozen_path.read_bytes()).hexdigest()
                != data["protocol"]["detector_code_sha256"]
            ):
                raise RuntimeError(
                    "Cannot refresh memory using a different detector implementation"
                )
            data["memory"] = _memory_measurements(cases)
            (args.output / "results.json").write_text(json.dumps(data, indent=2) + "\n")
        vision_path = args.output / "vision.json"
        if vision_path.exists():
            data["vision"] = json.loads(vision_path.read_text())
        _html_report(data, cases, args.output / "autostructure.html")
        return
    if args.development_only:
        cases = [case for case in cases if case.split == "development"]
    start = time.perf_counter()
    detector_path = (
        Path(__file__).resolve().parents[1] / "skills/foveate-images/autostructure.py"
    )
    detector_digest = hashlib.sha256(detector_path.read_bytes()).hexdigest()
    detector_spec = importlib.util.spec_from_file_location(
        "autostructure_benchmark", detector_path
    )
    assert detector_spec is not None and detector_spec.loader is not None
    detector = importlib.util.module_from_spec(detector_spec)
    detector_spec.loader.exec_module(detector)
    cold_import_ms = (time.perf_counter() - start) * 1000
    discover = detector.discover
    with Image.open(cases[0].image_path) as opened:
        warm = opened.convert("RGB")
    cold_start = time.perf_counter()
    discover(warm, [0, 0, *warm.size], "warmup", mode="auto")
    cold_first_call_ms = (time.perf_counter() - cold_start) * 1000
    results: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        with Image.open(case.image_path) as opened:
            image = opened.convert("RGB")
        for mode in MODES:
            results.append(_run(case, mode, discover, image))
        print(f"{index + 1}/{len(cases)} {case.name}", flush=True)
    import cv2
    import PIL

    auto = [r for r in results if r["mode"] == "auto"]
    heldout = [r for r in auto if r["split"] != "development"]
    negatives = [r for r in auto if r["negative"]]
    ungated = {r["name"]: r for r in results if r["mode"] == "auto-ungated"}
    data = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "opencv": cv2.__version__,
            "pillow": PIL.__version__,
            "cold_import_ms": cold_import_ms,
            "cold_first_call_ms": cold_first_call_ms,
        },
        "protocol": {
            "matching_iou": 0.5,
            "detector_code_sha256": detector_digest,
            "evaluation_note": args.retest_note,
            "detector_version": detector.DETECTOR_VERSION,
            "selection_policy_version": detector.POLICY_VERSION,
            "configuration_defaults": detector.DEFAULTS,
            "grid": [3, 3],
            "repeats": args.repeats,
            "template_holdout_families": [
                c.family for c in cases if c.split == "heldout-template"
            ],
        },
        "summary": {
            "auto_selected": sum(r["status"] == "selected" for r in auto),
            "auto_total": len(auto),
            "auto_fallback": sum(r["status"] == "fallback" for r in auto),
            "auto_errors": sum(r["status"] == "error" for r in auto),
            "heldout_auto_f1": _aggregate(heldout)["f1"],
            "heldout_grid_f1": _aggregate(
                [
                    r
                    for r in results
                    if r["mode"] == "grid" and r["split"] != "development"
                ]
            )["f1"],
            "negative_auto_selected": sum(r["status"] == "selected" for r in negatives),
            "negative_auto_total": len(negatives),
            "gating_candidate_misses": sum(
                ungated[r["name"]]["oracle_f1"] > r["oracle_f1"] + 0.05 for r in auto
            ),
        },
        "results": results,
        "integration": _integration(cases, args.output / "pipeline"),
        "memory": _memory_measurements(cases),
    }
    if hashlib.sha256(detector_path.read_bytes()).hexdigest() != detector_digest:
        raise RuntimeError(
            "Detector implementation changed during integration/memory evaluation; discard this run"
        )
    (args.output / "results.json").write_text(json.dumps(data, indent=2) + "\n")
    if not args.development_only:
        _html_report(data, cases, args.output / "autostructure.html")
    print(json.dumps(data["summary"], indent=2))


if __name__ == "__main__":
    main()
