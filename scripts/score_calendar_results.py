"""Score autonomous calendar answers against synthetic truth sidecars."""

from __future__ import annotations

import argparse
import collections
import datetime
import json
from pathlib import Path
from typing import Any


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _pairs(items: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {(str(item["date"]), str(item["color"]).strip().lower()) for item in items}


def score(truth: dict[str, Any], answer: dict[str, Any]) -> dict[str, Any]:
    truth_pairs = _pairs(truth["highlights"])
    raw_predictions = answer.get("highlights", [])
    predicted_pairs = _pairs(raw_predictions)
    truth_dates = {date for date, _ in truth_pairs}
    predicted_dates = {date for date, _ in predicted_pairs}
    date_tp = len(truth_dates & predicted_dates)
    pair_tp = len(truth_pairs & predicted_pairs)
    date_precision = date_tp / len(predicted_dates) if predicted_dates else 0.0
    date_recall = date_tp / len(truth_dates) if truth_dates else 0.0
    pair_precision = pair_tp / len(predicted_pairs) if predicted_pairs else 0.0
    pair_recall = pair_tp / len(truth_pairs) if truth_pairs else 0.0
    wrong_colors = sorted(
        date
        for date in truth_dates & predicted_dates
        if not any(
            (date, color) in truth_pairs for d, color in predicted_pairs if d == date
        )
    )
    return {
        "truth_count": len(truth_pairs),
        "prediction_count": len(predicted_pairs),
        "duplicate_predictions": len(raw_predictions) - len(predicted_pairs),
        "date": {
            "true_positive": date_tp,
            "precision": date_precision,
            "recall": date_recall,
            "f1": _f1(date_precision, date_recall),
        },
        "date_color": {
            "true_positive": pair_tp,
            "precision": pair_precision,
            "recall": pair_recall,
            "f1": _f1(pair_precision, pair_recall),
        },
        "color_accuracy_on_matched_dates": pair_tp / date_tp if date_tp else 0.0,
        "false_positive_dates": sorted(predicted_dates - truth_dates),
        "missed_dates": sorted(truth_dates - predicted_dates),
        "wrong_color_dates": wrong_colors,
        "exact_set": predicted_pairs == truth_pairs,
    }


def trace_metrics(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assistant = [
        event["message"]
        for event in events
        if event.get("type") == "message_end"
        and event.get("message", {}).get("role") == "assistant"
    ]
    tool_events = [
        event for event in events if event.get("type") == "tool_execution_start"
    ]
    tools = collections.Counter(event["toolName"] for event in tool_events)
    bash_commands = [
        event.get("args", {}).get("command", "")
        for event in tool_events
        if event["toolName"] == "bash"
    ]
    start = next(
        (event.get("timestamp") for event in events if event.get("type") == "session"),
        None,
    )
    end = max((message.get("timestamp", 0) for message in assistant), default=0)
    duration = None
    if start and end:
        started = (
            datetime.datetime.fromisoformat(start.replace("Z", "+00:00")).timestamp()
            * 1000
        )
        duration = round((end - started) / 1000, 2)
    return {
        "provider": assistant[-1].get("provider") if assistant else None,
        "model": assistant[-1].get("model") if assistant else None,
        "duration_seconds": duration,
        "model_calls": len(assistant),
        "input_tokens": sum(
            message.get("usage", {}).get("input", 0) for message in assistant
        ),
        "output_tokens": sum(
            message.get("usage", {}).get("output", 0) for message in assistant
        ),
        "thinking_blocks": sum(
            content.get("type") == "thinking"
            for message in assistant
            for content in message.get("content", [])
        ),
        "tool_calls": sum(tools.values()),
        "tools": dict(sorted(tools.items())),
        "image_reads": sum(
            event["toolName"] == "read"
            and event.get("args", {})
            .get("path", "")
            .lower()
            .endswith((".png", ".jpg", ".jpeg"))
            for event in tool_events
        ),
        "foveate_map_calls": sum(
            command.count("foveate map ") for command in bash_commands
        ),
        "foveate_focus_calls": sum(
            command.count("foveate focus ") for command in bash_commands
        ),
        "manual_pillow_crop_calls": sum(
            ".crop(" in command for command in bash_commands
        ),
        "tool_errors": sum(
            event.get("isError", False)
            for event in events
            if event.get("type") == "tool_execution_end"
        ),
    }


def score_directory(truth_dir: Path, results_dir: Path) -> dict[str, Any]:
    cases = {}
    all_truth: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    for truth_path in sorted(truth_dir.glob("*.truth.json")):
        slug = truth_path.name.removesuffix(".truth.json")
        answer_path = results_dir / slug / "answer.json"
        truth: dict[str, Any] = json.loads(truth_path.read_text())
        answer: dict[str, Any] = (
            json.loads(answer_path.read_text())
            if answer_path.exists()
            else {"highlights": []}
        )
        cases[slug] = score(truth, answer)
        agent = trace_metrics(results_dir / slug / "trace.jsonl")
        if agent:
            cases[slug]["agent"] = agent
        all_truth.extend(truth["highlights"])
        all_predictions.extend(answer.get("highlights", []))
    return {
        "cases": cases,
        "micro": score({"highlights": all_truth}, {"highlights": all_predictions}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("truth_dir", type=Path)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = score_directory(args.truth_dir, args.results_dir)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("calendar\ttruth\tpred\tdate_P\tdate_R\tpair_F1\texact")
    for slug, result in report["cases"].items():
        print(
            f"{slug}\t{result['truth_count']}\t{result['prediction_count']}\t"
            f"{result['date']['precision']:.4f}\t{result['date']['recall']:.4f}\t"
            f"{result['date_color']['f1']:.4f}\t{result['exact_set']}"
        )
    micro = report["micro"]
    print(
        f"MICRO\t{micro['truth_count']}\t{micro['prediction_count']}\t"
        f"{micro['date']['precision']:.4f}\t{micro['date']['recall']:.4f}\t"
        f"{micro['date_color']['f1']:.4f}\t{micro['exact_set']}"
    )


if __name__ == "__main__":
    main()
