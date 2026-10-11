"""Run an exploratory local-model calendar comparison, keeping results untracked."""

from __future__ import annotations

import argparse
import base64
import datetime
import importlib.util
import io
import json
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, cast

from PIL import Image

from foveate_tools.calendar import generate_set
from scripts.score_calendar_results import score

SYSTEM = (
    "Read highlighted calendar dates from raster images. Return only JSON with "
    'the form {"highlights":[{"date":"YYYY-MM-DD","color":"legend name"}]}. '
    "Use the exact color names printed in the legend, in lowercase. Include "
    "every highlighted day visible in the evidence image; include no unhighlighted "
    "day. An empty evidence region should produce an empty list."
)

FORMAT_NOTE = (
    "Secondary, post-hoc format diagnostic: accept a bare JSON list in addition "
    "to the requested highlights object. Validate the same date/color items; "
    "do not edit content or retry inference. Primary strict scores and parse "
    "failure counts remain unchanged. This separates wrapper compliance from "
    "visual reading and was added after inspecting format failures."
)


def validated_highlights(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TypeError("highlights is not a list")
    items: list[dict[str, Any]] = []
    for raw_item in cast(list[Any], value):
        if not isinstance(raw_item, dict):
            raise TypeError("invalid highlight object")
        item = cast(dict[str, Any], raw_item)
        if not isinstance(item.get("color"), str):
            raise TypeError("invalid highlight object")
        datetime.date.fromisoformat(item["date"])
        items.append(item)
    return items


def format_tolerant_predictions(response: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        content = response["choices"][0]["message"].get("content", "") or ""
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        decoded: Any = json.loads(content)
        return validated_highlights(
            decoded if isinstance(decoded, list) else decoded["highlights"]
        )
    except (ValueError, TypeError, KeyError, IndexError):
        return []


def image_item(image: Image.Image) -> tuple[dict[str, Any], dict[str, Any]]:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    data = buffer.getvalue()
    return (
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/png;base64," + base64.b64encode(data).decode("ascii")
            },
        },
        {"width": image.width, "height": image.height, "png_bytes": len(data)},
    )


def infer(job: dict[str, Any], endpoint: str, model: str) -> dict[str, Any]:
    started = time.perf_counter()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": job["content"]},
        ],
        "max_tokens": 4096,
        "temperature": 0,
        "seed": 20261011,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    result = {key: value for key, value in job.items() if key != "content"}
    result["predictions"] = []
    result["parse_error"] = None
    result["error"] = None
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=240) as response:
            raw = json.load(response)
        result["response"] = raw
        result["usage"] = raw.get("usage", {})
        choice = raw["choices"][0]
        result["finish_reason"] = choice.get("finish_reason")
        content = choice["message"].get("content", "") or ""
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            answer = json.loads(content)
            result["predictions"] = validated_highlights(answer["highlights"])
        except (ValueError, TypeError, KeyError) as exc:
            result["parse_error"] = str(exc)
    except (OSError, ValueError, KeyError, IndexError) as exc:
        result["error"] = str(exc)
    result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--endpoint", default="http://virili.:8000/v1")
    parser.add_argument("--model", default="Qwen3.8-27B")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="vision-", dir=args.output))
    truth_paths = generate_set(work / "fixtures", seed=20261011)
    path = Path(__file__).resolve().parents[1] / "skills/foveate-images/foveate.py"
    spec = importlib.util.spec_from_file_location("vision_foveate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    jobs: list[dict[str, Any]] = []
    truths: dict[str, Path] = {}
    for truth_path in truth_paths:
        case = truth_path.stem.removesuffix(".truth")
        truths[case] = truth_path
        source_path = truth_path.with_name(case + ".png")
        with Image.open(source_path) as opened:
            source = opened.copy()
        context, context_info = image_item(source)
        jobs.append(
            {
                "case": case,
                "mode": "full",
                "anchor": "ROOT",
                "images": [context_info],
                "content": [
                    {
                        "type": "text",
                        "text": "Extract all highlighted dates from this full evidence image.",
                    },
                    context,
                ],
            }
        )
        for mode in ("grid", "auto"):
            map_dir = work / case / mode
            module.create_map(source_path, map_dir, mode=mode)
            manifest = json.loads((map_dir / "manifest.json").read_text())
            anchors = manifest["anchors"]
            leaves = [
                (anchor, value)
                for anchor, value in anchors.items()
                if not any(other.startswith(anchor + "/") for other in anchors)
            ]
            for anchor, entry in leaves:
                crop = source.crop(entry["core_bounds_px"])
                evidence, evidence_info = image_item(crop)
                jobs.append(
                    {
                        "case": case,
                        "mode": mode,
                        "anchor": anchor,
                        "bounds_px": entry["core_bounds_px"],
                        "images": [context_info, evidence_info],
                        "content": [
                            {
                                "type": "text",
                                "text": "First image: full calendar context, for identifying month, year and legend only. Second image: evidence crop. Extract ONLY highlighted dates visible in the SECOND image. Do not copy highlighted dates seen only in the first image. Use the full context to resolve month/day labels cut by the crop.",
                            },
                            context,
                            evidence,
                        ],
                    }
                )
    responses: list[dict[str, Any]] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(infer, job, args.endpoint, args.model) for job in jobs
        ]
        for future in as_completed(futures):
            result = future.result()
            responses.append(result)
            print(
                f"{len(responses)}/{len(jobs)} {result['case']} {result['mode']} {result['anchor']} error={result['error']} parse={result['parse_error']}",
                flush=True,
            )
            (work / "responses.json").write_text(json.dumps(responses, indent=2) + "\n")
    cases: list[dict[str, Any]] = []
    for case, truth_path in truths.items():
        truth = json.loads(truth_path.read_text())
        for mode in ("full", "grid", "auto"):
            calls = [r for r in responses if r["case"] == case and r["mode"] == mode]
            predictions = [p for r in calls for p in r["predictions"]]
            tolerant = [
                p
                for r in calls
                for p in format_tolerant_predictions(r.get("response", {}))
            ]
            cases.append(
                {
                    "case": case,
                    "mode": mode,
                    "metrics": score(truth, {"highlights": predictions}),
                    "metrics_format_tolerant": score(truth, {"highlights": tolerant}),
                    "calls": len(calls),
                    "input_tokens": sum(
                        r.get("usage", {}).get("prompt_tokens", 0) for r in calls
                    ),
                    "output_tokens": sum(
                        r.get("usage", {}).get("completion_tokens", 0) for r in calls
                    ),
                    "elapsed_seconds": round(
                        sum(r["elapsed_seconds"] for r in calls), 3
                    ),
                    "parse_errors": sum(r["parse_error"] is not None for r in calls),
                    "errors": sum(r["error"] is not None for r in calls),
                    "image_count": sum(len(r["images"]) for r in calls),
                    "submitted_pixels": sum(
                        i["width"] * i["height"] for r in calls for i in r["images"]
                    ),
                }
            )
    data = {
        "model": args.model,
        "endpoint": args.endpoint,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "wall_seconds": round(time.perf_counter() - started, 3),
        "response_path": str(work / "responses.json"),
        "protocol": {
            "posthoc_format_note": FORMAT_NOTE,
            "description": "Five independently seeded calendar presets; full source vs all exact leaf cores of default 3x3 grid vs automatic structure. Crops are extracted from normalized source bytes without annotations or extra margins; inference never receives truth, source filename, or detector labels.",
            "max_tokens": 4096,
            "temperature": 0,
            "seed": 20261011,
            "server_image_limit": "At most 2 images per call; configured image width/height limit1280. Server preprocessing may resize images.",
            "note": "Exploratory end-to-end smoke test, not a matched-budget causal comparison. Full condition uses1 call; each cropped condition uses1 call per leaf with full context repeated. Token, call and submitted pixel budgets therefore differ. Both crop conditions use identical prompts and full context. Three concurrent workers; no retries or answer repair; parse/API errors scored as empty predictions; output truncation recorded in raw responses. Runtime is summed request latency, not parallel wall time. Geometry was frozen before these results; no tuning uses extraction scores.",
        },
        "cases": cases,
    }
    (args.output / "vision.json").write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(cases, indent=2), flush=True)


if __name__ == "__main__":
    main()
