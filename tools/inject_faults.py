"""Create deterministic, labeled 2D audit faults without invoking a model."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from scripts.eval_autolabel import Box, iou, read_boxes

FAULTS = ("possible_missing", "wrong_class", "bad_extent", "agree")


def fault_schedule(count: int, seed: int = 42) -> list[str]:
    """Balance error types while reserving about a third of frames as clean controls."""
    if count < 4:
        return [FAULTS[index % len(FAULTS)] for index in range(count)]
    agree = max(1, round(count * 0.35)) if count > 4 else 1
    errors = [FAULTS[index % 3] for index in range(count - agree)]
    schedule = errors + ["agree"] * agree
    if count > 4:
        random.Random(seed).shuffle(schedule)
    return schedule


def extent_level(option: str, occurrence: int) -> str:
    return ("light" if occurrence % 2 == 0 else "heavy") if option == "mixed" else option


def inject(
    ground_truth: Path,
    output: Path,
    *,
    frames_dir: Path | None = None,
    seed: int = 42,
    extent_severity: str = "heavy",
    selection_signals: dict[int, list[str]] | None = None,
) -> dict:
    if extent_severity not in {"light", "heavy", "mixed"}:
        raise ValueError("extent_severity must be light, heavy or mixed")
    boxes = read_boxes(ground_truth)
    by_frame: dict[int, list[Box]] = defaultdict(list)
    for box in boxes:
        by_frame[box.frame].append(box)
    if not by_frame:
        raise ValueError("ground truth contains no visible boxes")
    output.mkdir(parents=True, exist_ok=True)
    (output / "frames").mkdir(exist_ok=True)
    (output / "overlays").mkdir(exist_ok=True)
    labels = sorted({box.label for box in boxes})
    wrong_label = {label: next((other for other in labels if other != label), "__wrong_class__") for label in labels}
    altered: dict[int, list[Box]] = {}
    faults = []
    extent_count = 0
    schedule = fault_schedule(len(by_frame), seed)
    for index, frame in enumerate(sorted(by_frame)):
        original = by_frame[frame]
        target = original[0]
        fault = schedule[index]
        changed = list(original)
        severity = 0.0
        if fault == "possible_missing":
            changed.pop(0)
            severity = 1.0
        elif fault == "wrong_class":
            changed[0] = Box(frame, wrong_label[target.label], target.xyxy, target.track_id)
            severity = 0.85
        elif fault == "bad_extent":
            x1, y1, x2, y2 = target.xyxy
            width, height = x2 - x1, y2 - y1
            level = extent_level(extent_severity, extent_count)
            extent_count += 1
            fraction = 0.08 if level == "light" else 0.25
            changed[0] = Box(
                frame,
                target.label,
                (x1 + width * fraction, y1 + height * fraction, x2 - width * fraction, y2 - height * fraction),
                target.track_id,
            )
            severity = 1 - iou(target, changed[0])
        altered[frame] = changed
        raw_path = output / "frames" / f"frame_{frame:06d}.png"
        source = None
        if frames_dir is not None:
            matches = sorted(frames_dir.glob(f"*{frame:06d}.*"))
            source = matches[0] if matches else None
            if source is None:
                raise FileNotFoundError(f"no source image for frame {frame} in {frames_dir}")
        if source is not None:
            with Image.open(source) as image:
                image.convert("RGB").save(raw_path)
        else:
            width = max(256, int(max(box.xyxy[2] for box in original) + 32))
            height = max(256, int(max(box.xyxy[3] for box in original) + 32))
            Image.new("RGB", (width, height), (48, 48, 48)).save(raw_path)
        with Image.open(raw_path) as image:
            overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        for box in changed:
            draw.rectangle(box.xyxy, outline="red", width=3)
            draw.text((box.xyxy[0], box.xyxy[1]), f"{box.track_id or '-'} {box.label}", fill="yellow")
        overlay.save(output / "overlays" / f"frame_{frame:06d}.png")
        faults.append(
            {
                "frame_index": frame,
                "verdict": fault,
                "severity": round(severity, 6),
                "selection_reasons": (selection_signals or {}).get(frame, []),
                "extent_severity": level if fault == "bad_extent" else None,
                "affected_track_ids": [target.track_id] if target.track_id else [],
                "raw_file": f"frames/frame_{frame:06d}.png",
                "overlay_file": f"overlays/frame_{frame:06d}.png",
                "raw_is_synthetic": source is None,
            }
        )
    categories = {
        label: number
        for number, label in enumerate(sorted({box.label for items in altered.values() for box in items}), start=1)
    }
    payload = {
        "images": [
            {"id": frame, "frame_index": frame, "file_name": f"frame_{frame:06d}.png"} for frame in sorted(by_frame)
        ],
        "categories": [{"id": number, "name": label} for label, number in categories.items()],
        "annotations": [
            {
                "id": index,
                "image_id": box.frame,
                "category_id": categories[box.label],
                "bbox": [box.xyxy[0], box.xyxy[1], box.xyxy[2] - box.xyxy[0], box.xyxy[3] - box.xyxy[1]],
                **({"track_id": box.track_id} if box.track_id is not None else {}),
            }
            for index, box in enumerate((box for frame in sorted(altered) for box in altered[frame]), start=1)
        ],
    }
    (output / "predictions.coco.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output / "faults.json").write_text(
        json.dumps({"schema": "autolabel-faults/v1", "faults": faults}, indent=2), encoding="utf-8"
    )
    return {
        "frames": len(faults),
        "fault_counts": {name: sum(item["verdict"] == name for item in faults) for name in FAULTS},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=Path, help="Optional source images; absent frames use synthetic gray images")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--extent-severity", choices=("light", "heavy", "mixed"), default="heavy")
    parser.add_argument(
        "--selection-signals",
        type=Path,
        help="Optional JSON mapping frame index to existing selection reasons; never derive from faults",
    )
    args = parser.parse_args()
    signals = json.loads(args.selection_signals.read_text(encoding="utf-8")) if args.selection_signals else {}
    print(
        json.dumps(
            inject(
                args.ground_truth,
                args.output,
                frames_dir=args.frames,
                seed=args.seed,
                extent_severity=args.extent_severity,
                selection_signals={int(k): v for k, v in signals.items()},
            )
        )
    )


if __name__ == "__main__":
    main()
