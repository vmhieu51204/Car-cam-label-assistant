"""Evaluate stored SAM3 boxes after stitching and CVAT-style interpolation.

The intermediate frames contain linear interpolation, never SAM3 inference.
This local KITTI-style evaluator is not the official TrackEval HOTA protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.eval_autolabel import Box, tracking_metrics
from scripts.kitti_eval import KittiItem, KittiReference, evaluate_kitti_style
from scripts.stitch_sam3_benchmark import _boxes as direct_boxes
from scripts.stitch_sam3_benchmark import compare_stitch
from src.services.video.export_2d import Export2DConfig
from src.services.video.export_2d import _boxes as interpolated_boxes


def _source_reference(labels: Path, source_frames: set[int]) -> tuple[list[Box], KittiReference]:
    """Read KITTI source indices directly; keep raw and ignore-adjusted GT distinct."""
    raw = []
    adjusted = []
    dontcare: dict[int, list[tuple[float, float, float, float]]] = {}
    for line_number, line in enumerate(labels.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 17:
            raise ValueError(f"invalid KITTI label line {line_number}")
        frame = int(fields[0])
        if frame not in source_frames:
            continue
        label = fields[2]
        box = (float(fields[6]), float(fields[7]), float(fields[8]), float(fields[9]))
        if label == "DontCare":
            dontcare.setdefault(frame, []).append(box)
            continue
        normalized = {"Car": "car", "Van": "car", "Pedestrian": "person", "Person": "person"}.get(label)
        if normalized is None:
            continue
        item = Box(frame, normalized, box, fields[1])
        valid = label in {"Car", "Pedestrian"} and float(fields[3]) <= 0 and int(fields[4]) <= 2
        adjusted.append(KittiItem(item, valid))
        if label in {"Car", "Pedestrian"}:
            raw.append(item)
    return raw, KittiReference(tuple(adjusted), {frame: tuple(rows) for frame, rows in dontcare.items()})


def _metrics(gt: list[Box], pred: list[Box], frames: set[int], reference: KittiReference) -> dict:
    raw = tracking_metrics(gt, pred, frames)
    adjusted = evaluate_kitti_style(reference, pred, frames)
    return {
        "raw_iou_05": raw["thresholds"]["0.5"],
        "raw_iou_07": raw["thresholds"]["0.7"],
        "raw_id_switches": raw["id_switches"],
        "kitti_style_iou_05": adjusted["metrics"]["thresholds"]["0.5"],
        "kitti_style_iou_07": adjusted["metrics"]["thresholds"]["0.7"],
        "kitti_style_ignored_predictions": adjusted["ignored_predictions"],
    }


def evaluate(result: dict, manifest: dict, labels: Path, *, config_id: str = "fn-baseline") -> dict:
    item = next((row for row in result["configurations"] if row["id"] == config_id), None)
    if item is None or item.get("status") != "completed":
        raise ValueError("completed SAM3 configuration not found")
    records = sorted(manifest["frames"], key=lambda row: row["frame_index"])
    source = [int(row["source_frame_index"]) for row in records]
    if source != sorted(set(source)) or [row["frame_index"] for row in records] != list(range(len(records))):
        raise ValueError("manifest needs contiguous sampled indices and ascending source indices")
    if set(item["evaluated_source_frames"]) != set(range(len(records))):
        raise ValueError("SAM3 result does not cover all sampled frames")
    sampled_frames = set(source)
    full_frames = set(range(source[0], source[-1] + 1))
    intermediate_frames = full_frames - sampled_frames
    gt, reference = _source_reference(labels, full_frames)
    _, stitched = compare_stitch(item, [], reference=None)
    direct = [Box(source[box.frame], box.label, box.xyxy, box.track_id) for box in direct_boxes(item["predictions"])]
    stitched_sampled = [
        Box(source[obs.frame], track.label, obs.bbox, track.track_id)
        for track in stitched.tracks
        for obs in track.observations
    ]
    expanded = [
        Box(frame, track.label, box, track.track_id)
        for track in stitched.tracks
        for frame, box in interpolated_boxes(track, source, Export2DConfig())
    ]
    return {
        "schema": "autolabel-sam3-export-timeline-eval/v1",
        "config_id": config_id,
        "sampled_frames": len(sampled_frames),
        "interpolated_frames": len(intermediate_frames),
        "direct_sampled": _metrics(gt, direct, sampled_frames, reference),
        "stitched_sampled": _metrics(gt, stitched_sampled, sampled_frames, reference),
        "interpolated_only": _metrics(gt, expanded, intermediate_frames, reference) if intermediate_frames else None,
        "full_source_timeline": _metrics(gt, expanded, full_frames, reference),
        "note": "Intermediate source frames are CVAT-style linear interpolation, not SAM3 predictions.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--kitti-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config-id", default="fn-baseline")
    args = parser.parse_args()
    report = evaluate(
        json.loads(args.result.read_text(encoding="utf-8")),
        json.loads(args.manifest.read_text(encoding="utf-8")),
        args.kitti_labels,
        config_id=args.config_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sampled_frames": report["sampled_frames"],
                "interpolated_frames": report["interpolated_frames"],
            }
        )
    )


if __name__ == "__main__":
    main()
