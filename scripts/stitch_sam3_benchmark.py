"""Compare pre/post overlap ID stitching from one stored SAM3 GPU result."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from scripts.eval_autolabel import Box, read_boxes, tracking_metrics
from scripts.kitti_eval import KittiReference, evaluate_kitti_style, load_reference
from src.services.video.stitching import (
    ChunkObservation,
    ChunkTrack,
    MaskRLE,
    StitchConfig,
    StitchResult,
    stitch_tracks,
)


def _window_tracks(rows: list[dict]) -> list[ChunkTrack]:
    grouped: dict[tuple[int, str, str], list[ChunkObservation]] = defaultdict(list)
    for row in rows:
        raw = row.get("mask_rle")
        mask = MaskRLE(raw["height"], raw["width"], tuple(raw["counts"]), raw["sha256"]) if raw else None
        grouped[(row["window_id"], row["track_id"], row["label"])].append(
            ChunkObservation(row["frame_index"], tuple(row["bbox"]), mask, row.get("score"))
        )
    return [
        ChunkTrack(window, local_id, label, sorted(observations, key=lambda row: row.frame))
        for (window, local_id, label), observations in sorted(grouped.items())
    ]


def _boxes(rows: list[dict]) -> list[Box]:
    return [
        Box(row["frame_index"], row["label"], tuple(row["bbox"]), row["track_id"])
        for row in rows
        if not row.get("outside", False)
    ]


def compare_stitch(
    item: dict, ground_truth: list[Box], *, reference: KittiReference | None = None, config: StitchConfig | None = None
) -> tuple[dict, StitchResult]:
    if not item.get("window_predictions"):
        raise ValueError("GPU result has no preserved two-sided window overlap")
    frames = set(item["evaluated_source_frames"])
    stitched = stitch_tracks(_window_tracks(item["window_predictions"]), config)
    before = _boxes(item["predictions"])
    after = [
        Box(obs.frame, track.label, obs.bbox, track.track_id) for track in stitched.tracks for obs in track.observations
    ]
    before_metrics = tracking_metrics(ground_truth, before, frames)
    after_metrics = tracking_metrics(ground_truth, after, frames)
    report: dict = {
        "evaluated_frames": len(frames),
        "before": {
            "iou_05": before_metrics["thresholds"]["0.5"],
            "iou_07": before_metrics["thresholds"]["0.7"],
            "id_switches": before_metrics["id_switches"],
        },
        "after": {
            "iou_05": after_metrics["thresholds"]["0.5"],
            "iou_07": after_metrics["thresholds"]["0.7"],
            "id_switches": after_metrics["id_switches"],
        },
        "before_tracks": len({box.track_id for box in before}),
        "after_tracks": len(stitched.tracks),
        "decisions": [asdict(decision) for decision in stitched.decisions],
        "boundary_logs": stitched.boundary_logs,
    }
    if reference is not None:
        for name, boxes in (("before", before), ("after", after)):
            adjusted = evaluate_kitti_style(reference, boxes, frames)
            report[name]["kitti_style"] = {
                "iou_05": adjusted["metrics"]["thresholds"]["0.5"],
                "iou_07": adjusted["metrics"]["thresholds"]["0.7"],
                "id_switches": adjusted["metrics"]["id_switches"],
                "ignored_predictions": adjusted["ignored_predictions"],
            }
    return report, stitched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--kitti-labels", type=Path)
    parser.add_argument("--kitti-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.kitti_labels is None) != (args.kitti_manifest is None):
        parser.error("--kitti-labels and --kitti-manifest must be paired")
    result = json.loads(args.result.read_text(encoding="utf-8"))
    item = next((item for item in result["configurations"] if item["id"] == args.config_id), None)
    if item is None:
        parser.error("configuration not found in GPU result")
    report, stitched = compare_stitch(
        item,
        read_boxes(args.ground_truth),
        reference=load_reference(args.kitti_labels, args.kitti_manifest) if args.kitti_labels else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "schema": "autolabel-stitch-comparison/v1",
                "config_id": args.config_id,
                "comparison": report,
                "tracks": [asdict(track) for track in stitched.tracks],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"{args.config_id}: ID switches {report['before']['id_switches']} -> "
        f"{report['after']['id_switches']}; tracks {report['before_tracks']} -> "
        f"{report['after_tracks']}"
    )


if __name__ == "__main__":
    main()
