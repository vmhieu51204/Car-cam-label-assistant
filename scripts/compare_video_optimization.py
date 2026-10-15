"""Compare sampled predictions and CVAT-style rectangle interpolation on one timeline."""

from __future__ import annotations

import argparse
import csv
import json
from bisect import bisect_right
from pathlib import Path

from scripts.eval_autolabel import Box, read_boxes, tracking_metrics
from scripts.kitti_eval import KittiReference, evaluate_kitti_style, load_reference
from scripts.stitch_sam3_benchmark import _window_tracks
from src.services.video.stitching import stitch_tracks


def interpolate_tracks(tracks: list, selected: list[int], frames: set[int]) -> list[Box]:
    """Mirror CVAT rectangle-track linear points and left-keyframe outside semantics.

    Reference: cvat-core/src/annotations-objects/{rectangle-track,track}.ts.
    Missing sampled observations create outside at the next selected source frame.
    """
    result = []
    next_selected = dict(zip(selected, selected[1:]))
    for track in tracks:
        visible = {obs.frame: obs.bbox for obs in track.observations}
        keys = {frame: (box, False) for frame, box in visible.items()}
        for frame, box in visible.items():
            following = next_selected.get(frame)
            if following is not None and following not in visible:
                keys[following] = (box, True)
        ordered = sorted(keys)
        for frame in sorted(frames):
            left = bisect_right(ordered, frame) - 1
            if left < 0:
                continue
            a = ordered[left]
            box, outside = keys[a]
            if outside:
                continue
            if left + 1 < len(ordered):
                b = ordered[left + 1]
                right_box = keys[b][0]
                offset = (frame - a) / (b - a)
                box = tuple(x + (y - x) * offset for x, y in zip(box, right_box, strict=True))
            result.append(Box(frame, track.label, tuple(box), track.track_id))
    return result


def compare(
    result: dict,
    gt: list[Box],
    *,
    total_frames: int,
    reference: KittiReference | None = None,
    baseline_gap_frames: set[int] | None = None,
) -> dict:
    """Evaluate policies on the same timeline and, optionally, the same 5-FPS gaps.

    Per-policy ``interpolated_only`` covers different frames when FPS differs;
    ``baseline_gap_frames`` keeps the denominator fixed for a fair comparison.
    """
    report = []
    timeline = set(range(total_frames))
    if baseline_gap_frames is not None and not baseline_gap_frames <= timeline:
        raise ValueError("baseline gap frame indices must be inside the source timeline")
    for item in result.get("configurations", []):
        row = {"id": item["id"], "status": item["status"], "error": item.get("error"), "metrics": {}}
        runtime = item.get("runtime", {})
        row.update(
            inference_seconds=runtime.get("timing_s", {}).get("total_inference"),
            peak_cuda_allocated_bytes=runtime.get("peak_cuda_allocated_bytes"),
            sampled_frames=len(item.get("selected_source_frames", [])),
        )
        if item["status"] != "completed":
            report.append(row)
            continue
        selected = item["selected_source_frames"]
        config = item.get("config", {})
        cuts = set(config.get("scene_cut_frames", []))
        reset_windows = {
            entry["window_id"]
            for entry in item.get("window_checkpoints", [])
            if entry.get("frame_ids") and entry["frame_ids"][0] in cuts
        }
        stitched = stitch_tracks(_window_tracks(item["window_predictions"]), reset_windows=reset_windows)
        sampled = [
            Box(obs.frame, track.label, obs.bbox, track.track_id)
            for track in stitched.tracks
            for obs in track.observations
        ]
        dense = interpolate_tracks(stitched.tracks, selected, timeline)
        between = timeline - set(selected)
        subsets = [
            ("sampled", sampled, set(selected)),
            ("full_interpolated", dense, timeline),
            ("interpolated_only", dense, between),
        ]
        if baseline_gap_frames is not None:
            subsets.append(("baseline_gaps", dense, baseline_gap_frames))
        for name, boxes, frames in subsets:
            row["metrics"][name] = {"raw": tracking_metrics(gt, boxes, frames)}
            if reference is not None:
                row["metrics"][name]["kitti_style"] = evaluate_kitti_style(reference, boxes, frames)
        row["tracks_after_stitch"] = len(stitched.tracks)
        report.append(row)
    return {
        "schema": "autolabel-video-optimization-comparison/v1",
        "model_load_seconds": result.get("model_load_seconds"),
        "evaluated_source_frames": total_frames,
        "configurations": report,
        "caveat": "Single 40-second KITTI 0020 segment; interpolation is geometry, not model inference.",
    }


def save(report: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    rows = []
    modes = ["sampled", "full_interpolated", "interpolated_only"]
    if any("baseline_gaps" in item.get("metrics", {}) for item in report["configurations"]):
        modes.append("baseline_gaps")
    for item in report["configurations"]:
        row = {
            key: item.get(key)
            for key in ("id", "status", "sampled_frames", "inference_seconds", "peak_cuda_allocated_bytes")
        }
        for mode in modes:
            metric = item["metrics"].get(mode, {}).get("raw", {})
            overall = metric.get("thresholds", {}).get("0.5", {}).get("overall", {})
            row.update({f"{mode}_{key}": overall.get(key) for key in ("precision", "recall", "mean_iou")})
            row[f"{mode}_id_switches"] = metric.get("id_switches")
            row[f"{mode}_new_object_recall"] = metric.get("new_object_recall")
        rows.append(row)
    if rows:
        with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--kitti-labels", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = load_reference(args.kitti_labels, args.manifest) if args.kitti_labels and args.manifest else None
    report = compare(
        json.loads(args.result.read_text()),
        read_boxes(args.ground_truth),
        total_frames=args.frames,
        reference=reference,
    )
    save(report, args.output)
    print(
        json.dumps(
            [
                {key: row.get(key) for key in ("id", "status", "sampled_frames", "inference_seconds")}
                for row in report["configurations"]
            ]
        )
    )


if __name__ == "__main__":
    main()
