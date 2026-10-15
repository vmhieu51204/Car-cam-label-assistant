"""Compare offline SAM3 benchmark tracks with video ground truth on evaluated frames."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from scripts.eval_autolabel import Box, read_boxes, tracking_metrics
from scripts.kitti_eval import KittiReference, evaluate_kitti_style, load_reference


def _predictions(items: list[dict]) -> list[Box]:
    return [
        Box(frame=item["frame_index"], label=item["label"], xyxy=tuple(item["bbox"]), track_id=item["track_id"])
        for item in items
        if not item.get("outside", False)
    ]


def _discovery_boxes(discovery_path: Path, manifest_path: Path) -> list[Box]:
    discovery = json.loads(discovery_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mapping = {item["frame_index"]: item.get("source_frame_index", item["frame_index"]) for item in manifest["frames"]}
    return [
        Box(
            frame=mapping[obs["frame_index"]],
            label=track["label_name"],
            xyxy=tuple(obs["bbox"][key] for key in ("xtl", "ytl", "xbr", "ybr")),
            track_id=track["track_id"],
        )
        for track in discovery["proposal_tracks"]
        for obs in track["observations"]
        if not obs.get("outside", False) and obs["frame_index"] in mapping
    ]


def compare(
    result: dict,
    ground_truth: list[Box],
    *,
    discovery: list[Box] | None = None,
    discovery_frames: set[int] | None = None,
    kitti_reference: KittiReference | None = None,
) -> list[dict]:
    rows = []
    discovery_frames = discovery_frames or {box.frame for box in discovery or []}
    evaluated = [
        item
        for item in result["configurations"]
        if item["status"] in {"completed", "timed_out"} and item.get("evaluated_source_frames")
    ]
    baseline = next((item for item in evaluated if item["id"] in {"optical-current", "fn-baseline"}), None) or max(
        evaluated, key=lambda item: len(item["evaluated_source_frames"]), default=None
    )
    baseline_frames = set(baseline.get("evaluated_source_frames", [])) if baseline else set()
    baseline_boxes = _predictions(baseline.get("predictions", [])) if baseline else []
    for item in result["configurations"]:
        frames = set(item.get("evaluated_source_frames", []))
        predictions = _predictions(item.get("predictions", []))
        metrics = tracking_metrics(ground_truth, predictions, frames)
        overall05 = metrics["thresholds"]["0.5"]["overall"]
        overall07 = metrics["thresholds"]["0.7"]["overall"]
        track_frames: dict[str, set[int]] = defaultdict(set)
        for box in predictions:
            if box.frame in frames and box.track_id is not None:
                track_frames[box.track_id].add(box.frame)
        tiny_boxes = sum(
            box.frame in frames and min(box.xyxy[2] - box.xyxy[0], box.xyxy[3] - box.xyxy[1]) < 6 for box in predictions
        )
        runtime = item.get("runtime", {})
        row = {
            "id": item["id"],
            "status": item["status"],
            "error": item.get("error", ""),
            "selection": item["config"]["selection"],
            "window_size": item["config"]["window_size"],
            "overlap": item["config"]["overlap"],
            "reprompt": item["config"]["reprompt"],
            "mode": item["config"]["mode"],
            "evaluated_frames": len(frames),
            "precision_iou_05": overall05["precision"],
            "recall_iou_05": overall05["recall"],
            "precision_iou_07": overall07["precision"],
            "recall_iou_07": overall07["recall"],
            "raw_false_positive_boxes_iou_05": overall05["predicted"] - overall05["tp"],
            "window_local_track_ids": len(track_frames),
            "one_frame_track_ids": sum(len(observed) == 1 for observed in track_frames.values()),
            "tiny_prediction_boxes_lt6px": tiny_boxes,
            "mean_iou": overall05["mean_iou"],
            "id_switches": metrics["id_switches"],
            "recall_early": metrics["recall_by_segment"]["early"],
            "recall_middle": metrics["recall_by_segment"]["middle"],
            "recall_late": metrics["recall_by_segment"]["late"],
            "new_object_recall": metrics["new_object_recall"],
            "window_seconds_mean": runtime.get("window_seconds", {}).get("mean"),
            "total_seconds": runtime.get("total_seconds"),
            "vram_peak_bytes": runtime.get("peak_cuda_allocated_bytes"),
            "vram_by_device": json.dumps(runtime.get("peak_cuda_allocated_bytes_by_device")),
            "mask_to_box_fallbacks": runtime.get("mask_to_box_fallbacks"),
        }
        for label in sorted({box.label for box in ground_truth}):
            for threshold in ("0.5", "0.7"):
                rates = metrics["thresholds"][threshold]["by_class"].get(label, {})
                suffix = threshold.replace(".", "")
                row[f"{label}_precision_iou_{suffix}"] = rates.get("precision")
                row[f"{label}_recall_iou_{suffix}"] = rates.get("recall")
        if kitti_reference is not None and frames:
            adjusted = evaluate_kitti_style(kitti_reference, predictions, frames)
            adjusted_rates = adjusted["metrics"]["thresholds"]["0.5"]
            adjusted_rates_07 = adjusted["metrics"]["thresholds"]["0.7"]
            row["kitti_style_precision_iou_05"] = adjusted_rates["overall"]["precision"]
            row["kitti_style_recall_iou_05"] = adjusted_rates["overall"]["recall"]
            row["kitti_style_precision_iou_07"] = adjusted_rates_07["overall"]["precision"]
            row["kitti_style_recall_iou_07"] = adjusted_rates_07["overall"]["recall"]
            row["kitti_style_id_switches"] = adjusted["metrics"]["id_switches"]
            row["kitti_style_new_object_recall"] = adjusted["metrics"]["new_object_recall"]
            row["kitti_style_valid_gt"] = adjusted["valid_ground_truth"]
            row["kitti_style_retained_predictions"] = adjusted["retained_predictions"]
            row["kitti_style_ignored_predictions"] = json.dumps(adjusted["ignored_predictions"])
            for label in ("car", "person"):
                rates = adjusted_rates["by_class"].get(label, {})
                row[f"kitti_style_{label}_precision_iou_05"] = rates.get("precision")
                row[f"kitti_style_{label}_recall_iou_05"] = rates.get("recall")
        common = frames & baseline_frames
        row["baseline_id"] = baseline["id"] if baseline else None
        row["common_baseline_frames"] = len(common)
        row["frame_coverage_vs_baseline"] = len(common) / len(baseline_frames) if baseline_frames else None
        if common and item["status"] in {"completed", "timed_out"}:
            own_common = tracking_metrics(ground_truth, predictions, common)
            base_common = tracking_metrics(ground_truth, baseline_boxes, common)
            own_recall = own_common["thresholds"]["0.5"]["overall"]["recall"]
            base_recall = base_common["thresholds"]["0.5"]["overall"]["recall"]
            row["recall_iou_05_common"] = own_recall
            row["baseline_recall_iou_05_common"] = base_recall
            row["recall_delta_vs_baseline_common"] = (
                own_recall - base_recall if own_recall is not None and base_recall is not None else None
            )
        else:
            row["recall_iou_05_common"] = None
            row["baseline_recall_iou_05_common"] = None
            row["recall_delta_vs_baseline_common"] = None
        if discovery is not None and item["config"]["mode"] == "text_prompt":
            common = frames & discovery_frames
            # Report both methods on identical observed source frames only.
            row["text_new_object_recall_common"] = tracking_metrics(
                ground_truth, _predictions(item.get("predictions", [])), common
            )["new_object_recall"]
            row["rtdetr_new_object_recall_common"] = tracking_metrics(ground_truth, discovery, common)[
                "new_object_recall"
            ]
            row["common_discovery_frames"] = len(common)
        rows.append(row)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["id"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--discovery", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--kitti-labels", type=Path, help="Raw KITTI label_02 sequence file for KITTI-style ignore preprocessing"
    )
    parser.add_argument(
        "--kitti-manifest", type=Path, help="5-fps selection manifest corresponding to the raw KITTI labels"
    )
    args = parser.parse_args()
    if (args.discovery is None) != (args.manifest is None):
        parser.error("--discovery and --manifest must be supplied together")
    if (args.kitti_labels is None) != (args.kitti_manifest is None):
        parser.error("--kitti-labels and --kitti-manifest must be supplied together")
    discovery = _discovery_boxes(args.discovery, args.manifest) if args.discovery else None
    discovery_frames = None
    if args.manifest:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        discovery_frames = {item.get("source_frame_index", item["frame_index"]) for item in manifest["frames"]}
    rows = compare(
        json.loads(args.result.read_text(encoding="utf-8")),
        read_boxes(args.ground_truth),
        discovery=discovery,
        discovery_frames=discovery_frames,
        kitti_reference=(load_reference(args.kitti_labels, args.kitti_manifest) if args.kitti_labels else None),
    )
    write_csv(rows, args.csv)
    print("id\tstatus\tframes\tP/R@0.5\tP/R@0.7\tmean IoU\tID switches\tearly/middle/late\tnew recall\tseconds\tVRAM")
    for row in rows:
        print(
            f"{row['id']}\t{row['status']}\t{row['evaluated_frames']}\t"
            f"{row['precision_iou_05']}/{row['recall_iou_05']}\t"
            f"{row['precision_iou_07']}/{row['recall_iou_07']}\t{row['mean_iou']}\t"
            f"{row['id_switches']}\t{row['recall_early']}/{row['recall_middle']}/{row['recall_late']}\t"
            f"{row['new_object_recall']}\t{row['total_seconds']}\t{row['vram_peak_bytes']}"
        )
        if args.kitti_labels and row.get("kitti_style_valid_gt") is not None:
            print(
                f"  KITTI-style ignore-adjusted P/R@0.5: "
                f"{row['kitti_style_precision_iou_05']}/{row['kitti_style_recall_iou_05']} "
                f"(GT {row['kitti_style_valid_gt']}, predictions "
                f"{row['kitti_style_retained_predictions']})"
            )


if __name__ == "__main__":
    main()
