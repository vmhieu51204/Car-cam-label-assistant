"""Read frozen inputs; simulate export filters/sparse rectangles with the shared matcher."""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scripts.eval_autolabel import Box
from scripts.stitch_sam3_benchmark import _window_tracks
from src.models.video import BoundingBox2D, TrackObservation, VideoTrack
from src.services.evaluation.phase0 import canonical_predictions, file_hash, load_frozen_run
from src.services.evaluation.tracking import tracking_diagnostics
from src.services.video.export_2d import Export2DConfig, _boxes
from src.services.video.keyframe_sparsify import SparseConfig, box_iou, interpolate, sparsify_shapes
from src.services.video.sequence_review import score_tracks
from src.services.video.stitching import ChunkObservation, UnifiedTrack, stitch_tracks
from src.services.video.track_filter import TrackFilterConfig, classify_tracks


def export_experiments(protocol: Path, root: Path, output: Path) -> dict[str, Any]:
    data = json.loads(protocol.read_text(encoding="utf-8"))
    report: dict[str, Any] = {
        "schema": "autolabel-export-experiments/v1",
        "protocol_sha256": file_hash(protocol),
        "note": "Offline simulation, not CVAT verification. Filtering hides boxes ONLY in the counterfactual view.",
        "services_called": [],
        "runs": [],
    }
    for run in data["runs"]:
        frozen = load_frozen_run(run, root)
        windows = frozen.item.get("window_predictions", [])
        if not windows:
            raise ValueError("post-stitch export experiment requires preserved overlap predictions")
        reset_windows = {r["window_id"] for r in windows if frozen.records[r["frame_index"]].get("scene_cut_before")}
        stitched = stitch_tracks(_window_tracks(windows), reset_windows=reset_windows)
        sampled_boxes = [
            Box(frozen.source[o.frame], t.label, o.bbox, t.track_id) for t in stitched.tracks for o in t.observations
        ]
        dense_timeline = [
            Box(f, t.label, b, t.track_id)
            for t in stitched.tracks
            for f, b in _boxes(t, frozen.source, Export2DConfig())
        ]
        reviews = {r.track_id: r for r in score_tracks(stitched.tracks)}
        grouped: dict[tuple[str, str], list[Box]] = defaultdict(list)
        for box in sampled_boxes:
            grouped[(box.label, str(box.track_id))].append(box)
        source_to_sample = {frame: i for i, frame in enumerate(frozen.source)}
        labels = {name: i + 1 for i, name in enumerate(sorted({b.label for b in sampled_boxes}))}
        tracks = [
            VideoTrack(
                track_id=f"{label}:{identity}",
                label_id=labels[label],
                label_name=label,
                source="auto",
                observations=[
                    TrackObservation(
                        frame_index=source_to_sample[b.frame],
                        bbox=BoundingBox2D(xtl=b.xyxy[0], ytl=b.xyxy[1], xbr=b.xyxy[2], ybr=b.xyxy[3]),
                    )
                    for b in sorted(boxes, key=lambda b: b.frame)
                ],
            )
            for (label, identity), boxes in sorted(grouped.items())
        ]
        frame_metadata = frozen.records[0]
        width = int(frame_metadata["width"])
        height = int(frame_metadata["height"])
        filter_config = TrackFilterConfig(enabled=True)
        suspects = classify_tracks(tracks, filter_config, width=width, height=height)
        kept = [b for b in sampled_boxes if f"{b.label}:{b.track_id}" not in suspects]
        sparse_boxes: list[Box] = []
        deviations = []
        before_count = after_count = 0
        sidecars = {}
        for (label, identity), boxes in sorted(grouped.items()):
            rows: list[dict[str, Any]] = [
                {"frame": b.frame, "points": list(b.xyxy), "outside": False}
                for b in sorted(boxes, key=lambda b: b.frame)
            ]
            mandatory = {
                frozen.source[i]
                for i, record in enumerate(frozen.records)
                if record.get("scene_cut_before") or i % 45 in {0, 1, 2, 3, 4, 44}
            }
            if set(reviews[identity].reasons) - {"class_policy"}:
                mandatory.update(b.frame for b in boxes)
            reduced, reasons = sparsify_shapes(rows, SparseConfig(enabled=True), mandatory=mandatory)
            before_count += len(rows)
            after_count += len(reduced)
            sidecars[f"{label}:{identity}"] = reasons
            for row in rows:
                if row in reduced:
                    interpolated_box = row["points"]
                else:
                    left = max((r for r in reduced if r["frame"] < row["frame"]), key=lambda r: r["frame"])
                    right = min((r for r in reduced if r["frame"] > row["frame"]), key=lambda r: r["frame"])
                    interpolated_box = interpolate(left, right, row["frame"])
                deviations.append(box_iou(row["points"], interpolated_box))
                sparse_boxes.append(
                    Box(
                        row["frame"],
                        label,
                        (interpolated_box[0], interpolated_box[1], interpolated_box[2], interpolated_box[3]),
                        identity,
                    )
                )
        gt = tuple(b for b in frozen.gt if int(b.frame.frame_uid) in source_to_sample)
        frames = tuple(frozen.keys[f] for f in frozen.source)
        reconstructed: list[Box] = []
        for label, identity in grouped:
            observations = [
                ChunkObservation(source_to_sample[b.frame], b.xyxy)
                for b in sparse_boxes
                if b.label == label and b.track_id == identity
            ]
            t = UnifiedTrack(identity, label, sorted(observations, key=lambda o: o.frame))
            reconstructed.extend(Box(f, label, b, identity) for f, b in _boxes(t, frozen.source, Export2DConfig()))
        timeline_metrics = {}
        for view, boxes in {"dense_timeline": dense_timeline, "sparse_timeline": reconstructed}.items():
            timeline_metrics[view] = tracking_diagnostics(
                frozen.gt,
                canonical_predictions(boxes, frozen.keys, run["config_id"]),
                frames=tuple(frozen.keys[f] for f in sorted(frozen.keys)),
                occlusions=frozen.occlusions,
                tracking_ids_available=True,
            )
        metrics = {}
        for view, boxes in {
            "dense": sampled_boxes,
            "filter_counterfactual": kept,
            "sparse_sampled_simulation": sparse_boxes,
        }.items():
            metrics[view] = tracking_diagnostics(
                gt,
                canonical_predictions(boxes, frozen.keys, run["config_id"]),
                frames=frames,
                occlusions=frozen.occlusions,
                tracking_ids_available=True,
            )
        report["runs"].append(
            {
                "sequence": run["sequence"],
                "view": "stitched_sampled",
                "timeline_metrics": timeline_metrics,
                "input_hashes": frozen.input_hashes,
                "metrics": metrics,
                "suspects": suspects,
                "filter_config": filter_config.model_dump(),
                "hidden_boxes": len(sampled_boxes) - len(kept),
                "annotations_hidden": False,
                "tp_delta": {
                    k: metrics["filter_counterfactual"]["thresholds"][k]["tp"] - v["tp"]
                    for k, v in metrics["dense"]["thresholds"].items()
                },
                "keyframes_before": before_count,
                "keyframes_after": after_count,
                "mean_dense_box_iou": sum(deviations) / len(deviations) if deviations else None,
                "min_dense_box_iou": min(deviations) if deviations else None,
                "keyframe_reasons": sidecars,
                "reason_frequency": dict(Counter(reason for reasons in suspects.values() for reason in reasons)),
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
