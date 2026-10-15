"""Offline phase-0 reports over frozen, already downloaded artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.eval_autolabel import Box, read_cvat_xml
from scripts.eval_sam3_export_timeline import _source_reference
from scripts.stitch_sam3_benchmark import _window_tracks
from src.models.evaluation import (
    FrameKey,
    NativeKittiRow,
    ObservationSource,
    PredictionBox,
    ReferenceBox,
    Score,
    SequenceKey,
    SourceFile,
)
from src.services.evaluation.references.kitti import _preprocess_reasons
from src.services.evaluation.tracking import tracking_diagnostics
from src.services.video.export_2d import Export2DConfig
from src.services.video.export_2d import _boxes as interpolate_boxes
from src.services.video.stitching import stitch_tracks
from tools.public_gt.kitti import read_native_rows


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_predictions(boxes: list[Box], frames: dict[int, FrameKey], config_id: str) -> tuple[PredictionBox, ...]:
    return tuple(
        PredictionBox(
            prediction_id=f"{config_id}/prediction/{i}",
            frame=frames[x.frame],
            class_name=x.label,
            bbox_xyxy=x.xyxy,
            raw_track_id=x.track_id or f"unavailable/{i}",
            source=ObservationSource(
                mapping_id="phase0-source/v1", source_ordinal=i, config_id=config_id, source_observation_id=str(i)
            ),
            scoring_ordinal=i,
            score=Score(value=None, semantics="missing", source="phase0-adapter"),
            origin="export_unknown",
        )
        for i, x in enumerate(boxes)
    )


@dataclass(frozen=True)
class FrozenRunData:
    """One read of frozen artifacts shared by baseline and miss diagnostics."""

    result: dict[str, Any]
    item: dict[str, Any]
    manifest: dict[str, Any]
    records: list[dict[str, Any]]
    source: list[int]
    keys: dict[int, FrameKey]
    gt: tuple[ReferenceBox, ...]
    native_by_annotation: dict[str, NativeKittiRow]
    occlusions: dict[str, str]
    excluded_classes: dict[str, int]
    excluded_sampled_classes: dict[str, int]
    direct: list[Box]
    input_hashes: dict[str, str]
    native_rows: tuple[NativeKittiRow, ...]


def load_frozen_run(run: dict[str, Any], root: Path) -> FrozenRunData:
    result_path, mapping_path, labels_path = (root / run[k] for k in ("result", "mapping", "labels"))
    input_fields = {"result", "mapping", "labels"} | ({"cvat_export"} if run.get("cvat_export") else set())
    for field, expected in run.get("sha256", {}).items():
        if field not in input_fields or file_hash(root / run[field]) != expected:
            raise ValueError("frozen input checksum mismatch")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    item = next(c for c in result["configurations"] if c["id"] == run["config_id"])
    if item.get("status") != "completed":
        raise ValueError("baseline needs a completed configuration")
    manifest = json.loads(mapping_path.read_text(encoding="utf-8"))
    records = sorted(manifest["frames"], key=lambda x: x["frame_index"])
    if [x["frame_index"] for x in records] != list(range(len(records))):
        raise ValueError("mapping must have contiguous processed indices")
    source = [x["source_frame_index"] for x in records]
    if source != sorted(set(source)) or not source or source[0] < 0:
        raise ValueError("source mapping must be unique ascending")
    if set(item["evaluated_source_frames"]) != set(range(len(records))):
        raise ValueError("result and source mapping disagree")
    class_map = run["class_map"]
    if len(set(class_map.values())) != len(class_map):
        raise ValueError("phase0 raw class mapping may not silently merge classes")
    sequence = SequenceKey(dataset_id="kitti", split="training", sequence_uid=run["sequence"])
    keys = {i: FrameKey(sequence=sequence, frame_uid=str(i)) for i in range(source[0], source[-1] + 1)}
    gt: list[ReferenceBox] = []
    native_by_annotation: dict[str, NativeKittiRow] = {}
    occlusions: dict[str, str] = {}
    excluded_classes: Counter[str] = Counter()
    excluded_sampled_classes: Counter[str] = Counter()
    sampled_sources = set(source)
    native_count = run.get("native_frame_count")
    if type(native_count) is not int or native_count <= source[-1]:
        raise ValueError("declare native_frame_count covering the sampled source timeline")
    native_rows = read_native_rows(
        labels_path,
        sequence=sequence,
        source=SourceFile(relative_path=labels_path.name, sha256=file_hash(labels_path), role="labels"),
        frame_count=native_count,
    )
    for i, row in enumerate(native_rows):
        frame, native = row.source_frame_index, row.native_class
        if frame not in keys:
            continue
        if native not in class_map:
            excluded_classes[native] += 1
            if frame in sampled_sources:
                excluded_sampled_classes[native] += 1
            continue
        box = ReferenceBox(
            annotation_id=f"kitti/{run['sequence']}/{i}",
            native_annotation_id=str(i),
            frame=keys[frame],
            class_name=class_map[native],
            bbox_xyxy=row.bbox_xyxy,
            native_track_id=str(row.track_id),
            source_ordinal=i,
            reference_kind="native_2d",
            native_category=native,
        )
        gt.append(box)
        native_by_annotation[box.annotation_id] = row
        occluded = row.occluded
        if occluded not in (0, 1, 2, 3):
            raise ValueError("invalid native KITTI occlusion")
        occlusions[box.annotation_id] = ("none", "partial", "heavy", "unknown")[occluded]
    direct = [
        Box(source[x["frame_index"]], x["label"], tuple(x["bbox"]), x["track_id"])
        for x in item["predictions"]
        if not x.get("outside", False)
    ]
    return FrozenRunData(
        result=result,
        item=item,
        manifest=manifest,
        records=records,
        source=source,
        keys=keys,
        gt=tuple(gt),
        native_by_annotation=native_by_annotation,
        occlusions=occlusions,
        excluded_classes=dict(excluded_classes),
        direct=direct,
        excluded_sampled_classes=dict(excluded_sampled_classes),
        input_hashes={k: file_hash(root / run[k]) for k in sorted(input_fields)},
        native_rows=native_rows,
    )


def _stored_run(run: dict[str, Any], root: Path) -> dict[str, Any]:
    loaded = load_frozen_run(run, root)
    result, item, manifest = loaded.result, loaded.item, loaded.manifest
    records, source, keys = loaded.records, loaded.source, loaded.keys
    gt, occlusions = loaded.gt, loaded.occlusions
    direct, class_map = loaded.direct, run["class_map"]
    labels_path = root / run["labels"]
    views: dict[str, list[Box]] = {"direct_sampled": direct}
    # Do not invent the dropped second overlap from old artifacts.
    windows = item.get("window_predictions", [])
    if windows:
        resets = {x["window_id"] for x in windows if records[x["frame_index"]].get("scene_cut_before", False)}
        stitched = stitch_tracks(_window_tracks(windows), reset_windows=resets)
        views["stitched_sampled"] = [
            Box(source[x.frame], t.label, x.bbox, t.track_id) for t in stitched.tracks for x in t.observations
        ]
        views["interpolated_timeline"] = [
            Box(frame, t.label, box, t.track_id)
            for t in stitched.tracks
            for frame, box in interpolate_boxes(t, source, Export2DConfig())
        ]
        views["interpolated_only"] = [x for x in views["interpolated_timeline"] if x.frame not in set(source)]
    if run.get("cvat_export"):
        views["cvat_export"] = [x for x in read_cvat_xml(root / run["cvat_export"]) if x.frame in keys]
    report_views = {}
    for name, boxes in views.items():
        frame_set = (
            set(keys)
            if name in {"interpolated_timeline", "cvat_export"}
            else set(keys) - set(source)
            if name == "interpolated_only"
            else set(source)
        )
        selected_gt = tuple(x for x in gt if int(x.frame.frame_uid) in frame_set)
        pred = canonical_predictions(boxes, keys, run["config_id"])
        frames = tuple(keys[i] for i in sorted(frame_set))
        ids_available = all(x.track_id is not None for x in boxes)
        raw = tracking_diagnostics(
            selected_gt, pred, frames=frames, occlusions=occlusions, tracking_ids_available=ids_available
        )
        adjusted = None
        if class_map == {"Car": "car", "Pedestrian": "person"}:
            _, reference = _source_reference(labels_path, frame_set)
            valid_gt, retained, removed = _preprocess_reasons(reference, boxes, frame_set)
            valid = {(x.frame, x.track_id) for x in valid_gt}
            adjusted_gt = tuple(x for x in selected_gt if (int(x.frame.frame_uid), x.native_track_id) in valid)
            adjusted = {
                "diagnostics": tracking_diagnostics(
                    adjusted_gt,
                    canonical_predictions(retained, keys, run["config_id"]),
                    frames=frames,
                    occlusions=occlusions,
                    tracking_ids_available=ids_available,
                ),
                "ignored_predictions": dict(Counter(removed.values())),
                "note": "Project local ignore rules, not the official KITTI benchmark",
            }
        report_views[name] = {
            "raw": raw,
            "kitti_style": adjusted,
            "geometry": "CVAT_XML_export"
            if name == "cvat_export"
            else "offline_linear_interpolation"
            if name.startswith("interpolated")
            else "SAM3_mask_box",
        }
    hashes = loaded.input_hashes
    runtime = item.get("runtime", {})
    return {
        "sequence": run["sequence"],
        "config_id": run["config_id"],
        "config": item["config"],
        "input_hashes": hashes,
        "model_checkpoint_sha256": result.get("model_runtime", {}).get("checkpoint_sha256"),
        "model_commit": result.get("model_runtime", {}).get("environment", {}).get("versions", {}).get("sam3_commit"),
        "source_fps": manifest.get("source_fps", run.get("source_fps")),
        "fps_provenance": manifest.get("fps_method", run.get("fps_provenance", "unavailable")),
        "excluded_gt_classes": loaded.excluded_classes,
        "views": report_views,
        "missing_views": [] if windows else ["stitched_sampled", "interpolated_timeline"],
        "runtime": runtime_summary(result, item),
        "prediction_sha256": hashlib.sha256(json.dumps(item["predictions"], sort_keys=True).encode()).hexdigest(),
        "legacy_runtime_present": bool(runtime),
    }


def runtime_summary(result: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    runtime = item.get("runtime", {})
    timing = runtime.get("timing_s", {})
    windows = timing.get("windows", [])
    classes = timing.get("classes", [])
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in windows:
        predictions = [x for x in item.get("window_predictions", []) if x["window_id"] == row["window_id"]]
        if row["window_id"] in item.get("resumed_windows", []):
            continue  # Loading a checkpoint is not another model inference.
        confirmed = any(
            isinstance(x, dict) and x.get("window_id") == row["window_id"] for x in item.get("window_checkpoints", [])
        )
        if not predictions and not confirmed:
            continue  # Missing observations cannot prove a negative window.
        count = len({x["track_id"] for x in predictions})
        bucket = "0" if not count else "1-5" if count <= 5 else "6-10" if count <= 10 else ">10"
        buckets[bucket].append(row["seconds"])
    return {
        "model_load_s": result.get("model_load_seconds"),
        "inference_s": timing.get("total_inference", runtime.get("total_seconds")),
        "windows": windows,
        "classes": classes,
        "peak_gpu_memory_bytes": runtime.get("peak_cuda_allocated_bytes"),
        "memory_measurement": "torch.cuda.max_memory_allocated"
        if runtime.get("peak_cuda_allocated_bytes")
        else "unavailable",
        "object_count_basis": "unique_predicted_track_ids_per_window_not_GT",
        "by_object_count": {k: {"windows": len(v), "mean_window_s": sum(v) / len(v)} for k, v in buckets.items()},
    }


def collect_profile(job_dir: Path) -> dict[str, Any]:
    """Read allowlisted timing evidence; never dump env/keys or infer missing spans."""
    imports = job_dir / "import-runtime.json"
    imported = json.loads(imports.read_text(encoding="utf-8")) if imports.is_file() else {}
    events = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((job_dir / "profile-2d/events").glob("*.json"))]
    results = []
    for path in sorted(job_dir.glob("kaggle/*/output/result.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        for item in raw.get("configurations", []):
            results.append({"config_id": item["id"], "status": item.get("status"), **runtime_summary(raw, item)})
    return {
        "schema": "autolabel-profile-summary/v1",
        "import_total_s": imported.get("total_seconds"),
        "import_note": "CVAT delivery + image processing, not isolated video decode",
        "events": events,
        "sam3_runs": results,
        "decode_s": None,
        "end_to_end_s": None,
        "unavailable": ["isolated_video_decode", "end_to_end_without_complete_wall_spans"],
        "note": "Overlapping worker durations must not be summed as end-to-end",
    }


def claude_activity(database: Path, job_id: str) -> dict[str, Any]:
    """Read an existing DB in read-only mode, without Store startup/migrations."""
    with sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True) as connection:
        row = connection.execute(
            "SELECT status, attempts, partial_payload, proposals_payload FROM video_box_rescues WHERE job_id=?",
            (job_id,),
        ).fetchone()
    if row is None:
        return {"availability": "no_saved_claude_job", "accuracy": "unavailable_without_GT"}
    partial, proposals = json.loads(row[2]), json.loads(row[3])
    durations = [x["latency_s"] for x in partial if x.get("latency_s") is not None]
    usages = [x.get("usage", {}) for x in partial]
    costs = [x.get("cost_usd") for x in usages]
    return {
        "status": row[0],
        "worker_attempts": row[1],
        "completed_frames": len(partial),
        "displayed_proposals": len(proposals),
        "raw_proposals": sum(len(x.get("proposals", [])) for x in partial),
        "omitted_proposals": max(0, sum(len(x.get("proposals", [])) for x in partial) - len(proposals)),
        "models": sorted({x.get("model_name", "unknown") for x in partial}),
        "mean_latency_s": sum(durations) / len(durations) if durations else None,
        "saved_request_seconds": sum(durations) if durations else None,
        "prompt_tokens": sum(x.get("prompt_tokens", 0) for x in usages),
        "completion_tokens": sum(x.get("completion_tokens", 0) for x in usages),
        "reported_cost_usd": sum(costs) if costs and all(x is not None for x in costs) else None,
        "accuracy": "unavailable_without_independent_GT",
        "error_429_count": None,
        "note": "Partial results do not retain failed-request counts or full worker duration",
    }


def write_baseline(protocol: Path, *, root: Path, output: Path) -> dict[str, Any]:
    config = json.loads(protocol.read_text(encoding="utf-8"))
    if config.get("schema") != "autolabel-phase0-protocol/v1" or config.get("seed") != 42:
        raise ValueError("expected phase0 protocol v1 and evaluation seed 42")
    runs = [_stored_run(run, root) for run in config["runs"]]
    report: dict[str, Any] = {
        "schema": "autolabel-phase0-baseline/v1",
        "seed": 42,
        "inference_seed": "unavailable_for_old_artifacts",
        "protocol_sha256": file_hash(protocol),
        "runs": runs,
        "profiles": {name: collect_profile(root / path) for name, path in config.get("profile_jobs", {}).items()},
        "annotation_change": False,
        "services_called": [],
        "accuracy_claim": "baseline_only",
    }
    report["claude_activity"] = {
        name: claude_activity(root / entry["database"], entry["job_id"])
        for name, entry in config.get("claude_jobs", {}).items()
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "baseline.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    columns = [
        "sequence",
        "config_id",
        "view",
        "profile",
        "iou",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "mean_iou",
        "fp_per_frame",
        "idf1",
        "id_switches",
        "fragmentations",
        "track_recall",
    ]
    with (output / "baseline.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for run in runs:
            for view, evidence in run["views"].items():
                for profile in ("raw", "kitti_style"):
                    data = evidence.get(profile)
                    if not data:
                        continue
                    diagnostics = data if profile == "raw" else data["diagnostics"]
                    for threshold, values in diagnostics["thresholds"].items():
                        row = {
                            "sequence": run["sequence"],
                            "config_id": run["config_id"],
                            "view": view,
                            "profile": profile,
                            "iou": threshold,
                        }
                        row.update({k: values.get(k, values["tracking"].get(k)) for k in columns[5:]})
                        writer.writerow(row)
    slice_rows = []
    for run in runs:
        for view, evidence in run["views"].items():
            for threshold, metrics in evidence["raw"]["thresholds"].items():
                for dimension, groups in metrics["groups"].items():
                    for name, values in groups.items():
                        slice_rows.append(
                            {
                                "sequence": run["sequence"],
                                "view": view,
                                "iou": threshold,
                                "dimension": dimension,
                                "group": name,
                                **values,
                            }
                        )
    with (output / "slices.csv").open("w", encoding="utf-8", newline="") as stream:
        columns = [
            "sequence",
            "view",
            "iou",
            "dimension",
            "group",
            "tp",
            "gt",
            "recall",
            "mean_iou",
            "fp",
            "precision",
            "fp_per_frame",
        ]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(slice_rows)
    profile_rows = [
        {"run": run["sequence"], "stage": stage, "seconds": run["runtime"][field]}
        for run in runs
        for stage, field in (("model_load", "model_load_s"), ("SAM3_inference", "inference_s"))
    ]
    for name, profile in report["profiles"].items():
        profile_rows.append({"run": name, "stage": "import_total", "seconds": profile["import_total_s"]})
        profile_rows.extend(
            {"run": name, "stage": event["stage"], "seconds": event["seconds"]} for event in profile["events"]
        )
        for sam in profile["sam3_runs"]:
            profile_rows.extend(
                [
                    {"run": name, "stage": "model_load", "seconds": sam["model_load_s"]},
                    {"run": name, "stage": "SAM3_inference", "seconds": sam["inference_s"]},
                ]
            )
    with (output / "profile.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["run", "stage", "seconds"])
        writer.writeheader()
        writer.writerows(profile_rows)
    window_rows = [
        {
            "sequence": run["sequence"],
            "object_count_basis": run["runtime"]["object_count_basis"],
            "bucket": bucket,
            **values,
        }
        for run in runs
        for bucket, values in run["runtime"]["by_object_count"].items()
    ]
    window_rows.extend(
        {"sequence": name, "object_count_basis": sam["object_count_basis"], "bucket": bucket, **values}
        for name, profile in report["profiles"].items()
        for sam in profile["sam3_runs"]
        for bucket, values in sam["by_object_count"].items()
    )
    with (output / "window-runtime.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["sequence", "object_count_basis", "bucket", "windows", "mean_window_s"]
        )
        writer.writeheader()
        writer.writerows(window_rows)
    return report


def compare_baselines(before: Path, after: Path) -> dict[str, Any]:
    """Phase0 equivalence check, not an algorithm improvement claim."""
    a, b = (json.loads(path.read_text(encoding="utf-8")) for path in (before, after))
    if a["protocol_sha256"] != b["protocol_sha256"] or a["seed"] != b["seed"]:
        raise ValueError("baseline protocol or evaluation seed changed")
    left = {(x["sequence"], x["config_id"]): x for x in a["runs"]}
    right = {(x["sequence"], x["config_id"]): x for x in b["runs"]}
    if left.keys() != right.keys():
        raise ValueError("baseline run population changed")
    rows = []
    for key, run in left.items():
        other = right[key]
        if run["input_hashes"] != other["input_hashes"] or run["prediction_sha256"] != other["prediction_sha256"]:
            raise ValueError("baseline artifacts changed")
        if run["views"].keys() != other["views"].keys():
            raise ValueError("baseline view population changed")
        for view, evidence in run["views"].items():
            for profile in ("raw", "kitti_style"):
                if (evidence.get(profile) is None) != (other["views"][view].get(profile) is None):
                    rows.append(
                        {
                            "sequence": key[0],
                            "view": view,
                            "profile": profile,
                            "deltas": {},
                            "availability_unchanged": False,
                        }
                    )
                    continue
                if evidence.get(profile) is None:
                    continue
                old = evidence[profile] if profile == "raw" else evidence[profile]["diagnostics"]
                new_data = other["views"][view][profile]
                new = new_data if profile == "raw" else new_data["diagnostics"]
                if old["thresholds"].keys() != new["thresholds"].keys():
                    raise ValueError("baseline threshold population changed")
                for threshold, metrics in old["thresholds"].items():
                    updated = new["thresholds"][threshold]
                    deltas = {
                        name: updated[name] - metrics[name]
                        if updated[name] is not None and metrics[name] is not None
                        else None
                        for name in ("tp", "fp", "fn", "precision", "recall", "mean_iou", "fp_per_frame")
                    }
                    deltas.update(
                        {
                            name: updated["tracking"][name] - metrics["tracking"][name]
                            if updated["tracking"][name] is not None and metrics["tracking"][name] is not None
                            else None
                            for name in ("idf1", "id_switches", "fragmentations", "track_recall")
                        }
                    )
                    availability_unchanged = all(
                        (metrics[name] is None) == (updated[name] is None)
                        for name in ("precision", "recall", "mean_iou", "fp_per_frame")
                    ) and all(
                        (metrics["tracking"][name] is None) == (updated["tracking"][name] is None)
                        for name in ("idf1", "id_switches", "fragmentations", "track_recall")
                    )
                    rows.append(
                        {
                            "sequence": key[0],
                            "view": view,
                            "profile": profile,
                            "iou": threshold,
                            "deltas": deltas,
                            "availability_unchanged": availability_unchanged,
                        }
                    )
    return {
        "schema": "autolabel-phase0-equivalence/v1",
        "inputs_unchanged": True,
        "metrics_unchanged": all(
            row["availability_unchanged"] and all(value in (None, 0) for value in row["deltas"].values())
            for row in rows
        ),
        "comparisons": rows,
        "claim": "same stored predictions; no new inference or quality improvement",
    }
