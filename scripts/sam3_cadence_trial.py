"""One private SAM3 benchmark: 5 versus 7.5 FPS on KITTI 0013 and 0020.

The uploaded ZIP contains images and prompts only. KITTI labels remain local.
Both clips are padded at the right/bottom to a shared size without moving boxes.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import sys
import zipfile
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from PIL import Image

from kaggle.sam3_benchmark.main import select_frames, validate_config
from scripts.compare_video_optimization import compare, save
from scripts.eval_autolabel import read_boxes
from scripts.kitti_eval import load_reference
from src.config import Settings
from src.services.video.kaggle_batch import KaggleBatchError, KaggleBatchRunner

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "kaggle/notebooks/autolabel_sam3_benchmark.ipynb"
SOURCES = (("0013", 0), ("0020", 500))


def _config(frame_count: int) -> dict:
    boundary = frame_count
    total = frame_count * 2
    common = {
        "selection": "uniform",
        "gap": None,
        "source_fps": 10,
        "window_size": 50,
        "overlap": 5,
        "mode": "text_prompt",
        "seed_type": "none",
        "reprompt": "none",
        "text_prompts": ["car", "person"],
        "preserve_overlap": True,
        "scene_cut_frames": [boundary],
        "force_include_frames": [boundary - 1, boundary, total - 1],
        "score_threshold_detection": 0.5,
        "new_det_thresh": 0.7,
    }
    config = {
        "schema": "autolabel-sam3-benchmark-config/v1",
        "max_total_seconds": 5400,
        "max_config_seconds": 2400,
        "source_frame_limit": total,
        "configurations": [{**common, "id": f"fps-{fps:g}", "target_fps": fps} for fps in (5.0, 7.5)],
    }
    validate_config(config)
    return config


def _local_ground_truth(labels: Path, sequence: str, start: int, count: int, width: int, height: int) -> dict:
    images = [
        {"id": index, "frame_index": index, "file_name": f"{index:06d}.jpg", "width": width, "height": height}
        for index in range(count)
    ]
    annotations: list[dict] = []
    for number, line in enumerate(labels.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 17:
            raise ValueError(f"invalid KITTI {sequence} label line {number}")
        source = int(fields[0])
        category = {"Car": 1, "Pedestrian": 2}.get(fields[2])
        if not start <= source < start + count or category is None:
            continue
        x1, y1, x2, y2 = map(float, fields[6:10])
        if x2 <= x1 or y2 <= y1:
            continue
        annotations.append(
            {
                "id": len(annotations),
                "image_id": source - start,
                "category_id": category,
                "track_id": f"{sequence}:{fields[1]}",
                "bbox": [x1, y1, x2 - x1, y2 - y1],
            }
        )
    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "car"}, {"id": 2, "name": "person"}],
    }


def _archive_entry(archive: zipfile.ZipFile, name: str, content: bytes | str) -> None:
    """Stable bytes make a repeated command reuse the same private Kaggle run."""
    entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    entry.compress_type = zipfile.ZIP_DEFLATED
    archive.writestr(entry, content)


def prepare(
    dataset_root: Path,
    output: Path,
    *,
    frames_per_clip: int = 200,
    starts: tuple[int, int] = (0, 500),
    max_bytes: int = 500_000_000,
) -> dict:
    """Package two fixed 10-Hz spans; source-frame endpoints enter both policies."""
    if frames_per_clip < 4 or frames_per_clip % 4 or max_bytes < 1:
        raise ValueError("frames_per_clip must be a positive multiple of four")
    training = dataset_root / "training" if (dataset_root / "training").is_dir() else dataset_root
    clips = []
    for (sequence, _), start in zip(SOURCES, starts, strict=True):
        directory = training / "image_02" / sequence
        labels = training / "label_02" / f"{sequence}.txt"
        paths = [directory / f"{source:06d}.png" for source in range(start, start + frames_per_clip)]
        if not labels.is_file() or any(not path.is_file() for path in paths):
            raise FileNotFoundError(
                f"KITTI {sequence} requires source PNG frames {start}.."
                f"{start + frames_per_clip - 1} and label_02/{sequence}.txt"
            )
        with Image.open(paths[0]) as first:
            width, height = first.size
        clips.append((sequence, start, paths, labels, width, height))
    width = max(clip[4] for clip in clips)
    height = max(clip[5] for clip in clips)
    config = _config(frames_per_clip)
    output.mkdir(parents=True, exist_ok=True)
    frames = []
    total_bytes = 0
    with zipfile.ZipFile(output / "payload.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for clip_index, (sequence, start, paths, labels, clip_width, clip_height) in enumerate(clips):
            offset = clip_index * frames_per_clip
            manifest = {
                "frames": [
                    {"frame_index": local, "source_frame_index": start + local} for local in range(frames_per_clip)
                ]
            }
            (output / f"{sequence}-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (output / f"{sequence}-ground-truth.coco.json").write_text(
                json.dumps(_local_ground_truth(labels, sequence, start, frames_per_clip, clip_width, clip_height)),
                encoding="utf-8",
            )
            for local, path in enumerate(paths):
                with Image.open(path) as opened:
                    if opened.size != (clip_width, clip_height):
                        raise ValueError(f"KITTI {sequence} changed image size at {path.name}")
                    canvas = Image.new("RGB", (width, height), "black")
                    canvas.paste(opened.convert("RGB"), (0, 0))
                    stream = io.BytesIO()
                    canvas.save(stream, format="JPEG", quality=95)
                encoded = stream.getvalue()
                total_bytes += len(encoded)
                if total_bytes > max_bytes:
                    raise ValueError("cadence benchmark exceeds max_bytes")
                index = offset + local
                name = f"frames/frame_{index:06d}.jpg"
                _archive_entry(archive, name, encoded)
                frames.append(
                    {
                        "source_frame_index": index,
                        "original_source_index": start + local,
                        "sequence": sequence,
                        "relative_path": name,
                        "sha256": hashlib.sha256(encoded).hexdigest(),
                        "width": width,
                        "height": height,
                        "scene_cut_before": index == frames_per_clip,
                    }
                )
        _archive_entry(
            archive, "frames.json", json.dumps({"frames": frames, "optical_source_indices": list(range(len(frames)))})
        )
        _archive_entry(archive, "benchmark-config.json", json.dumps(config))
        _archive_entry(archive, "seed.json", '{"seeds": []}')
        _archive_entry(
            archive,
            "dataset-attribution.json",
            json.dumps(
                {
                    "dataset": "KITTI Tracking",
                    "sequences": ["0013", "0020"],
                    "source_fps": 10,
                    "timebase": "nominal 10-Hz KITTI cadence",
                    "use": "private non-commercial evaluation",
                    "geometry": "right/bottom padding only",
                }
            ),
        )
    metadata = {
        "schema": "sam3-cadence-trial/v1",
        "frames_per_clip": frames_per_clip,
        "image_size": [width, height],
        "source_fps": 10,
        "segments": [
            {
                "sequence": clip[0],
                "source_start": clip[1],
                "offset": index * frames_per_clip,
                "frame_count": frames_per_clip,
            }
            for index, clip in enumerate(clips)
        ],
        "configurations": [
            {"id": row["id"], "sampled_frames": len(select_frames(frames, [], row))} for row in config["configurations"]
        ],
        "zip_bytes": (output / "payload.zip").stat().st_size,
        "ground_truth_in_zip": False,
    }
    (output / "segments.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _clip_result(result: dict, segment: dict) -> dict:
    offset, count = segment["offset"], segment["frame_count"]
    rows = []
    for original in result.get("configurations", []):
        item = dict(original)
        item["selected_source_frames"] = [
            index - offset for index in original.get("selected_source_frames", []) if offset <= index < offset + count
        ]
        item["window_predictions"] = [
            {**prediction, "frame_index": prediction["frame_index"] - offset}
            for prediction in original.get("window_predictions", [])
            if offset <= prediction["frame_index"] < offset + count
        ]
        item["window_checkpoints"] = [
            {**entry, "frame_ids": [index - offset for index in entry["frame_ids"]]}
            for entry in original.get("window_checkpoints", [])
            if entry.get("frame_ids") and offset <= entry["frame_ids"][0] < offset + count
        ]
        item["config"] = {**original.get("config", {}), "scene_cut_frames": []}
        timing = original.get("runtime", {}).get("timing_s", {})
        window_ids = {entry["window_id"] for entry in item["window_checkpoints"]}
        seconds = sum(entry["seconds"] for entry in timing.get("windows", []) if entry["window_id"] in window_ids)
        item["runtime"] = {**original.get("runtime", {}), "timing_s": {**timing, "total_inference": seconds}}
        rows.append(item)
    return {
        "schema": result.get("schema"),
        "model_load_seconds": result.get("model_load_seconds"),
        "configurations": rows,
    }


def evaluate(result: dict, package: Path, dataset_root: Path) -> dict:
    metadata = json.loads((package / "segments.json").read_text(encoding="utf-8"))
    training = dataset_root / "training" if (dataset_root / "training").is_dir() else dataset_root
    reports = {}
    for segment in metadata["segments"]:
        sequence = segment["sequence"]
        clip = _clip_result(result, segment)
        baseline = next((item for item in clip["configurations"] if item["id"] == "fps-5"), None)
        baseline_gaps = (
            (set(range(segment["frame_count"])) - set(baseline["selected_source_frames"])) if baseline else None
        )
        reference = load_reference(training / "label_02" / f"{sequence}.txt", package / f"{sequence}-manifest.json")
        reports[sequence] = compare(
            clip,
            read_boxes(package / f"{sequence}-ground-truth.coco.json"),
            total_frames=segment["frame_count"],
            reference=reference,
            baseline_gap_frames=baseline_gaps,
        )
        reports[sequence]["caveat"] = (
            f"KITTI {sequence} source frames {segment['source_start']}-"
            f"{segment['source_start'] + segment['frame_count'] - 1}; "
            "intermediate boxes are CVAT-style interpolation, not SAM3 inference."
        )
    report = {
        "schema": "sam3-cadence-comparison/v1",
        "segments": reports,
        "note": "Same source frames per policy; raw/KITTI-style box metrics are not official HOTA.",
    }
    report["quality_gate"] = assess_quality_gate(report)
    return report


def assess_quality_gate(report: dict) -> dict:
    """Apply uncalibrated engineering thresholds, never approve an FPS change.

    The target group is the fixed set of frames the 5-FPS baseline did not
    infer. A 7.5-FPS prediction can be either inferred or interpolated there.
    Precision and class recall use the entire source timeline. Independent
    non-KITTI human review remains mandatory even if every condition passes.
    """
    checks = {}
    for sequence, segment in report["segments"].items():
        rows = {row["id"]: row for row in segment["configurations"]}
        baseline, candidate = rows.get("fps-5"), rows.get("fps-7.5")
        if (
            baseline is None
            or candidate is None
            or baseline["status"] != "completed"
            or candidate["status"] != "completed"
        ):
            checks[sequence] = {"passed": False, "reason": "both configurations must complete"}
            continue
        old = baseline["metrics"]
        new = candidate["metrics"]
        old_recall = old["baseline_gaps"]["raw"]["thresholds"]["0.5"]["overall"]["recall"]
        new_recall = new["baseline_gaps"]["raw"]["thresholds"]["0.5"]["overall"]["recall"]
        old_full = old["full_interpolated"]["raw"]["thresholds"]["0.5"]
        new_full = new["full_interpolated"]["raw"]["thresholds"]["0.5"]
        precision_delta = new_full["overall"]["precision"] - old_full["overall"]["precision"]
        class_deltas = {
            label: new_full["by_class"].get(label, {"recall": 0})["recall"] - row["recall"]
            for label, row in old_full["by_class"].items()
            if row["gt"] > 0
        }
        old_time, new_time = baseline.get("inference_seconds"), candidate.get("inference_seconds")
        has_time = (
            type(old_time) in (int, float)
            and type(new_time) in (int, float)
            and math.isfinite(old_time)
            and math.isfinite(new_time)
            and old_time > 0
        )
        time_ok = has_time and new_time <= old_time * 1.5
        conditions = {
            "baseline_gap_recall_gain_3pp": new_recall - old_recall >= 0.03,
            "raw_precision_loss_at_most_5pp": precision_delta >= -0.05,
            "inference_time_increase_at_most_50pct": time_ok,
            "no_class_recall_loss_over_2pp": all(value >= -0.02 for value in class_deltas.values()),
        }
        checks[sequence] = {
            "passed": all(conditions.values()),
            "conditions": conditions,
            "baseline_gap_recall_delta": new_recall - old_recall,
            "full_raw_precision_delta": precision_delta,
            "full_class_recall_deltas": class_deltas,
            "inference_time_ratio": new_time / old_time if has_time else None,
        }
    return {
        "numerical_gate_passed": bool(checks) and all(row["passed"] for row in checks.values()),
        "segments": checks,
        "outside_kitti_human_review_required": True,
        "change_production_default": False,
    }


def _resume_reserved_run(runner: KaggleBatchRunner, run_id: str, payload: Path, notebook: Path,
                         *, upload_missing: bool = False) -> dict:
    """Reuse the one reserved dataset/kernel identity after an interrupted upload.

    Never create a second dataset or kernel slug. A remote kernel with the
    reserved identity is collected even if the local push receipt was lost.
    """
    root = runner.work_root / run_id / "kaggle"
    candidates = list(root.glob(f"autolabel-sam3-benchmark-{run_id[:8]}-*"))
    if len(candidates) != 1:
        raise RuntimeError("reserved benchmark run directory is missing or ambiguous")
    run_dir = candidates[0]
    metadata = json.loads((run_dir / "kernel/kernel-metadata.json").read_text(encoding="utf-8"))
    dataset = json.loads((run_dir / "dataset/dataset-metadata.json").read_text(encoding="utf-8"))
    kernel_reference = metadata["id"]
    dataset_reference = dataset["id"]
    if (
        not isinstance(dataset_reference, str)
        or not re.fullmatch(rf"{re.escape(runner.username)}/[a-z0-9][a-z0-9-]{{1,99}}", dataset_reference)
        or kernel_reference != f"{runner.username}/{run_dir.name}-run"
        or metadata.get("dataset_sources") != [dataset_reference, runner.sam3_checkpoint_dataset]
        or metadata.get("code_file") != "main.ipynb"
        or hashlib.sha256((run_dir / "kernel/main.ipynb").read_bytes()).digest()
        != hashlib.sha256(notebook.read_bytes()).digest()
    ):
        raise RuntimeError("reserved Kaggle run metadata or notebook does not match this trial")
    with zipfile.ZipFile(run_dir / "dataset/payload.zip") as archive:
        if archive.namelist() != ["payload.zip"]:
            raise RuntimeError("reserved Kaggle input archive has unexpected members")
        digest = hashlib.sha256()
        with archive.open("payload.zip") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
    if digest.hexdigest() != hashlib.sha256(payload.read_bytes()).hexdigest():
        raise RuntimeError("reserved Kaggle payload does not match this trial")
    submitted = run_dir / "kernel-submitted.json"
    if not submitted.is_file():
        listed = json.loads(runner._command("kernels", "list", "--mine", "--format", "json", timeout=60).stdout)
        if not isinstance(listed, list):
            raise RuntimeError("Kaggle kernel inventory has an unexpected format")
        remote_exists = any(item.get("ref") == kernel_reference for item in listed if isinstance(item, dict))
        if not remote_exists:
            try:
                status = runner._command("datasets", "status", dataset_reference, timeout=60).stdout.strip().lower()
            except KaggleBatchError as error:
                if not upload_missing:
                    raise RuntimeError(f"reserved Kaggle dataset is not accessible yet: {dataset_reference}") from error
                inventory = json.loads(runner._command("datasets", "list", "--mine", "--file-type", "all",
                                                       "--format", "json", timeout=60).stdout)
                if not isinstance(inventory, list) or any(not isinstance(item, dict) for item in inventory):
                    raise RuntimeError("Kaggle dataset inventory has an unexpected format") from error
                if not any(item.get("ref") == dataset_reference for item in inventory):
                    runner._command("datasets", "create", "-p", str(run_dir / "dataset"), timeout=900)
                runner._wait_for_dataset(dataset_reference)
                status = "ready"
            if status != "ready":
                raise RuntimeError(f"reserved Kaggle dataset is {status}, not ready")
            runner._command("kernels", "push", "-p", str(run_dir / "kernel"), timeout=900)
        submitted.write_text(json.dumps({"kernel_reference": kernel_reference}), encoding="utf-8")
    return runner._collect_result("sam3-benchmark", run_id, kernel_reference, run_dir / "output")


def execute(dataset_root: Path, output: Path, settings: Settings) -> dict:
    if not settings.kaggle_api_token.get_secret_value() or not settings.kaggle_sam3_checkpoint_dataset:
        raise ValueError("Kaggle token and private SAM3 checkpoint dataset are required")
    package = output / "payload"
    metadata = prepare(dataset_root, package)
    payload = package / "payload.zip"
    identity = hashlib.sha256(payload.read_bytes() + NOTEBOOK.read_bytes()).hexdigest()
    run_id = str(uuid5(NAMESPACE_URL, f"sam3-cadence-trial:{identity}"))
    run_dir = output / "runs" / run_id
    marker = run_dir / "kernel-submission-started.json"
    global_marker = output / "submission.json"
    if global_marker.exists():
        previous = json.loads(global_marker.read_text(encoding="utf-8"))
        if previous.get("payload_sha256") != identity or previous.get("run_id") != run_id:
            raise RuntimeError("this trial has already reserved its single Kaggle kernel")
        if not marker.exists():
            raise RuntimeError("Kaggle submission state is incomplete; refusing a second kernel")
    runner = KaggleBatchRunner(
        username=settings.kaggle_username,
        work_root=settings.video_work_root,
        machine_shape=settings.kaggle_machine_shape,
        timeout_s=5700,
        poll_seconds=settings.kaggle_poll_seconds,
        sam3_checkpoint_dataset=settings.kaggle_sam3_checkpoint_dataset,
        api_token=settings.kaggle_api_token.get_secret_value(),
        command=str(Path(sys.executable).parent / ("kaggle.exe" if os.name == "nt" else "kaggle")),
    )
    if not marker.exists():
        runner._command("kernels", "list", "--mine", "--format", "json", timeout=60)
        run_dir.mkdir(parents=True, exist_ok=True)
        global_marker.write_text(json.dumps({"run_id": run_id, "payload_sha256": identity}), encoding="utf-8")
        marker.write_text(json.dumps({"run_id": run_id, "payload_sha256": identity}), encoding="utf-8")
        result = runner.run(
            stage="sam3-benchmark",
            job_id=run_id,
            files={"payload.zip": payload},
            code_file=NOTEBOOK,
            allow_retry_on_terminal_failure=False,
        )
    else:
        result = _resume_reserved_run(runner, run_id, payload, NOTEBOOK)
    (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    report = evaluate(result, package, dataset_root)
    (run_dir / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for sequence, segment_report in report["segments"].items():
        save(segment_report, run_dir / sequence)
    return {
        "run_id": run_id,
        "payload": metadata,
        "statuses": [row.get("status") for row in result.get("configurations", [])],
        "comparison": str(run_dir / "comparison.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT.parent / "AutoLabelDatasets")
    parser.add_argument("--output", type=Path, default=ROOT / "state/sam3-cadence-trial")
    parser.add_argument("--offline-only", action="store_true")
    parser.add_argument("--native-fps", action="store_true", help="Controlled 5/10-FPS trial over frozen phase0 spans")
    parser.add_argument("--protocol", type=Path, default=ROOT / "eval/phase0-protocol.json")
    args = parser.parse_args()
    if args.native_fps:
        from scripts.native_fps_trial import run_native

        output = args.output if args.output != ROOT / "state/sam3-cadence-trial" else ROOT / "state/evaluation/phase08-native-fps"
        print(json.dumps(run_native(args.dataset_root, output, args.protocol, offline=args.offline_only), indent=2))
        return
    if args.offline_only:
        print(json.dumps(prepare(args.dataset_root, args.output / "payload"), indent=2))
    else:
        print(json.dumps(execute(args.dataset_root, args.output, Settings()), indent=2))


if __name__ == "__main__":
    main()
