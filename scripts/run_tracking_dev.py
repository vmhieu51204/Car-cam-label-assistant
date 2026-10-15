"""Run the approved all-frame DEV payloads; dry-run never calls a provider."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import time
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from kaggle.sam3_benchmark.main import (
    checkpoint_identity,
    model_identity,
    validate_checkpoint,
    validate_config,
    windows,
)
from scripts.native_fps_trial import NativeTrialRunner
from scripts.sam3_cadence_trial import _resume_reserved_run
from src.config import Settings
from src.services.evaluation.phase0 import file_hash, load_frozen_run
from src.services.evaluation.tracking_payload import canonical_bytes, save_equal_or_new
from src.services.video.kaggle_batch import KaggleBatchError

ROOT = Path(__file__).resolve().parents[1]
SEQUENCES = ("0001", "0019", "0011")
NOTEBOOK = Path("kaggle/notebooks/autolabel_sam3_benchmark.ipynb")
EXPECTED_MODEL = {"sam3_commit": "2345a4ad109ac29c569da749c91d84f10dc08c40",
                  "checkpoint_sha256": "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"}


def save_state(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(canonical_bytes(value))
    temporary.replace(path)


@contextmanager
def live_lock(output: Path) -> Iterator[None]:
    """OS lock survives neither crashes nor process exit; persistent file is harmless."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / "live.lock").open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError("another DEV execution owns the live lock") from error
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            posix_lock: Any = import_module("fcntl")
            try:
                posix_lock.flock(stream, posix_lock.LOCK_EX | posix_lock.LOCK_NB)
            except OSError as error:
                raise RuntimeError("another DEV execution owns the live lock") from error
            try:
                yield
            finally:
                posix_lock.flock(stream, posix_lock.LOCK_UN)


def verify_payload(root: Path, protocol: dict[str, Any], row: dict[str, Any]) -> tuple[Path, dict, dict]:
    sequence = row["sequence"]
    if sequence not in SEQUENCES or sequence not in protocol["splits"]["dev"] or sequence in protocol["splits"]["held_out"]:
        raise ValueError("only approved DEV sequences may run")
    count = row["frames"]
    if row.get("split") != "dev" or row.get("cadence") != "all_frames" or row.get("source_frames") != list(range(count)):
        raise ValueError("one complete all-frame sequence per approved kernel is required")
    archive = root / row["prepared_directory"] / row["archive_name"]
    try:
        digest = file_hash(archive)
    except PermissionError:
        # Earlier sandbox identities can own unreadable ZIPs after switching to
        # the desktop user. Only accept a rebuilt copy with exactly the frozen hash.
        archive = root / "state/evaluation/phase1-tracking/live-dev/payloads" / sequence / row["archive_name"]
        digest = file_hash(archive)
    if digest != row["archive_sha256"]:
        raise ValueError("archive hash mismatch")
    with zipfile.ZipFile(archive) as stream:
        names = stream.namelist()
        expected = {"frames.json", "benchmark-config.json", "seed.json", *(f"frames/frame_{i:06d}.jpg" for i in range(count))}
        if len(names) != len(set(names)) or set(names) != expected:
            raise ValueError("payload contains GT, unexpected files or missing frames")
        mapping = json.loads(stream.read("frames.json"))
        config = json.loads(stream.read("benchmark-config.json"))
        if stream.read("frames.json") != canonical_bytes(mapping) or stream.read("benchmark-config.json") != canonical_bytes(config):
            raise ValueError("noncanonical payload metadata")
        for name, key in (("frames.json", "frames_sha256"), ("benchmark-config.json", "config_sha256")):
            if hashlib.sha256(stream.read(name)).hexdigest() != row[key]:
                raise ValueError("payload metadata hash mismatch")
        frames = mapping["frames"]
        if len(frames) != count or len(row["source_hashes"]) != count or mapping["optical_source_indices"] != list(range(count)):
            raise ValueError("payload mapping must cover every source frame")
        training = root / protocol["dataset_root"] / "training"
        for i, frame in enumerate(frames):
            if frame["source_frame_index"] != i or frame["original_source_index"] != i or frame["relative_path"] != f"frames/frame_{i:06d}.jpg":
                raise ValueError("payload frame mapping mismatch")
            if hashlib.sha256(stream.read(frame["relative_path"])).hexdigest() != frame["sha256"]:
                raise ValueError("packaged image hash mismatch")
            if file_hash(training / "image_02" / sequence / f"{i:06d}.png") != row["source_hashes"][i]:
                raise ValueError("source image hash mismatch")
        if json.loads(stream.read("seed.json")) != {"seeds": []}:
            raise ValueError("DEV text prompts require empty seeds")
    validate_config(config)
    configurations = config["configurations"]
    if len(configurations) != 1:
        raise ValueError("exactly one approved configuration is required")
    c = configurations[0]
    expected_config = {"id": "sequence-text", "model_variant": "sam3", "selection": "dense", "sampling_strategy": "all_frames",
                       "mode": "text_prompt", "seed_type": "none", "reprompt": "none", "text_prompts": ["car", "person"],
                       "window_size": 50, "overlap": 5, "preserve_overlap": True}
    if any(c.get(k) != v for k, v in expected_config.items()) or c.get("tracking_policy", "independent_windows") != "independent_windows":
        raise ValueError("DEV configuration differs from approved tracking settings")
    if config["source_frame_limit"] != count or mapping.get("source_fps") != 10 or mapping.get("target_fps") != 10:
        raise ValueError("DEV cadence or frame count mismatch")
    return archive, config, mapping


def prepare(root: Path, protocol_path: Path) -> tuple[dict, list[dict]]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    rows = protocol["gpu_plan"]
    if len(rows) != 3 or [r["sequence"] for r in rows] != list(SEQUENCES):
        raise ValueError("approved plan requires exactly three DEV kernels in priority order")
    for row in rows:
        verify_payload(root, protocol, row)
    return protocol, rows


def csv_names(value: str) -> list[str]:
    rows = list(csv.reader(io.StringIO(value)))
    if len(rows) < 2 or rows[0][0].lower() not in {"name", "file_name", "filename"}:
        raise KaggleBatchError("Kaggle file inventory is missing or unrecognized")
    return [row[0] for row in rows[1:] if row]


def remote_names(runner: NativeTrialRunner, reference: str) -> list[str]:
    names: list[str] = []
    token = ""
    seen: set[str] = set()
    for _ in range(20):
        args: tuple[str, ...] = ("datasets", "files", reference, "--csv", "--page-size", "200")
        if token:
            args += ("--page-token", token)
        value = runner._command(*args, timeout=60).stdout
        lines = value.splitlines()
        prefix = "Next Page Token = "
        token = lines[0][len(prefix):] if lines and lines[0].startswith(prefix) else ""
        names.extend(csv_names("\n".join(lines[1:] if token else lines)))
        if not token:
            if len(names) != len(set(names)):
                raise KaggleBatchError("duplicate remote input files")
            return names
        if token in seen:
            raise KaggleBatchError("repeated remote file page token")
        seen.add(token)
    raise KaggleBatchError("remote file inventory exceeds paging limit")


class TrackingDevRunner(NativeTrialRunner):
    """Verify flattened remote input before pushing the reserved kernel."""

    def _wait_for_dataset(self, reference: str) -> None:
        super()._wait_for_dataset(reference)
        names = remote_names(self, reference)
        run_dir = self.work_root / self.run_id / "kaggle" / reference.split("/")[1]
        with zipfile.ZipFile(run_dir / "dataset/payload.zip") as wrapper:
            with zipfile.ZipFile(io.BytesIO(wrapper.read("payload.zip"))) as payload:
                expected = set(payload.namelist())
        normalized = {n.removeprefix("payload/") for n in names}
        if normalized != expected:
            raise KaggleBatchError("remote input files disagree with validated payload; kernel not submitted")

    run_id: str = ""

    def _command(self, *args: str, timeout: float) -> subprocess.CompletedProcess[str]:
        if args[:2] == ("kernels", "push"):
            directory = Path(args[args.index("-p") + 1]).resolve().parent
            marker = directory / "kernel-submission-started.json"
            with marker.open("xb") as stream:
                stream.write(canonical_bytes({"started_at": datetime.now(UTC).isoformat()}))
        return super()._command(*args, timeout=timeout)


def make_runner(settings: Settings, work: Path) -> TrackingDevRunner:
    return TrackingDevRunner(username=settings.kaggle_username, work_root=work,
                             machine_shape=settings.kaggle_machine_shape, timeout_s=21600,
                             poll_seconds=settings.kaggle_poll_seconds,
                             sam3_checkpoint_dataset=settings.kaggle_sam3_checkpoint_dataset,
                             api_token=settings.kaggle_api_token.get_secret_value(),
                             command=str(Path(sys.executable).parent / ("kaggle.exe" if os.name == "nt" else "kaggle")))


def preflight(runner: TrackingDevRunner) -> dict:
    if not runner.api_token or not runner.sam3_checkpoint_dataset:
        raise ValueError("Kaggle token and private SAM3 checkpoint are required")
    inventory = json.loads(runner._command("kernels", "list", "--mine", "--format", "json", timeout=60).stdout)
    if not isinstance(inventory, list):
        raise KaggleBatchError("Kaggle account inventory is invalid")
    state = runner._command("datasets", "status", runner.sam3_checkpoint_dataset, timeout=60).stdout.strip().lower()
    names = remote_names(runner, runner.sam3_checkpoint_dataset)
    if state != "ready" or not any(n.endswith((".pt", ".pth")) for n in names):
        raise KaggleBatchError("SAM3 checkpoint input is not verified ready")
    return {"verified_at": datetime.now(UTC).isoformat(), "account": runner.username,
            "checkpoint": runner.sam3_checkpoint_dataset, "checkpoint_status": state, "checkpoint_files": names}


def validate_result(result: dict, config: dict, mapping: dict) -> None:
    if result.get("schema") != "autolabel-sam3-benchmark/v1" or result.get("completed") is not True:
        raise ValueError("incomplete benchmark result")
    items = result.get("configurations", [])
    if len(items) != 1 or items[0].get("status") != "completed" or items[0].get("config") != config["configurations"][0]:
        raise ValueError("DEV result configuration is incomplete or mismatched")
    item = items[0]
    count = len(mapping["frames"])
    if item.get("evaluated_source_frames") != list(range(count)):
        raise ValueError("DEV result did not evaluate every source frame")
    c = config["configurations"][0]
    batches = windows(mapping["frames"], c["window_size"], c["overlap"])
    checkpoints = item.get("window_checkpoints", [])
    if [p["window_id"] for p in checkpoints] != list(range(len(batches))):
        raise ValueError("DEV result is missing window checkpoints, including empty windows")
    identity = checkpoint_identity(mapping["frames"], c, config.get("code_sha256", ""))
    model = model_identity(result["model_runtime"])
    if model != EXPECTED_MODEL:
        raise ValueError("SAM3 runtime commit/checkpoint differs from the audited baseline")
    predictions = []
    accepted: dict[int, list[dict]] = {}
    for i, (_, batch) in enumerate(batches):
        validate_checkpoint(checkpoints[i], identity=identity, window_id=i,
                            frame_ids=[f["source_frame_index"] for f in batch], model=model)
        predictions.extend(checkpoints[i]["predictions"])
        for frame in checkpoints[i]["frame_ids"]:
            if frame not in accepted:
                accepted[frame] = [p for p in checkpoints[i]["predictions"] if p["frame_index"] == frame]
    if predictions != item.get("window_predictions"):
        raise ValueError("overlap predictions disagree with validated checkpoints")
    for prediction in item["predictions"] + predictions:
        if type(prediction["frame_index"]) is not int or not 0 <= prediction["frame_index"] < count or prediction["label"] not in {"car", "person"}:
            raise ValueError("invalid prediction source frame or class")
    if item.get("window_frame_count") != sum(len(batch) for _, batch in batches):
        raise ValueError("runtime window-frame denominator mismatch")
    def geometry(p: dict) -> tuple:
        return p["frame_index"], p["label"], p["track_id"], tuple(p["bbox"])

    if [geometry(p) for p in item["predictions"]] != [geometry(p) for f in sorted(accepted) for p in accepted[f]]:
        raise ValueError("direct predictions disagree with first accepted overlap observations")


def bounded_payload(archive: Path, config: dict, directory: Path, seconds: float) -> Path:
    """Only shorten execution deadlines; preserve all model parameters and image bytes."""
    bounded = copy.deepcopy(config)
    bounded["max_total_seconds"] = math.floor(seconds)
    bounded["max_config_seconds"] = math.floor(seconds) - 180
    validate_config(bounded)
    path = directory / "execution-payload.zip"
    buffer = io.BytesIO()
    with zipfile.ZipFile(archive) as source, zipfile.ZipFile(buffer, "w") as target:
        for info in source.infolist():
            target.writestr(info, canonical_bytes(bounded) if info.filename == "benchmark-config.json" else source.read(info.filename))
    save_equal_or_new(path, buffer.getvalue())
    save_equal_or_new(directory / "execution-limits.json", canonical_bytes({
        "prepared_archive_sha256": file_hash(archive), "execution_archive_sha256": file_hash(path),
        "prepared_config_sha256": hashlib.sha256(canonical_bytes(config)).hexdigest(),
        "execution_config_sha256": hashlib.sha256(canonical_bytes(bounded)).hexdigest(),
        "max_total_seconds": bounded["max_total_seconds"], "max_config_seconds": bounded["max_config_seconds"],
        "limitation": "Notebook checks deadlines between windows; one running window may overrun. Never start another kernel after budget overrun."}))
    return path


def execute_one(runner: TrackingDevRunner, archive: Path, notebook: Path, directory: Path) -> dict:
    binding = {"archive_sha256": file_hash(archive), "notebook_sha256": file_hash(notebook),
               "account": runner.username, "checkpoint": runner.sam3_checkpoint_dataset, "machine_shape": runner.machine_shape}
    run_id = str(uuid5(NAMESPACE_URL, canonical_bytes(binding).decode()))
    runner.run_id = run_id
    save_equal_or_new(directory / "reservation.json", canonical_bytes({"run_id": run_id, **binding}))
    runs = list((runner.work_root / run_id / "kaggle").glob("autolabel-sam3-benchmark-*"))
    if runs:
        if len(runs) != 1:
            raise ValueError("ambiguous reserved run; refusing a new kernel")
        # Verify input again even when resuming after upload/push receipt loss.
        metadata = json.loads((runs[0] / "dataset/dataset-metadata.json").read_text(encoding="utf-8"))
        runner._wait_for_dataset(metadata["id"])
        kernel = json.loads((runs[0] / "kernel/kernel-metadata.json").read_text(encoding="utf-8"))
        if (kernel.get("id") != f"{runner.username}/{runs[0].name}-run"
                or kernel.get("dataset_sources") != [metadata["id"], runner.sam3_checkpoint_dataset]
                or file_hash(runs[0] / "kernel/main.ipynb") != file_hash(notebook)):
            raise ValueError("reserved metadata or notebook mismatch")
        with zipfile.ZipFile(runs[0] / "dataset/payload.zip") as wrapper:
            if wrapper.namelist() != ["payload.zip"] or hashlib.sha256(wrapper.read("payload.zip")).hexdigest() != file_hash(archive):
                raise ValueError("reserved input mismatch")
        if (runs[0] / "kernel-submission-started.json").exists() or (runs[0] / "kernel-submitted.json").exists():
            # Exact identity lookup/collection only. A lost response must never
            # become a new Kaggle version based on eventual/paginated inventory.
            return runner._collect_result("sam3-benchmark", run_id, kernel["id"], runs[0] / "output")
        return _resume_reserved_run(runner, run_id, archive, notebook, upload_missing=False)
    return runner.run(stage="sam3-benchmark", job_id=run_id, files={"payload.zip": archive}, code_file=notebook,
                      allow_retry_on_terminal_failure=False)


def register(root: Path, protocol: dict, rows: list[dict], output: Path) -> Path:
    extended = copy.deepcopy(protocol)
    extended["revision"] = protocol["revision"] + 1
    extended["parent_protocol_sha256"] = file_hash(output / "source-protocol.json")
    registered = []
    for row in rows:
        directory = output / row["sequence"]
        path = directory / "result.json"
        if not path.is_file():
            continue
        _, config, mapping = verify_payload(root, protocol, row)
        result = json.loads(path.read_text(encoding="utf-8"))
        validate_result(result, config, mapping)
        mapping = {**mapping, "frames": [{**f, "frame_index": i, "source_frame_index": f["original_source_index"]}
                                        for i, f in enumerate(mapping["frames"])],
                   "fps_method": "KITTI nominal cadence, not measured timestamps"}
        save_equal_or_new(directory / "mapping.json", canonical_bytes(mapping))
        labels = root / protocol["dataset_root"] / "training/label_02" / (row["sequence"] + ".txt")
        if file_hash(labels) != protocol["input_hashes"][row["sequence"]]["labels"]:
            raise ValueError("GT input hash mismatch")
        run = {"sequence": row["sequence"], "native_frame_count": row["frames"],
               "result": path.relative_to(root).as_posix(), "mapping": (directory / "mapping.json").relative_to(root).as_posix(),
               "labels": os.path.relpath(labels, root), "class_map": protocol["class_map"], "config_id": "sequence-text",
               "source_fps": 10, "fps_provenance": mapping["fps_method"], "cadence": "all_frames",
               "sha256": {"result": file_hash(path), "mapping": file_hash(directory / "mapping.json"), "labels": file_hash(labels)}}
        load_frozen_run(run, root)
        extended["runs"].append(run)
        registered.append(row["sequence"])
    extended["prediction_sequences"] = [r["sequence"] for r in extended["runs"]]
    extended["observation_cohorts"] = {"legacy_sampled": ["0013", "0020"], "all_frames": registered}
    path = output / "dev-protocol.json"
    save_state(path, extended)
    return path


def run(root: Path, protocol_path: Path, *, live: bool = False) -> dict:
    protocol, rows = prepare(root, protocol_path)
    budget = sum(r["rough_budget_seconds"] for r in rows)
    summary: dict[str, Any] = {"sequences": list(SEQUENCES), "frames": sum(r["frames"] for r in rows),
                               "rough_budget_minutes": budget / 60, "live": live, "submitted": False}
    if not live:
        return summary
    with live_lock(root / "state/evaluation/phase1-tracking/live-dev"):
        return run_live(root, protocol_path, protocol, rows, summary)


def run_live(root: Path, protocol_path: Path, protocol: dict, rows: list[dict], summary: dict) -> dict:
    budget = sum(r["rough_budget_seconds"] for r in rows)
    output = root / "state/evaluation/phase1-tracking/live-dev"
    save_equal_or_new(output / "source-protocol.json", protocol_path.read_bytes())
    settings = Settings()
    runner = make_runner(settings, output / "runs")
    spent = 0.0
    try:
        save_state(output / "preflight.json", preflight(runner))
        measured_rate = 0.0
        measured_overhead = 180.0
        for index, row in enumerate(rows):
            directory = output / row["sequence"]
            archive, config, mapping = verify_payload(root, protocol, row)
            receipt = directory / "execution.json"
            if (directory / "result.json").exists():
                result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
                validate_result(result, config, mapping)
                spent += json.loads(receipt.read_text(encoding="utf-8"))["budget_accounted_seconds"]
                measured_rate = max(measured_rate, result["configurations"][0]["runtime"]["total_seconds"] / result["configurations"][0]["window_frame_count"])
                measured_overhead = max(measured_overhead, json.loads(receipt.read_text(encoding="utf-8"))["wall_seconds"] - result["configurations"][0]["runtime"]["total_seconds"])
                continue
            remaining_estimate = 0.0
            for future in rows[index:]:
                _, future_config, future_mapping = verify_payload(root, protocol, future)
                c = future_config["configurations"][0]
                denominator = sum(len(b) for _, b in windows(future_mapping["frames"], c["window_size"], c["overlap"]))
                remaining_estimate += max(future["rough_budget_seconds"], denominator * measured_rate * 1.25 + measured_overhead)
            summary["reestimated_remaining_minutes"] = remaining_estimate / 60
            if spent + remaining_estimate > budget + 0.01:
                summary.update(status="budget_reestimate_exceeded", completed_sequences=[r["sequence"] for r in rows[:index]])
                break
            started = time.monotonic()
            previous_receipt = json.loads(receipt.read_text(encoding="utf-8")) if receipt.exists() else {}
            started_at = previous_receipt.get("started_at", datetime.now(UTC).isoformat())
            save_state(receipt, {"status": "started", "started_at": started_at, "rough_budget_seconds": row["rough_budget_seconds"]})
            print(json.dumps({"sequence": row["sequence"], "status": "running", "rough_minutes": row["rough_budget_seconds"] / 60}), flush=True)
            execution_archive = bounded_payload(archive, config, directory, row["rough_budget_seconds"])
            result = execute_one(runner, execution_archive, root / NOTEBOOK, directory)
            # Keep raw output even when validation fails, without admitting it to the registry.
            save_equal_or_new(directory / "raw-result.json", canonical_bytes(result))
            validate_result(result, config, mapping)
            elapsed = max(time.monotonic() - started, (datetime.now(UTC) - datetime.fromisoformat(started_at)).total_seconds())
            inference = result["configurations"][0]["runtime"]["total_seconds"]
            accounted = max(elapsed, inference + result["model_load_seconds"])
            if not math.isfinite(accounted) or accounted < 0:
                raise ValueError("invalid runtime measurement")
            save_state(receipt, {"status": "completed", "started_at": started_at, "wall_seconds": elapsed, "inference_seconds": inference,
                                 "model_load_seconds": result["model_load_seconds"], "budget_accounted_seconds": accounted,
                                 "completed_at": datetime.now(UTC).isoformat()})
            save_equal_or_new(directory / "result.json", canonical_bytes(result))
            spent += accounted
            measured_rate = max(measured_rate, inference / result["configurations"][0]["window_frame_count"])
            measured_overhead = max(measured_overhead, elapsed - inference)
            print(json.dumps({"sequence": row["sequence"], "status": "completed", "accounted_minutes": accounted / 60}), flush=True)
        else:
            summary["status"] = "completed"
        summary.update(budget_accounted_minutes=spent / 60)
    except Exception as error:
        summary.update(status="incomplete", error=f"{type(error).__name__}: {str(error).replace(runner.api_token, '[redacted]')}")
    summary["budget_accounted_minutes"] = spent / 60
    try:
        summary["protocol"] = str(register(root, protocol, rows, output))
    except Exception as error:
        summary.update(status="incomplete", registration_error=f"{type(error).__name__}: {error}")
    summary["reserved_sequences"] = [s for s in SEQUENCES if (output / s / "reservation.json").exists()]
    summary["submitted"] = any((output / "runs").glob("*/kaggle/*/kernel-submitted.json"))
    save_state(output / "execution.json", summary)
    return summary


def diagnose_available(root: Path, protocol_path: Path) -> dict:
    """Report pooled DEV and separate cadence cohorts without changing evaluators."""
    from src.services.evaluation.tracking_diagnosis import diagnose

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    output = protocol_path.parent / "reports"
    results = {"all_dev": diagnose(protocol_path, root, output / "all-dev")}
    for cohort, sequences in protocol["observation_cohorts"].items():
        selected = [r for r in protocol["runs"] if r["sequence"] in sequences]
        if not selected:
            continue
        child = {**protocol, "runs": selected, "prediction_sequences": [r["sequence"] for r in selected]}
        path = output / cohort / "protocol.json"
        save_state(path, child)
        results[cohort] = diagnose(path, root, path.parent)
    summary: dict[str, Any] = {name: {"event_count": r["event_count"], "population_support": r["population_support"],
                      "fragmentation_diagnosis": {v: data["fragmentation_diagnosis"] for v, data in r["pooled"].items()}}
               for name, r in results.items()}
    summary["note"] = "Cadence/population differ; differences are not algorithm improvements. HELD-OUT not evaluated."
    save_state(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "eval/phase1-tracking-protocol.json")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    result = run(ROOT, args.protocol.resolve(), live=args.live)
    if args.live and result.get("protocol"):
        diagnose_available(ROOT, Path(result["protocol"]))
    print(json.dumps(result, ensure_ascii=False))
    if result.get("status") in {"incomplete", "budget_reestimate_exceeded"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
