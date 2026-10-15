"""Portable comparison exports. No CUDA dependency; absent telemetry stays unavailable."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import threading
import time
from pathlib import Path

from samlocal.metrics import stats


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_identity(frames) -> dict:
    """Hash ordered image bytes + source indices, independently of local file paths."""
    manifest = []
    digest = hashlib.sha256()
    for frame in frames:
        content = file_sha256(frame.path)
        record = {"frame": frame.index, "source_frame_index": frame.source_index,
                  "sha256": content, "width": frame.width, "height": frame.height}
        digest.update(f"{frame.index}:{frame.source_index}:{content}\n".encode())
        manifest.append(record)
    return {"algorithm": "sha256-ordered-frame-source-index-v1", "sha256": digest.hexdigest(),
            "frames": manifest}


class ResourceSampler:
    """Sample benchmark-process RSS and device-wide GPU counters during one repeat.

    nvidia-smi GPU index 0 corresponds to the current single-GPU backend. On multi-GPU
    machines CUDA_VISIBLE_DEVICES remapping must be resolved before comparing counters.
    Polling is approximate, not a substitute for CUDA allocator peaks or a profiler.
    """
    FIELDS = ("gpu_utilization_pct", "gpu_memory_used_mib", "gpu_power_w", "gpu_temperature_c")

    def __init__(self, interval: float = 1.0, gpu_index: str | int | None = 0):
        self.interval = interval
        self.gpu_index = gpu_index
        self.samples: list[dict] = []
        self.errors: dict[str, str] = {}
        self._stop = threading.Event()
        self._thread = None
        self._process = None
        self._gpu_available = gpu_index is not None
        if gpu_index is None:
            self.errors["gpu"] = "CUDA device mapping unavailable; GPU sampling disabled to avoid measuring a different device"

    def start(self):
        if self.interval <= 0:
            return self
        try:
            import psutil
            self._process = psutil.Process()
        except (ImportError, OSError) as exc:
            self.errors["process_rss"] = str(exc)
        self._started = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            row = {"elapsed_seconds": time.perf_counter() - self._started}
            if self._process is not None:
                try:
                    row["process_rss_gib"] = self._process.memory_info().rss / 1024**3
                except Exception as exc:
                    self.errors["process_rss"] = str(exc)
            if self._gpu_available:
                try:
                    result = subprocess.run(
                        ["nvidia-smi", f"--id={self.gpu_index}",
                         "--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu",
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=2, check=True)
                    values = result.stdout.strip().splitlines()[0].split(",")
                    for key, value in zip(self.FIELDS, values):
                        try:
                            number = float(value.strip())
                            row[key] = number if math.isfinite(number) else None
                        except ValueError:
                            row[key] = None
                except (OSError, subprocess.SubprocessError, IndexError) as exc:
                    self.errors["gpu"] = str(exc)
                    self._gpu_available = False
            row["sample_finished_seconds"] = time.perf_counter() - self._started
            # Discard a poll that completed after repeat timing ended.
            if not self._stop.is_set():
                self.samples.append(row)
            self._stop.wait(self.interval)

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        fields = ("process_rss_gib", *self.FIELDS)
        return {"enabled": self.interval > 0, "interval_seconds": self.interval,
                "scope": "process RSS; GPU device-wide, all processes; sampled, not exact peaks",
                "gpu_index": self.gpu_index, "errors": self.errors,
                "samples": self.samples,
                "metrics": {key: stats([r[key] for r in self.samples if r.get(key) is not None])
                            for key in fields}}


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


INVALID_SCORE_FIELDS = ["repeat", "window", "frame", "source_frame_index", "label", "local_id", "score", "bbox"]


def write_invalid_scores(path: Path, repeat: int, records: list[dict], frames=None) -> None:
    """Detections dropped because SAM3 gave no real score (-10000); kept here so none vanish silently."""
    source = {f.index: f.source_index for f in frames} if frames else {}
    write_csv(path, [{"repeat": repeat, "source_frame_index": source.get(r["frame"]), **r} for r in records],
              INVALID_SCORE_FIELDS)


def export_details(run_dir: Path, report: dict, result=None, frames=None) -> None:
    """Write plot-friendly rows; frame outputs describe the last completed repeat only."""
    windows, attempts, resources, classes = [], [], [], []
    for repeat in report["repeats"]:
        number = repeat["repeat"]
        for window in repeat.get("windows_detail", []):
            windows.append({"repeat": number, **window})
        for attempt in repeat.get("window_attempt_records", []):
            attempts.append({"repeat": number, **attempt})
        for sample in repeat.get("resources", {}).get("samples", []):
            resources.append({"repeat": number, **sample})
        for label, data in repeat.get("output_by_label", {}).items():
            classes.append({"repeat": number, "label": label, **data,
                            "prompt_total_seconds": repeat.get("prompt_total_seconds", {}).get(label),
                            "confidence_mean": data.get("confidence", {}).get("mean"),
                            "track_length_mean_frames": data.get("track_length_frames", {}).get("mean")})
    write_csv(run_dir / "windows.csv", windows,
              ["repeat", "index", "start", "end", "frames", "unique_frames", "seconds", "fps",
               "staging_seconds", "backend_seconds", "stitching_seconds", "prompt_seconds",
               "peak_allocated_gib", "peak_reserved_gib"])
    write_csv(run_dir / "attempts.csv", attempts,
              ["repeat", "index", "start", "end", "frames", "success", "error", "unique_frames",
               "seconds", "staging_seconds", "backend_seconds", "stitching_seconds",
               "peak_allocated_gib", "peak_reserved_gib"])
    write_csv(run_dir / "resources.csv", resources,
              ["repeat", "elapsed_seconds", "sample_finished_seconds", "process_rss_gib",
               *ResourceSampler.FIELDS])
    write_csv(run_dir / "classes.csv", classes,
              ["repeat", "label", "detections", "frames_with_detections", "zero_detection_frames",
               "tracks", "detections_per_frame", "confidence_mean", "track_length_mean_frames",
               "prompt_total_seconds"])
    if result is not None and frames is not None:
        rows = []
        for frame in frames:
            dets = result.frames.get(frame.index, [])
            for label in report["prompts"]:
                selected = [d for d in dets if d.label == label]
                rows.append({"repeat": report.get("last_completed_repeat"), "frame": frame.index,
                             "source_frame_index": frame.source_index, "label": label,
                             "detections": len(selected),
                             "confidence_mean": stats([d.score for d in selected])["mean"],
                             "track_ids": sorted({d.track_id for d in selected})})
        write_csv(run_dir / "frames.csv", rows,
                  ["repeat", "frame", "source_frame_index", "label", "detections",
                   "confidence_mean", "track_ids"])
