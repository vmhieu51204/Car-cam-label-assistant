"""Run SAM 3.1 over a whole payload with adaptive windows, OOM backoff and track stitching."""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from samlocal.backend import VideoBackend, WindowOOMError
from samlocal.metrics import stats
from samlocal.payload import Frame
from samlocal.scheduler import AdaptiveWindowScheduler
from samlocal.staging import remove_window, stage_window
from samlocal.stitch import Detection, TrackStitcher


def valid_score(score: float) -> bool:
    """SAM3 reports a tracker-only object (no detector score) as -10000; probabilities are in [0, 1]."""
    return math.isfinite(score) and 0.0 <= score <= 1.0


@dataclass
class WindowSettings:
    size: int = 48
    min_size: int = 8
    max_size: int = 160
    overlap: int = 4
    grow_below: float = 0.60
    stitch_iou: float = 0.5


@dataclass
class SequenceResult:
    frames: dict[int, list[Detection]] = field(default_factory=lambda: defaultdict(list))
    windows: list[dict[str, Any]] = field(default_factory=list)
    window_attempts: list[dict[str, Any]] = field(default_factory=list)
    prompt_seconds: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    scheduler_events: list[dict] = field(default_factory=list)
    total_seconds: float = 0.0
    final_window_size: int = 0
    tracks_by_label: dict[str, int] = field(default_factory=dict)
    num_frames: int = 0
    staging_seconds: float = 0.0
    backend_seconds: float = 0.0
    stitching_seconds: float = 0.0
    emitted_frames: int = 0
    invalid_scores: list[dict[str, Any]] = field(default_factory=list)

    def progress(self) -> dict[str, Any]:
        done = min(self.emitted_frames, self.num_frames)
        fraction = done / self.num_frames if self.num_frames else 0.0
        eta = self.total_seconds * (1 - fraction) / fraction if fraction > 0 else None
        return {"frames_done": done, "frames_total": self.num_frames, "percent": round(100 * fraction, 2),
                "elapsed_seconds": self.total_seconds, "eta_seconds": eta,
                "windows_done": len(self.windows), "window_size": self.final_window_size}

    def summary(self) -> dict[str, Any]:
        detections = sum(len(v) for v in self.frames.values())
        window_seconds = [w["seconds"] for w in self.windows]
        attempts = self.window_attempts or self.windows
        peaks_all = [w.get("peak_reserved_gib") for w in attempts if w.get("peak_reserved_gib") is not None]
        emitted = sum(w.get("unique_frames_emitted", 0) for w in self.windows)
        processed = sum(w.get("frames", 0) for w in self.windows)
        labels = sorted(set(self.prompt_seconds) | {d.label for ds in self.frames.values() for d in ds})
        detection_counts = {label: sum(1 for ds in self.frames.values() for d in ds if d.label == label)
                            for label in labels}
        track_frames: dict[tuple[str, int], set[int]] = defaultdict(set)
        for frame, ds in self.frames.items():
            for det in ds:
                track_frames[(det.label, det.track_id)].add(frame)
        track_lengths: dict[str, list[int]] = defaultdict(list)
        for (label, _), observed in track_frames.items():
            track_lengths[label].append(len(observed))
        label_stats = {}
        output_by_label = {}
        for label in labels:
            counts = [sum(1 for d in self.frames.get(frame, ()) if d.label == label)
                      for frame in range(emitted)]
            scores = [d.score for ds in self.frames.values() for d in ds if d.label == label]
            lengths = track_lengths.get(label, [])
            label_stats[label] = {"detections": detection_counts[label],
                                  "detections_per_frame": stats([float(c) for c in counts]),
                                  "confidence": stats(scores)}
            output_by_label[label] = {
                "detections": detection_counts[label],
                "detections_per_frame": detection_counts[label] / emitted if emitted else None,
                "frames_with_detections": sum(c > 0 for c in counts),
                "zero_detection_frames": sum(c == 0 for c in counts),
                "tracks": self.tracks_by_label.get(label, len(lengths)),
                "confidence": stats(scores),
                "track_length_frames": stats([float(n) for n in lengths]),
            }
        recorded = self.staging_seconds + self.backend_seconds + self.stitching_seconds
        return {
            "num_frames": self.num_frames,
            "total_seconds": self.total_seconds,
            "fps": self.num_frames / self.total_seconds if self.total_seconds else None,
            "unique_fps": emitted / self.total_seconds if self.total_seconds else None,
            "amortized_ms_per_unique_frame": (1000 * self.total_seconds / emitted) if emitted else None,
            "amortized_metric_note": "sequence wall time divided by unique emitted frames; not per-frame latency",
            "processed_window_frames": processed,
            "attempted_window_frames": sum(w.get("frames", 0) for w in attempts),
            "unique_frames_emitted": emitted,
            "overlap_frames": max(0, processed - emitted),
            "overlap_overhead_fraction": (max(0, processed - emitted) / processed) if processed else None,
            "window_attempts": len(attempts),
            "failed_window_attempts": sum(1 for a in attempts if not a.get("success", True)),
            "staging_seconds": self.staging_seconds,
            "backend_seconds": self.backend_seconds,
            "stitching_seconds": self.stitching_seconds,
            "callback_logging_residual_seconds": max(0.0, self.total_seconds - recorded),
            "windows": len(self.windows),
            "oom_backoffs": sum(1 for e in self.scheduler_events if e["type"] == "oom_backoff"),
            "final_window_size": self.final_window_size,
            "peak_reserved_gib": max(peaks_all, default=None),
            "peak_allocated_gib": max((w["peak_allocated_gib"] for w in attempts
                                        if w.get("peak_allocated_gib") is not None), default=None),
            "window_seconds": stats(window_seconds),
            "prompt_seconds": {label: stats(values) for label, values in self.prompt_seconds.items()},
            "prompt_total_seconds": {label: sum(values) for label, values in self.prompt_seconds.items()},
            "detections": detections,
            "invalid_score_detections": len(self.invalid_scores),
            "invalid_score_by_label": {label: sum(1 for r in self.invalid_scores if r["label"] == label)
                                       for label in sorted({r["label"] for r in self.invalid_scores})},
            "detections_per_frame": detections / self.num_frames if self.num_frames else None,
            "tracks_by_label": self.tracks_by_label,
            "detections_by_label": label_stats,
            "track_lengths_by_label": {label: stats([float(n) for n in lengths])
                                       for label, lengths in track_lengths.items()},
            "output_by_label": output_by_label,
            "window_attempt_records": attempts,
        }

    def predictions(self) -> dict[str, Any]:
        return {
            "schema": "local-sam-predictions/v1",
            "frames": {
                str(frame): [{"track_id": f"{d.label}:{d.track_id}", "label": d.label,
                              "bbox": [round(v, 2) for v in d.bbox], "score": round(d.score, 4),
                              **({"mask_rle": d.mask_rle} if d.mask_rle else {})}
                             for d in dets]
                for frame, dets in sorted(self.frames.items())
            },
        }


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    seconds = int(seconds)
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m{seconds % 60:02d}s"


def _drop_invalid_scores(detections: list[Detection], label: str, window_index: int, emitted_until: int,
                         result: SequenceResult) -> list[Detection]:
    """Remove detections without a real probability. Only frames this window will emit are recorded,
    so overlap frames are not logged twice."""
    kept = []
    for det in detections:
        if valid_score(det.score):
            kept.append(det)
        elif det.frame >= emitted_until:
            result.invalid_scores.append({
                "window": window_index, "frame": det.frame, "label": label, "local_id": det.local_id,
                "score": det.score, "bbox": [round(v, 2) for v in det.bbox]})
    return kept


def run_sequence(backend: VideoBackend, frames: list[Frame], prompts: list[str], settings: WindowSettings,
                 *, stage_root: Path, logger: logging.Logger,
                 on_window: Callable[[SequenceResult], None] | None = None) -> SequenceResult:
    """Process ``frames`` window by window. Frames already emitted by an earlier window keep
    the earlier result (it had more temporal context); overlap frames are only used to stitch."""
    scheduler = AdaptiveWindowScheduler(total=len(frames), size=settings.size, min_size=settings.min_size,
                                        max_size=settings.max_size, overlap=settings.overlap,
                                        grow_below=settings.grow_below)
    stitcher = TrackStitcher(iou_threshold=settings.stitch_iou)
    result = SequenceResult(num_frames=len(frames))
    for prompt in prompts:
        result.prompt_seconds[prompt] = []
    emitted_until = 0  # global frames [0, emitted_until) already have final detections
    started = time.perf_counter()
    window_index = 0
    while not scheduler.done:
        start, end = scheduler.current()
        window_frames = frames[start:end]
        stage_started = time.perf_counter()
        frame_dir = stage_window(window_frames, stage_root / f"window-{window_index:04d}")
        stage_seconds = time.perf_counter() - stage_started
        result.staging_seconds += stage_seconds
        backend.reset_peak()
        window_started = time.perf_counter()
        prompt_times: dict[str, float] = {}
        backend_elapsed = 0.0
        failure: Exception | None = None
        try:
            backend_call_started = time.perf_counter()
            per_prompt = backend.run_window(
                frame_dir, len(window_frames), prompts, frame_offset=start,
                on_prompt_done=lambda label, seconds: prompt_times.__setitem__(label, seconds))
            backend_elapsed = time.perf_counter() - backend_call_started
        except Exception as exc:
            failure = exc
            backend_elapsed = time.perf_counter() - window_started
            peak = backend.peak()
            result.backend_seconds += backend_elapsed
        finally:
            cleanup_started = time.perf_counter()
            remove_window(frame_dir)
            cleanup_seconds = time.perf_counter() - cleanup_started
            result.staging_seconds += cleanup_seconds
            stage_seconds += cleanup_seconds
        if failure is not None:
            failed_record = {"index": window_index, "start": start, "end": end,
                             "frames": end - start, "seconds": time.perf_counter() - window_started,
                             "success": False, "error": str(failure),
                             "error_type": type(failure).__name__, "prompt_seconds": prompt_times,
                             "unique_frames_emitted": 0, "unique_frames": 0,
                             "staging_seconds": stage_seconds, "backend_seconds": backend_elapsed,
                             "stitching_seconds": 0.0, **peak}
            result.window_attempts.append(failed_record)
            logger.warning("window %d frames [%d,%d) failed (peak reserved %.2f GiB): %s", window_index, start, end,
                           peak["peak_reserved_gib"], failure)
            # Preserve the terminal OOM event as well as recoverable backoffs.
            try:
                if isinstance(failure, WindowOOMError):
                    scheduler.on_oom()
            finally:
                result.total_seconds = time.perf_counter() - started
                result.scheduler_events = list(scheduler.events)
                result.final_window_size = scheduler.size
                if on_window:
                    on_window(result)
            if not isinstance(failure, WindowOOMError):
                raise failure
            logger.info("retrying from frame %d with window size %d", start, scheduler.size)
            continue
        result.backend_seconds += backend_elapsed
        seconds = time.perf_counter() - window_started
        peak = backend.peak()
        overlap_frames = set(range(start, min(end, emitted_until)))
        stitch_started = time.perf_counter()
        for label, detections in per_prompt.items():
            detections = _drop_invalid_scores(detections, label, window_index, emitted_until, result)
            for det in stitcher.stitch(label, detections, overlap_frames):
                if det.frame >= emitted_until:
                    result.frames[det.frame].append(det)
            result.prompt_seconds[label].append(prompt_times.get(label, 0.0))
        stitch_seconds = time.perf_counter() - stitch_started
        result.stitching_seconds += stitch_seconds
        unique_emitted = max(0, end - max(start, emitted_until))
        emitted_until = max(emitted_until, end)
        result.emitted_frames = emitted_until
        record = {"index": window_index, "start": start, "end": end, "frames": end - start,
                  "seconds": seconds, "fps": (end - start) / seconds if seconds else None,
                  "success": True, "unique_frames_emitted": unique_emitted,
                  "unique_frames": unique_emitted, "staging_seconds": stage_seconds,
                  "backend_seconds": backend_elapsed,
                  "stitching_seconds": stitch_seconds,
                  "prompt_seconds": prompt_times, **peak}
        result.windows.append(record)
        result.window_attempts.append(record)
        result.total_seconds = time.perf_counter() - started
        progress = result.progress()
        logger.info("window %d [%d,%d) %d frames in %.2fs (%.2f fps), peak %.2f GiB (%.0f%% of budget)",
                    window_index, start, end, end - start, seconds, record["fps"] or 0.0,
                    peak["peak_reserved_gib"], 100 * (peak["peak_fraction_of_budget"] or 0))
        logger.info("progress %.1f%% (%d/%d frames), elapsed %s, ETA %s", progress["percent"],
                    progress["frames_done"], progress["frames_total"], format_duration(progress["elapsed_seconds"]),
                    format_duration(progress["eta_seconds"]))
        scheduler.on_success(end, peak["peak_fraction_of_budget"])
        window_index += 1
        result.scheduler_events = list(scheduler.events)
        result.final_window_size = scheduler.size
        result.total_seconds = time.perf_counter() - started
        result.tracks_by_label = {label: len(ids) for label, ids in stitcher.track_ids_by_label.items()}
        if on_window:
            on_window(result)
    result.total_seconds = time.perf_counter() - started
    result.scheduler_events = list(scheduler.events)
    result.final_window_size = scheduler.size
    return result
