"""Deterministic FFmpeg frame extraction with timestamp and hash provenance."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image

from src.models.video import FrameManifest, FrameRecord, VideoJob
from src.services.video.workspace import VideoWorkspace


class VideoExtractionError(RuntimeError):
    """An actionable extraction failure safe to persist and return to the client."""


class VideoExtractionCancelledError(RuntimeError):
    """The persistent job was cancelled while FFmpeg was running."""


class FFmpegVideoExtractor:
    def __init__(
        self,
        workspace: VideoWorkspace,
        *,
        ffmpeg_path: str = "ffmpeg",
        ffprobe_path: str = "ffprobe",
        timeout_s: float = 3600,
    ) -> None:
        self.workspace = workspace
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path
        self.timeout_s = timeout_s

    def extract(self, job: VideoJob, progress: Callable[[int, int], None] | None = None) -> FrameManifest:
        progress = progress or _ignore_progress
        source = self.workspace.source_path(job)
        if not source.is_file():
            raise VideoExtractionError("uploaded video file is missing")
        if _sha256_file(source) != job.source_sha256:
            raise VideoExtractionError("uploaded video hash changed before extraction")
        probe = self._probe(source)
        frame_metadata = probe.get("frames")
        streams = probe.get("streams")
        if not isinstance(frame_metadata, list) or not frame_metadata:
            raise VideoExtractionError("ffprobe found no decodable video frames")
        if not isinstance(streams, list) or not streams or not isinstance(streams[0], dict):
            raise VideoExtractionError("ffprobe found no video stream")
        progress(0, len(frame_metadata))

        frames_dir = self.workspace.prepare_frames_dir(job.job_id)
        self._run_extraction(
            [
                self.ffmpeg_path,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-fps_mode",
                "passthrough",
                "-q:v",
                "2",
                "-start_number",
                "0",
                "-progress",
                "pipe:1",
                "-nostats",
                str(frames_dir / "frame_%06d.jpg"),
            ],
            frame_count=len(frame_metadata),
            progress=progress,
        )
        files = sorted(frames_dir.glob("frame_*.jpg"))
        if len(files) != len(frame_metadata):
            raise VideoExtractionError(
                f"frame count mismatch: ffprobe reported {len(frame_metadata)}, FFmpeg wrote {len(files)}"
            )

        source_times = [_frame_timestamp(frame, index) for index, frame in enumerate(frame_metadata)]
        normalized_times = _normalize_timestamps(source_times)
        records: list[FrameRecord] = []
        previous_histogram: list[int] | None = None
        for index, (path, metadata, timestamp, source_timestamp) in enumerate(
            zip(files, frame_metadata, normalized_times, source_times, strict=True)
        ):
            if path.name != f"frame_{index:06d}.jpg":
                raise VideoExtractionError("FFmpeg produced a non-contiguous frame sequence")
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    width, height = image.size
                    histogram = image.convert("L").resize((64, 36)).histogram()
            except OSError as error:
                raise VideoExtractionError(f"extracted frame {index} is corrupt") from error
            duration = _optional_positive_float(metadata.get("pkt_duration_time"))
            scene_change_score = _histogram_distance(previous_histogram, histogram)
            records.append(
                FrameRecord(
                    frame_index=index,
                    timestamp_s=timestamp,
                    source_timestamp_s=source_timestamp,
                    duration_s=duration,
                    relative_path=f"frames/{path.name}",
                    sha256=_sha256_file(path),
                    width=width,
                    height=height,
                    scene_change_score=scene_change_score,
                    scene_cut_before=scene_change_score is not None and scene_change_score >= 0.35,
                )
            )
            previous_histogram = histogram

        stream = streams[0]
        average_fps = _fps(stream, normalized_times)
        raw_format = probe.get("format")
        format_data: dict[str, Any] = raw_format if isinstance(raw_format, dict) else {}
        duration = _duration(format_data, stream, normalized_times, records[-1].duration_s, average_fps)
        version = self._ffmpeg_version()
        return FrameManifest(
            job_id=job.job_id,
            source_name=job.source_name,
            source_sha256=job.source_sha256,
            codec=str(stream.get("codec_name") or "unknown"),
            width=records[0].width,
            height=records[0].height,
            duration_s=duration,
            average_fps=average_fps,
            variable_frame_rate=_is_variable_frame_rate(normalized_times),
            rotation_degrees=_rotation_degrees(stream),
            ffmpeg_version=version,
            frames=records,
        )

    def _probe(self, source: Path) -> dict[str, Any]:
        result = self._run(
            [
                self.ffprobe_path,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_streams",
                "-show_format",
                "-show_frames",
                "-show_entries",
                (
                    "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,duration,nb_frames:"
                    "stream_tags=rotate:stream_side_data=rotation:format=duration:"
                    "frame=best_effort_timestamp_time,pts_time,pkt_duration_time,width,height"
                ),
                "-of",
                "json",
                str(source),
            ],
            operation="video inspection",
        )
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise VideoExtractionError("ffprobe returned invalid JSON") from error
        if not isinstance(payload, dict):
            raise VideoExtractionError("ffprobe returned an invalid payload")
        return payload

    def _ffmpeg_version(self) -> str:
        result = self._run([self.ffmpeg_path, "-version"], operation="FFmpeg version check")
        first_line = result.stdout.splitlines()[0] if result.stdout else ""
        if not first_line:
            raise VideoExtractionError("FFmpeg version could not be determined")
        return first_line[:200]

    def _run_extraction(self, command: list[str], *, frame_count: int, progress: Callable[[int, int], None]) -> None:
        started = time.monotonic()
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=False)
        except FileNotFoundError as error:
            raise VideoExtractionError("frame extraction tool is not installed") from error
        try:
            if process.stdout is None:
                raise VideoExtractionError("FFmpeg progress stream is unavailable")
            for line in process.stdout:
                if time.monotonic() - started > self.timeout_s:
                    raise VideoExtractionError("frame extraction exceeded the configured timeout")
                if line.startswith("frame="):
                    progress(min(_progress_frame(line), frame_count), frame_count)
            remaining = max(0.1, self.timeout_s - (time.monotonic() - started))
            return_code = process.wait(timeout=remaining)
        except Exception:
            _stop_process(process)
            raise
        if return_code != 0:
            detail = process.stderr.read().strip().splitlines() if process.stderr is not None else []
            suffix = f": {detail[-1][:300]}" if detail else ""
            raise VideoExtractionError(f"frame extraction failed with exit code {return_code}{suffix}")
        progress(frame_count, frame_count)

    def _run(self, command: list[str], *, operation: str) -> subprocess.CompletedProcess[str]:
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                shell=False,
            )
        except FileNotFoundError as error:
            raise VideoExtractionError(f"{operation} tool is not installed") from error
        except subprocess.TimeoutExpired as error:
            raise VideoExtractionError(f"{operation} exceeded the configured timeout") from error
        if result.returncode != 0:
            detail = (result.stderr or "").strip().splitlines()
            suffix = f": {detail[-1][:300]}" if detail else ""
            raise VideoExtractionError(f"{operation} failed with exit code {result.returncode}{suffix}")
        return result


def _frame_timestamp(metadata: object, index: int) -> float:
    if not isinstance(metadata, dict):
        raise VideoExtractionError(f"ffprobe frame {index} metadata is invalid")
    raw = metadata.get("best_effort_timestamp_time", metadata.get("pts_time"))
    if isinstance(raw, bool) or not isinstance(raw, (str, int, float)):
        raise VideoExtractionError(f"ffprobe frame {index} has no presentation timestamp")
    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise VideoExtractionError(f"ffprobe frame {index} has no presentation timestamp") from error
    if not math.isfinite(value):
        raise VideoExtractionError(f"ffprobe frame {index} has a non-finite timestamp")
    return value


def _normalize_timestamps(values: list[float]) -> list[float]:
    first = values[0]
    normalized = [round(value - first, 9) for value in values]
    if any(right <= left for left, right in zip(normalized, normalized[1:], strict=False)):
        raise VideoExtractionError("video presentation timestamps are not strictly increasing")
    normalized[0] = 0.0
    return normalized


def _optional_positive_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def _parse_ratio(value: object) -> float | None:
    if not isinstance(value, str) or "/" not in value:
        return None
    numerator, denominator = value.split("/", maxsplit=1)
    try:
        result = float(numerator) / float(denominator)
    except (ValueError, ZeroDivisionError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def _fps(stream: dict[str, Any], timestamps: list[float]) -> float:
    reported = _parse_ratio(stream.get("avg_frame_rate")) or _parse_ratio(stream.get("r_frame_rate"))
    if reported is not None:
        return reported
    if len(timestamps) > 1 and timestamps[-1] > timestamps[0]:
        return (len(timestamps) - 1) / (timestamps[-1] - timestamps[0])
    raise VideoExtractionError("video frame rate could not be determined")


def _duration(
    format_data: dict[str, Any],
    stream: dict[str, Any],
    timestamps: list[float],
    last_duration: float | None,
    fps: float,
) -> float:
    for raw in (stream.get("duration"), format_data.get("duration")):
        parsed = _optional_positive_float(raw)
        if parsed is not None:
            return parsed
    return timestamps[-1] + (last_duration or 1 / fps)


def _is_variable_frame_rate(timestamps: list[float]) -> bool:
    if len(timestamps) < 3:
        return False
    intervals = [right - left for left, right in zip(timestamps, timestamps[1:], strict=False)]
    median = statistics.median(intervals)
    tolerance = max(0.001, median * 0.02)
    return any(abs(interval - median) > tolerance for interval in intervals)


def _rotation_degrees(stream: dict[str, Any]) -> float:
    tags = stream.get("tags")
    if isinstance(tags, dict):
        parsed = _finite_float(tags.get("rotate"))
        if parsed is not None:
            return parsed
    side_data = stream.get("side_data_list")
    if isinstance(side_data, list):
        for item in side_data:
            if isinstance(item, dict):
                parsed = _finite_float(item.get("rotation"))
                if parsed is not None:
                    return parsed
    return 0.0


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _histogram_distance(previous: list[int] | None, current: list[int]) -> float | None:
    if previous is None:
        return None
    pixel_count = sum(current)
    if len(previous) != len(current) or pixel_count <= 0 or sum(previous) != pixel_count:
        raise VideoExtractionError("scene-cut histogram dimensions are inconsistent")
    return sum(abs(left - right) for left, right in zip(previous, current, strict=True)) / (2 * pixel_count)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _progress_frame(line: str) -> int:
    try:
        return max(0, int(line.partition("=")[2].strip()))
    except ValueError:
        return 0


def _stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _ignore_progress(_current: int, _total: int) -> None:
    return None
