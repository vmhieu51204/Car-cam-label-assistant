"""Bounded video redaction. No source frame is a public fallback."""

from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import IO, Protocol

import cv2
import numpy as np

from src.services.vm.contracts import file_sha256
from src.services.vm.storage import atomic_json


class Detector(Protocol):
    def detect(self, bgr: np.ndarray) -> list[list[float]]: ...


def redact(image: np.ndarray, boxes: list[list[float]], expansion: float = 1.25) -> np.ndarray:
    result = image.copy()
    height, width = result.shape[:2]
    for box in boxes:
        if len(box) != 4 or not all(math.isfinite(v) for v in box):
            raise ValueError("Invalid privacy box")
        x1, y1, x2, y2 = box
        if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
            raise ValueError("Privacy box outside image")
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        bw, bh = (x2 - x1) * expansion, (y2 - y1) * expansion
        left, top = max(0, math.floor(mx - bw / 2)), max(0, math.floor(my - bh / 2))
        right, bottom = min(width, math.ceil(mx + bw / 2)), min(height, math.ceil(my + bh / 2))
        roi = result[top:bottom, left:right]
        # Tiny regions become opaque, avoiding a blur kernel that preserves detail.
        if min(roi.shape[:2]) < 12:
            roi[:] = 0
        else:
            roi[:] = cv2.GaussianBlur(roi, (0, 0), sigmaX=max(8, min(roi.shape[:2]) / 3))
    return result


def probe(path: Path, ffprobe: str = "ffprobe") -> dict:
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_streams",
            "-show_frames",
            "-show_entries",
            "stream=width,height,r_frame_rate,avg_frame_rate:stream_tags=rotate:stream_side_data_list:frame=best_effort_timestamp_time,duration_time,pkt_duration_time",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        check=True,
        timeout=3600,
    )
    data = json.loads(result.stdout)
    stream = data["streams"][0]
    times = [float(row["best_effort_timestamp_time"]) for row in data["frames"]]
    fps = Fraction(stream["avg_frame_rate"])
    if fps <= 0 or not times or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Privacy requires a monotonic source frame timeline")
    if stream.get("tags", {}).get("rotate", "0") != "0" or any(
        row.get("rotation", 0) for row in stream.get("side_data_list", [])
    ):
        raise ValueError("Normalize video rotation explicitly before upload")
    return {
        "width": stream["width"],
        "height": stream["height"],
        "fps": str(fps),
        "total_frames": len(times),
        "timestamps": times,
        # Packet duration can contradict the presentation timeline in an edited
        # H.264 source. A frame stays visible until the next frame's timestamp;
        # MP4 muxers also derive non-final sample durations from that interval.
        "durations": [b - a for a, b in zip(times, times[1:])]
        + [float(data["frames"][-1].get("duration_time", data["frames"][-1].get("pkt_duration_time", float(1 / fps))))],
        "variable_frame_rate": any(abs((b - a) - float(1 / fps)) > 1e-4 for a, b in zip(times, times[1:])),
    }


def read_exact(handle: IO[bytes], size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        part = handle.read(size - len(chunks))
        if not part:
            break
        chunks.extend(part)
    return bytes(chunks)


def redact_video(
    source: Path,
    output: Path,
    detector: Detector,
    *,
    expected_sha256: str,
    manual_regions: list[dict] | None = None,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    progress: Callable[[int, int], None] = lambda _a, _b: None,
    model_metadata: dict | None = None,
) -> dict:
    if source.resolve() == output.resolve() or file_sha256(source) != expected_sha256:
        raise ValueError("Source identity changed or output would overwrite source")
    if output.exists():
        raise ValueError("Choose a new privacy output version")
    metadata = probe(source, ffprobe)
    width, height, total = metadata["width"], metadata["height"], metadata["total_frames"]
    manual_regions = manual_regions or []
    for region in manual_regions:
        if not 0 <= region["start"] <= region["end"] < total:
            raise ValueError("Manual privacy range outside video")
        redact(np.zeros((height, width, 3), dtype=np.uint8), [region["box"]])
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.stem + ".partial.mp4")
    ledger = output.with_suffix(".regions.jsonl")
    from src.services.vm.video_codec import encode_redacted

    encode_redacted(source, partial, ledger, metadata, detector, manual_regions, progress, ffmpeg=ffmpeg)
    actual = probe(partial, ffprobe)
    if any(actual[key] != metadata[key] for key in ("width", "height", "total_frames")):
        raise ValueError("Redacted video mapping differs from source")
    if any(
        abs((a - metadata["timestamps"][0]) - (b - actual["timestamps"][0])) > 1e-4
        for a, b in zip(metadata["timestamps"], actual["timestamps"], strict=True)
    ):
        raise ValueError("Redacted video timestamps differ")
    if any(abs(a - b) > 1e-4 for a, b in zip(metadata["durations"], actual["durations"], strict=True)):
        raise ValueError("Redacted frame duration differs")
    partial.replace(output)
    receipt = {
        **metadata,
        "schema": "autolabel-privacy/v1",
        "source_sha256": expected_sha256,
        "output_sha256": file_sha256(output),
        "regions_sha256": file_sha256(ledger),
        "status": "complete",
        "expansion": 1.25,
        "manual_regions": manual_regions,
        "anonymity_guaranteed": False,
        "audio": "omitted",
        "encoded_fps": actual["fps"],
    }
    if model_metadata is not None:
        receipt["model"] = model_metadata
        receipt["score_thresholds"] = {"face": 0.1, "plate": 0.1, "calibrated": False}
    atomic_json(output.with_suffix(".privacy.json"), receipt)
    return receipt
