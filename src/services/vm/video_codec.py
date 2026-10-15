"""Sequential RGB encoding that preserves source PTS, including variable frame rates."""

import json
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Callable, Generator
from contextlib import closing
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from src.services.vm.privacy import Detector, read_exact, redact


def _decode_frames(source: Path, metadata: dict, ffmpeg: str) -> Generator[np.ndarray, None, None]:
    # Use the same FFmpeg decoder as ffprobe's best-effort timeline. PyAV's
    # raw frame.pts can repeat even when that timeline is valid.
    width, height = metadata["width"], metadata["height"]
    frame_size = width * height * 3
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(
            [
                ffmpeg,
                "-v",
                "error",
                "-nostdin",
                "-xerror",
                "-threads",
                "1",
                "-noautorotate",
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-vsync",
                "0",
                "-threads",
                "1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=errors,
        )
        assert process.stdout is not None
        try:
            for _ in range(metadata["total_frames"]):
                data = read_exact(process.stdout, frame_size)
                if len(data) != frame_size:
                    raise ValueError("Privacy decoder coverage incomplete or truncated frame")
                yield np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
            if process.stdout.read(1):
                raise ValueError("Privacy decoder has extra frames outside source timeline")
            if process.wait():
                size = errors.seek(0, 2)
                errors.seek(max(0, size - 8192))
                detail = errors.read(8192).decode("utf-8", errors="replace")
                raise ValueError(f"Privacy FFmpeg decoder failed: {detail}")
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            process.stdout.close()


def encode_redacted(
    source: Path,
    output: Path,
    ledger: Path,
    metadata: dict,
    detector: Detector,
    regions: list[dict],
    progress: Callable[[int, int], None],
    *,
    ffmpeg: str = "ffmpeg",
) -> None:
    import av

    count = 0
    starts: dict[int, list[int]] = defaultdict(list)
    ends: dict[int, list[int]] = defaultdict(list)
    for i, region in enumerate(regions):
        starts[region["start"]].append(i)
        ends[region["end"] + 1].append(i)
    active: dict[int, dict] = {}
    with (
        av.open(str(source)) as reader,
        av.open(str(output), mode="w", format="mp4") as writer,
        ledger.open("w", encoding="utf-8") as rows,
        closing(_decode_frames(source, metadata, ffmpeg)) as frames,
    ):
        original = reader.streams.video[0]
        original.codec_context.thread_count = 1
        time_base = Fraction(original.time_base)
        stream = writer.add_stream("libx264rgb", rate=Fraction(metadata["fps"]))
        stream.width, stream.height = metadata["width"], metadata["height"]
        stream.pix_fmt = "bgr24"
        stream.time_base = stream.codec_context.time_base = time_base
        stream.codec_context.thread_count = 1
        stream.options = {"crf": "0", "preset": "fast", "bf": "0", "tune": "zerolatency"}
        durations: dict[int, int] = {}

        def mux(packet: Any) -> None:
            key = round(float(packet.pts * packet.time_base) / time_base)
            packet.duration = round(durations.pop(key) * time_base / packet.time_base)
            writer.mux(packet)

        for index, image in enumerate(frames):
            if image.shape != (metadata["height"], metadata["width"], 3):
                raise ValueError("Privacy decoded dimensions differ")
            for i in ends[index]:
                active.pop(i, None)
            for i in starts[index]:
                active[i] = regions[i]
            boxes = detector.detect(image) + [r["box"] for r in active.values()]
            clean = redact(image, boxes)
            encoded = av.VideoFrame.from_ndarray(clean, format="bgr24")
            encoded.pts = round((metadata["timestamps"][index] - metadata["timestamps"][0]) / time_base)
            encoded.time_base = time_base
            encoded.duration = round(metadata["durations"][index] / time_base)
            durations[encoded.pts] = encoded.duration
            for packet in stream.encode(encoded):
                mux(packet)
            rows.write(json.dumps({"frame": index, "timestamp": metadata["timestamps"][index], "boxes": boxes}) + "\n")
            count = index + 1
            progress(count, metadata["total_frames"])
        if count != metadata["total_frames"]:
            raise ValueError("Privacy decoder coverage incomplete")
        for packet in stream.encode():
            mux(packet)
