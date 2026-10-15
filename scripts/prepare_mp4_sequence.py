"""Extract every MP4 frame for CVAT and prepare a truthful 2D sequence timebase.

The numbered images are suitable for ``package_sam3_sequence``. CVAT should
receive the original video, so its player retains every source frame.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from src.models.video import VideoJob, VideoJobStatus
from src.services.video.extractor import FFmpegVideoExtractor
from src.services.video.workspace import VideoWorkspace


def prepare_video(video: Path, output: Path) -> dict:
    if not video.is_file() or video.suffix.lower() not in {".mp4", ".mov", ".mkv"}:
        raise ValueError("supply an existing MP4/MOV/MKV video")
    output.mkdir(parents=True, exist_ok=True)
    workspace = VideoWorkspace(output / "extraction")
    job_id = str(uuid4())
    job_dir = workspace.job_dir(job_id, create=True)
    source = workspace.source_path(
        VideoJob(
            job_id=job_id,
            status=VideoJobStatus.QUEUED,
            source_name=video.name,
            source_sha256="0" * 64,
            source_size_bytes=video.stat().st_size,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            stage="preparing",
        )
    )
    shutil.copyfile(video, source)
    with source.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    job = VideoJob(
        job_id=job_id,
        status=VideoJobStatus.QUEUED,
        source_name=video.name,
        source_sha256=digest,
        source_size_bytes=source.stat().st_size,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        stage="preparing",
    )
    manifest = FFmpegVideoExtractor(workspace).extract(job)
    if manifest.variable_frame_rate:
        raise ValueError("variable-frame-rate video needs timestamp-aware sampling; no constant FPS was invented")
    if manifest.frame_count < 2:
        raise ValueError("SAM3 needs at least two video frames")
    source_fps = manifest.average_fps
    images = output / "images"
    if images.exists() and any(images.iterdir()):
        raise ValueError("output/images must be empty")
    images.mkdir(exist_ok=True)
    for frame in manifest.frames:
        shutil.move(str(job_dir / frame.relative_path), images / f"{frame.frame_index:06d}.jpg")
    result = {
        "schema": "autolabel-mp4-sequence/v1",
        "source_video": str(video.resolve()),
        "source_sha256": digest,
        "source_fps": source_fps,
        "fps_source": "ffprobe",
        "frame_count": manifest.frame_count,
        "images": str(images.resolve()),
        "timestamps_s": [frame.timestamp_s for frame in manifest.frames],
        "cvat_media": str(video.resolve()),
    }
    (output / "video-sequence.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_video(args.video, args.output)
    print(json.dumps({key: value for key, value in result.items() if key != "timestamps_s"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
