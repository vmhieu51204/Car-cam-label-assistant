"""Read an existing CVAT job, select motion-aware keyframes, and retain source indexes."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from io import BytesIO

from PIL import Image

from src.models.video import FrameManifest, FrameRecord, VideoJob
from src.services.cvat.read_client import CvatReadClient
from src.services.video.extractor import _histogram_distance
from src.services.video.profile import Profile2D, current_profile
from src.services.video.sampling import MAX_MODEL_FRAMES, choose_keyframes, motion_image, motion_score, uniform_indexes
from src.services.video.workspace import VideoWorkspace


async def import_cvat_frames(
    job: VideoJob,
    *,
    client: CvatReadClient,
    workspace: VideoWorkspace,
    progress: Callable[[int, int], None],
    profile_enabled: bool = False,
) -> FrameManifest:
    profile = Profile2D(workspace.job_dir(job.job_id), enabled=profile_enabled)
    with profile.activate(), profile.stage("import.total"):
        return await _import_cvat_frames(job, client=client, workspace=workspace, progress=progress)


async def _import_cvat_frames(
    job: VideoJob,
    *,
    client: CvatReadClient,
    workspace: VideoWorkspace,
    progress: Callable[[int, int], None],
) -> FrameManifest:
    started = time.perf_counter()
    if job.cvat_job_id is None or job.cvat_task_id is None:
        raise ValueError("CVAT import job has no CVAT binding")
    with current_profile().stage("import.metadata"):
        cvat_job = await client.get_job(job.cvat_job_id)
        meta = await client.get_job_data_meta(job.cvat_job_id)
    count = meta.get("size")
    frames_meta = meta.get("frames")
    if (
        not isinstance(count, int)
        or isinstance(count, bool)
        or count < 1
        or not isinstance(frames_meta, list)
        or not frames_meta
        or not isinstance(frames_meta[0], dict)
    ):
        raise ValueError("CVAT frame metadata changed or is incomplete")
    if (
        cvat_job.get("task_id") != job.cvat_task_id
        or cvat_job.get("start_frame") != 0
        or cvat_job.get("stop_frame") != count - 1
    ):
        raise ValueError("CVAT job range changed after import was queued")
    source_metadata = json.dumps(
        {"job": job.cvat_job_id, "task": job.cvat_task_id, "frames": frames_meta},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if job.source_sha256 != "0" * 64 and hashlib.sha256(source_metadata).hexdigest() != job.source_sha256:
        raise ValueError("CVAT frame metadata changed after import was queued")

    request = workspace.read_sampling_request(job.job_id)
    if request.source_fps is None:
        raise ValueError("CVAT import requires a declared source_fps")
    scores = [0.0]
    if request.strategy in {"uniform", "all_frames"}:
        indexes = (
            list(range(count))
            if request.strategy == "all_frames"
            else uniform_indexes(count, request.source_fps, request.target_fps or request.source_fps)
        )
        effective_threshold = None
        offset = 0
    else:
        previous = None
        total = count + min(count, MAX_MODEL_FRAMES)
        async for encoded in client.iter_job_frames(job.cvat_job_id, count, quality="compressed"):
            current = motion_image(encoded)
            if previous is not None:
                scores.append(motion_score(previous, current))
            previous = current
            progress(len(scores), total)
        indexes, effective_threshold = choose_keyframes(scores, request)
        offset = count
    total = offset + len(indexes)
    progress(offset, total)
    frames_dir = workspace.prepare_frames_dir(job.job_id)
    records: list[FrameRecord] = []
    width = height = 0
    previous_preview = None
    previous_histogram = None
    delivery_tick = time.perf_counter()
    async for encoded in client.iter_selected_job_frames(job.cvat_job_id, indexes):
        current_profile().record(
            "import.frame_delivery_wait", time.perf_counter() - delivery_tick, frame_index=len(records)
        )
        if len(records) >= len(indexes):
            raise ValueError("CVAT returned more selected frames than requested")
        with current_profile().stage("import.image_processing_and_save"):
            source_index = indexes[len(records)]
            preview = motion_image(encoded)
            with Image.open(BytesIO(encoded)) as image:
                frame_width, frame_height = image.size
                histogram = image.convert("L").resize((64, 36)).histogram()
                if image.format != "JPEG":
                    buffer = BytesIO()
                    image.convert("RGB").save(buffer, format="JPEG", quality=95)
                    encoded = buffer.getvalue()
            if records and (frame_width, frame_height) != (width, height):
                raise ValueError("CVAT selected frames have inconsistent dimensions")
            width, height = frame_width, frame_height
            scene_change = _histogram_distance(previous_histogram, histogram)
            optical = (
                (
                    motion_score(previous_preview, preview, include_appearance=False)
                    if previous_preview is not None
                    else 0.0
                )
                if request.strategy in {"uniform", "all_frames"}
                else scores[source_index]
            )
            previous_preview, previous_histogram = preview, histogram
            local_index = len(records)
            relative_path = f"frames/frame_{local_index:06d}.jpg"
            (frames_dir / f"frame_{local_index:06d}.jpg").write_bytes(encoded)
            records.append(
                FrameRecord(
                    frame_index=local_index,
                    source_frame_index=source_index,
                    timestamp_s=source_index / request.source_fps,
                    source_timestamp_s=source_index / request.source_fps,
                    relative_path=relative_path,
                    sha256=hashlib.sha256(encoded).hexdigest(),
                    width=width,
                    height=height,
                    optical_flow_score=optical,
                    scene_change_score=scene_change,
                    scene_cut_before=scene_change is not None and scene_change >= request.scene_cut_threshold,
                )
            )
        progress(offset + len(records), total)
        delivery_tick = time.perf_counter()
    if len(records) != len(indexes):
        raise ValueError("CVAT did not return all selected frames")
    manifest = FrameManifest(
        job_id=job.job_id,
        source_name=job.source_name,
        source_sha256=job.source_sha256,
        source_frame_count=count,
        sampling_strategy=request.strategy,
        sampling_threshold=request.threshold if request.strategy == "threshold" else None,
        effective_sampling_threshold=effective_threshold,
        sampling_budget=request.budget if request.strategy == "budget" else None,
        source_fps=request.source_fps,
        target_fps=request.source_fps
        if request.strategy == "all_frames"
        else request.target_fps
        if request.strategy == "uniform"
        else None,
        fps_source="operator",
        codec="cvat-original-jpeg",
        width=width,
        height=height,
        duration_s=count / request.source_fps,
        average_fps=(len(records) - 1) * request.source_fps / (indexes[-1] - indexes[0])
        if len(records) > 1
        else request.source_fps,
        variable_frame_rate=False,
        scene_cut_threshold=request.scene_cut_threshold,
        ffmpeg_version="CVAT frame-index mapping; FPS supplied by operator",
        frames=records,
    )

    (workspace.job_dir(job.job_id) / "import-runtime.json").write_text(
        json.dumps(
            {
                "schema": "autolabel-import-runtime/v1",
                "total_seconds": time.perf_counter() - started,
                "source_frame_count": count,
                "selected_frame_count": len(records),
                "frame_requests": total,
                "sampling_strategy": request.strategy,
                "source_fps": request.source_fps,
                "target_fps": request.target_fps,
                "scene_cuts": [r.frame_index for r in records if r.scene_cut_before],
                "scene_cut_threshold": request.scene_cut_threshold,
                "threshold_calibrated": False,
            }
        ),
        encoding="utf-8",
    )
    return manifest
