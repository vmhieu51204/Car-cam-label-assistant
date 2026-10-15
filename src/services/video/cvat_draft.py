"""Prepare editable pending 2D tracks for CVAT REST without contacting CVAT."""

from __future__ import annotations

import hashlib
import json

from src.models.video import (
    BoundingBox2D,
    ComponentVersion,
    FrameManifest,
    FrameRecord,
    TrackObservation,
    TrackSet,
    VideoTrack,
)
from src.services.cvat.track_client import build_cvat_annotation_payload
from src.services.video.sequence_review import TrackReview
from src.services.video.stitching import UnifiedTrack


def build_sequence_cvat_draft(
    *,
    job_id: str,
    source_sha256: str,
    source_name: str,
    source_frames: list[int],
    frame_sha256: list[str],
    width: int,
    height: int,
    source_fps: float,
    labels: list[dict],
    text_prompts: tuple[str, ...],
    tracks: list[UnifiedTrack],
    review: list[TrackReview],
    existing_annotations: dict,
    model_version: str,
) -> dict:
    """Return a CVAT annotation update, still requiring human review to publish labels.

    The caller must supply a read-only snapshot of CVAT's label IDs, attribute IDs,
    and existing annotations. This function never performs an HTTP request.
    """
    if not source_frames or len(source_frames) != len(frame_sha256) or source_fps <= 0:
        raise ValueError("invalid CVAT source timeline")
    manifest = FrameManifest(
        job_id=job_id,
        source_name=source_name,
        source_sha256=source_sha256,
        source_frame_count=source_frames[-1] + 1,
        sampling_strategy="uniform",
        source_fps=source_fps,
        fps_source="operator",
        codec="sequence-jpeg",
        width=width,
        height=height,
        duration_s=(source_frames[-1] + 1) / source_fps,
        average_fps=(
            (len(source_frames) - 1) * source_fps / source_frames[-1] if len(source_frames) > 1 else source_fps
        ),
        variable_frame_rate=False,
        ffmpeg_version="sequence frames; FPS supplied by operator",
        frames=[
            FrameRecord(
                frame_index=index,
                source_frame_index=source,
                timestamp_s=source / source_fps,
                source_timestamp_s=source / source_fps,
                relative_path=f"frames/frame_{index:06d}.jpg",
                sha256=frame_sha256[index],
                width=width,
                height=height,
            )
            for index, source in enumerate(source_frames)
        ],
    )
    label_ids = {row["name"]: row["id"] for row in labels}
    if any(track.label not in label_ids for track in tracks):
        raise ValueError("CVAT task lacks one or more predicted classes")
    converted = []
    for track in tracks:
        observations = []
        ordered = sorted(track.observations, key=lambda item: item.frame)
        for index, item in enumerate(ordered):
            box = BoundingBox2D(xtl=item.bbox[0], ytl=item.bbox[1], xbr=item.bbox[2], ybr=item.bbox[3])
            observations.append(
                TrackObservation(
                    frame_index=item.frame,
                    bbox=box,
                    score=item.score,
                    mask_area_px=sum(item.mask.counts[1::2]) if item.mask else None,
                )
            )
            next_frame = ordered[index + 1].frame if index + 1 < len(ordered) else None
            if (next_frame is None and item.frame + 1 < len(source_frames)) or (
                next_frame is not None and next_frame > item.frame + 1
            ):
                observations.append(
                    TrackObservation(frame_index=item.frame + 1, bbox=box, outside=True, mask_area_px=0)
                )
        converted.append(
            VideoTrack(
                track_id=track.track_id,
                label_id=label_ids[track.label],
                label_name=track.label,
                source="auto",
                review_status="pending",
                observations=observations,
            )
        )
    prompt_hash = hashlib.sha256(json.dumps(text_prompts, separators=(",", ":")).encode()).hexdigest()
    track_set = TrackSet(
        job_id=job_id,
        manifest_sha256=hashlib.sha256(manifest.model_dump_json(by_alias=True).encode()).hexdigest(),
        prompt_sha256=prompt_hash,
        component_versions=[ComponentVersion(component="sam3-video-text", version=model_version)],
        tracks=converted,
    )
    attribute_ids = {row["id"]: {spec["name"]: spec["id"] for spec in row.get("attributes", [])} for row in labels}
    mutable_names = {
        row["id"]: {spec["name"] for spec in row.get("attributes", []) if spec.get("mutable") is True} for row in labels
    }
    return build_cvat_annotation_payload(
        existing=existing_annotations,
        manifest=manifest,
        tracks=track_set,
        label_attributes=attribute_ids,
        mutable_attributes=mutable_names,
        job_id=job_id,
        review_priority_by_track={item.track_id: item.priority for item in review},
        review_score_by_track={item.track_id: item.score for item in review},
    )
