"""Seed-free SAM3 Kaggle batch contract for a sampled CVAT video job."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.models.video import (
    BoundingBox2D,
    ComponentVersion,
    FrameManifest,
    TrackObservation,
    TrackSet,
    VideoTrack,
)
from src.services.video.chunks import plan_chunks
from src.services.video.stitching import (
    ChunkObservation,
    ChunkTrack,
    MaskRLE,
    StitchConfig,
    StitchResult,
    stitch_tracks,
)
from src.services.video.workspace import VideoWorkspace


def sequence_payload(
    manifest: FrameManifest,
    workspace: VideoWorkspace,
    classes: tuple[str, ...],
    *,
    window_size: int = 50,
    overlap: int = 5,
    model_variant: str = "sam3",
) -> tuple[dict[str, Path | bytes], dict]:
    """Build the benchmark notebook's single-config payload without GT seeds."""
    if manifest.frame_count < 1 or manifest.source_fps is None or manifest.target_fps is None:
        raise ValueError("seed-free sequence needs at least one frame and a declared uniform timebase")
    if manifest.frame_count > 10_000:
        raise ValueError(
            "SAM3 Kaggle notebook accepts at most 10000 input frames; split the video without dropping frames"
        )
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("SAM3 text prompts must be unique and nonempty")
    if model_variant not in {"sam3", "sam3.1"}:
        raise ValueError("unsupported SAM model variant")
    if not 2 <= window_size <= 250 or not 0 <= overlap < window_size:
        raise ValueError("invalid SAM3 window settings")
    paths = workspace.frame_paths(manifest)
    config: dict[str, Any] = {
        "schema": "autolabel-sam3-benchmark-config/v1",
        "max_total_seconds": 21_600,
        "max_config_seconds": 20_000,
        "source_frame_limit": manifest.frame_count,
        "configurations": [
            {
                "id": "sequence-text",
                "model_variant": model_variant,
                "selection": "dense",
                "gap": None,
                "window_size": window_size,
                "overlap": overlap,
                "mode": "text_prompt",
                "seed_type": "none",
                "reprompt": "none",
                "text_prompts": list(classes),
                "preserve_overlap": True,
                "scene_cut_frames": [frame.frame_index for frame in manifest.frames[1:] if frame.scene_cut_before],
            }
        ],
    }
    frames = []
    if manifest.sampling_strategy == "all_frames":
        if [r.source_frame_index for r in manifest.frames] != list(range(manifest.source_frame_count or 0)):
            raise ValueError("all_frames manifest must contain every source frame in order")
        config["configurations"][0]["sampling_strategy"] = "all_frames"
    files: dict[str, Path | bytes] = {}
    for record, path in zip(manifest.frames, paths, strict=True):
        name = f"frames/frame_{record.frame_index:06d}.jpg"
        files[name] = path
        frames.append(
            {
                "source_frame_index": record.frame_index,
                "original_source_index": record.source_frame_index,
                "timestamp_s": record.timestamp_s,
                "relative_path": name,
                "sha256": record.sha256,
                "width": record.width,
                "height": record.height,
                "scene_cut_before": record.scene_cut_before,
            }
        )
    files["frames.json"] = json.dumps(
        {
            "frames": frames,
            "optical_source_indices": list(range(manifest.frame_count)),
            "source_fps": manifest.source_fps,
            "target_fps": manifest.target_fps,
            "fps_source": manifest.fps_source,
        }
    ).encode()
    files["benchmark-config.json"] = json.dumps(config).encode()
    files["seed.json"] = b'{"seeds": []}'
    return files, config["configurations"][0]


def parse_sequence_result(
    result: dict, *, expected_config: dict, frame_count: int, decision_log_enabled: bool = True
) -> tuple[StitchResult, ComponentVersion]:
    """Reject partial or differently configured kernels before tracks reach CVAT."""
    rows = result.get("configurations")
    expected_frames = list(range(frame_count))
    if result.get("schema") != "autolabel-sam3-benchmark/v1" or not isinstance(rows, list) or len(rows) != 1:
        raise ValueError("SAM3 sequence result has the wrong schema")
    item = rows[0]
    if (
        item.get("id") != "sequence-text"
        or item.get("config") != expected_config
        or item.get("status") != "completed"
        or item.get("selected_source_frames") != expected_frames
        or item.get("evaluated_source_frames") != expected_frames
        or not isinstance(item.get("window_predictions"), list)
    ):
        raise ValueError("SAM3 sequence result is incomplete or mismatched")
    grouped: dict[tuple[int, str, str], list[ChunkObservation]] = defaultdict(list)
    for row in item["window_predictions"]:
        frame = row["frame_index"]
        if not isinstance(frame, int) or not 0 <= frame < frame_count:
            raise ValueError("SAM3 observation is outside the sampled timeline")
        raw = row.get("mask_rle")
        mask = MaskRLE(raw["height"], raw["width"], tuple(raw["counts"]), raw["sha256"]) if raw else None
        grouped[(row["window_id"], row["track_id"], row["label"])].append(
            ChunkObservation(frame, tuple(row["bbox"]), mask, row.get("score"))
        )
    chunks = [
        ChunkTrack(window, local_id, label, sorted(observations, key=lambda obs: obs.frame))
        for (window, local_id, label), observations in sorted(grouped.items())
    ]
    cuts = tuple(expected_config.get("scene_cut_frames", []))
    plans = plan_chunks(
        frame_count,
        window_size=expected_config["window_size"],
        overlap=expected_config["overlap"],
        scene_cut_frames=cuts,
    )
    reset_windows = {plan.index for plan in plans if plan.start in cuts}
    stitched = stitch_tracks(
        chunks,
        StitchConfig(decision_log_enabled=decision_log_enabled),
        reset_windows=reset_windows,
        planned_window_ids=[plan.index for plan in plans],
    )
    runtime = result.get("model_runtime") or {}
    environment = runtime.get("environment") or {}
    versions = environment.get("versions") or {}
    commit = versions.get("sam3_commit") or "unreported"
    model_variant = expected_config.get("model_variant", "sam3")
    checksum = runtime.get("checkpoint_sha256")
    if not isinstance(checksum, str) or len(checksum) != 64:
        checksum = None
    if model_variant == "sam3.1":
        if runtime.get("model_variant") != "sam3.1":
            raise ValueError("Kaggle result does not confirm SAM 3.1 model identity")
        if checksum is None or any(character not in "0123456789abcdef" for character in checksum.lower()):
            raise ValueError("Kaggle result is missing a valid SAM 3.1 checkpoint SHA-256")
        if not isinstance(commit, str) or commit == "unreported":
            raise ValueError("Kaggle result is missing the SAM 3.1 code commit")
    if checksum is not None:
        checksum = checksum.lower()
    model_name = "sam3.1" if model_variant == "sam3.1" else "sam3"
    return stitched, ComponentVersion(
        component="sam3-video-text", version=f"{model_name}@{commit}", checkpoint_sha256=checksum
    )


def make_track_set(
    *,
    manifest: FrameManifest,
    manifest_sha256: str,
    stitched: StitchResult,
    labels: list[dict],
    classes: tuple[str, ...],
    component: ComponentVersion,
) -> TrackSet:
    """Keep sampled indices internally; CVAT publisher maps them to source frames."""
    label_ids = {row["name"]: row["id"] for row in labels}
    if any(name not in label_ids for name in classes):
        raise ValueError("CVAT job lacks one or more requested SAM3 classes")
    tracks = []
    for track in stitched.tracks:
        if track.label not in label_ids or track.label not in classes:
            raise ValueError("SAM3 returned a class that was not requested in CVAT")
        ordered = sorted(track.observations, key=lambda obs: obs.frame)
        observations = []
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
            if (next_frame is None and item.frame + 1 < manifest.frame_count) or (
                next_frame is not None and next_frame > item.frame + 1
            ):
                observations.append(
                    TrackObservation(
                        frame_index=item.frame + 1,
                        bbox=box,
                        outside=True,
                        mask_area_px=0,
                    )
                )
        tracks.append(
            VideoTrack(
                track_id=track.track_id,
                label_id=label_ids[track.label],
                label_name=track.label,
                source="auto",
                needs_review=track.needs_review,
                review_reasons=track.review_reasons,
                observations=observations,
            )
        )
    prompt_sha256 = hashlib.sha256(json.dumps(classes, separators=(",", ":")).encode()).hexdigest()
    return TrackSet(
        job_id=manifest.job_id,
        manifest_sha256=manifest_sha256,
        prompt_sha256=prompt_sha256,
        component_versions=[component],
        tracks=tracks,
    )
