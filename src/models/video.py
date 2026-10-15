"""Versioned contracts for the video-first annotation workflow."""

from __future__ import annotations

import math
from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SHA256_PATTERN = r"^[0-9a-f]{64}$"
UUID_PATTERN = r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"


class StrictContract(BaseModel):
    """Reject unknown fields at every persisted/API contract boundary."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class VideoJobStatus(StrEnum):
    UPLOADED = "uploaded"
    QUEUED = "queued"
    EXTRACTING = "extracting"
    AWAITING_SEED = "awaiting_seed"
    SEEDING = "seeding"
    SEED_READY = "seed_ready"
    PROPAGATING = "propagating"
    DISCOVERING = "discovering"
    AUDITING = "auditing"
    AWAITING_REVIEW = "awaiting_review"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ComponentVersion(StrictContract):
    component: str = Field(min_length=1, max_length=64)
    version: str = Field(min_length=1, max_length=200)
    checkpoint_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    config_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)


class VideoJob(StrictContract):
    """Small public job record; runtime file paths stay behind the workspace service."""

    schema_version: Literal[1] = 1
    job_id: str = Field(pattern=UUID_PATTERN)
    status: VideoJobStatus
    source_name: str = Field(min_length=1, max_length=255)
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    source_size_bytes: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime
    stage: str = Field(min_length=1, max_length=64)
    progress_current: int = Field(default=0, ge=0)
    progress_total: int = Field(default=0, ge=0)
    manifest_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    cvat_job_id: int | None = Field(default=None, ge=1)
    seed_revision: int | None = Field(default=None, ge=0)
    seed_count: int = Field(default=0, ge=0)
    seed_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    cvat_task_id: int | None = Field(default=None, ge=1)
    component_versions: list[ComponentVersion] = Field(default_factory=list)
    track_set_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    discovery_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    audit_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    review_queue_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    publish_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    cvat_annotation_revision: int | None = Field(default=None, ge=0)
    published_track_count: int = Field(default=0, ge=0)
    error_code: str | None = Field(default=None, max_length=64)
    error_message: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def progress_is_bounded(self) -> VideoJob:
        if self.progress_total and self.progress_current > self.progress_total:
            raise ValueError("progress_current cannot exceed progress_total")
        return self


class FrameRecord(StrictContract):
    frame_index: int = Field(ge=0)
    source_frame_index: int | None = Field(default=None, ge=0)
    timestamp_s: float = Field(ge=0, allow_inf_nan=False)
    source_timestamp_s: float = Field(allow_inf_nan=False)
    duration_s: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    relative_path: str = Field(pattern=r"^frames/frame_[0-9]{6}\.jpg$")
    sha256: str = Field(pattern=SHA256_PATTERN)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    scene_change_score: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    optical_flow_score: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    scene_cut_before: bool = False


class FrameManifest(StrictContract):
    schema_id: Literal["autolabel-frame-manifest/v1"] = Field(default="autolabel-frame-manifest/v1", alias="schema")
    job_id: str = Field(pattern=UUID_PATTERN)
    source_name: str = Field(min_length=1, max_length=255)
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    source_frame_count: int | None = Field(default=None, ge=1)
    sampling_strategy: Literal["budget", "threshold", "uniform", "all_frames"] | None = None
    sampling_threshold: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    effective_sampling_threshold: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    sampling_budget: int | None = Field(default=None, ge=2, le=300)
    source_fps: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    target_fps: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    fps_source: Literal["operator", "ffprobe"] | None = None
    codec: str = Field(min_length=1, max_length=64)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    duration_s: float = Field(ge=0, allow_inf_nan=False)
    average_fps: float = Field(gt=0, allow_inf_nan=False)
    variable_frame_rate: bool
    rotation_degrees: float = Field(default=0, ge=-360, le=360, allow_inf_nan=False)
    scene_cut_method: Literal["luma_histogram_l1/v1"] = "luma_histogram_l1/v1"
    scene_cut_threshold: float = Field(default=0.35, gt=0, lt=1, allow_inf_nan=False)
    ffmpeg_version: str = Field(min_length=1, max_length=200)
    frames: list[FrameRecord] = Field(min_length=1)

    @model_validator(mode="after")
    def frames_are_ordered(self) -> FrameManifest:
        expected = list(range(len(self.frames)))
        actual = [frame.frame_index for frame in self.frames]
        if actual != expected:
            raise ValueError("frame indices must be contiguous and zero-based")
        timestamps = [frame.timestamp_s for frame in self.frames]
        if any(not math.isfinite(value) for value in timestamps):
            raise ValueError("frame timestamps must be finite")
        if any(right <= left for left, right in zip(timestamps, timestamps[1:], strict=False)):
            raise ValueError("frame timestamps must be strictly increasing")
        source_indices = [
            frame.source_frame_index if frame.source_frame_index is not None else frame.frame_index
            for frame in self.frames
        ]
        if source_indices != sorted(set(source_indices)):
            raise ValueError("source frame indices must be unique and increasing")
        if self.source_frame_count is not None and source_indices[-1] >= self.source_frame_count:
            raise ValueError("source frame index exceeds source frame count")
        return self

    @property
    def frame_count(self) -> int:
        return len(self.frames)


class CvatSamplingRequest(StrictContract):
    strategy: Literal["budget", "threshold", "uniform", "all_frames"] = "all_frames"
    budget: int = Field(default=300, ge=2, le=300)
    threshold: float = Field(default=0.02, gt=0, le=1, allow_inf_nan=False)
    source_fps: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    target_fps: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    auto_start_sequence: bool = False
    sam3_model_variant: Literal["sam3", "sam3.1"] = "sam3"
    sam_execution_provider: Literal["kaggle", "local"] = "kaggle"
    vlm_box_rescue: bool = False
    # Initial engineering threshold, not calibrated on real video cuts.
    scene_cut_threshold: float = Field(default=0.35, gt=0, lt=1, allow_inf_nan=False)
    sequence_classes: list[str] = Field(default_factory=lambda: ["car", "person"], min_length=1, max_length=8)

    @model_validator(mode="after")
    def uniform_has_timebase(self) -> CvatSamplingRequest:
        if self.sam_execution_provider == "local" and not self.auto_start_sequence:
            raise ValueError("Local SAM requires automatic sequence labeling")
        if any(not label.strip() or len(label) > 128 for label in self.sequence_classes):
            raise ValueError("sequence class names must be nonempty and at most 128 characters")
        if len(set(self.sequence_classes)) != len(self.sequence_classes):
            raise ValueError("sequence class names must be unique")
        if self.auto_start_sequence and self.strategy not in {"uniform", "all_frames"}:
            raise ValueError("automatic sequence labeling requires uniform sampling")
        if self.sam3_model_variant != "sam3" and not self.auto_start_sequence:
            raise ValueError("SAM 3.1 model selection requires automatic sequence labeling")
        if self.vlm_box_rescue and not self.auto_start_sequence:
            raise ValueError("VLM box rescue requires automatic sequence labeling")
        if self.strategy == "all_frames":
            if self.source_fps is None:
                raise ValueError("all_frames requires source_fps")
            if self.target_fps is not None and self.target_fps != self.source_fps:
                raise ValueError("all_frames cannot reduce source FPS")
        if self.strategy == "uniform":
            if self.source_fps is None:
                raise ValueError("uniform sampling requires source_fps")
            if self.target_fps is None:
                raise ValueError("uniform sampling requires target_fps")
            if self.target_fps > self.source_fps:
                raise ValueError("target_fps cannot exceed source_fps")
        return self


class BoundingBox2D(StrictContract):
    xtl: float = Field(ge=0, allow_inf_nan=False)
    ytl: float = Field(ge=0, allow_inf_nan=False)
    xbr: float = Field(gt=0, allow_inf_nan=False)
    ybr: float = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def corners_are_ordered(self) -> BoundingBox2D:
        if self.xbr <= self.xtl or self.ybr <= self.ytl:
            raise ValueError("rectangle bottom-right must be below and right of top-left")
        return self


class SeedRectangle(StrictContract):
    seed_id: str = Field(min_length=1, max_length=128)
    object_id: str = Field(min_length=1, max_length=128)
    source: Literal["human_seed", "auto_seed"] = "human_seed"
    frame_index: Literal[0] = 0
    label_id: int = Field(ge=1)
    label_name: str = Field(min_length=1, max_length=128)
    shape_id: int = Field(ge=0)
    track_id: int | None = Field(default=None, ge=0)
    bbox: BoundingBox2D
    occluded: bool = False


class SeedSet(StrictContract):
    schema_id: Literal["autolabel-seed-set/v1"] = Field(default="autolabel-seed-set/v1", alias="schema")
    job_id: str = Field(pattern=UUID_PATTERN)
    cvat_job_id: int = Field(ge=1)
    annotation_revision: int = Field(ge=0)
    frame_index: Literal[0] = 0
    seeds: list[SeedRectangle] = Field(min_length=1)

    @model_validator(mode="after")
    def identities_are_unique(self) -> SeedSet:
        identities = [seed.object_id for seed in self.seeds]
        if len(identities) != len(set(identities)):
            raise ValueError("seed object identities must be unique")
        return self


class AttachSeedRequest(StrictContract):
    cvat_job_id: int = Field(ge=1)
    annotation_revision: int = Field(ge=0)


class CvatAttributeSpec(StrictContract):
    name: str = Field(min_length=1, max_length=128)
    mutable: bool
    input_type: Literal["select", "radio", "checkbox", "text", "number"]
    default_value: str = Field(max_length=256)
    values: list[str]


class CvatLabelSpec(StrictContract):
    name: str = Field(min_length=1, max_length=128)
    color: str | None = Field(default=None, pattern=r"^#[0-9A-Fa-f]{6}$")
    type: Literal["rectangle"] = "rectangle"
    attributes: list[CvatAttributeSpec] | None = None


class CreateCvatTaskRequest(StrictContract):
    task_name: str | None = Field(default=None, min_length=1, max_length=256)
    labels: list[CvatLabelSpec] = Field(min_length=1, max_length=100)


class CvatTaskBinding(StrictContract):
    schema_id: Literal["autolabel-cvat-task-binding/v1"] = Field(
        default="autolabel-cvat-task-binding/v1", alias="schema"
    )
    job_id: str = Field(pattern=UUID_PATTERN)
    cvat_task_id: int = Field(ge=1)
    cvat_job_id: int = Field(ge=1)
    manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    frame_count: int = Field(ge=1)


class TrackObservation(StrictContract):
    frame_index: int = Field(ge=0)
    bbox: BoundingBox2D
    score: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    mask_area_px: int | None = Field(default=None, ge=0)
    occluded: bool = False
    outside: bool = False
    truncated: bool = False


class VideoTrack(StrictContract):
    track_id: str = Field(min_length=1, max_length=128)
    label_id: int = Field(ge=1)
    label_name: str = Field(min_length=1, max_length=128)
    source: Literal["human_seed", "auto"]
    review_status: Literal["pending", "accepted", "rejected"] = "pending"
    needs_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)
    observations: list[TrackObservation] = Field(min_length=1)


class TrackSet(StrictContract):
    schema_id: Literal["autolabel-track-set/v1"] = Field(default="autolabel-track-set/v1", alias="schema")
    job_id: str = Field(pattern=UUID_PATTERN)
    manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    seed_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    prompt_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    component_versions: list[ComponentVersion] = Field(min_length=1)
    tracks: list[VideoTrack]

    @model_validator(mode="after")
    def track_identities_are_unique(self) -> TrackSet:
        if (self.seed_sha256 is None) == (self.prompt_sha256 is None):
            raise ValueError("track set must have exactly one seed or text-prompt provenance hash")
        identities = [track.track_id for track in self.tracks]
        if len(identities) != len(set(identities)):
            raise ValueError("track identities must be unique")
        for track in self.tracks:
            frames = [observation.frame_index for observation in track.observations]
            if frames != sorted(set(frames)):
                raise ValueError(f"track {track.track_id} observations must have unique sorted frames")
        return self


class PropagationFrameInput(StrictContract):
    frame_index: int = Field(ge=0)
    relative_path: str = Field(pattern=r"^frames/frame_[0-9]{6}\.jpg$")
    sha256: str = Field(pattern=SHA256_PATTERN)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class PropagationModelRequest(StrictContract):
    schema_id: Literal["autolabel-propagation-request/v1"] = Field(
        default="autolabel-propagation-request/v1", alias="schema"
    )
    job_id: str = Field(pattern=UUID_PATTERN)
    manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    seed_sha256: str = Field(pattern=SHA256_PATTERN)
    end_frame_index: int = Field(ge=0)
    frames: list[PropagationFrameInput] = Field(min_length=1)
    seed_set: SeedSet

    @model_validator(mode="after")
    def inputs_are_consistent(self) -> PropagationModelRequest:
        if self.seed_set.job_id != self.job_id:
            raise ValueError("seed set belongs to a different video job")
        indices = [frame.frame_index for frame in self.frames]
        if indices != list(range(len(indices))):
            raise ValueError("propagation frame inputs must be contiguous and zero-based")
        if self.end_frame_index >= len(self.frames):
            raise ValueError("end_frame_index exceeds the supplied frame sequence")
        return self


class DetectorModelRequest(StrictContract):
    schema_id: Literal["autolabel-detector-request/v1"] = Field(default="autolabel-detector-request/v1", alias="schema")
    job_id: str = Field(pattern=UUID_PATTERN)
    manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    score_threshold: float = Field(ge=0, le=1, allow_inf_nan=False)
    frames: list[PropagationFrameInput] = Field(min_length=1)


class RawDetection2D(StrictContract):
    frame_index: int = Field(ge=0)
    model_label_id: int = Field(ge=0)
    model_label_name: str = Field(min_length=1, max_length=128)
    bbox: BoundingBox2D
    score: float = Field(ge=0, le=1, allow_inf_nan=False)


class DetectorModelResponse(StrictContract):
    schema_id: Literal["autolabel-detector-response/v1"] = Field(
        default="autolabel-detector-response/v1", alias="schema"
    )
    job_id: str = Field(pattern=UUID_PATTERN)
    manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    component_version: ComponentVersion
    detections: list[RawDetection2D]


class Detection2D(StrictContract):
    frame_index: int = Field(ge=0)
    label_id: int = Field(ge=1)
    label_name: str = Field(min_length=1, max_length=128)
    model_label_name: str = Field(min_length=1, max_length=128)
    bbox: BoundingBox2D
    score: float = Field(ge=0, le=1, allow_inf_nan=False)


class DetectionSet(StrictContract):
    schema_id: Literal["autolabel-detection-set/v1"] = Field(default="autolabel-detection-set/v1", alias="schema")
    job_id: str = Field(pattern=UUID_PATTERN)
    manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    component_version: ComponentVersion
    detections: list[Detection2D]


class DiscoveryEvent(StrictContract):
    event_id: str = Field(min_length=1, max_length=128)
    event_type: Literal[
        "possible_missing_object",
        "lost_seed",
        "unexpected_reappearance",
        "scene_cut_without_seed",
    ]
    frame_index: int = Field(ge=0)
    track_ids: list[str]
    label_name: str | None = Field(default=None, max_length=128)
    bbox: BoundingBox2D | None = None
    score: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    reason: str = Field(min_length=1, max_length=500)


class DiscoveryResult(StrictContract):
    schema_id: Literal["autolabel-discovery-result/v1"] = Field(default="autolabel-discovery-result/v1", alias="schema")
    job_id: str = Field(pattern=UUID_PATTERN)
    manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    propagated_track_sha256: str = Field(pattern=SHA256_PATTERN)
    component_versions: list[ComponentVersion] = Field(min_length=1)
    proposal_tracks: list[VideoTrack]
    events: list[DiscoveryEvent]


class AuditEvidence(StrictContract):
    schema_id: Literal["autolabel-audit-evidence/v1"] = Field(default="autolabel-audit-evidence/v1", alias="schema")
    job_id: str = Field(pattern=UUID_PATTERN)
    frame_index: int = Field(ge=0)
    raw_frame_sha256: str = Field(pattern=SHA256_PATTERN)
    overlay_sha256: str = Field(pattern=SHA256_PATTERN)
    track_ids: list[str]
    verdict: Literal["agree", "possible_missing", "wrong_class", "bad_extent", "uncertain"]
    reason: str = Field(min_length=1, max_length=500)
    component_version: ComponentVersion


class VlmAuditResponse(StrictContract):
    verdict: Literal["agree", "possible_missing", "wrong_class", "bad_extent", "uncertain"]
    affected_track_ids: list[str]
    reason: str = Field(min_length=1, max_length=500)


class AuditResult(StrictContract):
    schema_id: Literal["autolabel-audit-result/v1"] = Field(default="autolabel-audit-result/v1", alias="schema")
    job_id: str = Field(pattern=UUID_PATTERN)
    manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    track_set_sha256: str = Field(pattern=SHA256_PATTERN)
    discovery_sha256: str = Field(pattern=SHA256_PATTERN)
    prompt_version: str = Field(min_length=1, max_length=64)
    component_versions: list[ComponentVersion] = Field(min_length=1)
    evidence: list[AuditEvidence]


class SequenceAuditUsage(StrictContract):
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class SequenceAuditEvidence(StrictContract):
    """One bounded OpenRouter question; a negative region answer only flags review."""

    job_id: str = Field(pattern=UUID_PATTERN)
    frame_index: int = Field(ge=0)
    source_frame_index: int = Field(ge=0)
    audit_type: Literal["full_frame", "region"]
    track_id: str | None = None
    label_name: str | None = None
    bbox: BoundingBox2D | None = None
    missing_decision: Literal["yes", "no", "uncertain"] | None = None
    class_decision: Literal["yes", "no", "uncertain"] | None = None
    extent_decision: Literal["yes", "no", "uncertain"] | None = None
    reason: str = Field(min_length=1, max_length=500)
    model_name: str = Field(min_length=1, max_length=200)
    prompt_version: str = Field(min_length=1, max_length=64)
    timestamp: datetime
    raw_frame_sha256: str = Field(pattern=SHA256_PATTERN)
    visual_sha256: str = Field(pattern=SHA256_PATTERN)
    selection_reasons: list[str] = Field(default_factory=list)
    usage: SequenceAuditUsage = Field(default_factory=SequenceAuditUsage)
    cache_hit: bool = False

    @model_validator(mode="after")
    def decisions_match_question(self) -> SequenceAuditEvidence:
        if self.audit_type == "full_frame":
            if (
                self.track_id is not None
                or self.missing_decision is None
                or self.class_decision is not None
                or self.extent_decision is not None
            ):
                raise ValueError("full-frame evidence needs only a missing decision")
        elif (
            not self.track_id
            or self.bbox is None
            or self.class_decision is None
            or self.extent_decision is None
            or self.missing_decision is not None
        ):
            raise ValueError("region evidence needs a track, box, class and extent decisions")
        return self


class SequenceAuditResult(StrictContract):
    schema_id: Literal["autolabel-sequence-audit/v2"] = Field(default="autolabel-sequence-audit/v2", alias="schema")
    job_id: str = Field(pattern=UUID_PATTERN)
    manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    track_set_sha256: str = Field(pattern=SHA256_PATTERN)
    discovery_sha256: str = Field(pattern=SHA256_PATTERN)
    prompt_version: str = Field(min_length=1, max_length=64)
    model_name: str = Field(min_length=1, max_length=200)
    evidence: list[SequenceAuditEvidence]


class ReviewUnit(StrictContract):
    schema_id: Literal["autolabel-review-unit/v1"] = Field(default="autolabel-review-unit/v1", alias="schema")
    review_unit_id: str = Field(min_length=1, max_length=128)
    job_id: str = Field(pattern=UUID_PATTERN)
    priority: Literal["critical", "high", "normal", "low"]
    review_score: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    frame_start: int = Field(ge=0)
    frame_end: int = Field(ge=0)
    track_ids: list[str]
    reason_codes: list[str] = Field(min_length=1)
    review_status: Literal["pending", "accepted", "rejected"] = "pending"

    @model_validator(mode="after")
    def range_is_ordered(self) -> ReviewUnit:
        if self.frame_end < self.frame_start:
            raise ValueError("frame_end cannot be before frame_start")
        return self


class ReviewQueue(StrictContract):
    schema_id: Literal["autolabel-review-queue/v1"] = Field(default="autolabel-review-queue/v1", alias="schema")
    job_id: str = Field(pattern=UUID_PATTERN)
    track_set_sha256: str = Field(pattern=SHA256_PATTERN)
    discovery_sha256: str = Field(pattern=SHA256_PATTERN)
    audit_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    scoring_version: str | None = Field(default=None, max_length=64)
    units: list[ReviewUnit]


class PublishCvatTracksRequest(StrictContract):
    expected_annotation_revision: int | None = Field(default=None, ge=0)


class AcceptBoxRescueRequest(StrictContract):
    expected_annotation_revision: int = Field(ge=0)


class CvatPublishResult(StrictContract):
    schema_id: Literal["autolabel-cvat-publish-result/v1"] = Field(
        default="autolabel-cvat-publish-result/v1", alias="schema"
    )
    job_id: str = Field(pattern=UUID_PATTERN)
    cvat_job_id: int = Field(ge=1)
    revision_before: int = Field(ge=0)
    revision_after: int = Field(ge=0)
    published_track_count: int = Field(ge=0)
    payload_sha256: str = Field(pattern=SHA256_PATTERN)
