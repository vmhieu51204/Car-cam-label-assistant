"""Portable immutable contracts for offline camera-box evaluation (no computation)."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Annotated, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

CONTRACT_VERSION: Literal["autolabel-evaluation/v1"] = "autolabel-evaluation/v1"
Name = Annotated[str, Field(min_length=1)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Finite = Annotated[float, Field(allow_inf_nan=False)]
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Nonnegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Ordinal = Annotated[int, Field(ge=0)]
Vec3 = tuple[Finite, Finite, Finite]
Quaternion = tuple[Finite, Finite, Finite, Finite]
BBox = tuple[Finite, Finite, Finite, Finite]
RunStatus = Literal["completed", "failed", "timed_out", "unsupported", "skipped_budget", "pending", "invalid_input"]
PredictionView = Literal["raw", "stitched", "interpolated"]
Evidence = Literal["wrapper_declared", "checkpoint_verified", "model_yield_verified", "unavailable"]
Availability = Literal[
    "available",
    "undefined_denominator",
    "unavailable_input",
    "unavailable_dependency",
    "not_requested",
    "invalid_input",
]
ScoreSemantics = Literal["probability", "sigmoid_logit", "fixed_visible", "missing", "inherited_export"]
Cohort = Literal["text_discovery", "seeded_propagation", "human_assisted"]
Camera = Literal["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]


class Contract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        populate_by_name=True,
        revalidate_instances="always",
    )


def _unique(values: tuple[object, ...], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {label}")


def _bbox(box: BBox) -> None:
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("bbox must have positive continuous xyxy extents")


def _relative(path: str) -> None:
    if path.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", path) or ".." in path.replace("\\", "/").split("/"):
        raise ValueError("portable paths must be relative without parent traversal")


class SequenceKey(Contract):
    dataset_id: Literal["nuscenes", "kitti", "coco"]
    split: Name
    sequence_uid: Name
    camera_channel: Camera | None = None

    @model_validator(mode="after")
    def identity(self) -> SequenceKey:
        if self.dataset_id == "kitti" and (
            not re.fullmatch(r"[0-9]{4}", self.sequence_uid) or self.camera_channel is not None
        ):
            raise ValueError("KITTI requires native four-digit sequence and no nuScenes camera")
        if self.dataset_id == "nuscenes" and (
            not re.fullmatch(r"[0-9a-f]{32}", self.sequence_uid) or self.camera_channel is None
        ):
            raise ValueError("nuScenes requires scene token and camera channel")
        return self


class FrameKey(Contract):
    sequence: SequenceKey
    frame_uid: Name

    @model_validator(mode="after")
    def identity(self) -> FrameKey:
        if self.sequence.dataset_id == "kitti" and not re.fullmatch(r"0|[1-9][0-9]*", self.frame_uid):
            raise ValueError("KITTI frame UID is unpadded native decimal index")
        if self.sequence.dataset_id == "nuscenes" and not re.fullmatch(r"[0-9a-f]{32}", self.frame_uid):
            raise ValueError("nuScenes frame UID is sample token")
        return self


class TrackKey(Contract):
    sequence: SequenceKey
    track_id: Name
    config_id: Name | None = None
    window_id: Name | None = None

    @model_validator(mode="after")
    def namespace(self) -> TrackKey:
        if self.window_id is not None and self.config_id is None:
            raise ValueError("window-local tracks require config namespace")
        return self


class RunKey(Contract):
    """A config/view identity without delimiter collisions."""

    config_id: Name
    view: PredictionView


class SourceFile(Contract):
    relative_path: Name
    sha256: Sha256
    role: Literal["image", "labels", "metadata", "result", "payload", "config", "checkpoint", "code", "mapping"]

    @model_validator(mode="after")
    def portable(self) -> SourceFile:
        _relative(self.relative_path)
        return self


class CameraMetadata(Contract):
    sample_data_token: Name
    ego_pose_token: Name
    calibrated_sensor_token: Name
    log_token: Name
    sample_timestamp_us: int
    sensor_timestamp_us: int
    ego_translation: Vec3
    ego_rotation_wxyz: Quaternion
    camera_translation: Vec3
    camera_rotation_wxyz: Quaternion
    camera_intrinsic: tuple[Vec3, Vec3, Vec3]

    @model_validator(mode="after")
    def rotations(self) -> CameraMetadata:
        for rotation in (self.ego_rotation_wxyz, self.camera_rotation_wxyz):
            if abs(sum(x * x for x in rotation) - 1) > 1e-5:
                raise ValueError("camera/ego rotations must be unit quaternions")
        return self


class FrameReference(Contract):
    key: FrameKey
    source_ordinal: Ordinal
    width: Annotated[int, Field(gt=0)]
    height: Annotated[int, Field(gt=0)]
    media: SourceFile
    timestamp_us: int | None = None
    timestamp_provenance: Literal["recorded_sensor", "declared_fps", "unavailable"]
    camera: CameraMetadata | None = None

    @model_validator(mode="after")
    def timestamps(self) -> FrameReference:
        if (self.timestamp_us is None) != (self.timestamp_provenance == "unavailable"):
            raise ValueError("timestamp and provenance disagree")
        if self.key.sequence.dataset_id == "nuscenes" and self.camera is None:
            raise ValueError("nuScenes camera observation metadata is required")
        if self.camera is not None and self.timestamp_us != self.camera.sensor_timestamp_us:
            raise ValueError("camera scoring timestamp must be sensor timestamp")
        return self


class NuScenesAnnotation(Contract):
    annotation_token: Name
    sample_token: Name
    instance_token: Name
    category_token: Name
    category_name: Name
    translation_global: Vec3
    size_wlh: tuple[Positive, Positive, Positive]
    rotation_wxyz: Quaternion
    attribute_tokens: tuple[Name, ...] = ()
    visibility_token: Name
    num_lidar_pts: Ordinal
    num_radar_pts: Ordinal
    prev_token: str = ""
    next_token: str = ""

    @model_validator(mode="after")
    def rotation(self) -> NuScenesAnnotation:
        if abs(sum(x * x for x in self.rotation_wxyz) - 1) > 1e-5:
            raise ValueError("native rotation must be a unit quaternion")
        return self


class NativeKittiRow(Contract):
    frame: FrameKey
    source_id: Name
    source_file: SourceFile
    line_number: Annotated[int, Field(gt=0)]
    source_frame_index: Ordinal
    track_id: int
    native_class: Literal["Car", "Pedestrian", "Van", "Cyclist", "Truck", "Misc", "Person", "Tram", "DontCare"]
    truncated: Finite
    occluded: int
    alpha: Finite
    bbox_xyxy: BBox
    dimensions_hwl: Vec3
    location_xyz: Vec3
    rotation_y: Finite

    @model_validator(mode="after")
    def native_fields(self) -> NativeKittiRow:
        _bbox(self.bbox_xyxy)
        if self.frame.sequence.dataset_id != "kitti" or self.frame.frame_uid != str(self.source_frame_index):
            raise ValueError("native KITTI row frame mismatch")
        if self.native_class == "DontCare":
            if self.track_id != -1:
                raise ValueError("DontCare track ID must be -1")
        elif (
            self.track_id < 0
            or not 0 <= self.truncated <= 2
            or self.occluded not in (0, 1, 2, 3)
            or min(self.dimensions_hwl) <= 0
        ):
            raise ValueError("invalid native object fields")
        return self


class BoxMetadata(Contract):
    nuscenes: NuScenesAnnotation | None = None
    kitti_source_id: Name | None = None
    original_bbox_xyxy: BBox | None = None
    clipping_policy: Name | None = None
    ego_distance_m: Nonnegative | None = None
    camera_visibility_revision: Name | None = None

    @model_validator(mode="after")
    def clipped(self) -> BoxMetadata:
        if self.original_bbox_xyxy is not None:
            _bbox(self.original_bbox_xyxy)
            if self.clipping_policy is None:
                raise ValueError("derived clipping requires a named policy")
        return self


class ReferenceBox(Contract):
    annotation_id: Name
    native_annotation_id: Name
    frame: FrameKey
    class_name: Name
    bbox_xyxy: BBox
    native_track_id: Name | None = None
    source_ordinal: Ordinal
    reference_kind: Literal["native_2d", "projected_3d_box", "human_reviewed_2d_box"]
    native_category: Name
    metadata: BoxMetadata = BoxMetadata()

    @model_validator(mode="after")
    def extent(self) -> ReferenceBox:
        _bbox(self.bbox_xyxy)
        if self.reference_kind == "projected_3d_box" and self.metadata.nuscenes is None:
            raise ValueError("projected boxes require native annotation metadata")
        return self


class IgnoredRegion(Contract):
    source_id: Name
    frame: FrameKey
    role: Literal["distractor", "ignore", "out_of_task"]
    native_category: Name
    bbox_xyxy: BBox
    native_kitti_source_id: Name | None = None
    native_nuscenes: NuScenesAnnotation | None = None

    @model_validator(mode="after")
    def extent(self) -> IgnoredRegion:
        _bbox(self.bbox_xyxy)
        return self


class ReasonCount(Contract):
    reason: Name
    count: Ordinal


class ReferenceRejection(Contract):
    frame: FrameKey
    native_annotation_id: Name
    reason: Name
    policy_revision: Name


class ClassMapping(Contract):
    native_class: Name
    class_name: Name


class ProjectionPolicy(Contract):
    revision: Literal["reject-any-corner-near-plane-hull/v1"] = "reject-any-corner-near-plane-hull/v1"
    min_depth_m: Positive = 0.1
    canvas_policy: Literal["convex_hull_intersection"] = "convex_hull_intersection"


class ReferenceProfile(Contract):
    profile_id: Name
    revision: Name
    designation: Literal["raw", "local_ignore_adjusted", "standard_backend"]
    classes: tuple[Name, ...]
    class_map: tuple[ClassMapping, ...]
    eligibility_policy: Name
    geometry_policy: Name
    projection: ProjectionPolicy | None = None

    @model_validator(mode="after")
    def taxonomy(self) -> ReferenceProfile:
        _unique(self.classes, "classes")
        _unique(tuple(x.native_class for x in self.class_map), "native class map")
        if not self.classes or any(x.class_name not in self.classes for x in self.class_map):
            raise ValueError("class map must target declared classes")
        return self


class DatasetReference(Contract):
    schema_id: Literal["autolabel-evaluation-reference/v1"] = Field(
        default="autolabel-evaluation-reference/v1", alias="schema"
    )
    reference_revision: Name
    frames: tuple[FrameReference, ...]
    boxes: tuple[ReferenceBox, ...] = ()
    ignored_regions: tuple[IgnoredRegion, ...] = ()
    native_kitti_rows: tuple[NativeKittiRow, ...] = ()
    native_nuscenes_annotations: tuple[NuScenesAnnotation, ...] = ()
    sources: tuple[SourceFile, ...] = ()
    profiles: tuple[ReferenceProfile, ...]
    rejections: tuple[ReferenceRejection, ...] = ()
    rejection_counts: tuple[ReasonCount, ...] = ()
    content_sha256: Sha256 | None = None
    generated_at: Name | None = None

    @model_validator(mode="after")
    def joins(self) -> DatasetReference:
        keys = tuple(x.key for x in self.frames)
        _unique(keys, "frame keys")
        _unique(tuple(x.annotation_id for x in self.boxes), "annotation IDs")
        _unique(tuple(x.source_id for x in self.ignored_regions), "ignored region IDs")
        _unique(tuple(x.source_id for x in self.native_kitti_rows), "native row IDs")
        _unique(tuple(x.annotation_token for x in self.native_nuscenes_annotations), "native annotation tokens")
        _unique(tuple((x.key.sequence, x.source_ordinal) for x in self.frames), "sequence frame ordinals")
        _unique(tuple((x.frame, x.source_ordinal) for x in self.boxes), "box ordinals")
        _unique(tuple(x.profile_id for x in self.profiles), "profile IDs")
        _unique(tuple(x.relative_path for x in self.sources), "source paths")
        known = set(keys)
        for item in (*self.boxes, *self.ignored_regions, *self.native_kitti_rows, *self.rejections):
            if item.frame not in known:
                raise ValueError("reference row points to unknown frame")
        objects = tuple(x for x in self.native_kitti_rows if x.native_class != "DontCare")
        _unique(tuple((x.frame, x.track_id) for x in objects), "native object frame/track IDs")
        track_classes: dict[tuple[SequenceKey, int], str] = {}
        for row in objects:
            track_key = (row.frame.sequence, row.track_id)
            if track_key in track_classes and track_classes[track_key] != row.native_class:
                raise ValueError("native KITTI track changes class")
            track_classes[track_key] = row.native_class
        return self


class BindingStep(Contract):
    mapping_id: Name
    domain: Literal["payload", "converted", "selected", "window_local", "native_export"]
    index: Ordinal
    config_id: Name | None = None
    window_id: Name | None = None

    @model_validator(mode="after")
    def namespace(self) -> BindingStep:
        if self.domain == "window_local" and (self.config_id is None or self.window_id is None):
            raise ValueError("window-local indices require config and window namespaces")
        if self.window_id is not None and self.config_id is None:
            raise ValueError("window requires config")
        return self


class FrameBinding(BindingStep):
    frame: FrameKey
    upstream: tuple[BindingStep, ...] = ()

    @model_validator(mode="after")
    def chain(self) -> FrameBinding:
        identities = tuple((x.mapping_id, x.domain, x.index, x.config_id, x.window_id) for x in self.upstream)
        _unique(identities, "binding steps")
        _unique(
            tuple((x.mapping_id, x.domain, x.config_id, x.window_id) for x in self.upstream), "binding chain domains"
        )
        if (self.mapping_id, self.domain, self.index, self.config_id, self.window_id) in identities:
            raise ValueError("binding composition cycle")
        return self


class BindingOptions(Contract):
    input_kind: Literal["sequence-native", "public-converted", "nuscenes-native", "coco-explicit"]
    sequence: SequenceKey
    mapping_id: Name
    require_exhaustive: Literal[True] = True
    endpoint_policy: Literal["sampled_only", "include_final_source_frame", "explicit"]


class StitchOptions(Contract):
    min_overlap_iou: Probability = 0.5
    ambiguity_margin: Probability = 0.1
    min_common_frames: Annotated[int, Field(ge=1)] = 2
    revision: Literal["shared-stitch-config/v1"] = "shared-stitch-config/v1"


class InterpolationPolicy(Contract):
    revision: Name
    algorithm: Literal["none", "legacy_cvat_linear", "video_optimization_linear"]
    max_gap_source_frames: Ordinal | None = None
    outside_policy: Literal["terminate", "preserve_state"] = "terminate"
    endpoint_policy: Literal["observations_only", "legacy_final_extent"] = "observations_only"


class ResetBoundary(Contract):
    config_id: Name
    frame: FrameKey
    mapping_id: Name
    source: Literal["config_declared", "payload_declared"]
    window_id: Name | None = None


class PredictionNormalizationOptions(Contract):
    revision: Literal["sam3-normalization/v1"] = "sam3-normalization/v1"
    views: tuple[PredictionView, ...] = ("raw", "stitched")
    stitch: StitchOptions = StitchOptions()
    scene_reset_policy: Literal["sequence_and_declared_cuts"] = "sequence_and_declared_cuts"
    raw_overlap_geometry: Literal["first_window"] = "first_window"
    stitched_overlap_geometry: Literal["last_window", "identity_only"] = "last_window"
    identity_geometry_view: Literal["first_window", "last_window"] | None = None
    interpolation: InterpolationPolicy = InterpolationPolicy(revision="none/v1", algorithm="none")
    bounds_policy: Literal["preserve_and_flag", "reject"] = "preserve_and_flag"
    unknown_class_policy: Literal["reject", "count_out_of_task"] = "reject"
    reset_boundaries: tuple[ResetBoundary, ...] = ()

    @model_validator(mode="after")
    def policies(self) -> PredictionNormalizationOptions:
        _unique(self.views, "prediction views")
        _unique(tuple((x.config_id, x.frame) for x in self.reset_boundaries), "reset boundaries")
        _unique(
            tuple((x.config_id, x.window_id) for x in self.reset_boundaries if x.window_id is not None),
            "window reset assignments",
        )
        if not self.views:
            raise ValueError("at least one view required")
        if (self.stitched_overlap_geometry == "identity_only") != (self.identity_geometry_view is not None):
            raise ValueError("identity-only stitching requires fixed geometry view")
        if "interpolated" in self.views and self.interpolation.algorithm == "none":
            raise ValueError("interpolated view requires named interpolation")
        return self


class Score(Contract):
    value: Probability | None
    semantics: ScoreSemantics
    source: Name

    @model_validator(mode="after")
    def missing(self) -> Score:
        if (self.value is None) != (self.semantics == "missing"):
            raise ValueError("null scores must have missing semantics")
        if self.semantics == "fixed_visible" and self.value != 1:
            raise ValueError("production fixed-visible score must be 1")
        return self


class MaskReference(Contract):
    relative_path: Name
    sha256: Sha256
    format: Literal["row_major_uncompressed_rle/v1"] = "row_major_uncompressed_rle/v1"
    width: Annotated[int, Field(gt=0)]
    height: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def portable(self) -> MaskReference:
        _relative(self.relative_path)
        return self


class ObservationSource(Contract):
    mapping_id: Name
    source_ordinal: Ordinal
    config_id: Name
    window_id: Name | None = None
    source_observation_id: Name


class PredictionBox(Contract):
    prediction_id: Name
    frame: FrameKey
    class_name: Name
    bbox_xyxy: BBox
    raw_track_id: Name
    stitched_track_id: Name | None = None
    source: ObservationSource
    scoring_ordinal: Ordinal
    score: Score
    origin: Literal["model", "interpolated", "seed", "export_unknown"]
    mask: MaskReference | None = None
    interpolation_revision: Name | None = None

    @model_validator(mode="after")
    def extent(self) -> PredictionBox:
        _bbox(self.bbox_xyxy)
        if (self.origin == "interpolated") != (self.interpolation_revision is not None):
            raise ValueError("interpolated boxes require algorithm revision")
        return self


class TrackStateObservation(Contract):
    observation_id: Name
    frame: FrameKey
    track: TrackKey
    class_name: Name
    source: ObservationSource
    outside: bool
    bbox_xyxy: BBox | None = None
    score: Score

    @model_validator(mode="after")
    def state(self) -> TrackStateObservation:
        if self.track.sequence != self.frame.sequence:
            raise ValueError("track and frame namespaces differ")
        if self.bbox_xyxy is not None:
            _bbox(self.bbox_xyxy)
        if not self.outside and self.bbox_xyxy is None:
            raise ValueError("visible state requires geometry")
        return self


class FrameProcessing(Contract):
    frame: FrameKey
    selected: bool
    state: Literal["processed", "not_processed", "unknown"]
    evidence: Evidence
    mapping_id: Name
    window_ids: tuple[Name, ...] = ()

    @model_validator(mode="after")
    def evidence_state(self) -> FrameProcessing:
        _unique(self.window_ids, "processing windows")
        if self.state == "processed" and self.evidence == "unavailable":
            raise ValueError("processed frames require processing evidence")
        if self.state == "unknown" and self.evidence != "unavailable":
            raise ValueError("unknown processing cannot assert evidence")
        return self


class RuntimeRecord(Contract):
    load_s: Nonnegative | None = None
    warmup_s: Nonnegative | None = None
    inference_s: Nonnegative | None = None
    transfer_s: Nonnegative | None = None
    stitching_s: Nonnegative | None = None
    end_to_end_s: Nonnegative | None = None
    peak_gpu_memory_bytes: Ordinal | None = None
    memory_measurement: Name | None = None
    hardware: Name | None = None
    precision: Name | None = None


class WindowOutcome(Contract):
    window_id: Name
    status: RunStatus
    checkpoint_sha256: Sha256 | None = None
    retries: Ordinal = 0
    resumed: bool = False
    failure_reason: Name | None = None


class RunProvenance(Contract):
    result_sha256: Sha256
    payload_sha256: Sha256 | None = None
    config_sha256: Sha256
    checkpoint_sha256: Sha256 | None = None
    model_revision: Name
    evaluator_revision: Name
    inference_variant: Name
    random_seeds: tuple[int, ...] = ()
    input_resize: tuple[Annotated[int, Field(gt=0)], Annotated[int, Field(gt=0)]] | None = None


class PayloadSeed(Contract):
    seed_id: Name
    model_seed_id: Name
    frame: FrameKey
    class_name: Name
    bbox_xyxy: BBox
    mapping_id: Name

    @model_validator(mode="after")
    def extent(self) -> PayloadSeed:
        _bbox(self.bbox_xyxy)
        return self


class PredictionRun(Contract):
    schema_id: Literal["autolabel-evaluation-prediction/v1"] = Field(
        default="autolabel-evaluation-prediction/v1", alias="schema"
    )
    config_id: Name
    view: PredictionView
    status: RunStatus
    raw_status: RunStatus
    processing: tuple[FrameProcessing, ...]
    boxes: tuple[PredictionBox, ...] = ()
    track_states: tuple[TrackStateObservation, ...] = ()
    payload_seeds: tuple[PayloadSeed, ...] = ()
    normalization: PredictionNormalizationOptions
    normalization_sha256: Sha256
    provenance: RunProvenance
    windows: tuple[WindowOutcome, ...] = ()
    runtime: RuntimeRecord = RuntimeRecord()
    failure_reason: Name | None = None
    warnings: tuple[Name, ...] = ()
    content_sha256: Sha256 | None = None
    generated_at: Name | None = None

    @model_validator(mode="after")
    def rows(self) -> PredictionRun:
        _unique(tuple(x.frame for x in self.processing), "processing frame keys")
        _unique(tuple(x.prediction_id for x in self.boxes), "prediction IDs")
        _unique(tuple((x.frame, x.scoring_ordinal) for x in self.boxes), "prediction scoring ordinals")
        _unique(tuple(x.observation_id for x in self.track_states), "track-state IDs")
        _unique(tuple(x.window_id for x in self.windows), "window IDs")
        _unique(tuple(x.seed_id for x in self.payload_seeds), "payload seed IDs")
        known = {x.frame for x in self.processing}
        for row in (*self.boxes, *self.track_states):
            if row.frame not in known or row.source.config_id != self.config_id:
                raise ValueError("prediction lacks frame record or config identity")
        if self.view not in self.normalization.views:
            raise ValueError("prediction view not frozen in normalization options")
        if self.normalization_sha256 != content_hash(self.normalization):
            raise ValueError("normalization options hash mismatch")
        if self.status == "completed" and self.raw_status != "completed":
            raise ValueError("normalization cannot promote failed source to completed")
        if self.status in ("failed", "timed_out", "invalid_input") and self.failure_reason is None:
            raise ValueError("failed runs require a reason")
        return self


class SeedReference(PayloadSeed):
    reference_annotation_id: Name
    reference_track: TrackKey
    reference_revision: Name
    review_revision: Name
    source_sha256: Sha256

    @model_validator(mode="after")
    def seed_namespace(self) -> SeedReference:
        if self.reference_track.sequence != self.frame.sequence:
            raise ValueError("seed reference track must be in initial frame sequence")
        return self


class SeedSet(Contract):
    revision: Name
    seeds: tuple[SeedReference, ...] = ()
    content_sha256: Sha256 | None = None

    @model_validator(mode="after")
    def keys(self) -> SeedSet:
        _unique(tuple(x.seed_id for x in self.seeds), "seed IDs")
        _unique(tuple((x.frame.sequence, x.model_seed_id) for x in self.seeds), "model seed IDs")
        _unique(tuple(x.reference_track for x in self.seeds), "seed reference tracks")
        return self


class CadencePolicy(Contract):
    sequence: SequenceKey
    source_fps: Positive | None = None
    source_fps_provenance: Literal["recorded_timestamps", "kitti_dataset_fallback", "operator", "unavailable"]
    input_fps: Positive | None = None
    target_fps: Positive | None = None
    sampling_policy: Name
    endpoint_policy: Literal["sampled_only", "include_final_source_frame", "explicit"]

    @model_validator(mode="after")
    def fallback(self) -> CadencePolicy:
        if self.source_fps_provenance in ("kitti_dataset_fallback", "operator") and self.source_fps is None:
            raise ValueError("declared cadence requires FPS")
        if self.source_fps_provenance == "unavailable" and self.source_fps is not None:
            raise ValueError("unavailable cadence cannot contain FPS")
        return self


class MatchingPolicy(Contract):
    revision: Literal["descending-iou-greedy/v1"] = "descending-iou-greedy/v1"
    iou_thresholds: tuple[Probability, ...] = (0.5, 0.7)
    tie_policy: Literal["gt_ordinal_then_prediction_ordinal"] = "gt_ordinal_then_prediction_ordinal"
    threshold_rule: Literal["inclusive"] = "inclusive"


class ArrivalPolicy(Contract):
    revision: Literal["full-reference-distinct-arrivals/v1"] = "full-reference-distinct-arrivals/v1"
    deadline_s: Positive = 1.0
    iou_threshold: Probability = 0.5
    first_appearance: Literal["full_eligible_reference"] = "full_eligible_reference"
    censor_policy: Literal["report_censored_separately"] = "report_censored_separately"


class ReviewPolicy(Contract):
    revision: Name
    budgets: tuple[Probability, ...] = (0.1, 0.2, 0.4)
    rounding: Literal["ceil_capped"] = "ceil_capped"
    tie_policy: Literal["unit_id"] = "unit_id"
    severity_policy: Name
    extent_candidate_min_iou: Probability = 0.1
    random_seeds: tuple[int, ...] = (0,)


class DiagnosticGroupingOptions(Contract):
    dimensions: tuple[Literal["micro", "per_class", "per_sequence", "per_camera", "slice"], ...] = (
        "micro",
        "per_class",
        "per_sequence",
    )
    slices: tuple[Literal["temporal_third", "box_height", "occlusion", "truncation", "ego_distance"], ...] = ()
    slice_revision: Name | None = None
    absent_classes: Literal["include_zero_support"] = "include_zero_support"
    unmatched_prediction_attribution: Literal["prediction_frame_class_and_box", "separate_unattributed"] = (
        "prediction_frame_class_and_box"
    )

    @model_validator(mode="after")
    def grouping(self) -> DiagnosticGroupingOptions:
        _unique(self.dimensions, "grouping dimensions")
        _unique(self.slices, "grouping slices")
        if not self.dimensions:
            raise ValueError("grouping requires dimensions")
        if bool(self.slices) != ("slice" in self.dimensions) or bool(self.slices) != (self.slice_revision is not None):
            raise ValueError("slice grouping requires named slices and frozen revision")
        return self


class BenchmarkManifest(Contract):
    schema_id: Literal["autolabel-evaluation-manifest/v1"] = Field(
        default="autolabel-evaluation-manifest/v1", alias="schema"
    )
    protocol_revision: Name
    reference_revision: Name
    reference_sha256: Sha256
    sequences: tuple[SequenceKey, ...]
    planned_frames: tuple[FrameKey, ...]
    cohort: Cohort
    seed_set: SeedSet
    excluded_seed_frames: tuple[FrameKey, ...] = ()
    profile: ReferenceProfile
    bindings: tuple[FrameBinding, ...] = ()
    normalization: PredictionNormalizationOptions
    cadence: tuple[CadencePolicy, ...]
    matching: MatchingPolicy = MatchingPolicy()
    grouping: DiagnosticGroupingOptions = DiagnosticGroupingOptions()
    arrival: ArrivalPolicy = ArrivalPolicy()
    review: ReviewPolicy | None = None
    sources: tuple[SourceFile, ...] = ()
    evaluator_revision: Name
    content_sha256: Sha256 | None = None
    generated_at: Name | None = None

    @model_validator(mode="after")
    def frozen_scope(self) -> BenchmarkManifest:
        _unique(self.sequences, "manifest sequences")
        _unique(self.planned_frames, "planned frame keys")
        _unique(self.excluded_seed_frames, "seed excluded frames")
        _unique(tuple(x.sequence for x in self.cadence), "cadence sequences")
        _unique(
            tuple((x.mapping_id, x.domain, x.config_id, x.window_id, x.index) for x in self.bindings), "binding indices"
        )
        _unique(
            tuple((x.mapping_id, x.domain, x.config_id, x.window_id, x.frame) for x in self.bindings),
            "binding target frames",
        )
        index_targets: dict[tuple[str, str, str | None, str | None, int], FrameKey] = {}
        target_indices: dict[tuple[str, str, str | None, str | None, FrameKey], int] = {}
        for binding in self.bindings:
            for step in (binding, *binding.upstream):
                index_key = (step.mapping_id, step.domain, step.config_id, step.window_id, step.index)
                target_key = (step.mapping_id, step.domain, step.config_id, step.window_id, binding.frame)
                if index_key in index_targets and index_targets[index_key] != binding.frame:
                    raise ValueError("contradictory binding composition target")
                if target_key in target_indices and target_indices[target_key] != step.index:
                    raise ValueError("contradictory binding composition index")
                index_targets[index_key] = binding.frame
                target_indices[target_key] = step.index
        _unique(tuple(x.relative_path for x in self.sources), "manifest source paths")
        if any(x.sequence not in self.sequences for x in self.planned_frames):
            raise ValueError("planned frame outside declared sequences")
        if any(x not in self.planned_frames for x in self.excluded_seed_frames):
            raise ValueError("seed exclusions outside E")
        if self.cohort == "text_discovery" and (self.seed_set.seeds or self.excluded_seed_frames):
            raise ValueError("discovery cohort cannot use GT seeds")
        for seed in self.seed_set.seeds:
            if seed.frame not in self.planned_frames or seed.frame not in self.excluded_seed_frames:
                raise ValueError("seeds must map to explicitly excluded planned initial frames")
        if any(x.frame.sequence not in self.sequences for x in self.bindings):
            raise ValueError("binding outside declared sequences")
        if set(x.sequence for x in self.cadence) != set(self.sequences):
            raise ValueError("cadence must describe every declared sequence")
        return self


class EvaluationScope(Contract):
    scope_id: Name
    kind: Literal["fixed_set", "observed", "pairwise_common"]
    frames: tuple[FrameKey, ...]
    profile_id: Name
    view: PredictionView
    cohort: Cohort
    target_population: Literal["all_objects", "seeded_objects"]
    excluded_seed_frames: tuple[FrameKey, ...] = ()
    interpolation: InterpolationPolicy
    run_ids: tuple[RunKey, ...]
    eligible_annotation_ids: tuple[Name, ...] = ()

    @model_validator(mode="after")
    def scope(self) -> EvaluationScope:
        _unique(self.frames, "scope frames")
        _unique(self.eligible_annotation_ids, "eligible annotation IDs")
        _unique(self.run_ids, "scope run IDs")
        if set(self.frames) & set(self.excluded_seed_frames):
            raise ValueError("headline frames contain excluded seeds")
        if self.target_population == "seeded_objects" and self.cohort == "text_discovery":
            raise ValueError("discovery cannot have seeded population")
        if self.kind == "pairwise_common" and len(self.run_ids) != 2:
            raise ValueError("pairwise scope requires two runs")
        return self


class ScopeSet(Contract):
    scopes: tuple[EvaluationScope, ...]

    @model_validator(mode="after")
    def keys(self) -> ScopeSet:
        _unique(tuple(x.scope_id for x in self.scopes), "scope IDs")
        return self


class ExcludedRecord(Contract):
    record_id: Name
    kind: Literal["reference", "prediction"]
    frame: FrameKey
    reason: Name


class PreparedEvaluationView(Contract):
    schema_id: Literal["autolabel-evaluation-prepared-view/v1"] = Field(
        default="autolabel-evaluation-prepared-view/v1", alias="schema"
    )
    scope: EvaluationScope
    profile: ReferenceProfile
    reference_boxes: tuple[ReferenceBox, ...]
    prediction_boxes: tuple[PredictionBox, ...]
    ignored_regions: tuple[IgnoredRegion, ...] = ()
    excluded: tuple[ExcludedRecord, ...] = ()
    reference_frames: tuple[FrameReference, ...] = ()
    native_kitti_rows: tuple[NativeKittiRow, ...] = ()

    @model_validator(mode="after")
    def rows(self) -> PreparedEvaluationView:
        _unique(tuple(x.annotation_id for x in self.reference_boxes), "prepared annotation IDs")
        _unique(tuple(x.prediction_id for x in self.prediction_boxes), "prepared prediction IDs")
        _unique(tuple((x.frame, x.source_ordinal) for x in self.reference_boxes), "prepared reference ordinals")
        _unique(
            tuple((x.frame, x.scoring_ordinal) for x in self.prediction_boxes), "prepared prediction scoring ordinals"
        )
        _unique(tuple(x.key for x in self.reference_frames), "prepared frame keys")
        _unique(tuple(x.source_id for x in self.native_kitti_rows), "prepared native row IDs")
        if any(x.frame not in self.scope.frames for x in (*self.reference_boxes, *self.prediction_boxes)):
            raise ValueError("prepared box outside scope")
        if self.profile.profile_id != self.scope.profile_id:
            raise ValueError("prepared profile differs from scope")
        return self


class MatchedPair(Contract):
    annotation_id: Name
    prediction_id: Name
    frame: FrameKey
    iou: Probability


class MatchResult(Contract):
    policy: Literal["descending-iou-greedy/v1"] = "descending-iou-greedy/v1"
    iou_threshold: Probability
    class_aware: bool
    pairs: tuple[MatchedPair, ...]
    unmatched_annotation_ids: tuple[Name, ...]
    unmatched_prediction_ids: tuple[Name, ...]

    @model_validator(mode="after")
    def one_to_one(self) -> MatchResult:
        _unique(
            tuple(x.annotation_id for x in self.pairs) + self.unmatched_annotation_ids, "matched/residual annotations"
        )
        _unique(
            tuple(x.prediction_id for x in self.pairs) + self.unmatched_prediction_ids, "matched/residual predictions"
        )
        if any(x.iou < self.iou_threshold for x in self.pairs):
            raise ValueError("match below inclusive threshold")
        return self


class SupportCount(Contract):
    name: Name
    count: Ordinal


class MetricValue(Contract):
    value: Finite | None
    unit: Name
    scale: Name
    numerator: Finite | None = None
    denominator: Nonnegative | None = None
    support: tuple[SupportCount, ...] = ()
    availability: Availability
    reason: Name | None = None
    definition_id: Name

    @model_validator(mode="after")
    def available(self) -> MetricValue:
        _unique(tuple(x.name for x in self.support), "support names")
        if (self.value is not None) != (self.availability == "available"):
            raise ValueError("available metrics need values; unavailable metrics need null")
        if self.availability != "available" and self.reason is None:
            raise ValueError("null metrics require availability reason")
        if self.availability == "undefined_denominator" and self.denominator != 0:
            raise ValueError("undefined denominator must be explicitly zero")
        return self


class NamedMetric(Contract):
    name: Name
    metric: MetricValue


class DiagnosticMetrics(Contract):
    tp: Ordinal
    fp: Ordinal
    fn: Ordinal
    iou_sum: Nonnegative
    matched_count: Ordinal
    precision: MetricValue
    recall: MetricValue
    f1: MetricValue
    matched_mean_iou: MetricValue


class MetricGroup(Contract):
    group_id: Name
    aggregation: Literal[
        "micro", "per_class", "per_sequence", "per_camera", "slice", "sequence_mean", "upstream_combined"
    ]
    class_name: Name | None = None
    sequence: SequenceKey | None = None
    slice_name: Name | None = None
    metrics: tuple[NamedMetric, ...]
    diagnostic: DiagnosticMetrics | None = None

    @model_validator(mode="after")
    def keys(self) -> MetricGroup:
        _unique(tuple(x.name for x in self.metrics), "metric names")
        return self


class CoverageResult(Contract):
    frame_coverage: MetricValue
    object_observation_coverage: MetricValue
    evidence: tuple[Evidence, ...]
    planned_frames: Ordinal
    processed_frames: Ordinal
    planned_gt_observations: Ordinal
    observed_gt_observations: Ordinal


class ArrivalTrackResult(Contract):
    track: TrackKey
    first_frame: FrameKey
    detected_frame: FrameKey | None
    deadline_source_frames: Ordinal | None
    delay_source_frames: Ordinal | None
    delay_s: Nonnegative | None
    time_provenance: Literal["recorded_sensor", "declared_fps", "unavailable"]
    within_deadline: bool | None
    censored: bool


class ArrivalMetrics(Contract):
    distinct_object_recall: MetricValue
    observation_recall: MetricValue
    median_delay_s: MetricValue
    p90_delay_s: MetricValue
    median_delay_frames: MetricValue
    p90_delay_frames: MetricValue
    eligible_tracks: Ordinal
    never_detected_tracks: Ordinal
    censored_tracks: Ordinal
    tracks: tuple[ArrivalTrackResult, ...] = ()


class TrackEvalOptions(Contract):
    backend_revision: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    environment_revision: Name
    metrics: tuple[Literal["HOTA", "CLEAR", "Identity"], ...] = ("HOTA", "CLEAR", "Identity")
    classes: tuple[Literal["Car", "Pedestrian"], ...] = ("Car", "Pedestrian")
    timeline: Literal["full_native", "custom_sampled"] = "full_native"
    required: bool = False
    scoreless_export_value: Probability = 1.0
    preprocessing_revision: Name
    metric_config_id: Name
    metric_config_sha256: Sha256


class BackendIdBinding(Contract):
    track: TrackKey
    export_id: Ordinal


class BackendFrameBinding(Contract):
    frame: FrameKey
    export_frame_index: Ordinal


class BackendMetric(Contract):
    name: Name
    scalar: Finite | None = None
    threshold_values: tuple[Finite, ...] = ()
    thresholds: tuple[Probability, ...] = ()
    unit: Name
    scale: Name

    @model_validator(mode="after")
    def thresholds_align(self) -> BackendMetric:
        if len(self.threshold_values) != len(self.thresholds):
            raise ValueError("backend threshold/value arrays must align")
        _unique(self.thresholds, "backend thresholds")
        if (self.scalar is not None) == bool(self.threshold_values):
            raise ValueError("native metric requires exactly one scalar or threshold array")
        return self


class BackendMetricGroup(Contract):
    class_name: Name
    sequence: SequenceKey | None = None
    aggregation: Literal["per_sequence", "upstream_combined"]
    metrics: tuple[BackendMetric, ...]
    support: tuple[SupportCount, ...] = ()

    @model_validator(mode="after")
    def keys(self) -> BackendMetricGroup:
        _unique(tuple(x.name for x in self.metrics), "backend metric names")
        _unique(tuple(x.name for x in self.support), "backend support names")
        return self


class BackendResult(Contract):
    backend: Literal["trackeval"] = "trackeval"
    availability: Availability
    scope: EvaluationScope
    timeline: Literal["full_native", "custom_sampled"]
    reason: Name | None = None
    revision: Name | None = None
    options: TrackEvalOptions | None = None
    command: tuple[Name, ...] = ()
    export_sources: tuple[SourceFile, ...] = ()
    id_bindings: tuple[BackendIdBinding, ...] = ()
    frame_bindings: tuple[BackendFrameBinding, ...] = ()
    groups: tuple[BackendMetricGroup, ...] = ()
    metrics: tuple[NamedMetric, ...] = ()

    @model_validator(mode="after")
    def status(self) -> BackendResult:
        _unique(tuple(x.frame for x in self.frame_bindings), "backend frame keys")
        _unique(
            tuple((x.frame.sequence, x.export_frame_index) for x in self.frame_bindings), "backend export frame indices"
        )
        if any(x.frame not in self.scope.frames for x in self.frame_bindings):
            raise ValueError("backend frame binding outside scope")
        if self.options is not None and self.options.timeline != self.timeline:
            raise ValueError("backend timeline contradicts frozen options")
        _unique(tuple(x.name for x in self.metrics), "backend metric names")
        _unique(tuple((x.class_name, x.sequence, x.aggregation) for x in self.groups), "backend metric groups")
        _unique(tuple(x.track for x in self.id_bindings), "backend track IDs")
        _unique(tuple((x.track.sequence, x.export_id) for x in self.id_bindings), "backend export IDs")
        if self.availability != "available" and self.reason is None:
            raise ValueError("unavailable backend requires reason")
        if self.availability != "available" and self.groups:
            raise ValueError("unavailable backend cannot assert native results")
        return self


class ScopeEvaluation(Contract):
    scope: EvaluationScope
    coverage: CoverageResult
    matches: tuple[MatchResult, ...]
    groups: tuple[MetricGroup, ...]
    arrivals: ArrivalMetrics | None = None
    prepared_view: PreparedEvaluationView | None = None

    @model_validator(mode="after")
    def keys(self) -> ScopeEvaluation:
        _unique(tuple(x.group_id for x in self.groups), "scope metric groups")
        _unique(tuple((x.policy, x.iou_threshold, x.class_aware) for x in self.matches), "scope match policies")
        return self


class RunEvaluation(Contract):
    config_id: Name
    view: PredictionView
    cohort: Cohort
    status: RunStatus
    raw_status: RunStatus
    provenance: RunProvenance
    runtime: RuntimeRecord
    scopes: tuple[ScopeEvaluation, ...]
    processing: tuple[FrameProcessing, ...]
    windows: tuple[WindowOutcome, ...]
    normalization: PredictionNormalizationOptions
    normalization_sha256: Sha256
    reference_frames: tuple[FrameReference, ...] = ()
    native_kitti_rows: tuple[NativeKittiRow, ...] = ()
    backend: BackendResult | None = None
    ignore_counts: tuple[ReasonCount, ...] = ()
    warnings: tuple[Name, ...] = ()
    failure_reason: Name | None = None

    @model_validator(mode="after")
    def keys(self) -> RunEvaluation:
        _unique(tuple(x.frame for x in self.processing), "evaluation processing frames")
        _unique(tuple(x.window_id for x in self.windows), "evaluation window IDs")
        if self.normalization_sha256 != content_hash(self.normalization):
            raise ValueError("evaluation normalization hash mismatch")
        if self.view not in self.normalization.views:
            raise ValueError("evaluation view absent from frozen options")
        _unique(tuple(x.scope.scope_id for x in self.scopes), "run scopes")
        _unique(tuple(x.key for x in self.reference_frames), "evaluation media frame keys")
        _unique(tuple(x.source_id for x in self.native_kitti_rows), "evaluation native row IDs")
        _unique(tuple(x.reason for x in self.ignore_counts), "ignore reasons")
        return self


class PairedComparison(Contract):
    left_config_id: Name
    right_config_id: Name
    scope: EvaluationScope
    availability: Availability
    reason: Name | None = None
    left_groups: tuple[MetricGroup, ...] = ()
    right_groups: tuple[MetricGroup, ...] = ()
    left_status: RunStatus
    right_status: RunStatus


class EvaluationContext(Contract):
    run: RunKey
    scope_id: Name
    profile_id: Name
    target_population: Literal["all_objects", "seeded_objects"]


class ErrorEvent(Contract):
    error_id: Name
    evaluation_context: EvaluationContext | None = None
    category: Literal[
        "miss", "extra", "duplicate", "wrong_class", "extent_candidate", "identity_switch", "fragmentation"
    ]
    frame: FrameKey
    annotation_ids: tuple[Name, ...] = ()
    prediction_ids: tuple[Name, ...] = ()
    iou_threshold: Probability
    severity: Literal["unrated", "minor", "severe"]
    severity_policy: Name
    origin: Literal["automatic_diagnostic", "injected", "human_confirmed"]
    reference_revision: Name
    review_revision: Name | None = None
    evidence_path: Name | None = None
    ignore_context_ids: tuple[Name, ...] = ()


class ReviewUnit(Contract):
    unit_id: Name
    kind: Literal["frame", "region", "track"]
    frames: tuple[FrameKey, ...]
    bbox_xyxy: BBox | None = None
    track: TrackKey | None = None
    confirmed_error_ids: tuple[Name, ...]
    clean: bool
    annotator: Name

    @model_validator(mode="after")
    def unit(self) -> ReviewUnit:
        _unique(self.frames, "review span frames")
        if not self.frames or self.clean == bool(self.confirmed_error_ids):
            raise ValueError("review unit must be clean or contain confirmed errors")
        if self.kind == "region" and self.bbox_xyxy is None:
            raise ValueError("region review needs bbox")
        if self.bbox_xyxy is not None:
            _bbox(self.bbox_xyxy)
        if self.kind == "track" and self.track is None:
            raise ValueError("track review needs identity")
        return self


class RankingInput(Contract):
    unit_id: Name
    method: Literal["confidence", "cheap_review", "vlm", "random"]
    score: Finite | None
    verdict: Literal["error", "clean", "uncertain"] | None = None
    random_seed: int | None = None


class ReviewTruth(Contract):
    schema_id: Literal["autolabel-evaluation-review-truth/v1"] = Field(
        default="autolabel-evaluation-review-truth/v1", alias="schema"
    )
    revision: Name
    reference_revision: Name
    severity_policy: Name
    units: tuple[ReviewUnit, ...]
    errors: tuple[ErrorEvent, ...]
    rankings: tuple[RankingInput, ...] = ()
    content_sha256: Sha256 | None = None

    @model_validator(mode="after")
    def independent(self) -> ReviewTruth:
        _unique(tuple(x.unit_id for x in self.units), "review unit IDs")
        _unique(tuple(x.error_id for x in self.errors), "review error IDs")
        known_errors = {x.error_id for x in self.errors}
        if any(x.origin != "human_confirmed" for x in self.errors):
            raise ValueError("review truth requires independently human-confirmed errors")
        if any(y not in known_errors for x in self.units for y in x.confirmed_error_ids):
            raise ValueError("review unit links unknown error")
        if any(x.unit_id not in {u.unit_id for u in self.units} for x in self.rankings):
            raise ValueError("ranking links unknown unit")
        return self


class ReviewBudgetResult(Contract):
    method: Name
    budget: Probability
    reviewed_units: Ordinal
    metrics: tuple[NamedMetric, ...]
    unresolved_errors: Ordinal
    severe_unresolved_errors: Ordinal
    ranking_coverage: MetricValue
    random_seed: int | None = None


class ExampleArtifact(Contract):
    error_id: Name
    frame: FrameKey
    view: PredictionView
    media: SourceFile
    width: Annotated[int, Field(gt=0)]
    height: Annotated[int, Field(gt=0)]
    relative_path: Name | None
    availability: Availability
    reason: Name | None = None


class EvaluationReport(Contract):
    schema_id: Literal["autolabel-evaluation-report/v1"] = Field(
        default="autolabel-evaluation-report/v1", alias="schema"
    )
    contract_version: Literal["autolabel-evaluation/v1"] = CONTRACT_VERSION
    protocol_revision: Name
    manifest_sha256: Sha256
    reference_sha256: Sha256
    evaluations: tuple[RunEvaluation, ...]
    paired_comparisons: tuple[PairedComparison, ...] = ()
    errors: tuple[ErrorEvent, ...] = ()
    examples: tuple[ExampleArtifact, ...] = ()
    review: tuple[ReviewBudgetResult, ...] = ()
    optional_metrics: tuple[NamedMetric, ...] = ()
    warnings: tuple[Name, ...] = ()
    content_sha256: Sha256 | None = None
    generated_at: Name | None = None

    @model_validator(mode="after")
    def keys(self) -> EvaluationReport:
        _unique(tuple((x.config_id, x.view) for x in self.evaluations), "report evaluations")
        _unique(tuple(x.error_id for x in self.errors), "report errors")
        _unique(tuple(x.name for x in self.optional_metrics), "optional metric names")
        _unique(
            tuple((x.left_config_id, x.right_config_id, x.scope.scope_id) for x in self.paired_comparisons),
            "paired comparisons",
        )
        return self


class ReportArtifacts(Contract):
    json_path: Name
    csv_path: Name
    examples_index_path: Name | None = None
    example_paths: tuple[Name, ...] = ()
    warnings: tuple[Name, ...] = ()


class ExampleOptions(Contract):
    examples_per_category: Annotated[int, Field(ge=0, le=100)] = 5
    max_total_examples: Annotated[int, Field(ge=0, le=1000)] = 100
    selection_revision: Literal["category-sequence-error-id/v1"] = "category-sequence-error-id/v1"


class CocoImageAssociation(Contract):
    image_id: int | Name
    frame: FrameReference


class CocoReferenceOptions(Contract):
    schema_id: Literal["autolabel-evaluation-coco-options/v1"] = Field(
        default="autolabel-evaluation-coco-options/v1", alias="schema"
    )
    reference_revision: Name
    profile: ReferenceProfile
    images: tuple[CocoImageAssociation, ...]

    @model_validator(mode="after")
    def images_are_explicit(self) -> CocoReferenceOptions:
        _unique(tuple(x.image_id for x in self.images), "COCO image IDs")
        _unique(tuple(x.frame.key for x in self.images), "COCO canonical frames")
        return self


class NuScenesReferenceOptions(Contract):
    version: Name = "v1.0-mini"
    split: Name
    scenes: tuple[Name, ...]
    cameras: tuple[Camera, ...] = ("CAM_FRONT",)
    reference_revision: Name
    profile: ReferenceProfile
    projection: ProjectionPolicy = ProjectionPolicy()


class KittiReferenceOptions(Contract):
    split: Literal["training", "testing"] = "training"
    sequences: tuple[Annotated[str, Field(pattern=r"^[0-9]{4}$")], ...]
    reference_revision: Name
    profiles: tuple[ReferenceProfile, ...]
    source_fps: Positive | None = None
    timestamp_provenance: Literal["declared_fps", "unavailable"] = "unavailable"


TContract = TypeVar("TContract", bound=Contract)


def canonical_json(model: Contract, *, logical: bool = False) -> bytes:
    """Stable UTF-8 JSON; logical hashing excludes self hashes and generated time."""

    def clean(value: object) -> object:
        if isinstance(value, dict):
            return {
                key: clean(item)
                for key, item in value.items()
                if not logical or key not in ("content_sha256", "generated_at")
            }
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    return json.dumps(
        clean(model.model_dump(mode="json", by_alias=True)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def content_hash(model: Contract) -> str:
    return hashlib.sha256(canonical_json(model, logical=True)).hexdigest()


def parse_document(model_type: type[TContract], data: str | bytes) -> TContract:
    """Reject duplicate JSON object keys before strict schema/version validation."""

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    json.loads(data, object_pairs_hook=pairs)
    return model_type.model_validate_json(data)
