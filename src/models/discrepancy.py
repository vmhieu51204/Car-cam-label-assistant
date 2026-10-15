"""Evidence contracts for HCL. None means unavailable, never agreement."""

from typing import Literal

from pydantic import Field, model_validator

from src.models.perception import Box3D, Finite, Positive, Probability, SemanticVerdict, StrictModel

SignalName = Literal["detector", "points", "projection", "dimensions", "temporal", "semantic"]
Status = Literal["invalid", "insufficient_evidence", "ambiguous", "discrepancy", "consistent_candidate"]


class DetectorEvidence(StrictModel):
    raw_score: Probability
    probability: Probability | None = None
    calibration_id: str | None = None


class PointEvidence(StrictModel):
    robust_z: Finite
    calibration_id: str = Field(min_length=1)
    reliability: Probability = 1.0


class ProjectionEvidence(StrictModel):
    iou: Probability
    calibration_id: str = Field(min_length=1)
    calibration_valid: bool = False
    reliability: Probability = 1.0
    source: Literal["2d_detector", "mask_envelope"] = "2d_detector"


class DimensionEvidence(StrictModel):
    size_lwh: tuple[Positive, Positive, Positive]
    source: Literal["vlm_estimate", "verified_catalogue"] = "vlm_estimate"
    source_id: str = Field(min_length=1)
    identifiable: bool = False
    modified: bool = False
    reliability: Probability = 1.0


class SemanticEvidence(StrictModel):
    verdict: SemanticVerdict
    model_id: str = Field(min_length=1)
    reliability: Probability = 1.0


class FrameEvidence(StrictModel):
    frame_index: int = Field(ge=0)
    timestamp_s: Finite
    label: str
    box: Box3D
    detector: DetectorEvidence | None = None
    points: PointEvidence | None = None
    projection: ProjectionEvidence | None = None
    dimensions: DimensionEvidence | None = None
    semantic: SemanticEvidence | None = None
    vlm_status: Literal["not_requested", "completed", "failed", "abstained"] = "not_requested"
    occlusion: Literal["none", "partial", "heavy", "unknown"] = "unknown"
    timestamps_aligned: bool = True

    @model_validator(mode="after")
    def validate_stage(self) -> "FrameEvidence":
        """Prevent failed or unrequested VLM stages from injecting stale evidence."""
        if (self.dimensions or self.semantic) and self.vlm_status != "completed":
            raise ValueError("Semantic/dimension evidence requires vlm_status=completed")
        return self


class TrackEvidence(StrictModel):
    track_id: str = Field(min_length=1)
    label: str
    data_source: Literal["mock", "real"]
    frames: list[FrameEvidence] = Field(min_length=1, max_length=1000)


class SignalWeights(StrictModel):
    detector: Positive = 0.15
    points: Positive = 0.15
    projection: Positive = 0.20
    dimensions: Positive = 0.20
    temporal: Positive = 0.20
    semantic: Positive = 0.10


class HCLConfig(StrictModel):
    version: str = "hcl-v1-experimental"
    minimum_coverage: Probability = 0.50
    vlm_escalation_low: Probability = 0.15
    review_threshold: Probability = 0.30
    max_blend: Probability = 0.60
    high_detector_probability: Probability = 0.95
    weights: SignalWeights = Field(default_factory=SignalWeights)
    dimension_log_tolerance: tuple[Positive, Positive, Positive] = (0.20, 0.15, 0.15)
    point_z_min: Finite = -1.0
    point_z_span: Positive = 2.0
    temporal_yaw_tolerance: Positive = 0.5
    temporal_size_tolerance: Positive = 0.20
    temporal_speed_mps: dict[str, Positive] = Field(
        default_factory=lambda: {"car": 20.0, "truck": 15.0, "pedestrian": 3.0}
    )

    @model_validator(mode="after")
    def check_thresholds(self) -> "HCLConfig":
        """Reject nonsensical thresholds rather than producing silent policy changes."""
        if not 0 < self.vlm_escalation_low < self.review_threshold <= 1:
            raise ValueError("Require 0 < vlm_escalation_low < review_threshold <= 1")
        if self.minimum_coverage == 0:
            raise ValueError("minimum_coverage must be positive")
        if set(self.temporal_speed_mps) != {"car", "truck", "pedestrian"}:
            raise ValueError("Supply speed priors for car, truck and pedestrian")
        return self


class SignalResult(StrictModel):
    available: bool
    value: Probability | None = None
    reliability: Probability = 0.0
    reason: str
    provenance: str | None = None


class FrameResult(StrictModel):
    frame_index: int
    discrepancy_score: Probability | None
    evidence_coverage: Probability
    signals: dict[str, SignalResult]


class DiscrepancyReason(StrictModel):
    code: str
    frame_index: int | None = None
    score: Probability | None = None
    detail: str


class DiscrepancyResult(StrictModel):
    track_id: str
    data_source: Literal["mock", "real"]
    discrepancy_score: Probability | None
    evidence_coverage: Probability
    status: Status
    suggested_action: Literal["review", "request_vlm", "exclude"]
    p_correct: None = None
    review_status: Literal["pending"] = "pending"
    needs_vlm: bool = False
    worst_frame: int | None
    reasons: list[DiscrepancyReason]
    hard_flags: list[str]
    frame_results: list[FrameResult]
    config_version: str
    config_digest: str
    config: HCLConfig
    thresholds_calibrated: Literal[False] = False
