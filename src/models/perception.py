from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator

Finite = Annotated[float, Field(allow_inf_nan=False)]
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Box3D(StrictModel):
    # Internal upright box: local x follows length, local y follows width.
    # Yaw is counterclockwise around global +z. This is NOT a CVAT mapping.
    center: tuple[Finite, Finite, Finite]
    size_lwh: tuple[Positive, Positive, Positive]
    yaw: Finite
    coordinate_frame: Literal["nuscenes_global"] = "nuscenes_global"


class Detection(StrictModel):
    label: str
    box: Box3D
    confidence: Probability


class Frame(StrictModel):
    scene: str
    sample_token: str
    frame_index: int = Field(ge=0)
    timestamp_s: Finite
    detections: list[Detection]


class Observation(StrictModel):
    sample_token: str
    frame_index: int
    timestamp_s: Finite
    box: Box3D
    confidence: Probability


class Track(StrictModel):
    track_id: str
    scene: str
    label: str
    confidence: Probability
    p_correct: Probability | None = None
    routing: Literal["auto-accept", "review", "drop"] = "review"
    source: Literal["auto"] = "auto"
    review_status: Literal["pending", "accepted", "rejected"] = "pending"
    data_source: Literal["mock", "real"] = "mock"
    observations: list[Observation]
    discrepancy: dict | None = None


class DetectRequest(StrictModel):
    scene: str = Field(default="scene-0061", min_length=1)
    max_frames: int = Field(default=5, ge=1, le=1000)


class QueueQuery(StrictModel):
    run_id: str = Field(min_length=1)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=100, ge=1, le=500)
    order: Literal["hcl", "track_id"] = "hcl"


class TelemetryRequest(StrictModel):
    event_id: str = Field(min_length=1, max_length=128)
    run_id: str
    track_id: str
    event_type: Literal["review_opened", "edit_recorded", "review_closed"]
    duration_ms: int = Field(ge=0)
    edit_count: int = Field(default=0, ge=0)


class ValidationItem(StrictModel):
    frame_token: str = Field(min_length=1, max_length=128)
    box_id: str = Field(min_length=1, max_length=128)
    expected_class: Literal["car", "truck", "pedestrian"]
    crop_data_url: str | None = Field(default=None, max_length=4_000_000)
    context_data_url: str | None = Field(default=None, max_length=4_000_000)

    @field_validator("crop_data_url", "context_data_url")
    @classmethod
    def require_image_data_url(cls, value: str | None) -> str | None:
        """Accept bounded inline PNG/JPEG content only."""
        if value is not None:
            import base64

            if not value.startswith(("data:image/jpeg;base64,", "data:image/png;base64,")):
                raise ValueError("Only inline PNG/JPEG images are supported")
            try:
                base64.b64decode(value.split(",", 1)[1], validate=True)
            except ValueError as error:
                raise ValueError("Invalid base64 image") from error
        return value


class ValidateRequest(StrictModel):
    items: list[ValidationItem] = Field(min_length=1, max_length=16)


class SemanticVerdict(StrictModel):
    is_real_object: StrictBool
    predicted_class: Literal["car", "truck", "pedestrian", "other"] = Field(alias="class")
    occlusion: Literal["none", "partial", "heavy"]


class PredictionBox(StrictModel):
    center: tuple[Finite, Finite, Finite]
    size: tuple[Positive, Positive, Positive]
    size_order: Literal["lwh"] = "lwh"
    yaw: Finite
    coordinate_frame: Literal["nuscenes_global"] = "nuscenes_global"


class Prediction(StrictModel):
    frame_token: str
    frame_index: int = Field(ge=0)
    box3d: PredictionBox
    label: str = Field(alias="class")
    score: Probability
    track_id: str | None = None
    source: Literal["auto"] = "auto"
    data_source: Literal["mock", "real"]
    review_status: Literal["pending"] = "pending"
