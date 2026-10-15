"""Pinned detectors and common metric LiDAR boxes for uploaded datasets."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.models.lidar import LidarPrediction

CHECKPOINT_SHA = "191a38226d2ce990f26cc3f78c454310cc562d81d29e86d8ca8fa39c0a44215d"
BEVFUSION_CHECKPOINT_SHA = "e68c6c084d8971f4f879751b0ed3981f575c24ee3cf385a1e39197c26a9bb3ab"
CLASSES = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]
KITTI_CLASSES = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc"]
CHUNK_BYTES = 4 * 1024 * 1024
Finite = Annotated[float, Field(allow_inf_nan=False)]


class UploadPrediction(LidarPrediction):
    rotation_lidar_xyz_rad: tuple[Finite, Finite, Finite] | None = None
    track_id: str | None = Field(default=None, min_length=1, max_length=160)
    linked_2d_tracks: dict[str, str] = Field(default_factory=dict)


class UploadOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["autolabel-lidar3d-upload/v1"]
    dataset_format: Literal["nuscenes", "kitti"]
    scene: str
    model_name: Literal["BEVFusion", "Imported"]
    checkpoint_sha256: str
    frames: list[str]
    predictions: list[UploadPrediction]
