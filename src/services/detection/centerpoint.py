"""Import validated CenterPoint LiDAR predictions into the common scene schema."""

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.config import Settings
from src.models.lidar import LidarPrediction as LidarPrediction
from src.models.perception import Box3D, Detection, Frame
from src.services.detection.mock import SceneReader
from src.services.geometry.transforms import lidar_box_to_global

Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
SUPPORTED_LABELS = {"car", "truck", "pedestrian"}


class DetectorModel(BaseModel):
    """Reproducibility metadata emitted by the isolated GPU worker."""

    model_config = ConfigDict(extra="forbid")
    name: Literal["CenterPoint"]
    framework: Literal["MMDetection3D"]
    config: str
    checkpoint: str
    device: str
    score_threshold: Probability
    multi_sweep: bool


class CenterPointOutput(BaseModel):
    """Versioned interchange contract between the GPU worker and backend."""

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["autolabel3d-centerpoint/v1"]
    data_source: Literal["real"]
    dataset: str
    scene: str
    coordinate_frame: Literal["nuscenes_lidar"]
    size_order: Literal["lwh"]
    model: DetectorModel
    predictions: list[LidarPrediction]


@dataclass(frozen=True)
class CenterPointResult:
    """Common frames plus non-oracle runtime diagnostics."""

    frames: list[Frame]
    stats: dict[str, object]
    model: dict[str, object]


class NuScenesLidarIndex:
    """Resolve each keyframe's LiDAR calibration and ego pose from metadata."""

    def __init__(self, reader: SceneReader) -> None:
        sensors = {row["token"]: row for row in reader.table("sensor")}
        self.calibrated = {row["token"]: row for row in reader.table("calibrated_sensor")}
        self.ego_poses = {row["token"]: row for row in reader.table("ego_pose")}
        self.lidar_by_sample: dict[str, dict] = {}
        for row in reader.table("sample_data"):
            calibrated = self.calibrated[row["calibrated_sensor_token"]]
            if sensors[calibrated["sensor_token"]]["channel"] == "LIDAR_TOP" and row["is_key_frame"]:
                self.lidar_by_sample[row["sample_token"]] = row

    def pose(self, sample_token: str) -> tuple[dict, dict]:
        """Return ego pose and LiDAR calibration for a keyframe sample."""
        try:
            sample_data = self.lidar_by_sample[sample_token]
        except KeyError as error:
            raise ValueError(f"No LIDAR_TOP keyframe for sample {sample_token}") from error
        return (
            self.ego_poses[sample_data["ego_pose_token"]],
            self.calibrated[sample_data["calibrated_sensor_token"]],
        )


def load_centerpoint_output(settings: Settings, path: Path, scene: str) -> CenterPointResult:
    """Validate, transform, and group a real detector output without any mock fallback."""
    payload = CenterPointOutput.model_validate_json(path.read_text(encoding="utf-8"))
    if payload.dataset != settings.data_version:
        raise ValueError(f"Detector dataset {payload.dataset!r} != configured {settings.data_version!r}")
    if payload.scene != scene:
        raise ValueError(f"Detector scene {payload.scene!r} != requested {scene!r}")

    reader = SceneReader(settings)
    samples = reader.samples(scene, None)
    sample_by_token = {row["token"]: (index, row) for index, row in enumerate(samples)}
    lidar_index = NuScenesLidarIndex(reader)
    grouped: dict[str, list[Detection]] = defaultdict(list)
    skipped = Counter[str]()

    for prediction in payload.predictions:
        try:
            expected_index, _ = sample_by_token[prediction.sample_token]
        except KeyError as error:
            raise ValueError(f"Prediction references a sample outside {scene}: {prediction.sample_token}") from error
        if prediction.frame_index != expected_index:
            raise ValueError(
                f"Frame index mismatch for {prediction.sample_token}: "
                f"got {prediction.frame_index}, expected {expected_index}"
            )
        if prediction.label not in SUPPORTED_LABELS:
            skipped[prediction.label] += 1
            continue
        ego_pose, calibration = lidar_index.pose(prediction.sample_token)
        center, yaw = lidar_box_to_global(
            prediction.center_lidar_xyz_m,
            prediction.yaw_lidar_rad,
            ego_pose,
            calibration,
        )
        grouped[prediction.sample_token].append(
            Detection(
                label=prediction.label,
                confidence=prediction.score,
                box=Box3D(center=center, size_lwh=prediction.size_lwh_m, yaw=yaw),
            )
        )

    emitted_tokens = {prediction.sample_token for prediction in payload.predictions}
    frames = [
        Frame(
            scene=scene,
            sample_token=sample["token"],
            frame_index=index,
            timestamp_s=sample["timestamp"] / 1_000_000,
            detections=grouped[sample["token"]],
        )
        for index, sample in enumerate(samples)
        if sample["token"] in emitted_tokens
    ]
    if not frames:
        raise ValueError("Detector output contains no frames for the requested scene")
    stats: dict[str, object] = {
        "frames": len(frames),
        "raw_boxes": len(payload.predictions),
        "predicted_boxes": sum(len(frame.detections) for frame in frames),
        "skipped_unsupported": dict(sorted(skipped.items())),
        "supported_classes": sorted(SUPPORTED_LABELS),
    }
    return CenterPointResult(frames=frames, stats=stats, model=json.loads(payload.model.model_dump_json()))
