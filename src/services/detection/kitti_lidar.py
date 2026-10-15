"""Validated LiDAR detector output and conservative 2D/3D linking.

Association thresholds are uncalibrated engineering defaults. Detector boxes
remain untouched and pending review; no image mask is used to invent geometry.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.optimize import linear_sum_assignment

from src.services.geometry.kitti_projection import project_lidar_box
from src.services.geometry.kitti_tracking import KittiSequence

Finite = Annotated[float, Field(allow_inf_nan=False)]
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LidarBox(Contract):
    label: Literal["car", "person"]
    center_lidar_xyz_m: tuple[Finite, Finite, Finite]
    size_lwh_m: tuple[Positive, Positive, Positive]
    yaw_lidar_rad: Finite
    score: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class DetectorFrame(Contract):
    source_frame_index: Annotated[int, Field(ge=0, strict=True)]
    lidar_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    boxes: list[LidarBox]


class DetectorOutput(Contract):
    schema_id: Literal["autolabel-kitti-detector/v1"] = Field(alias="schema")
    sequence_id: str
    coordinate_frame: Literal["kitti_velodyne_x_forward_y_left_z_up"]
    model_name: str = Field(min_length=1)
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    framework_version: str = Field(min_length=1)
    status: Literal["completed", "partial"] = "completed"
    runtime: dict = Field(default_factory=dict)
    frames: list[DetectorFrame]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def validate_output(path: Path, sequence: KittiSequence) -> DetectorOutput:
    result = DetectorOutput.model_validate_json(path.read_bytes())
    if result.sequence_id != sequence.sequence_id or result.status != "completed":
        raise ValueError("detector sequence mismatch")
    if [f.source_frame_index for f in result.frames] != list(range(len(sequence.frames))):
        raise ValueError("detector must report every source frame, including empty frames")
    for frame in result.frames:
        if sha256(sequence.frames[frame.source_frame_index].lidar) != frame.lidar_sha256:
            raise ValueError("detector LiDAR input hash mismatch")
    return result


def box_iou(a: tuple, b: tuple) -> float:
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area - intersection) if area > intersection else 0.0


def associate(
    boxes: list[LidarBox],
    observations: list[dict],
    sequence: KittiSequence,
    *,
    min_iou: float = 0.3,
    ambiguity_margin: float = 0.05,
) -> list[dict]:
    """Same-class gated Hungarian matching; close alternatives remain unlinked."""
    if not 0 < min_iou <= 1 or not 0 <= ambiguity_margin < 1:
        raise ValueError("invalid uncalibrated association thresholds")
    projected = [
        project_lidar_box(
            b.center_lidar_xyz_m,
            b.size_lwh_m,
            b.yaw_lidar_rad,
            calibration=sequence.calibration,
            image_size=sequence.image_size,
        )
        for b in boxes
    ]
    scores = np.zeros((len(boxes), len(observations)))
    for i, box in enumerate(boxes):
        projected_box = projected[i]
        for j, observation in enumerate(observations):
            if projected_box is not None and box.label == observation["label"]:
                scores[i, j] = box_iou(projected_box, tuple(observation["bbox"]))
    costs = np.full((len(boxes), len(observations) + len(boxes)), 2.0)
    costs[:, : len(observations)] = np.where(scores >= min_iou, 1 - scores, 3.0)
    links = {}
    if boxes:
        rows, cols = linear_sum_assignment(costs)
        links = {
            int(i): int(j) for i, j in zip(rows, cols, strict=True) if j < len(observations) and scores[i, j] >= min_iou
        }
    result = []
    for i in range(len(boxes)):
        candidate: int | None = links.get(i)
        reason = "unmatched_2d" if projected[i] is not None else "projection_unavailable"
        if candidate is not None:
            alternatives = [*np.delete(scores[i], candidate), *np.delete(scores[:, candidate], i)]
            if any(v >= min_iou and v >= scores[i, candidate] - ambiguity_margin for v in alternatives):
                candidate, reason = None, "ambiguous_2d_3d"
            else:
                reason = "linked_2d_3d_pending_review"
        result.append(
            {
                "track_id_2d": observations[candidate]["track_id"] if candidate is not None else None,
                "projected_bbox": projected[i],
                "iou": float(scores[i, candidate]) if candidate is not None else None,
                "needs_review": True,
                "review_reasons": [reason],
            }
        )
    return result
