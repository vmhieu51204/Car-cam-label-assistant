"""Optional CVAT display preprocessing; detector inputs and cuboids stay untouched."""

from typing import Any

import numpy as np
from pydantic import Field

from src.models.video import StrictContract


class PointCloudExportConfig(StrictContract):
    enabled: bool = False
    max_range_m: float = Field(default=60, gt=0, allow_inf_nan=False)
    horizontal_fov_degrees: float = Field(default=90, gt=0, le=360, allow_inf_nan=False)
    voxel_size_m: float = Field(default=0.15, ge=0, allow_inf_nan=False)
    max_task_frames: int = Field(default=100, ge=1, strict=True)


def display_points(
    points: np.ndarray,
    config: PointCloudExportConfig,
    *,
    calibration: dict[str, np.ndarray] | None = None,
    image_size: tuple[int, int] | None = None,
) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] not in {3, 4} or not np.isfinite(points).all():
        raise ValueError("finite Nx3/Nx4 LiDAR points required")
    if not config.enabled:
        return points
    xyz = points[:, :3]
    radius = np.linalg.norm(xyz, axis=1)
    angles = np.abs(np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0])))
    valid = (radius <= config.max_range_m) & (angles <= config.horizontal_fov_degrees / 2)
    if calibration is not None:
        from src.services.geometry.kitti_projection import project_lidar

        if image_size is None:
            raise ValueError("camera dimensions required with calibration")
        valid &= project_lidar(xyz, calibration=calibration, image_size=image_size).visible
    selected = points[valid]
    if not len(selected) or config.voxel_size_m == 0:
        return selected
    # Keep the first actual point per voxel (no invented centroids); preserve stable source order.
    grid = np.floor(selected[:, :3] / config.voxel_size_m).astype(np.int64)
    _, indices = np.unique(grid, axis=0, return_index=True)
    return selected[np.sort(indices)]


def split_cuboid_draft(draft: dict[str, Any], max_frames: int) -> list[dict[str, Any]]:
    import copy

    if max_frames < 1:
        raise ValueError("positive task size required")
    total = draft["source_frame_count"]
    result = []
    for start in range(0, total, max_frames):
        stop = min(total, start + max_frames)
        task = copy.deepcopy(draft)
        task.update(
            source_frame_count=stop - start, source_start=start, source_frames=list(range(start, stop)), tracks=[]
        )
        for track in draft["tracks"]:
            shapes = [copy.deepcopy(s) for s in track["shapes"] if start <= s["frame"] < stop]
            previous = [s for s in track["shapes"] if s["frame"] < start]
            if previous and (not shapes or shapes[0]["frame"] > start):
                anchor = max(previous, key=lambda s: s["frame"])
                if not anchor.get("outside"):
                    carry = copy.deepcopy(anchor)
                    carry["frame"] = start
                    shapes.insert(0, carry)
            if not shapes:
                continue
            for shape in shapes:
                shape["frame"] -= start
            converted = copy.deepcopy(track)
            converted.update(frame=shapes[0]["frame"], shapes=shapes)
            task["tracks"].append(converted)
        result.append(task)
    return result
