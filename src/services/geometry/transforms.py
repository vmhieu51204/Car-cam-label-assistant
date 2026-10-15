"""Rigid transforms shared by detector adapters and coordinate audits."""

import math
from collections.abc import Sequence

import numpy as np


def quaternion_rotation(quaternion_wxyz: Sequence[float]) -> np.ndarray:
    """Return the active right-handed rotation matrix for a wxyz quaternion."""
    if len(quaternion_wxyz) != 4:
        raise ValueError("Quaternion must contain [w, x, y, z]")
    values = np.asarray(quaternion_wxyz, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Quaternion must be finite")
    norm = float(np.linalg.norm(values))
    if norm == 0:
        raise ValueError("Quaternion norm must be positive")
    w, x, y, z = values / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def yaw_rotation(yaw: float) -> np.ndarray:
    """Return a right-handed rotation around +z."""
    if not math.isfinite(yaw):
        raise ValueError("Yaw must be finite")
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


def planar_yaw(rotation_box_to_frame: np.ndarray) -> float:
    """Return the right-handed yaw of a box's local +x axis in the target frame."""
    rotation = np.asarray(rotation_box_to_frame, dtype=float)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("Expected a finite 3 x 3 rotation matrix")
    heading = rotation @ np.array([1.0, 0.0, 0.0])
    if math.hypot(float(heading[0]), float(heading[1])) < 1e-12:
        raise ValueError("Box heading has no projection onto the target x-y plane")
    return math.atan2(float(heading[1]), float(heading[0]))


def lidar_box_to_global(
    center_lidar: Sequence[float],
    yaw_lidar: float,
    ego_pose: dict,
    calibrated_lidar: dict,
) -> tuple[tuple[float, float, float], float]:
    """Transform an upright LiDAR box center/yaw into nuScenes global coordinates."""
    center = np.asarray(center_lidar, dtype=float)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("LiDAR center must contain three finite values")
    lidar_to_ego = quaternion_rotation(calibrated_lidar["rotation"])
    ego_to_global = quaternion_rotation(ego_pose["rotation"])
    lidar_translation = np.asarray(calibrated_lidar["translation"], dtype=float)
    ego_translation = np.asarray(ego_pose["translation"], dtype=float)
    center_ego = lidar_to_ego @ center + lidar_translation
    center_global = ego_to_global @ center_ego + ego_translation
    box_to_global = ego_to_global @ lidar_to_ego @ yaw_rotation(yaw_lidar)
    global_xyz = (float(center_global[0]), float(center_global[1]), float(center_global[2]))
    return global_xyz, planar_yaw(box_to_global)
