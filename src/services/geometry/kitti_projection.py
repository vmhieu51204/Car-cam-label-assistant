"""Explicit LiDAR-to-image and ego-motion geometry for KITTI-style calibration.

All 3D box centers here are in LiDAR coordinates (x forward, y left, z up).
These functions do not infer calibration or pose from an image.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Projection:
    uv: np.ndarray  # N x 2; NaN for points behind the camera
    depth: np.ndarray  # camera rectified z, one value per input point
    visible: np.ndarray  # positive depth and inside the given image bounds


def transform_xyz(points: np.ndarray, destination_from_source: np.ndarray) -> np.ndarray:
    """Transform Nx3 metric points with a rigid 4x4 homogeneous matrix."""
    xyz = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(destination_from_source, dtype=np.float64)
    if (
        xyz.ndim != 2
        or xyz.shape[1] != 3
        or matrix.shape != (4, 4)
        or not np.isfinite(xyz).all()
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0, 0, 0, 1])
    ):
        raise ValueError("finite Nx3 points and a rigid 4x4 transform are required")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError("transform rotation must be orthonormal")
    return xyz @ rotation.T + matrix[:3, 3]


def compensate_ego_motion(
    points: np.ndarray, *, world_from_lidar_source: np.ndarray, world_from_lidar_target: np.ndarray
) -> np.ndarray:
    """Express source-frame LiDAR points in the target LiDAR coordinate frame."""
    target_from_source = np.linalg.inv(world_from_lidar_target) @ world_from_lidar_source
    return transform_xyz(points, target_from_source)


def world_from_oxts_imu(packet: tuple[float, ...], origin_packet: tuple[float, ...]) -> np.ndarray:
    """Mercator East/North/Up pose from KITTI OXTS GPS and roll/pitch/yaw.

    This follows the coordinate construction in the KITTI-oriented pykitti
    reader: Mercator scale at first latitude, translation origin at first GPS
    position, and Rz(yaw) @ Ry(pitch) @ Rx(roll). It requires genuine packets;
    no pose is inferred from frame number or box motion.
    """
    if (
        len(packet) != 30
        or len(origin_packet) != 30
        or not np.isfinite(packet).all()
        or not np.isfinite(origin_packet).all()
        or abs(packet[0]) >= 89.9
        or abs(origin_packet[0]) >= 89.9
    ):
        raise ValueError("two finite 30-field OXTS packets away from the poles are required")
    earth_radius_m = 6_378_137.0
    scale = np.cos(np.deg2rad(origin_packet[0]))

    def position(row: tuple[float, ...]) -> np.ndarray:
        latitude, longitude, altitude = row[:3]
        return np.array(
            [
                scale * earth_radius_m * np.deg2rad(longitude),
                scale * earth_radius_m * np.log(np.tan(np.pi / 4 + np.deg2rad(latitude) / 2)),
                altitude,
            ]
        )

    roll, pitch, yaw = packet[3:6]
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    pose = np.eye(4)
    pose[:3, :3] = rz @ ry @ rx
    pose[:3, 3] = position(packet) - position(origin_packet)
    return pose


def world_from_lidar_pose(world_from_imu: np.ndarray, imu_to_lidar: np.ndarray) -> np.ndarray:
    """Compose world<-IMU with KITTI's `Tr_imu_velo` (LiDAR<-IMU)."""
    transform = np.asarray(imu_to_lidar, dtype=np.float64)
    if transform.shape != (3, 4) or not np.isfinite(transform).all():
        raise ValueError("Tr_imu_velo must be a finite 3x4 matrix")
    lidar_from_imu = np.eye(4)
    lidar_from_imu[:3] = transform
    # Check both rigid parts before inversion to reject bad calibration.
    transform_xyz(np.zeros((0, 3)), world_from_imu)
    transform_xyz(np.zeros((0, 3)), lidar_from_imu)
    return world_from_imu @ np.linalg.inv(lidar_from_imu)


def project_lidar(points: np.ndarray, *, calibration: dict[str, np.ndarray], image_size: tuple[int, int]) -> Projection:
    """Apply P2 * R_rect * Tr_velo_cam with positive-depth and image clipping."""
    xyz = np.asarray(points, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("LiDAR points must be finite Nx3")
    width, height = image_size
    if width < 1 or height < 1:
        raise ValueError("image dimensions must be positive")
    try:
        projection = np.asarray(calibration["P2"], dtype=np.float64)
        rectification = np.asarray(calibration["R_rect"], dtype=np.float64)
        lidar_to_camera = np.asarray(calibration["Tr_velo_cam"], dtype=np.float64)
    except KeyError as error:
        raise ValueError("P2, R_rect and Tr_velo_cam calibration are required") from error
    if (
        projection.shape != (3, 4)
        or rectification.shape != (3, 3)
        or lidar_to_camera.shape != (3, 4)
        or not all(np.isfinite(item).all() for item in (projection, rectification, lidar_to_camera))
    ):
        raise ValueError("invalid KITTI calibration matrix shape or values")
    camera = xyz @ lidar_to_camera[:, :3].T + lidar_to_camera[:, 3]
    rectified = camera @ rectification.T
    homogeneous = rectified @ projection[:, :3].T + projection[:, 3]
    depth = rectified[:, 2]
    positive = (depth > 0) & (homogeneous[:, 2] > 0)
    uv = np.full((len(xyz), 2), np.nan, dtype=np.float64)
    uv[positive] = homogeneous[positive, :2] / homogeneous[positive, 2:3]
    visible = positive & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    return Projection(uv=uv, depth=depth, visible=visible)


def lidar_box_corners(
    center: tuple[float, float, float], size_lwh: tuple[float, float, float], yaw: float
) -> np.ndarray:
    """Return eight corners of a z-up oriented LiDAR box in meters."""
    if (
        len(center) != 3
        or len(size_lwh) != 3
        or not np.isfinite((*center, *size_lwh, yaw)).all()
        or any(length <= 0 for length in size_lwh)
    ):
        raise ValueError("finite center, positive length/width/height and yaw required")
    length, width, height = size_lwh
    offsets = np.array(
        [
            [x, y, z]
            for x in (-length / 2, length / 2)
            for y in (-width / 2, width / 2)
            for z in (-height / 2, height / 2)
        ],
        dtype=np.float64,
    )
    cosine, sine = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])
    return offsets @ rotation.T + np.asarray(center)


def project_lidar_box(
    center: tuple[float, float, float],
    size_lwh: tuple[float, float, float],
    yaw: float,
    *,
    calibration: dict[str, np.ndarray],
    image_size: tuple[int, int],
) -> tuple[float, float, float, float] | None:
    """Return a 2D extent only when all corners are in front of the camera.

    Near-plane clipping is intentionally deferred; a partly behind-camera box
    must be marked for review rather than projected to a misleading rectangle.
    """
    result = project_lidar(lidar_box_corners(center, size_lwh, yaw), calibration=calibration, image_size=image_size)
    if np.any(result.depth <= 0):
        return None
    return (
        float(np.min(result.uv[:, 0])),
        float(np.min(result.uv[:, 1])),
        float(np.max(result.uv[:, 0])),
        float(np.max(result.uv[:, 1])),
    )
