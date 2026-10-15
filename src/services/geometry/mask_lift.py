"""Conservative image-mask to LiDAR cuboid proposals.

This is geometry, not a trained 3D detector. Dimensions and point thresholds are
uncalibrated starting values. Every fitted cuboid remains pending human review.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.services.geometry.kitti_projection import Projection


@dataclass(frozen=True)
class LiftConfig:
    """Uncalibrated KITTI-scale priors; never treat them as measured dimensions."""

    min_points: int = 6
    depth_radius_m: float = 3.0
    car_min_lwh_m: tuple[float, float, float] = (3.5, 1.5, 1.3)
    person_min_lwh_m: tuple[float, float, float] = (0.4, 0.4, 1.4)
    yaw_min_points: int = 15
    yaw_eigen_ratio: float = 1.5

    def __post_init__(self) -> None:
        if (
            self.min_points < 1
            or self.depth_radius_m <= 0
            or self.yaw_min_points < 2
            or self.yaw_eigen_ratio < 1
            or any(value <= 0 for value in (*self.car_min_lwh_m, *self.person_min_lwh_m))
        ):
            raise ValueError("invalid uncalibrated LiDAR lift configuration")


@dataclass(frozen=True)
class LiftedBox:
    center_lidar_xyz_m: tuple[float, float, float]
    size_lwh_m: tuple[float, float, float]
    yaw_lidar_rad: float
    point_count: int
    needs_review: bool = True
    review_reasons: tuple[str, ...] = field(default_factory=tuple)


def lift_observation(
    points_xyzi: np.ndarray,
    projection: Projection,
    *,
    image_size: tuple[int, int],
    label: str,
    bbox_xyxy: tuple[float, float, float, float],
    mask: np.ndarray | None,
    config: LiftConfig | None = None,
) -> tuple[LiftedBox | None, tuple[str, ...]]:
    """Select projected points in a mask, isolate one depth cluster and fit a draft.

    When the scan cannot support a box, return None with an explicit reason. A
    rectangle-only fallback is allowed but flagged; no point is invented.
    """
    config = config or LiftConfig()
    if label not in {"car", "person"}:
        return None, ("unsupported_class",)
    points = np.asarray(points_xyzi)
    width, height = image_size
    if (
        points.ndim != 2
        or points.shape[1] != 4
        or not np.isfinite(points).all()
        or projection.uv.shape != (len(points), 2)
        or projection.depth.shape != (len(points),)
        or projection.visible.shape != (len(points),)
        or width <= 0
        or height <= 0
    ):
        raise ValueError("scan, projection and image size are inconsistent")
    if mask is not None and (mask.shape != (height, width) or mask.dtype != np.bool_):
        raise ValueError("mask must be a boolean image in camera coordinates")
    x1, y1, x2, y2 = bbox_xyxy
    if not 0 <= x1 < x2 or not 0 <= y1 < y2:
        raise ValueError("invalid 2D bounding box")
    visible_indexes = np.flatnonzero(projection.visible)
    uv = projection.uv[visible_indexes]
    if mask is not None:
        u = np.floor(uv[:, 0]).astype(int)
        v = np.floor(uv[:, 1]).astype(int)
        inside = mask[v, u]
    else:
        inside = (uv[:, 0] >= x1) & (uv[:, 0] < x2) & (uv[:, 1] >= y1) & (uv[:, 1] < y2)
    selected = visible_indexes[inside]
    if len(selected) < config.min_points:
        return None, ("insufficient_lidar_points",)
    depths = projection.depth[selected]
    median_depth = float(np.median(depths))
    selected = selected[np.abs(depths - median_depth) <= config.depth_radius_m]
    if len(selected) < config.min_points:
        return None, ("insufficient_depth_cluster",)
    xyz = points[selected, :3].astype(np.float64)
    xy_center = np.median(xyz[:, :2], axis=0)
    centered = xyz[:, :2] - xy_center
    reasons = ["uncalibrated_geometric_lift"]
    if mask is None:
        reasons.append("box_only_point_selection")
    yaw = 0.0
    if len(xyz) >= config.yaw_min_points:
        covariance = centered.T @ centered / len(centered)
        values, vectors = np.linalg.eigh(covariance)
        if values[0] > 0 and values[1] / values[0] >= config.yaw_eigen_ratio:
            axis = vectors[:, 1]
            yaw = float(np.arctan2(axis[1], axis[0]))
        else:
            reasons.append("ambiguous_yaw")
    else:
        reasons.append("sparse_yaw")
    cosine, sine = np.cos(yaw), np.sin(yaw)
    local_x = centered[:, 0] * cosine + centered[:, 1] * sine
    local_y = -centered[:, 0] * sine + centered[:, 1] * cosine
    span = np.array(
        [
            np.percentile(local_x, 90) - np.percentile(local_x, 10),
            np.percentile(local_y, 90) - np.percentile(local_y, 10),
            np.percentile(xyz[:, 2], 90) - np.percentile(xyz[:, 2], 10),
        ]
    )
    prior = config.car_min_lwh_m if label == "car" else config.person_min_lwh_m
    size = np.maximum(span, prior)
    if np.any(span < prior):
        reasons.append("size_prior_used")
    center_z = float(np.median(xyz[:, 2]))
    box = LiftedBox(
        center_lidar_xyz_m=(float(xy_center[0]), float(xy_center[1]), center_z),
        size_lwh_m=(float(size[0]), float(size[1]), float(size[2])),
        yaw_lidar_rad=yaw,
        point_count=len(xyz),
        review_reasons=tuple(reasons),
    )
    return box, tuple(reasons)
