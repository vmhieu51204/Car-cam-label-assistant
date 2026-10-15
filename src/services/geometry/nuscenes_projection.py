"""Full-quaternion nuScenes projection; derived envelopes are not visible-object GT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.services.geometry.transforms import quaternion_rotation


def box_corners_global(annotation: dict[str, Any]) -> np.ndarray:
    """Return corners using the ordering in the official nuScenes Box implementation."""
    size = _vector(annotation["size"], 3)
    if np.any(size <= 0):
        raise ValueError("Box dimensions must be positive")
    width, length, height = size
    local = np.array(
        [
            [length / 2, width / 2, height / 2],
            [length / 2, -width / 2, height / 2],
            [length / 2, -width / 2, -height / 2],
            [length / 2, width / 2, -height / 2],
            [-length / 2, width / 2, height / 2],
            [-length / 2, -width / 2, height / 2],
            [-length / 2, -width / 2, -height / 2],
            [-length / 2, width / 2, -height / 2],
        ]
    )
    rotation = quaternion_rotation(annotation["rotation"])
    center = _vector(annotation["translation"], 3)
    return local @ rotation.T + center


def global_points_to_camera(
    points_global: np.ndarray,
    camera_ego_pose: dict[str, Any],
    calibrated_camera: dict[str, Any],
) -> np.ndarray:
    """Transform global XYZ rows into a nuScenes camera sensor frame."""
    points = np.asarray(points_global, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("Expected finite N x 3 global points")
    ego_to_global = quaternion_rotation(camera_ego_pose["rotation"])
    camera_to_ego = quaternion_rotation(calibrated_camera["rotation"])
    ego_translation = _vector(camera_ego_pose["translation"], 3)
    camera_translation = _vector(calibrated_camera["translation"], 3)
    points_ego = (points - ego_translation) @ ego_to_global
    return (points_ego - camera_translation) @ camera_to_ego


def project_camera_points(points_camera: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    """Project positive-depth camera XYZ rows to pixel UV rows."""
    points = np.asarray(points_camera, dtype=float)
    camera_matrix = np.asarray(intrinsic, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or camera_matrix.shape != (3, 3):
        raise ValueError("Expected N x 3 camera points and a 3 x 3 intrinsic")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(camera_matrix)):
        raise ValueError("Projection inputs must be finite")
    if not np.allclose(camera_matrix[2], [0.0, 0.0, 1.0], rtol=0, atol=1e-12):
        raise ValueError("Intrinsic must have homogeneous row [0,0,1]")
    if camera_matrix[0, 0] <= 0 or camera_matrix[1, 1] <= 0 or abs(np.linalg.det(camera_matrix)) < 1e-12:
        raise ValueError("Intrinsic must have positive focal lengths and be invertible")
    if np.any(points[:, 2] <= 0):
        raise ValueError("Camera projection requires positive depth")
    homogeneous = points @ camera_matrix.T
    pixels = homogeneous[:, :2] / homogeneous[:, 2:3]
    if not np.all(np.isfinite(pixels)):
        raise ValueError("Projected pixels must be finite")
    return pixels


def _vector(values: Any, size: int) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"Expected {size} finite values")
    return result


@dataclass(frozen=True)
class ProjectionResult:
    """One geometric decision; clipping has explicit pre-intersection bounds."""

    bbox_xyxy: tuple[float, float, float, float] | None = None
    original_bbox_xyxy: tuple[float, float, float, float] | None = None
    rejection_reason: str | None = None
    clipped: bool = False


def intersect_projected_hull(points: np.ndarray, width: int, height: int) -> ProjectionResult:
    """Intersect the convex polygon with the canvas, then compute bounds.

    Clamping corner coordinates alone would retain triangles outside the canvas
    whose axis-aligned envelopes overlap it. Monotone hull plus polygon clipping
    makes that distinction without a runtime geometry dependency.
    """
    pixels = np.asarray(points, dtype=float)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or not np.all(np.isfinite(pixels)):
        return ProjectionResult(rejection_reason="invalid_projection")
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    vertices = sorted(set((float(x), float(y)) for x, y in pixels))

    def cross(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def half(values: list[tuple[float, float]]) -> list[tuple[float, float]]:
        result: list[tuple[float, float]] = []
        for vertex in values:
            while len(result) >= 2 and cross(result[-2], result[-1], vertex) <= 0:
                result.pop()
            result.append(vertex)
        return result

    polygon = half(vertices)[:-1] + half(vertices[::-1])[:-1]
    if len(polygon) < 3:
        return ProjectionResult(rejection_reason="degenerate_projection")
    original = (
        min(x for x, _ in polygon),
        min(y for _, y in polygon),
        max(x for x, _ in polygon),
        max(y for _, y in polygon),
    )
    clipped = any(x < 0 or y < 0 or x > width or y > height for x, y in polygon)
    for axis, boundary, lower in ((0, 0.0, True), (0, float(width), False), (1, 0.0, True), (1, float(height), False)):
        output: list[tuple[float, float]] = []
        if not polygon:
            break
        previous = polygon[-1]
        previous_inside = previous[axis] >= boundary if lower else previous[axis] <= boundary
        for current in polygon:
            inside = current[axis] >= boundary if lower else current[axis] <= boundary
            if inside != previous_inside:
                ratio = (boundary - previous[axis]) / (current[axis] - previous[axis])
                intersection = (
                    previous[0] + ratio * (current[0] - previous[0]),
                    previous[1] + ratio * (current[1] - previous[1]),
                )
                output.append(intersection)
            if inside:
                output.append(current)
            previous, previous_inside = current, inside
        polygon = output
    if len(polygon) < 3:
        return ProjectionResult(rejection_reason="outside_canvas")
    area_twice = abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(polygon, polygon[1:] + polygon[:1], strict=True)))
    if area_twice <= 1e-12:
        return ProjectionResult(rejection_reason="degenerate_projection")
    bounds = (
        max(0.0, min(x for x, _ in polygon)),
        max(0.0, min(y for _, y in polygon)),
        min(float(width), max(x for x, _ in polygon)),
        min(float(height), max(y for _, y in polygon)),
    )
    return ProjectionResult(bbox_xyxy=bounds, original_bbox_xyxy=original, clipped=clipped)


def project_nuscenes_box(
    annotation: dict[str, Any],
    camera_ego_pose: dict[str, Any],
    calibrated_camera: dict[str, Any],
    *,
    width: int,
    height: int,
    min_depth_m: float = 0.1,
) -> ProjectionResult:
    """Conservatively reject any corner at/below the frozen near plane.

    This is not the nuScenes SDK positive-corner exporter policy. Malformed
    geometry is reported as invalid; loaders additionally reject malformed
    native metadata instead of treating it as a legitimate unsupported target.
    """
    if not np.isfinite(min_depth_m) or min_depth_m <= 0:
        raise ValueError("Minimum depth must be finite and positive")
    try:
        points = global_points_to_camera(box_corners_global(annotation), camera_ego_pose, calibrated_camera)
        if np.all(points[:, 2] <= 0):
            return ProjectionResult(rejection_reason="behind_camera")
        if np.any(points[:, 2] <= min_depth_m):
            return ProjectionResult(rejection_reason="near_plane")
        pixels = project_camera_points(points, np.asarray(calibrated_camera["camera_intrinsic"], dtype=float))
    except (ValueError, KeyError, TypeError, FloatingPointError):
        return ProjectionResult(rejection_reason="invalid_projection")
    return intersect_projected_hull(pixels, width, height)
