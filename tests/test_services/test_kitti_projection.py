"""Known-answer camera projection and ego-motion tests use no dataset or GPU."""

import numpy as np
import pytest

from src.services.geometry.kitti_projection import (
    compensate_ego_motion,
    lidar_box_corners,
    project_lidar,
    project_lidar_box,
    world_from_lidar_pose,
    world_from_oxts_imu,
)


def _calibration():
    return {
        "P2": np.array([[10, 0, 5, 0], [0, 10, 5, 0], [0, 0, 1, 0]], dtype=float),
        "R_rect": np.eye(3),
        "Tr_velo_cam": np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=float),
    }


def test_projection_preserves_correspondence_and_rejects_behind_camera():
    points = np.array([[1, 2, 10], [0, 0, -2], [100, 0, 10]], dtype=float)
    result = project_lidar(points, calibration=_calibration(), image_size=(20, 20))
    np.testing.assert_allclose(result.uv[0], [6, 7])
    assert result.visible.tolist() == [True, False, False]
    assert np.isnan(result.uv[1]).all()


def test_ego_motion_compensation_uses_target_frame():
    source = np.eye(4)
    target = np.eye(4)
    target[0, 3] = 2
    transformed = compensate_ego_motion(
        np.array([[5, 0, 0]], dtype=float), world_from_lidar_source=source, world_from_lidar_target=target
    )
    np.testing.assert_allclose(transformed, [[3, 0, 0]])


def test_lidar_box_corners_and_2d_extent():
    corners = lidar_box_corners((0, 0, 10), (2, 2, 2), 0)
    assert corners.shape == (8, 3)
    assert project_lidar_box(
        (0, 0, 10), (2, 2, 2), 0, calibration=_calibration(), image_size=(20, 20)
    ) == pytest.approx((3.888888889, 3.888888889, 6.111111111, 6.111111111))
    assert project_lidar_box((0, 0, 0), (2, 2, 2), 0, calibration=_calibration(), image_size=(20, 20)) is None


def test_oxts_mercator_pose_and_imu_to_lidar_composition():
    origin = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0) + (0.0,) * 24
    moved = (0.0, 0.00001, 12.0, 0.0, 0.0, np.pi / 2) + (0.0,) * 24
    world_from_imu = world_from_oxts_imu(moved, origin)
    np.testing.assert_allclose(world_from_imu[:3, 3], [1.1131949079, 0, 2], atol=1e-5)
    np.testing.assert_allclose(world_from_imu[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-8)
    lidar_from_imu = np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]])
    world_from_lidar = world_from_lidar_pose(world_from_imu, lidar_from_imu)
    np.testing.assert_allclose(world_from_lidar[:3, 3], [1.1131949079, -1, 2], atol=1e-5)
