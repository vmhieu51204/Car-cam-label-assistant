"""Conservative same-class camera evidence for 3D identities; no forced links."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

from src.services.geometry.kitti_projection import project_lidar
from src.services.geometry.nuscenes_projection import intersect_projected_hull, project_nuscenes_box
from src.services.vm.lidar3d_contracts import UploadPrediction
from src.services.vm.lidar3d_exchange import lidar_pose

COCO_MAPPING = {"pedestrian": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "bus": 5, "truck": 7}
KITTI_MAPPING = {"Car": "car", "Truck": "truck", "Pedestrian": "pedestrian", "Cyclist": "bicycle"}


def projected_box(row: UploadPrediction, frame: dict, channel: str, width: int, height: int) -> tuple | None:
    orientation = Rotation.from_euler("xyz", row.rotation_lidar_xyz_rad or (0, 0, row.yaw_lidar_rad)).as_matrix()
    if "camera_info" in frame:
        rotation, translation = lidar_pose(frame)
        quaternion = Rotation.from_matrix(rotation @ orientation).as_quat()
        length, box_width, box_height = row.size_lwh_m
        camera = frame["camera_info"][channel]
        return project_nuscenes_box(
            {
                "translation": (rotation @ row.center_lidar_xyz_m + translation).tolist(),
                "size": [box_width, length, box_height],
                "rotation": [quaternion[3], *quaternion[:3]],
            },
            camera["ego_pose"],
            camera["calibration"],
            width=width,
            height=height,
        ).bbox_xyxy
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    corners = (signs * np.asarray(row.size_lwh_m) / 2) @ orientation.T + row.center_lidar_xyz_m
    projection = project_lidar(
        corners,
        calibration={key: np.asarray(value) for key, value in frame["calibration"].items()},
        image_size=(width, height),
    )
    if np.any(projection.depth <= 0.1):
        return None
    return intersect_projected_hull(projection.uv, width, height).bbox_xyxy


def box_iou(left: tuple | list, right: tuple | list) -> float:
    a, b = np.asarray(left), np.asarray(right)
    intersection = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    union = np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - intersection
    return float(intersection / union) if union > 0 else 0.0


def associate(rows: list, observations: list[dict], frame: dict, channel: str, width: int, height: int) -> list[dict]:
    """Hungarian matching after class/IoU gates, rejecting ambiguous alternatives."""
    projections = [projected_box(row, frame, channel, width, height) for row in rows]
    scores = np.zeros((len(rows), len(observations)))
    for i, row in enumerate(rows):
        projection = projections[i]
        for j, observation in enumerate(observations):
            label = KITTI_MAPPING.get(row.label, row.label)
            if projection is not None and label == observation["label"]:
                scores[i, j] = box_iou(projection, observation["detector_bbox"])
    costs = np.full((len(rows), len(observations) + len(rows)), 2.0)
    costs[:, : len(observations)] = np.where(scores >= 0.3, 1 - scores, 2.0)
    assigned = dict(zip(*linear_sum_assignment(costs), strict=True)) if len(rows) else {}
    result = []
    for i, row in enumerate(rows):
        j = assigned[i]
        status, linked = "unmatched", None
        label = KITTI_MAPPING.get(row.label, row.label)
        if label not in COCO_MAPPING:
            status = "unsupported_class"
        elif projections[i] is None:
            status = "not_projectable"
        elif j < len(observations) and scores[i, j] >= 0.3:
            alternatives = [*np.delete(scores[i], j), *np.delete(scores[:, j], i)]
            if alternatives and scores[i, j] - max(alternatives) < 0.1:
                status = "ambiguous"
            else:
                status, linked = "matched", observations[j]["track_id"]
        result.append(
            {
                "prediction_index": i,
                "track_3d_id": row.track_id,
                "camera": channel,
                "frame": frame["frame"],
                "status": status,
                "track_2d_id": linked,
                "projected_bbox": projections[i],
                "iou": float(scores[i, j]) if linked else None,
            }
        )
    return result
