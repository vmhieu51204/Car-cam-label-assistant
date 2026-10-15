"""Export pending LiDAR cuboid proposals as KITTI tracking rows for inspection.

The file is format-compatible but not an evaluated 3D detector submission. Box
dimensions and orientation from geometric lift require review on real LiDAR.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from src.services.geometry.kitti_tracking import load_sequence


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def kitti_row(
    source_frame: int, track_id: int, label: str, observation: dict, calibration: dict[str, np.ndarray]
) -> str:
    """Map LiDAR z-up center/yaw to KITTI rectified-camera bottom/yaw fields."""
    raw = observation["box_lidar"]
    if raw is None:
        raise ValueError("cannot export a missing 3D box")
    center = np.asarray(raw["center_lidar_xyz_m"], dtype=float)
    length, width, height = map(float, raw["size_lwh_m"])
    yaw = float(raw["yaw_lidar_rad"])
    lidar_to_camera = calibration["Tr_velo_cam"]
    rectification = calibration["R_rect"]
    if (
        center.shape != (3,)
        or lidar_to_camera.shape != (3, 4)
        or rectification.shape != (3, 3)
        or not np.isfinite((*center, length, width, height, yaw)).all()
        or min(length, width, height) <= 0
    ):
        raise ValueError("invalid 3D geometry or calibration")
    bottom_lidar = center + np.array([0.0, 0.0, -height / 2])
    bottom_camera = rectification @ (lidar_to_camera[:, :3] @ bottom_lidar + lidar_to_camera[:, 3])
    forward_lidar = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    forward_camera = rectification @ lidar_to_camera[:, :3] @ forward_lidar
    rotation_y = _wrap(math.atan2(-forward_camera[2], forward_camera[0]))
    alpha = _wrap(rotation_y - math.atan2(bottom_camera[0], bottom_camera[2]))
    if bottom_camera[2] <= 0:
        raise ValueError("3D box lies behind the rectified camera")
    x1, y1, x2, y2 = map(float, observation["bbox_2d_xyxy"])
    if not x1 < x2 or not y1 < y2:
        raise ValueError("invalid 2D extent for KITTI row")
    score = observation.get("detector_score", observation.get("source_2d_score"))
    score = float(score) if score is not None else 0.0
    if not math.isfinite(score):
        raise ValueError("non-finite source 2D score")
    category = {"car": "Car", "person": "Pedestrian"}[label]
    fields = [
        source_frame,
        track_id,
        category,
        -1,
        -1,
        f"{alpha:.6f}",
        f"{x1:.2f}",
        f"{y1:.2f}",
        f"{x2:.2f}",
        f"{y2:.2f}",
        f"{height:.3f}",
        f"{width:.3f}",
        f"{length:.3f}",
        *(f"{value:.3f}" for value in bottom_camera),
        f"{rotation_y:.6f}",
        f"{score:.6f}",
    ]
    return " ".join(map(str, fields))


def export_kitti_draft(draft_path: Path, calibration: dict[str, np.ndarray], output: Path) -> dict:
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    if draft.get("schema") != "autolabel-lidar-3d-draft/v1":
        raise ValueError("expected a LiDAR 3D draft artifact")
    tracks = sorted(draft["tracks"], key=lambda row: row["track_id"])
    rows = []
    skipped = 0
    for track_id, track in enumerate(tracks):
        for observation in track["observations"]:
            if observation["box_lidar"] is None or observation.get("bbox_2d_xyxy") is None:
                skipped += 1
                continue
            rows.append(
                (
                    observation["source_frame_index"],
                    track_id,
                    kitti_row(observation["source_frame_index"], track_id, track["label"], observation, calibration),
                )
            )
    rows.sort()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(item[2] for item in rows) + ("\n" if rows else ""), encoding="utf-8")
    return {
        "output": str(output),
        "rows": len(rows),
        "skipped_no_cuboid": skipped,
        "track_id_map": {track["track_id"]: index for index, track in enumerate(tracks)},
        "review_status": "pending",
        "calibrated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-fps", type=float, required=True)
    parser.add_argument("--fps-source", required=True)
    args = parser.parse_args()
    sequence = load_sequence(args.dataset_root, args.sequence, source_fps=args.source_fps, fps_source=args.fps_source)
    print(json.dumps(export_kitti_draft(args.draft, sequence.calibration, args.output)))


if __name__ == "__main__":
    main()
