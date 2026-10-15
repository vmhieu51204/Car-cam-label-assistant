"""Run official MMDetection3D CenterPoint on nuScenes keyframes."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from mmengine.config import Config

CONFIG = Path(
    os.getenv(
        "CENTERPOINT_CONFIG",
        "/opt/mmdetection3d/configs/centerpoint/"
        "centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py",
    )
)
CHECKPOINT = Path(os.getenv("CENTERPOINT_CHECKPOINT", "/models/centerpoint_pillar02_nuscenes_191a3822.pth"))
CHECKPOINT_SHA256 = "191a38226d2ce990f26cc3f78c454310cc562d81d29e86d8ca8fa39c0a44215d"
CLASS_NAMES = (
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
)


def load_table(version_root: Path, name: str) -> list[dict[str, Any]]:
    """Load one nuScenes JSON table."""
    return json.loads((version_root / f"{name}.json").read_text(encoding="utf-8"))


@dataclass(frozen=True)
class NuScenesIndex:
    """Metadata required to align earlier sweeps to the current LiDAR."""

    sample_data: dict[str, dict[str, Any]]
    calibrated: dict[str, dict[str, Any]]
    ego_poses: dict[str, dict[str, Any]]


def scene_keyframes(
    dataset_root: Path, version: str, scene_name: str, max_frames: int | None = None
) -> tuple[list[dict[str, Any]], NuScenesIndex]:
    """Return ordered sample tokens and LIDAR_TOP files for a scene."""
    version_root = dataset_root / version
    scenes = {row["name"]: row for row in load_table(version_root, "scene")}
    if scene_name not in scenes:
        raise ValueError(f"Unknown scene: {scene_name}")
    samples = {row["token"]: row for row in load_table(version_root, "sample")}
    sensors = {row["token"]: row for row in load_table(version_root, "sensor")}
    calibrated = {row["token"]: row for row in load_table(version_root, "calibrated_sensor")}
    ego_poses = {row["token"]: row for row in load_table(version_root, "ego_pose")}
    all_sample_data = {row["token"]: row for row in load_table(version_root, "sample_data")}
    lidar_by_sample = {}
    for row in all_sample_data.values():
        calibration = calibrated[row["calibrated_sensor_token"]]
        if sensors[calibration["sensor_token"]]["channel"] == "LIDAR_TOP" and row["is_key_frame"]:
            lidar_by_sample[row["sample_token"]] = row

    frames = []
    token = scenes[scene_name]["first_sample_token"]
    seen = set()
    while token and (max_frames is None or len(frames) < max_frames):
        if token in seen:
            raise ValueError("Cycle in sample chain")
        seen.add(token)
        sample = samples[token]
        lidar_data = lidar_by_sample[token]
        lidar_path = dataset_root / Path(lidar_data["filename"])
        if not lidar_path.is_file():
            raise FileNotFoundError(lidar_path)
        frames.append(
            {
                "sample_token": token,
                "frame_index": len(frames),
                "timestamp": sample["timestamp"],
                "lidar_path": lidar_path,
                "lidar_data": lidar_data,
            }
        )
        token = sample["next"]
    return frames, NuScenesIndex(sample_data=all_sample_data, calibrated=calibrated, ego_poses=ego_poses)


def quaternion_rotation(quaternion_wxyz: list[float]) -> np.ndarray:
    """Return the active rotation for one nuScenes wxyz quaternion."""
    values = np.asarray(quaternion_wxyz, dtype=np.float64)
    values /= np.linalg.norm(values)
    w, x, y, z = values
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def read_lidar(path: Path) -> np.ndarray:
    """Read one raw nuScenes x/y/z/intensity/ring point cloud."""
    values = np.fromfile(path, dtype=np.float32)
    if values.size % 5:
        raise ValueError(f"Invalid nuScenes point cloud shape: {path}")
    return values.reshape(-1, 5)


def aggregate_sweeps(
    dataset_root: Path,
    current_data: dict[str, Any],
    index: NuScenesIndex,
    sweeps_num: int = 9,
) -> np.ndarray:
    """Align earlier sweeps to current LIDAR_TOP and encode their time lag."""
    current_points = read_lidar(dataset_root / Path(current_data["filename"]))
    current_points[:, 4] = 0
    aligned = [current_points]
    current_calibration = index.calibrated[current_data["calibrated_sensor_token"]]
    current_ego = index.ego_poses[current_data["ego_pose_token"]]
    lidar_current_to_ego = quaternion_rotation(current_calibration["rotation"])
    ego_current_to_global = quaternion_rotation(current_ego["rotation"])
    current_lidar_translation = np.asarray(current_calibration["translation"], dtype=np.float64)
    current_ego_translation = np.asarray(current_ego["translation"], dtype=np.float64)

    token = current_data["prev"]
    loaded = 0
    while token and loaded < sweeps_num:
        sweep_data = index.sample_data[token]
        sweep_points = read_lidar(dataset_root / Path(sweep_data["filename"]))
        keep = np.logical_or(np.abs(sweep_points[:, 0]) >= 1.0, np.abs(sweep_points[:, 1]) >= 1.0)
        sweep_points = sweep_points[keep]
        sweep_calibration = index.calibrated[sweep_data["calibrated_sensor_token"]]
        sweep_ego = index.ego_poses[sweep_data["ego_pose_token"]]
        lidar_sweep_to_ego = quaternion_rotation(sweep_calibration["rotation"])
        ego_sweep_to_global = quaternion_rotation(sweep_ego["rotation"])
        sweep_lidar_translation = np.asarray(sweep_calibration["translation"], dtype=np.float64)
        sweep_ego_translation = np.asarray(sweep_ego["translation"], dtype=np.float64)

        rotation = (
            lidar_current_to_ego.T
            @ ego_current_to_global.T
            @ ego_sweep_to_global
            @ lidar_sweep_to_ego
        )
        translation = lidar_current_to_ego.T @ (
            ego_current_to_global.T
            @ (
                ego_sweep_to_global @ sweep_lidar_translation
                + sweep_ego_translation
                - current_ego_translation
            )
            - current_lidar_translation
        )
        sweep_points[:, :3] = sweep_points[:, :3] @ rotation.T + translation
        sweep_points[:, 4] = (current_data["timestamp"] - sweep_data["timestamp"]) / 1_000_000
        aligned.append(sweep_points)
        loaded += 1
        token = sweep_data["prev"]

    if loaded == 0:
        without_ego = current_points[
            np.logical_or(np.abs(current_points[:, 0]) >= 1.0, np.abs(current_points[:, 1]) >= 1.0)
        ]
        aligned.extend(without_ego.copy() for _ in range(sweeps_num))
    return np.concatenate(aligned).astype(np.float32, copy=False)


def transform_name(item: dict[str, Any]) -> str:
    """Normalize string or imported-class transform declarations."""
    transform_type = item.get("type")
    return transform_type if isinstance(transform_type, str) else transform_type.__name__


def standalone_config(path: Path) -> Config:
    """Use pre-aligned arrays in place of the metadata-dependent sweep loader."""
    from mmengine.config import Config

    config = Config.fromfile(path)
    pipeline = config.test_dataloader.dataset.pipeline
    config.test_dataloader.dataset.pipeline = [
        item for item in pipeline if transform_name(item) != "LoadPointsFromMultiSweeps"
    ]
    return config


def prediction_rows(
    result: Any,
    *,
    frame: dict[str, Any],
    score_threshold: float,
    class_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Convert MMDetection3D instances to the versioned LiDAR interchange rows."""
    instances = result.pred_instances_3d
    boxes = instances.bboxes_3d
    centers = boxes.gravity_center.detach().cpu().tolist()
    dimensions = boxes.dims.detach().cpu().tolist()
    yaws = boxes.yaw.detach().cpu().tolist()
    labels = instances.labels_3d.detach().cpu().tolist()
    scores = instances.scores_3d.detach().cpu().tolist()
    rows = []
    for center, size, yaw, label_index, score in zip(centers, dimensions, yaws, labels, scores):
        if score < score_threshold:
            continue
        rows.append(
            {
                "sample_token": frame["sample_token"],
                "frame_index": frame["frame_index"],
                "label": class_names[label_index],
                "score": score,
                "center_lidar_xyz_m": center,
                "size_lwh_m": size,
                "yaw_lidar_rad": yaw,
            }
        )
    return rows


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("/datasets/nuscenes"))
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--scene", default="scene-0061")
    parser.add_argument("--max-frames", type=int, default=1, help="0 runs every frame")
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("/outputs"))
    args = parser.parse_args()
    if args.max_frames < 0:
        parser.error("--max-frames must be non-negative")
    if not 0 <= args.score_threshold <= 1:
        parser.error("--score-threshold must be between zero and one")
    return args


def main() -> None:
    """Load the model once and run serial batch-one inference."""
    import torch
    from mmdet3d.apis import inference_detector, init_model

    args = arguments()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside the CenterPoint container")
    frames, index = scene_keyframes(args.dataset_root, args.version, args.scene)
    if args.max_frames:
        frames = frames[: args.max_frames]
    config = standalone_config(CONFIG)
    model = init_model(config, str(CHECKPOINT), device=args.device)
    class_names = tuple(model.dataset_meta.get("classes", CLASS_NAMES))
    rows = []
    for frame in frames:
        points = aggregate_sweeps(args.dataset_root, frame["lidar_data"], index)
        try:
            result, _ = inference_detector(model, points)
        except torch.cuda.OutOfMemoryError as error:
            raise RuntimeError(
                "CenterPoint exhausted GPU memory. Close other GPU applications or run on a GPU with at least 6 GB VRAM."
            ) from error
        frame_rows = prediction_rows(
            result,
            frame=frame,
            score_threshold=args.score_threshold,
            class_names=class_names,
        )
        rows.extend(frame_rows)
        print(
            f"frame={frame['frame_index']} token={frame['sample_token']} "
            f"points={len(points)} boxes={len(frame_rows)}",
            flush=True,
        )

    payload = {
        "schema_version": "autolabel3d-centerpoint/v1",
        "data_source": "real",
        "dataset": args.version,
        "scene": args.scene,
        "coordinate_frame": "nuscenes_lidar",
        "size_order": "lwh",
        "model": {
            "name": "CenterPoint",
            "framework": "MMDetection3D",
            "config": CONFIG.name,
            "checkpoint": f"{CHECKPOINT.name}@sha256:{CHECKPOINT_SHA256}",
            "device": args.device,
            "score_threshold": args.score_threshold,
            "multi_sweep": True,
        },
        "predictions": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / f"{args.scene}.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(target)
    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if args.device.startswith("cuda") else 0
    print(f"wrote={target} frames={len(frames)} boxes={len(rows)} peak_cuda_mb={peak_mb:.1f}", flush=True)


if __name__ == "__main__":
    main()
