"""BEVFusion LiDAR + six-camera inference from validated native nuScenes data."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from mmdet3d.apis import init_model
from mmdet3d.structures import get_box_type
from mmengine.config import Config
from mmengine.dataset import Compose, pseudo_collate
from run import aggregate_sweeps, prediction_rows, quaternion_rotation, scene_keyframes, transform_name

CHECKPOINT = Path("/models/bevfusion-5239b1af.pth")
CHECKPOINT_SHA = "e68c6c084d8971f4f879751b0ed3981f575c24ee3cf385a1e39197c26a9bb3ab"
CONFIG = Path("/opt/mmdetection3d/projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py")
CAMERAS = ("CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")


def pose(calibration, ego):
    rotation_ego = quaternion_rotation(ego["rotation"])
    matrix = np.eye(4)
    matrix[:3, :3] = rotation_ego @ quaternion_rotation(calibration["rotation"])
    matrix[:3, 3] = rotation_ego @ np.asarray(calibration["translation"]) + np.asarray(ego["translation"])
    return matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest["format"] != "nuscenes" or not torch.cuda.is_available():
        raise ValueError("BEVFusion requires the native nuScenes sensor bundle and CUDA")
    for name, expected in manifest["files"].items():
        path = (args.dataset_root / name).resolve()
        if not path.is_relative_to(args.dataset_root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Dataset input checksum changed")
    config = Config.fromfile(str(CONFIG))
    config.model.img_backbone.init_cfg = None
    transforms = [item for item in config.test_dataloader.dataset.pipeline if transform_name(item) != "LoadPointsFromMultiSweeps"]
    for item in transforms:
        if transform_name(item) == "BEVLoadMultiViewImageFromFiles":
            item["num_views"] = 6
    model = init_model(config, str(CHECKPOINT), device="cuda:0")
    # init_model establishes MMDetection3D's registry scope for project transforms.
    pipeline = Compose(transforms)
    if type(model).__name__ != "BEVFusion":
        raise ValueError("Loaded model is not BEVFusion")
    source_frames, index = scene_keyframes(args.dataset_root, manifest["version"], manifest["scene"], max_frames=len(manifest["frames"]))
    if [f["sample_token"] for f in source_frames] != [f["sample_token"] for f in manifest["frames"]]:
        raise ValueError("Native sample identity changed")
    rows = []
    temporary_cloud = args.manifest.parent / "aligned-inference.bin"
    try:
        for frame, sensors in zip(source_frames, manifest["frames"], strict=True):
            if set(sensors["cameras"]) != set(CAMERAS):
                raise ValueError("BEVFusion checkpoint requires all six native nuScenes cameras")
            points = aggregate_sweeps(args.dataset_root, frame["lidar_data"], index)
            points.tofile(temporary_cloud)
            world_from_lidar = pose(sensors["lidar_calibration"], sensors["ego_pose"])
            images = {}
            for name in CAMERAS:
                info = sensors["camera_info"][name]
                camera_from_lidar = np.linalg.inv(pose(info["calibration"], info["ego_pose"])) @ world_from_lidar
                images[name] = {"img_path": str(args.dataset_root / sensors["cameras"][name]),
                                "cam2img": info["calibration"]["camera_intrinsic"], "lidar2cam": camera_from_lidar.tolist()}
            box_type, box_mode = get_box_type("LiDAR")
            data = pipeline({"lidar_points": {"lidar_path": str(temporary_cloud), "num_pts_feats": 5},
                             "images": images, "box_type_3d": box_type, "box_mode_3d": box_mode,
                             "sample_idx": frame["frame_index"], "timestamp": frame["timestamp"] / 1e6})
            with torch.no_grad():
                result = model.test_step(pseudo_collate([data]))[0]
            frame_rows = prediction_rows(result, frame=frame, score_threshold=0.3,
                                         class_names=tuple(model.dataset_meta["classes"]))
            rows.extend(frame_rows)
            print(f"frame={frame['frame_index']} token={frame['sample_token']} boxes={len(frame_rows)} model=BEVFusion", flush=True)
    finally:
        temporary_cloud.unlink(missing_ok=True)
    payload = {"schema_version": "autolabel-lidar3d-upload/v1", "dataset_format": "nuscenes",
               "scene": manifest["scene"], "model_name": "BEVFusion", "checkpoint_sha256": CHECKPOINT_SHA,
               "frames": [f["sample_token"] for f in manifest["frames"]], "predictions": rows}
    target = args.manifest.parent / "uploaded-predictions.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload))
    temporary.replace(target)


if __name__ == "__main__":
    main()
