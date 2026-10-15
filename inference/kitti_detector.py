"""Isolated MMDetection3D KITTI worker; no automatic dependency/weight download.

Use the official PointPillars KITTI 3-class config and its matching checkpoint.
The backend never imports MMDetection3D. First smoke run: --max-frames 1.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.services.detection.kitti_lidar import DetectorFrame, DetectorOutput, LidarBox, sha256
from src.services.geometry.kitti_tracking import load_sequence


def run(root: Path, sequence_id: str, config: Path, checkpoint: Path, output: Path,
        *, source_fps: float, fps_source: str, max_frames: int | None = None,
        device: str = "cuda:0", score_threshold: float = .3) -> dict:
    sequence = load_sequence(root, sequence_id, source_fps=source_fps, fps_source=fps_source)
    if not config.is_file() or not checkpoint.is_file():
        raise FileNotFoundError("supply official KITTI detector config and matching pretrained checkpoint")
    if not 0 <= score_threshold <= 1 or max_frames is not None and max_frames < 1:
        raise ValueError("invalid frame limit/uncalibrated detector threshold")
    # Heavy dependencies live only in the GPU environment.
    import mmdet3d
    import torch
    from mmdet3d.apis import inference_detector, init_model

    start = time.perf_counter()
    checkpoint_hash, config_hash = sha256(checkpoint), sha256(config)
    model = init_model(str(config), str(checkpoint), device=device)
    names = tuple(model.dataset_meta["classes"])
    if not {"Car", "Pedestrian"} <= set(names):
        raise ValueError("checkpoint must support KITTI Car and Pedestrian")
    metadata = {"load_seconds": time.perf_counter() - start, "device": device,
                "score_threshold": score_threshold, "threshold_calibrated": False,
                "excluded_native_classes": [n for n in names if n not in {"Car", "Pedestrian"}],
                "inference_seconds": [], "peak_cuda_allocated_bytes": None}
    frames = []
    output.parent.mkdir(parents=True, exist_ok=True)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    for frame in sequence.frames[:max_frames]:
        start = time.perf_counter()
        result, _ = inference_detector(model, str(frame.lidar))
        instances = result.pred_instances_3d
        native = instances.bboxes_3d
        if type(native).__name__ != "LiDARInstance3DBoxes":
            raise ValueError("expected native LiDAR boxes; cannot infer coordinate conversion")
        boxes = []
        for center, size, yaw, label, score in zip(native.gravity_center.cpu().tolist(), native.dims.cpu().tolist(),
                                                 native.yaw.cpu().tolist(), instances.labels_3d.cpu().tolist(),
                                                 instances.scores_3d.cpu().tolist(), strict=True):
            name = names[label]
            if name in {"Car", "Pedestrian"} and score >= score_threshold:
                boxes.append(LidarBox(label="car" if name == "Car" else "person", center_lidar_xyz_m=center,
                                      size_lwh_m=size, yaw_lidar_rad=yaw, score=score))
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
            metadata["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        metadata["inference_seconds"].append(time.perf_counter() - start)
        frames.append(DetectorFrame(source_frame_index=frame.index, lidar_sha256=sha256(frame.lidar), boxes=boxes))
        artifact = DetectorOutput.model_validate({"schema": "autolabel-kitti-detector/v1", "sequence_id": sequence_id,
            "coordinate_frame": "kitti_velodyne_x_forward_y_left_z_up", "model_name": type(model).__name__,
            "checkpoint_sha256": checkpoint_hash, "config_sha256": config_hash,
            "framework_version": mmdet3d.__version__, "runtime": metadata, "frames": frames,
            "status": "completed" if len(frames) == len(sequence.frames) else "partial"})
        temp = output.with_suffix(".tmp")
        temp.write_text(artifact.model_dump_json(by_alias=True, indent=2), encoding="utf-8")
        temp.replace(output)
    return artifact.model_dump(by_alias=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset-root", "config", "checkpoint", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--source-fps", type=float, required=True)
    parser.add_argument("--fps-source", required=True)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-threshold", type=float, default=.3)
    args = parser.parse_args()
    result = run(args.dataset_root, args.sequence, args.config, args.checkpoint, args.output,
                 source_fps=args.source_fps, fps_source=args.fps_source, max_frames=args.max_frames,
                 device=args.device, score_threshold=args.score_threshold)
    print(json.dumps({"status": result["status"], "frames": len(result["frames"]), "runtime": result["runtime"]}))


if __name__ == "__main__":
    main()
