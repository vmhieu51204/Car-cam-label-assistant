"""Local CUDA YOLO/ByteTrack evidence on native cameras, in the SAM environment."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from src.services.vm.contracts import file_sha256
from src.services.vm.hybrid import IndexedBoxes, TrackerPool, indexed_tracker
from src.services.vm.hybrid_contracts import HybridProfile
from src.services.vm.lidar3d_contracts import UploadOutput
from src.services.vm.lidar3d_datasets import safe_path
from src.services.vm.lidar3d_links import associate
from src.services.vm.storage import atomic_json


def run(directory: Path, base: Path) -> None:
    os.environ["YOLO_AUTOINSTALL"] = "false"
    (directory / "yolo-settings/Ultralytics").mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(directory / "yolo-settings")
    import torch
    from ultralytics import YOLO, settings
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.byte_tracker import BYTETracker

    profile = HybridProfile.model_validate_json((directory / "camera-profile.json").read_bytes())
    if not torch.cuda.is_available() or importlib.metadata.version("ultralytics") != profile.ultralytics_version:
        raise ValueError("Pinned camera CUDA detector runtime unavailable")
    if file_sha256(Path(profile.checkpoint)) != profile.checkpoint_sha256:
        raise ValueError("Camera detector checkpoint differs")
    settings.update({"sync": False})
    model = YOLO(profile.checkpoint, task="detect")
    if {int(k): v for k, v in model.names.items()} != profile.detector_names:
        raise ValueError("Camera detector taxonomy differs")
    manifest = json.loads((directory / "dataset.json").read_bytes())
    output = UploadOutput.model_validate_json((directory / "tracked-predictions.json").read_bytes())
    tracker_type = indexed_tracker(BYTETracker)
    pools = {}
    links, camera_tracks = [], []
    for frame in manifest["frames"]:
        if (directory.parent / "cancelled.json").exists():
            raise ValueError("3D job cancelled")
        rows = [row for row in output.predictions if row.frame_index == frame["frame"]]
        for channel, name in sorted(frame["cameras"].items()):
            image = safe_path(base, name)
            if file_sha256(image) != manifest["files"][name]:
                raise ValueError("Native camera identity changed")
            with Image.open(image) as source:
                width, height = source.size
            result = model.predict(str(image), device=0, verbose=False, imgsz=profile.image_size,
                conf=profile.detector_conf, iou=profile.detector_iou, classes=list(profile.class_mapping.values()))[0]
            detections = [{"box": row[:4].tolist(), "score": float(row[4]), "class_id": int(row[5])}
                for row in result.boxes.cpu().numpy().data]
            if channel not in pools:
                pools[channel] = TrackerPool(profile, lambda: tracker_type(SimpleNamespace(**profile.tracker_config)),
                    lambda data, shape: IndexedBoxes(Boxes(data, shape)))
            independent = manifest["format"] == "kitti" and not manifest.get("tracking")
            discontinuity = (manifest["format"] == "kitti" and frame["frame"] > 0 and
                int(frame["source_id"]) != int(manifest["frames"][frame["frame"] - 1]["source_id"]) + 1)
            observations = pools[channel].update({"frame_index": frame["frame"], "width": width,
                "height": height, "scene_cut_before": independent or discontinuity}, detections)
            for observation in observations:
                namespace = f"{manifest['scene']}:{frame['source_id']}" if independent else manifest["scene"]
                observation["track_id"] = f"{namespace}:{channel}:{observation['track_id']}"
            camera_tracks.append({"frame": frame["frame"], "camera": channel, "observations": observations})
            linked = associate(rows, observations, frame, channel, width, height)
            links.extend(linked)
            for link in linked:
                if link["status"] == "matched":
                    rows[link["prediction_index"]].linked_2d_tracks[channel] = link["track_2d_id"]
    atomic_json(directory / "camera-links.json", {"schema": "autolabel-3d-camera-links/v1",
        "dataset_sha256": file_sha256(directory / "dataset.json"), "detector_sha256": profile.checkpoint_sha256,
        "tracker_sha256": profile.tracker_sha256, "links": links, "camera_tracks": camera_tracks})
    atomic_json(directory / "linked-predictions.json", output.model_dump(mode="json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    args = parser.parse_args()
    run(args.directory, args.base)
