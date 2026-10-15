"""Bounded native dataset ingestion; never unpickle uploaded model/data files."""

from __future__ import annotations

import json
import re
import stat
import tarfile
from pathlib import Path, PurePosixPath
from typing import IO
from zipfile import ZipFile, is_zipfile

import numpy as np
from PIL import Image

from src.services.geometry.kitti_tracking import load_calibration
from src.services.geometry.transforms import quaternion_rotation
from src.services.vm.contracts import file_sha256
from src.services.vm.lidar3d_contracts import CLASSES, KITTI_CLASSES

CAMERAS = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]


def safe_path(root: Path, name: str) -> Path:
    parts = PurePosixPath(name)
    if (
        not name
        or len(name) > 512
        or "\\" in name
        or ":" in name
        or parts.is_absolute()
        or ".." in parts.parts
        or "\x00" in name
    ):
        raise ValueError("Dataset contains an unsafe path")
    target = (root / str(parts)).resolve()
    if not target.is_relative_to(root.resolve()) or target == root.resolve():
        raise ValueError("Dataset path escaped its root")
    return target


def unpack(source: Path, destination: Path, max_bytes: int) -> None:
    """Extract regular files only, enforcing both declared and streamed limits."""
    destination.mkdir(mode=0o700)
    seen: set[str] = set()
    total = 0

    def copy(name: str, size: int, stream: IO[bytes]) -> None:
        nonlocal total
        target = safe_path(destination, name)
        key = str(target).casefold()
        if key in seen or len(seen) >= 20000 or size < 0 or size > 128 * 1024 * 1024:
            raise ValueError("Dataset has duplicate files, too many entries or an oversized file")
        seen.add(key)
        total += size
        if total > max_bytes:
            raise ValueError("Expanded dataset exceeds the configured limit")
        target.parent.mkdir(parents=True, exist_ok=True)
        copied = 0
        with target.open("xb") as output:
            while block := stream.read(min(1024 * 1024, size - copied + 1)):
                copied += len(block)
                if copied > size:
                    raise ValueError("Archive file exceeds its declared size")
                output.write(block)
        if copied != size:
            raise ValueError("Truncated dataset archive")

    if is_zipfile(source):
        with ZipFile(source) as archive:
            if len(archive.infolist()) > 20000:
                raise ValueError("Too many dataset archive entries")
            for item in archive.infolist():
                safe_path(destination, item.orig_filename.rstrip("/"))
                mode = item.external_attr >> 16
                if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode))):
                    raise ValueError("Dataset links/devices are not allowed")
                if not item.is_dir():
                    with archive.open(item) as stream:
                        copy(item.filename, item.file_size, stream)
    else:
        with tarfile.open(source, "r:gz") as tar_archive:
            for entry, tar_item in enumerate(tar_archive):
                if entry >= 20000:
                    raise ValueError("Too many dataset archive entries")
                if tar_item.name in (".", "./") and tar_item.isdir():
                    continue
                safe_path(destination, tar_item.name.rstrip("/"))
                if tar_item.isdir():
                    continue
                if not tar_item.isfile():
                    raise ValueError("Dataset links/devices are not allowed")
                tar_stream = tar_archive.extractfile(tar_item)
                if tar_stream is None:
                    raise ValueError("Unreadable archive member")
                with tar_stream:
                    copy(tar_item.name, tar_item.size, tar_stream)
    if not seen:
        raise ValueError("Empty dataset archive")


def table(root: Path, name: str) -> list[dict]:
    path = root / f"{name}.json"
    if not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError(f"Missing or oversized nuScenes {name}.json")
    rows = json.loads(path.read_bytes())
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Invalid nuScenes {name} table")
    return rows


def token_index(rows: list[dict]) -> dict[str, dict]:
    indexed = {row["token"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate nuScenes metadata token")
    return indexed


def validate_points(path: Path, dimensions: int) -> None:
    if not path.is_file() or not 0 < path.stat().st_size <= 64 * 1024 * 1024:
        raise ValueError("Missing or oversized LiDAR frame")
    points = np.fromfile(path, dtype="<f4")
    if points.size % dimensions or not np.isfinite(points).all():
        raise ValueError("LiDAR BIN must contain finite native float32 point channels")


def image_size(path: Path) -> list[int]:
    with Image.open(path) as image:
        if image.format not in ("JPEG", "PNG") or max(image.size) > 12000:
            raise ValueError("Camera images must be bounded PNG/JPEG files")
        size = list(image.size)
        image.verify()
        return size


def nuscenes(root: Path, scene_name: str, max_frames: int) -> tuple[Path, dict]:
    versions = [
        p.parent for p in root.rglob("scene.json") if p.parent.name in ("v1.0-mini", "v1.0-trainval", "v1.0-test")
    ]
    if len(versions) != 1:
        raise ValueError("Upload exactly one native nuScenes version directory")
    version_root = versions[0]
    base = version_root.parent
    scenes = table(version_root, "scene")
    if not scene_name and len(scenes) == 1:
        scene_name = scenes[0]["name"]
    scene = next((row for row in scenes if row["name"] == scene_name), None)
    if scene is None or not re.fullmatch(r"scene-\d{4}", scene_name):
        raise ValueError("Choose a scene name present in the uploaded nuScenes scene.json")
    samples = token_index(table(version_root, "sample"))
    data = token_index(table(version_root, "sample_data"))
    sensors = token_index(table(version_root, "sensor"))
    calibrated = token_index(table(version_root, "calibrated_sensor"))
    poses = token_index(table(version_root, "ego_pose"))
    keyframes: dict[str, dict] = {}
    for row in data.values():
        if row["is_key_frame"]:
            channel = sensors[calibrated[row["calibrated_sensor_token"]]["sensor_token"]]["channel"]
            if channel in keyframes.setdefault(row["sample_token"], {}):
                raise ValueError("Duplicate nuScenes keyframe channel")
            keyframes[row["sample_token"]][channel] = row
    frames: list[dict] = []
    seen: set[str] = set()
    token = scene["first_sample_token"]
    while token and len(frames) < max_frames:
        if token in seen:
            raise ValueError("Cycle in nuScenes sample chain")
        seen.add(token)
        sample = samples[token]
        if sample["scene_token"] != scene["token"]:
            raise ValueError("nuScenes sample belongs to another scene")
        channels = keyframes[token]
        lidar = channels["LIDAR_TOP"]
        calibration = calibrated[lidar["calibrated_sensor_token"]]
        ego = poses[lidar["ego_pose_token"]]
        quaternion_rotation(calibration["rotation"])
        quaternion_rotation(ego["rotation"])
        validate_points(safe_path(base, lidar["filename"]), 5)
        sweep = lidar
        sweep_seen: set[str] = set()
        for _ in range(9):
            if not sweep["prev"]:
                break
            sweep = data[sweep["prev"]]
            if sweep["token"] in sweep_seen or sweep["calibrated_sensor_token"] != lidar["calibrated_sensor_token"]:
                raise ValueError("Invalid nuScenes sweep chain")
            sweep_seen.add(sweep["token"])
            validate_points(safe_path(base, sweep["filename"]), 5)
        cameras, camera_info = {}, {}
        for channel in CAMERAS:
            if channel not in channels:
                continue
            row = channels[channel]
            camera_cal = calibrated[row["calibrated_sensor_token"]]
            intrinsic = np.asarray(camera_cal["camera_intrinsic"], dtype=float)
            if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
                raise ValueError("Invalid nuScenes camera intrinsic")
            image_size(safe_path(base, row["filename"]))
            cameras[channel] = row["filename"]
            camera_info[channel] = {"calibration": camera_cal, "ego_pose": poses[row["ego_pose_token"]]}
        timestamp = float(sample["timestamp"]) / 1_000_000
        if not np.isfinite(timestamp) or (frames and timestamp <= frames[-1]["timestamp_s"]):
            raise ValueError("nuScenes sample timestamps must increase")
        frames.append(
            {
                "frame": len(frames),
                "sample_token": token,
                "source_id": token,
                "timestamp_s": timestamp,
                "lidar": lidar["filename"],
                "point_dimensions": 5,
                "cameras": cameras,
                "lidar_calibration": calibration,
                "ego_pose": ego,
                "camera_info": camera_info,
            }
        )
        token = sample["next"]
    if not frames:
        raise ValueError("nuScenes scene has no frames")
    return base, {"version": version_root.name, "scene": scene_name, "frames": frames, "classes": CLASSES}


def kitti(root: Path, sequence: str, max_frames: int) -> tuple[Path, dict]:
    clouds = sorted(root.rglob("*.bin"))
    clouds = [p for p in clouds if "velodyne" in p.parts]
    if not clouds:
        raise ValueError("KITTI requires velodyne/*.bin")
    directories = {p.parent for p in clouds}
    if sequence:
        directories = {p for p in directories if p.name == sequence}
    if len(directories) != 1:
        raise ValueError("Choose one KITTI sequence per upload")
    directory = next(iter(directories))
    tracking = directory.parent.name == "velodyne"
    base = directory.parent.parent if tracking else directory.parent
    if tracking:
        sequence = directory.name
        if not re.fullmatch(r"\d{4}", sequence):
            raise ValueError("KITTI tracking sequence names must contain four digits")
    frames: list[dict] = []
    for cloud in sorted(directory.glob("*.bin"))[:max_frames]:
        if not re.fullmatch(r"\d{6}", cloud.stem):
            raise ValueError("KITTI frame names must contain six digits")
        image = base / ("image_02" if tracking else "image_2")
        image = image / sequence if tracking else image
        matches = [p for p in (image / f"{cloud.stem}.png", image / f"{cloud.stem}.jpg") if p.is_file()]
        if len(matches) != 1:
            raise ValueError("Every KITTI LiDAR frame requires its matching camera image")
        calibration_path = base / "calib" / f"{sequence if tracking else cloud.stem}.txt"
        calibration = load_calibration(calibration_path)
        rotation = calibration["R_rect"] @ calibration["Tr_velo_cam"][:, :3]
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-3) or np.linalg.det(rotation) < 0:
            raise ValueError("KITTI calibration rotation must be rigid")
        validate_points(cloud, 4)
        size = image_size(matches[0])
        frames.append(
            {
                "frame": len(frames),
                "sample_token": cloud.stem,
                "source_id": cloud.stem,
                "lidar": cloud.relative_to(base).as_posix(),
                "point_dimensions": 4,
                "cameras": {"CAM_FRONT": matches[0].relative_to(base).as_posix()},
                "image_size": size,
                "calibration": {key: value.tolist() for key, value in calibration.items()},
            }
        )
    return base, {
        "version": "kitti",
        "scene": sequence or "kitti",
        "tracking": tracking,
        "frames": frames,
        "classes": KITTI_CLASSES,
    }


def ingest(source: Path, directory: Path, *, dataset_format: str, scene: str, max_frames: int, max_bytes: int) -> dict:
    destination = directory / "input"
    unpack(source, destination, max_bytes)
    try:
        base, payload = (nuscenes if dataset_format == "nuscenes" else kitti)(destination, scene, max_frames)
        files = {p.relative_to(base).as_posix(): file_sha256(p) for p in base.rglob("*") if p.is_file()}
        return {
            "schema": "autolabel-lidar3d-dataset/v1",
            "format": dataset_format,
            "root": base.relative_to(destination).as_posix(),
            "files": files,
            **payload,
        }
    except (KeyError, TypeError, IndexError, json.JSONDecodeError) as error:
        raise ValueError("Invalid or incomplete native dataset metadata") from error
