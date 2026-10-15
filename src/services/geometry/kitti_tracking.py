"""Strict preflight for a KITTI-tracking-style synchronized 2D/3D sequence.

The OXTS archive layout has not yet been checked against the group's download.
Both a per-sequence text file and numbered per-frame text files are accepted;
timestamps use the explicitly declared cadence until measured timestamps exist.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class KittiFrame:
    index: int
    image: Path
    lidar: Path
    oxts: tuple[float, ...]
    timestamp_s: float


@dataclass(frozen=True)
class KittiSequence:
    sequence_id: str
    frames: tuple[KittiFrame, ...]
    calibration: dict[str, np.ndarray]
    image_size: tuple[int, int]
    source_fps: float
    fps_source: str


def required_paths(root: Path, sequence_id: str) -> dict[str, Path]:
    """Return expected KITTI tracking locations; does not access the network."""
    if len(sequence_id) != 4 or not sequence_id.isdigit():
        raise ValueError("KITTI tracking sequence ID must have four digits")
    base = root / "training"
    return {
        "images": base / "image_02" / sequence_id,
        "lidar": base / "velodyne" / sequence_id,
        "calibration": base / "calib" / f"{sequence_id}.txt",
        "oxts_file": base / "oxts" / f"{sequence_id}.txt",
        "oxts_frames": base / "oxts" / sequence_id,
    }


def missing_paths(root: Path, sequence_id: str) -> list[str]:
    paths = required_paths(root, sequence_id)
    missing = [str(paths[key]) for key in ("images", "lidar", "calibration") if not paths[key].exists()]
    if not paths["oxts_file"].is_file() and not paths["oxts_frames"].is_dir():
        missing.append(f"{paths['oxts_file']} OR {paths['oxts_frames']}/<frame>.txt")
    return missing


def _numbered_files(directory: Path, suffix: str) -> list[Path]:
    files = sorted(directory.glob(f"*.{suffix}"))
    if not files or [path.stem for path in files] != [f"{i:06d}" for i in range(len(files))]:
        raise ValueError(f"{directory} must have contiguous six-digit frame names from 000000")
    return files


def load_calibration(path: Path) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        key, separator, raw = line.partition(":")
        if not separator or key in values:
            raise ValueError(f"invalid or duplicate calibration key in {path}")
        try:
            values[key] = np.asarray([float(item) for item in raw.split()], dtype=np.float64)
        except ValueError as error:
            raise ValueError(f"invalid numeric calibration {key}") from error
    aliases = {"P2": ("P2",), "R_rect": ("R_rect", "R0_rect"), "Tr_velo_cam": ("Tr_velo_cam", "Tr_velo_to_cam")}
    result: dict[str, np.ndarray] = {}
    for name, options in aliases.items():
        selected = next((values[key] for key in options if key in values), None)
        expected = 9 if name == "R_rect" else 12
        if selected is None or selected.size != expected or not np.isfinite(selected).all():
            raise ValueError(f"calibration needs finite {name} with {expected} values")
        result[name] = selected.reshape(3, 3 if name == "R_rect" else 4)
    imu_to_lidar = next((values[key] for key in ("Tr_imu_velo", "Tr_imu_to_velo") if key in values), None)
    if imu_to_lidar is not None:
        if imu_to_lidar.size != 12 or not np.isfinite(imu_to_lidar).all():
            raise ValueError("Tr_imu_velo needs 12 finite values")
        result["Tr_imu_velo"] = imu_to_lidar.reshape(3, 4)
    return result


def _pose_tokens(line: str, index: int) -> tuple[float, ...]:
    try:
        packet = tuple(float(item) for item in line.split())
    except ValueError as error:
        raise ValueError(f"OXTS frame {index} is not numeric") from error
    if len(packet) != 30 or not all(math.isfinite(item) for item in packet):
        raise ValueError(f"OXTS frame {index} needs 30 finite native packet fields")
    return packet


def load_sequence(root: Path, sequence_id: str, *, source_fps: float, fps_source: str) -> KittiSequence:
    """Validate alignment and native fields before any GPU or CVAT operation."""
    if not math.isfinite(source_fps) or source_fps <= 0 or not fps_source.strip():
        raise ValueError("positive declared FPS and its provenance are required")
    paths = required_paths(root, sequence_id)
    missing = missing_paths(root, sequence_id)
    if missing:
        raise FileNotFoundError("KITTI sequence is incomplete:\n" + "\n".join(missing))
    images = _numbered_files(paths["images"], "png")
    scans = _numbered_files(paths["lidar"], "bin")
    if len(images) != len(scans):
        raise ValueError("image and Velodyne frame counts differ")
    for scan in scans:
        if scan.stat().st_size == 0 or scan.stat().st_size % 16:
            raise ValueError(f"Velodyne scan is not Nx4 float32: {scan}")
    with Image.open(images[0]) as opened:
        image_size = opened.size
    for image in images[1:]:
        with Image.open(image) as opened:
            if opened.size != image_size:
                raise ValueError(f"image dimensions changed at {image}")
    if paths["oxts_file"].is_file():
        lines = paths["oxts_file"].read_text(encoding="utf-8").splitlines()
    else:
        packets = _numbered_files(paths["oxts_frames"], "txt")
        lines = [path.read_text(encoding="utf-8").strip() for path in packets]
    if len(lines) != len(images):
        raise ValueError("image and OXTS frame counts differ")
    calibration = load_calibration(paths["calibration"])
    return KittiSequence(
        sequence_id=sequence_id,
        frames=tuple(
            KittiFrame(index, images[index], scans[index], _pose_tokens(lines[index], index), index / source_fps)
            for index in range(len(images))
        ),
        calibration=calibration,
        image_size=image_size,
        source_fps=source_fps,
        fps_source=fps_source,
    )
