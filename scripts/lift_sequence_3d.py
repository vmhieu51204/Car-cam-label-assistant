"""Lift an existing SAM3 2D draft onto synchronized KITTI LiDAR scans.

The output is a provisional 3D artifact with explicit review flags. It does
not publish to CVAT and cannot be validated without real Velodyne/calib/OXTS.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.services.geometry.kitti_projection import (
    project_lidar,
    transform_xyz,
    world_from_lidar_pose,
    world_from_oxts_imu,
)
from src.services.geometry.kitti_tracking import load_sequence
from src.services.geometry.mask_lift import LiftConfig, lift_observation
from src.services.video.stitching import MaskRLE, decode_mask


def _source_map(path: Path, *, source_fps: float) -> dict[int, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapped_fps = payload.get("source_fps")
    if mapped_fps is not None and abs(float(mapped_fps) - source_fps) > 1e-6:
        raise ValueError("2D source FPS differs from LiDAR sequence FPS")
    if "frames" in payload:
        mapping = {int(row["selected_index"]): int(row["original_source_index"]) for row in payload["frames"]}
    elif "selected_source_frames" in payload:
        mapping = {index: int(source) for index, source in enumerate(payload["selected_source_frames"])}
    else:
        raise ValueError("source map needs frames or selected_source_frames")
    sources = [mapping[index] for index in sorted(mapping)]
    if sorted(mapping) != list(range(len(mapping))) or sources != sorted(set(sources)):
        raise ValueError("source map must have contiguous selected and increasing source indices")
    return mapping


def _mask(raw: dict | None, image_size: tuple[int, int]) -> np.ndarray | None:
    if raw is None:
        return None
    encoded = MaskRLE(
        int(raw["height"]), int(raw["width"]), tuple(int(value) for value in raw["counts"]), raw["sha256"]
    )
    if (encoded.width, encoded.height) != image_size:
        raise ValueError("SAM3 mask size differs from KITTI camera image")
    return decode_mask(encoded)


def lift_sequence(
    dataset_root: Path,
    sequence_id: str,
    tracks_path: Path,
    source_map_path: Path,
    output: Path,
    *,
    source_fps: float,
    fps_source: str,
    config: LiftConfig | None = None,
) -> dict:
    """Produce review-only 3D observations with inherited 2D IDs and world centers."""
    started = time.perf_counter()
    config = config or LiftConfig()
    sequence = load_sequence(dataset_root, sequence_id, source_fps=source_fps, fps_source=fps_source)
    mapping = _source_map(source_map_path, source_fps=source_fps)
    if not mapping or max(mapping.values()) >= len(sequence.frames):
        raise ValueError("2D source index exceeds KITTI sequence")
    payload = json.loads(tracks_path.read_text(encoding="utf-8"))
    if payload.get("schema") not in {
        "autolabel-stitched-tracks/v1",
        "autolabel-stitch-comparison/v1",
        "autolabel-sam3-benchmark/v1",
    } or not isinstance(payload.get("tracks"), list):
        raise ValueError("expected a stitched SAM3 track artifact")
    tracks = payload["tracks"]
    by_source: dict[int, list[tuple[int, dict]]] = defaultdict(list)
    output_tracks = []
    for track_index, track in enumerate(tracks):
        if not track.get("track_id") or track.get("label") not in {"car", "person"}:
            raise ValueError("3D lift supports identified car/person tracks only")
        result = {
            "track_id": track["track_id"],
            "label": track["label"],
            "needs_review": True,
            "review_reasons": list(track.get("review_reasons", [])),
            "observations": [],
        }
        output_tracks.append(result)
        observed_frames = set()
        for observation in track.get("observations", []):
            selected = int(observation["frame"])
            if selected not in mapping or selected in observed_frames:
                raise ValueError("track has duplicate or unmapped selected frame")
            observed_frames.add(selected)
            by_source[mapping[selected]].append((track_index, observation))

    origin = sequence.frames[0].oxts
    imu_to_lidar = sequence.calibration.get("Tr_imu_velo")
    read_s = project_s = lift_s = 0.0
    for source_index, observations in sorted(by_source.items()):
        frame = sequence.frames[source_index]
        tick = time.perf_counter()
        scan = np.fromfile(frame.lidar, dtype="<f4")
        if scan.size % 4:
            raise ValueError(f"invalid Nx4 scan: {frame.lidar}")
        scan = scan.reshape(-1, 4)
        read_s += time.perf_counter() - tick
        tick = time.perf_counter()
        projection = project_lidar(scan[:, :3], calibration=sequence.calibration, image_size=sequence.image_size)
        project_s += time.perf_counter() - tick
        world_from_lidar = None
        if imu_to_lidar is not None:
            world_from_lidar = world_from_lidar_pose(world_from_oxts_imu(frame.oxts, origin), imu_to_lidar)
        for track_index, raw in observations:
            tick = time.perf_counter()
            track = output_tracks[track_index]
            bbox = tuple(float(value) for value in raw["bbox"])
            box, reasons = lift_observation(
                scan,
                projection,
                image_size=sequence.image_size,
                label=track["label"],
                bbox_xyxy=bbox,
                mask=_mask(raw.get("mask"), sequence.image_size),
                config=config,
            )
            note = {
                "selected_frame_index": int(raw["frame"]),
                "source_frame_index": source_index,
                "timestamp_s": frame.timestamp_s,
                "bbox_2d_xyxy": bbox,
                "source_2d_score": raw.get("score"),
                "box_lidar": asdict(box) if box else None,
                "center_world_xyz_m": None,
                "point_count": box.point_count if box else 0,
                "needs_review": True,
                "review_reasons": list(reasons),
            }
            if box and world_from_lidar is not None:
                world_center = transform_xyz(np.asarray([box.center_lidar_xyz_m]), world_from_lidar)[0]
                note["center_world_xyz_m"] = [float(value) for value in world_center]
            elif imu_to_lidar is None:
                note["review_reasons"].append("missing_imu_to_lidar_calibration")
            track["observations"].append(note)
            lift_s += time.perf_counter() - tick
    for track in output_tracks:
        track["observations"].sort(key=lambda row: row["source_frame_index"])
    observations = [obs for track in output_tracks for obs in track["observations"]]
    result = {
        "schema": "autolabel-lidar-3d-draft/v1",
        "sequence_id": sequence_id,
        "coordinate_frame": "kitti_velodyne_x_forward_y_left_z_up",
        "source_fps": source_fps,
        "fps_source": fps_source,
        "review_status": "pending",
        "calibrated": False,
        "config": asdict(config),
        "tracks": output_tracks,
        "counts": {
            "tracks": len(output_tracks),
            "observations": len(observations),
            "cuboids": sum(item["box_lidar"] is not None for item in observations),
            "missing_cuboids_by_reason": dict(
                Counter(
                    reason for item in observations if item["box_lidar"] is None for reason in item["review_reasons"]
                )
            ),
            "source_frames_read": len(by_source),
        },
        "timing_s": {
            "read_lidar": read_s,
            "project_lidar": project_s,
            "lift_observations": lift_s,
            "total": time.perf_counter() - started,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--source-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-fps", type=float, required=True)
    parser.add_argument("--fps-source", required=True)
    parser.add_argument("--min-points", type=int, default=6)
    parser.add_argument("--depth-radius-m", type=float, default=3.0)
    args = parser.parse_args()
    result = lift_sequence(
        args.dataset_root,
        args.sequence,
        args.tracks,
        args.source_map,
        args.output,
        source_fps=args.source_fps,
        fps_source=args.fps_source,
        config=LiftConfig(min_points=args.min_points, depth_radius_m=args.depth_radius_m),
    )
    print(
        json.dumps(
            {"output": str(args.output), "counts": result["counts"], "timing_s": result["timing_s"]}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
