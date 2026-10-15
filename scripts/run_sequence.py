"""Run the current 2D→LiDAR-3D draft path for one KITTI tracking sequence.

Requires synchronized KITTI image_02, velodyne, calib and oxts. Use --tracks
and --source-map to reuse a completed Kaggle 2D draft without rerunning SAM3.
All outputs remain pending review; this command does not publish to CVAT.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from scripts.export_cvat_3d import build_cvat_3d_draft
from scripts.export_kitti_3d import export_kitti_draft
from scripts.import_kitti_detector import detector_draft
from scripts.lift_sequence_3d import lift_sequence
from scripts.run_video_2d import run_sequence as run_2d
from src.services.geometry.kitti_tracking import load_sequence
from src.services.geometry.pointcloud_export import PointCloudExportConfig, display_points, split_cuboid_draft


def run_complete_sequence(
    dataset_root: Path,
    sequence_id: str,
    output: Path,
    *,
    source_fps: float,
    fps_source: str,
    tracks_path: Path | None = None,
    source_map_path: Path | None = None,
    checkpoint: Path | None = None,
    target_fps: float | None = None,
    classes: tuple[str, ...] = ("car", "person"),
    cvat_3d_labels: Path | None = None,
    detector_output: Path | None = None,
    experimental_mask_lift: bool | None = None,
    detector_checkpoint_sha256: str | None = None,
    detector_config_sha256: str | None = None,
    display_config: PointCloudExportConfig | None = None,
) -> dict:
    """Preflight first, then make 2D, provisional 3D, and KITTI artifacts."""
    if (tracks_path is None) != (source_map_path is None):
        raise ValueError("--tracks and --source-map must be provided together")
    if tracks_path is None and checkpoint is None:
        raise ValueError("supply an existing 2D draft or a private SAM3 checkpoint")
    labels = None
    if cvat_3d_labels is not None:
        raw_labels = json.loads(cvat_3d_labels.read_text(encoding="utf-8"))
        labels = raw_labels.get("labels") if isinstance(raw_labels, dict) else raw_labels
        if not isinstance(labels, list):
            raise ValueError("CVAT 3D label snapshot must be a list")
    started = time.perf_counter()
    sequence = load_sequence(dataset_root, sequence_id, source_fps=source_fps, fps_source=fps_source)
    if "Tr_imu_velo" not in sequence.calibration:
        raise ValueError("Tr_imu_velo calibration is required for world-frame track centers")
    # None preserves callers of the old Python API; the CLI explicitly passes False.
    if detector_output is None and experimental_mask_lift is False:
        raise ValueError("primary 3D path requires --detector-output; mask lift needs --experimental-mask-lift")
    if detector_output is not None:
        from src.services.detection.kitti_lidar import validate_output

        validated = validate_output(detector_output, sequence)
        if detector_checkpoint_sha256 is not None and validated.checkpoint_sha256 != detector_checkpoint_sha256:
            raise ValueError("detector checkpoint hash differs from configured pin")
        if detector_config_sha256 is not None and validated.config_sha256 != detector_config_sha256:
            raise ValueError("detector config hash differs from configured pin")
    preflight_s = time.perf_counter() - started
    output.mkdir(parents=True, exist_ok=True)
    two_d_s = 0.0
    if tracks_path is None:
        tick = time.perf_counter()
        two_d = run_2d(
            dataset_root / "training" / "image_02" / sequence_id,
            output / "2d",
            source_fps=source_fps,
            target_fps=target_fps,
            labels=classes,
            checkpoint=checkpoint,
        )
        tracks_path = two_d.stitched
        source_map_path = output / "2d" / "sequence-manifest.json"
        two_d_s = time.perf_counter() - tick
    assert tracks_path is not None and source_map_path is not None
    draft_path = output / "3d" / "lidar-draft.json"
    draft = detector_draft(sequence, detector_output, tracks_path, source_map_path, draft_path) if detector_output else lift_sequence(
        dataset_root,
        sequence_id,
        tracks_path,
        source_map_path,
        draft_path,
        source_fps=source_fps,
        fps_source=fps_source,
    )
    kitti_path = output / "3d" / f"{sequence_id}.txt"
    kitti = export_kitti_draft(draft_path, sequence.calibration, kitti_path)
    cvat_draft_path = None
    if labels is not None:
        cvat_draft_path = output / "3d" / "cvat-3d-draft.json"
        cvat_draft = build_cvat_3d_draft(draft, labels, source_frame_count=len(sequence.frames))
        cvat_draft_path.write_text(json.dumps(cvat_draft, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if display_config is not None and display_config.enabled:
            import hashlib

            import numpy as np
            display_dir = output / "3d" / "display"
            display_dir.mkdir(parents=True, exist_ok=True)
            frames = []
            for frame in sequence.frames:
                points = np.fromfile(frame.lidar, dtype="<f4").reshape(-1, 4)
                selected = display_points(points, display_config, calibration=sequence.calibration, image_size=sequence.image_size)
                pcd = display_dir / f"{frame.index:06d}.pcd"
                header = ("VERSION .7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n"
                          f"WIDTH {len(selected)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(selected)}\nDATA binary\n")
                pcd.write_bytes(header.encode("ascii") + selected.astype("<f4").tobytes())
                frames.append({"source_frame":frame.index,"before":len(points),"after":len(selected),
                               "source_sha256":hashlib.sha256(frame.lidar.read_bytes()).hexdigest(),"pcd":pcd.name})
            tasks = split_cuboid_draft(cvat_draft, display_config.max_task_frames)
            for i, task in enumerate(tasks):
                (display_dir / f"task-{i:04d}.json").write_text(json.dumps(task), encoding="utf-8")
            (display_dir / "manifest.json").write_text(json.dumps({"config":display_config.model_dump(),"frames":frames,
                "tasks":[{"index":i,"source_frames":t["source_frames"]} for i,t in enumerate(tasks)],
                "cuboids_changed":False,"visually_verified":False}, indent=2), encoding="utf-8")
    summary = {
        "schema": "autolabel-sequence-run/v1",
        "sequence_id": sequence_id,
        "source_frames": len(sequence.frames),
        "2d_tracks": str(tracks_path),
        "3d_draft": str(draft_path),
        "kitti_tracking_draft": str(kitti_path),
        "cvat_3d_draft": str(cvat_draft_path) if cvat_draft_path else None,
        "counts_3d": draft["counts"],
        "geometry_source": draft.get("geometry_source", "mask_lift_experimental"),
        "kitti_rows": kitti["rows"],
        "review_status": "pending",
        "3d_geometry_calibrated": False,
        "cvat_3d_published": False,
        "timing_s": {
            "preflight": preflight_s,
            "2d_inference_and_export": two_d_s,
            "3d_geometry": draft["timing_s"]["total"],
            "total": time.perf_counter() - started,
        },
    }
    (output / "run-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-fps", type=float, required=True)
    parser.add_argument("--fps-source", required=True)
    parser.add_argument("--tracks", type=Path)
    parser.add_argument("--source-map", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--target-fps", type=float, help="Legacy sampled runs only; omit for all frames")
    parser.add_argument("--classes", default="car,person")
    parser.add_argument("--cvat-3d-labels", type=Path, help="Read-only CVAT 3D job label snapshot for an offline draft")
    parser.add_argument("--detector-output", type=Path, help="Validated full-frame KITTI detector JSON; no nuScenes fallback")
    parser.add_argument("--experimental-mask-lift", action="store_true")
    parser.add_argument("--detector-checkpoint-sha256")
    parser.add_argument("--detector-config-sha256")
    parser.add_argument("--display-preprocess", action="store_true")
    parser.add_argument("--display-max-range", type=float, default=60)
    parser.add_argument("--display-voxel", type=float, default=0.15)
    parser.add_argument("--display-task-frames", type=int, default=100)
    args = parser.parse_args()
    if args.detector_output and (not args.detector_checkpoint_sha256 or not args.detector_config_sha256):
        parser.error("detector import requires configured checkpoint/config SHA-256 pins")
    try:
        result = run_complete_sequence(
            args.dataset_root,
            args.sequence,
            args.output,
            source_fps=args.source_fps,
            fps_source=args.fps_source,
            tracks_path=args.tracks,
            source_map_path=args.source_map,
            checkpoint=args.checkpoint,
            target_fps=args.target_fps,
            classes=tuple(name.strip() for name in args.classes.split(",")),
            cvat_3d_labels=args.cvat_3d_labels,
            detector_output=args.detector_output,
            experimental_mask_lift=args.experimental_mask_lift,
            detector_checkpoint_sha256=args.detector_checkpoint_sha256,
            detector_config_sha256=args.detector_config_sha256,
            display_config=PointCloudExportConfig(enabled=args.display_preprocess,
                max_range_m=args.display_max_range, voxel_size_m=args.display_voxel, max_task_frames=args.display_task_frames),
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"Sequence cannot start: {error}", file=sys.stderr)
        raise SystemExit(2) from None
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
