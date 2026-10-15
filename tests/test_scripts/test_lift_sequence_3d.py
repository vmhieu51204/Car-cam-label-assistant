from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict

import numpy as np
import pytest
from PIL import Image

from scripts.export_kitti_3d import export_kitti_draft, kitti_row
from scripts.lift_sequence_3d import lift_sequence
from scripts.run_sequence import main as run_sequence_main
from scripts.run_sequence import run_complete_sequence
from src.services.geometry.kitti_tracking import load_sequence
from src.services.video.stitching import encode_mask


def _native_sequence(root):
    base = root / "training"
    for folder in ("image_02/0013", "velodyne/0013", "calib", "oxts"):
        (base / folder).mkdir(parents=True, exist_ok=True)
    scan = np.asarray(
        [
            [9.8, -0.3, 0.2, 1],
            [9.9, 0.3, 0.3, 1],
            [10.0, -0.2, 0.5, 1],
            [10.1, 0.2, 0.7, 1],
            [10.2, -0.3, 0.9, 1],
            [10.3, 0.3, 1.1, 1],
            [10.4, -0.1, 1.2, 1],
            [10.5, 0.1, 0.6, 1],
            [-5.0, 0, 0, 1],
        ],
        dtype="<f4",
    )
    for index in range(2):
        Image.new("RGB", (100, 100), "gray").save(base / "image_02/0013" / f"{index:06d}.png")
        scan.tofile(base / "velodyne/0013" / f"{index:06d}.bin")
    base.joinpath("calib/0013.txt").write_text(
        "P2: 10 0 50 0 0 10 50 0 0 0 1 0\n"
        "R_rect: 1 0 0 0 1 0 0 0 1\n"
        "Tr_velo_cam: 0 -1 0 0 0 0 -1 0 1 0 0 0\n"
        "Tr_imu_velo: 1 0 0 0 0 1 0 0 0 0 1 0\n",
        encoding="utf-8",
    )
    packet_a = [0.0] * 30
    packet_b = [0.0] * 30
    packet_b[1] = 0.00001
    base.joinpath("oxts/0013.txt").write_text(
        " ".join(map(str, packet_a)) + "\n" + " ".join(map(str, packet_b)) + "\n", encoding="utf-8"
    )


def test_lifts_2d_ids_to_review_only_lidar_boxes(tmp_path):
    _native_sequence(tmp_path)
    full_mask = asdict(encode_mask(np.ones((100, 100), dtype=bool)))
    empty_mask = asdict(encode_mask(np.zeros((100, 100), dtype=bool)))
    tracks = tmp_path / "stitched.json"
    tracks.write_text(
        json.dumps(
            {
                "schema": "autolabel-stitched-tracks/v1",
                "tracks": [
                    {
                        "track_id": "auto:000001",
                        "label": "car",
                        "observations": [
                            {"frame": 0, "bbox": [40, 40, 60, 60], "mask": full_mask},
                            {"frame": 1, "bbox": [40, 40, 60, 60], "mask": full_mask},
                        ],
                    },
                    {
                        "track_id": "auto:000002",
                        "label": "person",
                        "observations": [
                            {"frame": 1, "bbox": [40, 40, 60, 60], "mask": empty_mask},
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    source_map = tmp_path / "source-index-map.json"
    source_map.write_text(
        json.dumps(
            {
                "frames": [
                    {"selected_index": 0, "original_source_index": 0},
                    {"selected_index": 1, "original_source_index": 1},
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "3d-draft.json"
    result = lift_sequence(tmp_path, "0013", tracks, source_map, output, source_fps=10, fps_source="synthetic")
    assert result["review_status"] == "pending"
    assert result["counts"] == {
        "tracks": 2,
        "observations": 3,
        "cuboids": 2,
        "missing_cuboids_by_reason": {"insufficient_lidar_points": 1},
        "source_frames_read": 2,
    }
    car = result["tracks"][0]
    assert car["track_id"] == "auto:000001"
    assert [row["source_frame_index"] for row in car["observations"]] == [0, 1]
    assert all(row["box_lidar"]["point_count"] == 8 for row in car["observations"])
    assert all(row["needs_review"] for row in car["observations"])
    assert car["observations"][0]["center_world_xyz_m"] != car["observations"][1]["center_world_xyz_m"]
    assert result["tracks"][1]["observations"][0]["box_lidar"] is None
    assert json.loads(output.read_text(encoding="utf-8"))["counts"] == result["counts"]
    calibration = load_sequence(tmp_path, "0013", source_fps=10, fps_source="synthetic").calibration
    exported = export_kitti_draft(output, calibration, tmp_path / "predictions" / "0013.txt")
    assert exported["rows"] == 2
    assert exported["skipped_no_cuboid"] == 1
    for line in (tmp_path / "predictions/0013.txt").read_text().splitlines():
        fields = line.split()
        assert len(fields) == 18
        assert fields[2] == "Car"
        assert float(fields[12]) >= 3.5
    cvat_labels = tmp_path / "cvat-labels.json"
    cvat_labels.write_text(
        json.dumps(
            {
                "labels": [
                    {
                        "id": index,
                        "name": name,
                        "attributes": [
                            {"id": index * 10 + 1, "name": "autolabel_track_id", "mutable": False},
                            {"id": index * 10 + 2, "name": "review_status", "mutable": True},
                        ],
                    }
                    for index, name in ((1, "car"), (2, "person"))
                ]
            }
        ),
        encoding="utf-8",
    )
    complete = run_complete_sequence(
        tmp_path,
        "0013",
        tmp_path / "complete",
        source_fps=10,
        fps_source="synthetic",
        tracks_path=tracks,
        source_map_path=source_map,
        cvat_3d_labels=cvat_labels,
    )
    assert complete["counts_3d"]["cuboids"] == 2
    assert complete["kitti_rows"] == 2
    assert complete["cvat_3d_published"] is False
    assert (
        json.loads((tmp_path / "complete/3d/cvat-3d-draft.json").read_text())["tracks"][0]["shapes"][0]["type"]
        == "cuboid"
    )
    assert (tmp_path / "complete/run-summary.json").is_file()
    source_map.write_text(
        json.dumps(
            {
                "source_fps": 20,
                "frames": [
                    {"selected_index": 0, "original_source_index": 0},
                    {"selected_index": 1, "original_source_index": 1},
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="FPS differs"):
        lift_sequence(
            tmp_path, "0013", tracks, source_map, tmp_path / "bad.json", source_fps=10, fps_source="synthetic"
        )


def test_kitti_yaw_and_bottom_center_are_in_camera_coordinates():
    calibration = {
        "Tr_velo_cam": np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0]], dtype=float),
        "R_rect": np.eye(3),
    }
    observation = {
        "bbox_2d_xyxy": [40, 40, 60, 60],
        "box_lidar": {"center_lidar_xyz_m": [10, 0, 0], "size_lwh_m": [4, 2, 2], "yaw_lidar_rad": 0},
        "source_2d_score": 0.8,
    }
    fields = kitti_row(7, 3, "car", observation, calibration).split()
    assert fields[:3] == ["7", "3", "Car"]
    assert [float(item) for item in fields[10:16]] == [2, 2, 4, 0, 1, 10]
    assert float(fields[16]) == -math.pi / 2 or abs(float(fields[16]) + math.pi / 2) < 1e-6


def test_sequence_cli_reports_missing_lidar_before_gpu(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_sequence",
            "--dataset-root",
            str(tmp_path),
            "--sequence",
            "0013",
            "--source-fps",
            "10",
            "--fps-source",
            "synthetic",
            "--tracks",
            str(tmp_path / "tracks.json"),
            "--source-map",
            str(tmp_path / "source-map.json"),
            "--output",
            str(tmp_path / "output"),
        ],
    )
    with pytest.raises(SystemExit) as stopped:
        run_sequence_main()
    assert stopped.value.code == 2
    assert "velodyne" in capsys.readouterr().err
    assert not (tmp_path / "output").exists()
