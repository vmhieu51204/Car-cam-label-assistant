import json
import math
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile

import pytest

from src.services.vm.lidar3d import cuboid_shapes
from src.services.vm.lidar3d_contracts import UploadOutput
from src.services.vm.lidar3d_datasets import kitti, nuscenes
from src.services.vm.lidar3d_exchange import export_annotations, imported_annotations, saved_boxes
from src.services.vm.lidar3d_links import associate
from src.services.vm.lidar3d_review import review_objects, reviewed_annotations, revision
from src.services.vm.lidar3d_tracking import iter_cuboids, native_tracks, track_predictions
from tests.test_services.test_lidar3d_datasets import native_kitti as native_kitti
from tests.test_services.test_lidar3d_datasets import native_nuscenes as native_nuscenes


def labels(name="car"):
    return [
        {
            "id": 1,
            "name": name,
            "attributes": [
                {"id": index, "name": name}
                for index, name in enumerate(
                    ("review_status", "confidence", "detector", "autolabel_3d_track_id", "linked_2d_tracks"), 1
                )
            ],
        }
    ]


def test_isolated_camera_worker_does_not_import_backend_only_geometry():
    script = """
import builtins
original = builtins.__import__
def no_backend_geometry(name, *args, **kwargs):
    if name == 'shapely' or name.startswith('shapely.'):
        raise ImportError('Backend-only geometry unavailable in camera runtime')
    return original(name, *args, **kwargs)
builtins.__import__ = no_backend_geometry
import scripts.run_vm_3d_camera
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def output(frames=4):
    return UploadOutput.model_validate(
        {
            "schema_version": "autolabel-lidar3d-upload/v1",
            "dataset_format": "nuscenes",
            "scene": "scene-0001",
            "model_name": "BEVFusion",
            "checkpoint_sha256": "a" * 64,
            "frames": [str(i) for i in range(frames)],
            "predictions": [
                {
                    "frame_index": i,
                    "sample_token": str(i),
                    "label": "car",
                    "score": 0.9,
                    "center_lidar_xyz_m": [10 - i, 0, 0],
                    "size_lwh_m": [4, 2, 2],
                    "yaw_lidar_rad": 0,
                }
                for i in range(frames)
                if i != 2
            ],
        }
    )


def manifest(frames=4):
    return {
        "format": "nuscenes",
        "scene": "scene-0001",
        "frames": [
            {
                "frame": i,
                "sample_token": str(i),
                "timestamp_s": i * 0.5,
                "ego_pose": {"rotation": [1, 0, 0, 0], "translation": [i, 0, 0]},
                "lidar_calibration": {"rotation": [1, 0, 0, 0], "translation": [0, 0, 0]},
            }
            for i in range(frames)
        ],
    }


def test_global_tracking_compensates_ego_motion_and_does_not_fill_gaps():
    dataset = manifest()
    tracked = track_predictions(output(), dataset)
    assert len({row.track_id for row in tracked.predictions}) == 1
    assert tracked.predictions[1].center_lidar_xyz_m == (9, 0, 0)  # Association never smooths published geometry.
    shapes = cuboid_shapes(tracked, dataset["frames"], labels(), "BEVFusion")
    payload = native_tracks(shapes, tracked, labels(), 4)
    assert len(payload["tracks"]) == 1 and not payload["shapes"]
    assert next(shape for shape in payload["tracks"][0]["shapes"] if shape["frame"] == 2)["outside"]
    assert [shape["frame"] for shape in iter_cuboids(payload, 4)] == [0, 1, 3]
    assert len(saved_boxes(payload, labels(), dataset["frames"])) == 3


def test_native_import_keeps_nuscenes_instance_identity(native_nuscenes):
    base, dataset = nuscenes(native_nuscenes, "", 200)
    dataset["format"] = "nuscenes"
    imported = imported_annotations(base, dataset)
    assert imported.predictions[0].track_id == "instance"
    assert track_predictions(imported, dataset) is imported


def test_saved_track_edits_short_arc_angles_and_training_gate(tmp_path):
    dataset = manifest(3)
    tracked = track_predictions(output(3), dataset)
    payload = native_tracks(cuboid_shapes(tracked, dataset["frames"], labels(), "BEVFusion"), tracked, labels(), 3)
    track = payload["tracks"][0]
    track["id"] = 50
    first = track["shapes"][0]
    last = deepcopy(first)
    first["points"][5] = math.radians(179)
    last.update(frame=2)
    last["points"][0] = 14
    last["points"][5] = math.radians(-179)
    track["shapes"] = [first, last]
    expanded = list(iter_cuboids(payload, 3))
    assert expanded[1]["points"][0] == 12
    assert abs(expanded[1]["points"][5]) == pytest.approx(math.pi)
    path = tmp_path / "results.json"
    with pytest.raises(ValueError, match="Review required"):
        export_annotations(dataset, payload, labels(), path, purpose="training")
    assert not path.exists()
    pending_revision = revision(payload)
    accepted = reviewed_annotations(payload, labels(), "tracks:50", "accepted")
    assert revision(accepted) != pending_revision
    assert payload["tracks"][0]["shapes"][0]["attributes"][0]["value"] == "pending"
    assert review_objects(accepted, labels())[0]["review_status"] == "accepted"
    receipt = export_annotations(dataset, accepted, labels(), path, purpose="training", annotation_type="tracking")
    boxes = json.loads(path.read_text())["results"]
    assert receipt["boxes"] == 3 and receipt["pending_review"] == 0
    assert boxes["1"][0]["tracking_id"] == tracked.predictions[0].track_id
    assert boxes["1"][0]["translation"][0] == 13  # User edit + that frame's ego pose.
    rejected = reviewed_annotations(accepted, labels(), "tracks:50", "rejected")
    assert export_annotations(dataset, rejected, labels(), path, purpose="training")["boxes"] == 0


def test_kitti_tracking_import_and_export_preserve_source_frame_and_id(native_kitti, tmp_path):
    root = native_kitti
    for name in ("velodyne", "image_02", "calib", "label_02"):
        (root / name / "0000" if name in ("velodyne", "image_02") else root / name).mkdir(exist_ok=True, parents=True)
    (root / "velodyne/000123.bin").rename(root / "velodyne/0000/000123.bin")
    (root / "image_2/000123.png").rename(root / "image_02/0000/000123.png")
    (root / "calib/000123.txt").rename(root / "calib/0000.txt")
    text = (root / "label_2/000123.txt").read_text()
    (root / "label_02/0000.txt").write_text("123 42 " + text)
    base, dataset = kitti(root, "0000", 200)
    dataset["format"] = "kitti"
    imported = imported_annotations(base, dataset)
    assert imported.predictions[0].track_id == "0000:42"
    payload = native_tracks(
        cuboid_shapes(imported, dataset["frames"], labels("Car"), "Imported"), imported, labels("Car"), 1
    )
    path = tmp_path / "tracking.zip"
    export_annotations(dataset, payload, labels("Car"), path, annotation_type="tracking")
    with ZipFile(path) as archive:
        fields = archive.read("label_02/0000.txt").decode().split()
        assert fields[:3] == ["123", "42", "Car"] and len(fields) == 18


def test_camera_matching_rejects_ambiguity_wrong_class_and_low_overlap(monkeypatch):
    rows = output(1).predictions
    monkeypatch.setattr("src.services.vm.lidar3d_links.projected_box", lambda *args: (10, 10, 30, 30))
    observation = {"label": "car", "track_id": "CAM_FRONT:byte:1", "detector_bbox": [10, 10, 30, 30]}
    link = associate(rows, [observation], {"frame": 0}, "CAM_FRONT", 100, 100)[0]
    assert link["status"] == "matched" and link["iou"] == 1
    assert associate(rows, [observation, observation], {"frame": 0}, "CAM_FRONT", 100, 100)[0]["status"] == "ambiguous"
    for change in ({"label": "truck"}, {"detector_bbox": [60, 60, 80, 80]}):
        assert (
            associate(rows, [{**observation, **change}], {"frame": 0}, "CAM_FRONT", 100, 100)[0]["track_2d_id"] is None
        )
