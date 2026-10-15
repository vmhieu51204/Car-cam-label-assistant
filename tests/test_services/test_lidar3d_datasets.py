import json
import math
from zipfile import ZipFile

import numpy as np
import pytest
from PIL import Image
from scipy.spatial.transform import Rotation

from inference.centerpoint.run import scene_keyframes
from src.services.vm.lidar3d import cuboid_shapes
from src.services.vm.lidar3d_datasets import ingest, kitti, nuscenes, unpack
from src.services.vm.lidar3d_exchange import export_annotations, imported_annotations


@pytest.fixture
def native_kitti(tmp_path):
    root = tmp_path / "training"
    for name in ("velodyne", "image_2", "calib", "label_2"):
        (root / name).mkdir(parents=True)
    np.asarray([[10, 0, 0, 0.5]], dtype="<f4").tofile(root / "velodyne/000123.bin")
    Image.new("RGB", (100, 80)).save(root / "image_2/000123.png")
    (root / "calib/000123.txt").write_text(
        "P2: 20 0 50 0 0 20 40 0 0 0 1 0\nR0_rect: 1 0 0 0 1 0 0 0 1\nTr_velo_to_cam: 0 -1 0 0 0 0 -1 0 1 0 0 0\n"
    )
    (root / "label_2/000123.txt").write_text(f"Car 0 0 0 40 30 60 50 2 2 4 0 1 10 {-math.pi / 2}\n")
    return root


@pytest.fixture
def native_nuscenes(tmp_path):
    root = tmp_path / "nusc"
    version = root / "v1.0-mini"
    version.mkdir(parents=True)
    q = [math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)]
    tables = {
        "scene": [{"token": "scene-token", "name": "scene-0001", "first_sample_token": "s"}],
        "sample": [{"token": "s", "scene_token": "scene-token", "timestamp": 1000000, "next": ""}],
        "sensor": [{"token": "lidar", "channel": "LIDAR_TOP"}],
        "calibrated_sensor": [{"token": "lidar-cal", "sensor_token": "lidar", "rotation": q, "translation": [1, 0, 0]}],
        "ego_pose": [{"token": "ego", "rotation": q, "translation": [10, 20, 0]}],
        "sample_data": [
            {
                "token": "d",
                "sample_token": "s",
                "is_key_frame": True,
                "prev": "",
                "timestamp": 1000000,
                "calibrated_sensor_token": "lidar-cal",
                "ego_pose_token": "ego",
                "filename": "samples/LIDAR_TOP/frame.bin",
            }
        ],
        "category": [{"token": "cat", "name": "vehicle.car"}],
        "instance": [{"token": "instance", "category_token": "cat"}],
        "sample_annotation": [
            {
                "token": "ann",
                "sample_token": "s",
                "instance_token": "instance",
                "translation": [8, 21, 1],
                "size": [2, 4, 1.5],
                "rotation": [0, 0, 0, 1],
            }
        ],
    }
    cloud = root / "samples/LIDAR_TOP/frame.bin"
    cloud.parent.mkdir(parents=True)
    np.asarray([[2, 0, 1, 0.5, 3]], dtype="<f4").tofile(cloud)
    for name, value in tables.items():
        (version / f"{name}.json").write_text(json.dumps(value))
    return root


def labels(name):
    return [
        {
            "id": 1,
            "name": name,
            "attributes": [
                {"id": i, "name": key} for i, key in enumerate(("review_status", "confidence", "detector"), 1)
            ],
        }
    ]


def test_native_kitti_import_and_export_uses_saved_edits_and_original_frame_id(native_kitti, tmp_path):
    base, manifest = kitti(native_kitti, "", 200)
    manifest["format"] = "kitti"
    result = imported_annotations(base, manifest)
    assert np.allclose(result.predictions[0].center_lidar_xyz_m, [10, 0, 0])
    assert np.allclose(result.predictions[0].size_lwh_m, [4, 2, 2])
    assert result.predictions[0].yaw_lidar_rad == pytest.approx(0)
    shapes = cuboid_shapes(result, manifest["frames"], labels("Car"), "Imported")
    shapes[0]["points"][0] = 12  # A user's saved correction must reach the native export.
    target = tmp_path / "export.zip"
    receipt = export_annotations(manifest, {"shapes": shapes, "tracks": []}, labels("Car"), target)
    assert receipt["boxes"] == 1
    with ZipFile(target) as archive:
        fields = archive.read("label_2/000123.txt").decode().split()
        assert len(fields) == 16 and fields[0] == "Car"
        assert list(map(float, fields[8:14])) == pytest.approx([2, 2, 4, 0, 1, 12])
        assert float(fields[14]) == pytest.approx(-math.pi / 2)
    export_annotations(manifest, {"shapes": [], "tracks": []}, labels("Car"), target)
    with ZipFile(target) as archive:
        assert archive.read("label_2/000123.txt") == b""


def test_nuscenes_native_labels_round_trip_full_pose_and_wlh(native_nuscenes, tmp_path):
    base, manifest = nuscenes(native_nuscenes, "", 200)
    manifest["format"] = "nuscenes"
    result = imported_annotations(base, manifest)
    assert np.allclose(result.predictions[0].center_lidar_xyz_m, [2, 0, 1])
    assert np.allclose(result.predictions[0].size_lwh_m, [4, 2, 1.5])
    shapes = cuboid_shapes(result, manifest["frames"], labels("car"), "Imported")
    target = tmp_path / "results.json"
    export_annotations(manifest, {"shapes": shapes}, labels("car"), target)
    box = json.loads(target.read_text())["results"]["s"][0]
    assert box["translation"] == pytest.approx([8, 21, 1])
    assert box["size"] == [2, 4, 1.5]
    assert abs(box["rotation"][3]) == pytest.approx(1)
    assert box["detection_name"] == "car"


def test_kitti_calibration_pitch_roll_preserves_native_camera_boxes(native_kitti, tmp_path):
    calibration = native_kitti / "calib/000123.txt"
    original = calibration.read_text().splitlines()
    rotation = Rotation.from_euler("xyz", [0.013, -0.021, 0.007]).as_matrix() @ np.array(
        [[0, -1, 0], [0, 0, -1], [1, 0, 0]]
    )
    transform = np.column_stack((rotation, [0.2, -0.1, 0.3]))
    calibration.write_text(
        "\n".join(original[:2]) + "\nTr_velo_to_cam: " + " ".join(map(str, transform.ravel())) + "\n"
    )
    base, manifest = kitti(native_kitti, "", 200)
    manifest["format"] = "kitti"
    output = imported_annotations(base, manifest)
    assert not np.allclose(output.predictions[0].rotation_lidar_xyz_rad[:2], 0)
    shapes = cuboid_shapes(output, manifest["frames"], labels("Car"), "Imported")
    path = tmp_path / "tilted.zip"
    export_annotations(manifest, {"shapes": shapes}, labels("Car"), path)
    with ZipFile(path) as archive:
        fields = archive.read("label_2/000123.txt").decode().split()
    assert list(map(float, fields[8:15])) == pytest.approx([2, 2, 4, 0, 1, 10, -math.pi / 2], abs=1e-7)


def test_limited_inference_does_not_require_media_for_unselected_frames(native_nuscenes):
    version = native_nuscenes / "v1.0-mini"
    samples = json.loads((version / "sample.json").read_text())
    samples[0]["next"] = "later"
    samples.append({**samples[0], "token": "later", "next": ""})
    (version / "sample.json").write_text(json.dumps(samples))
    data = json.loads((version / "sample_data.json").read_text())
    data.append({**data[0], "token": "later-data", "sample_token": "later", "filename": "unselected.bin"})
    (version / "sample_data.json").write_text(json.dumps(data))
    frames, _ = scene_keyframes(native_nuscenes, "v1.0-mini", "scene-0001", max_frames=1)
    assert len(frames) == 1 and frames[0]["sample_token"] == "s"
    with pytest.raises(FileNotFoundError):
        scene_keyframes(native_nuscenes, "v1.0-mini", "scene-0001")


@pytest.mark.parametrize("name", ["../escape.txt", "/escape.txt", "C:/escape.txt", "folder\\escape.txt"])
def test_dataset_archive_cannot_escape_job_directory(tmp_path, name):
    source = tmp_path / "input.zip"
    with ZipFile(source, "w") as archive:
        archive.writestr(name, "bad")
    if "\\" in name:
        # Windows' ZIP writer normalizes separators; restore hostile on-wire bytes.
        source.write_bytes(source.read_bytes().replace(name.replace("\\", "/").encode(), name.encode()))
    with pytest.raises(ValueError, match="unsafe|escaped"):
        unpack(source, tmp_path / "dataset", 1000)


def test_archive_rejects_expansion_limit_and_duplicate_entries(tmp_path):
    source = tmp_path / "input.zip"
    with ZipFile(source, "w") as archive:
        archive.writestr("large.bin", b"x" * 1000)
    with pytest.raises(ValueError, match="Expanded"):
        unpack(source, tmp_path / "small", 10)
    with ZipFile(source, "w") as archive:
        archive.writestr("data.bin", b"a")
        archive.writestr("DATA.bin", b"b")
    with pytest.raises(ValueError, match="duplicate"):
        unpack(source, tmp_path / "duplicate", 1000)


def test_native_archive_inference_mapping_is_generated_from_actual_files(native_kitti, tmp_path):
    source = tmp_path / "kitti.zip"
    with ZipFile(source, "w") as archive:
        for path in native_kitti.rglob("*"):
            if path.is_file():
                archive.write(path, "training/" + path.relative_to(native_kitti).as_posix())
    directory = tmp_path / "job"
    directory.mkdir()
    manifest = ingest(source, directory, dataset_format="kitti", scene="", max_frames=200, max_bytes=1000000)
    assert manifest["format"] == "kitti" and manifest["root"] == "training"
    assert manifest["frames"][0]["source_id"] == "000123"
    assert len(manifest["files"]["velodyne/000123.bin"]) == 64
