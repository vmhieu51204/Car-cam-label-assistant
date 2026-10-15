import hashlib
import json
import os
from zipfile import ZipFile

import numpy as np
import pytest

from src.config import Settings
from src.services.detection.centerpoint import CenterPointOutput
from src.services.vm.lidar3d import CHECKPOINT_SHA, cuboid_shapes, pcd_bytes, prepare_media


def test_native_cuboid_uses_lidar_center_lwh_and_positive_yaw():
    raw = {
        "schema_version": "autolabel3d-centerpoint/v1",
        "data_source": "real",
        "dataset": "v1.0-mini",
        "scene": "scene-0061",
        "coordinate_frame": "nuscenes_lidar",
        "size_order": "lwh",
        "model": {
            "name": "CenterPoint",
            "framework": "MMDetection3D",
            "config": "pinned.py",
            "checkpoint": "model@sha256:" + CHECKPOINT_SHA,
            "device": "cuda:0",
            "score_threshold": 0.3,
            "multi_sweep": True,
        },
        "predictions": [
            {
                "sample_token": "frame-a",
                "frame_index": 0,
                "label": "car",
                "score": 0.9,
                "center_lidar_xyz_m": [10, -3, 2],
                "size_lwh_m": [4, 2, 1.5],
                "yaw_lidar_rad": 0.7,
            }
        ],
    }
    output = CenterPointOutput.model_validate_json(json.dumps(raw))
    labels = [
        {
            "name": "car",
            "id": 4,
            "attributes": [
                {"name": name, "id": index} for index, name in enumerate(("review_status", "confidence", "detector"), 1)
            ],
        }
    ]
    shapes = cuboid_shapes(output, [{"sample_token": "frame-a"}], labels)
    assert shapes[0]["points"] == [10, -3, 2, 0, 0, 0.7, 4, 2, 1.5, 0, 0, 0, 0, 0, 0, 0]
    assert shapes[0]["attributes"][0] == {"spec_id": 1, "value": "pending"}
    with pytest.raises(ValueError, match="identity"):
        cuboid_shapes(output, [{"sample_token": "another-frame"}], labels)


def test_pointcloud_keeps_sensor_xyz_and_drops_only_ring_channel(tmp_path):
    path = tmp_path / "lidar.bin"
    np.asarray([[1, -2, 3, 0.5, 7], [4, 5, 6, 0.7, 2]], dtype="<f4").tofile(path)
    encoded = pcd_bytes(path)
    header, payload = encoded.split(b"DATA binary\n")
    assert b"POINTS 2\n" in header
    assert np.allclose(np.frombuffer(payload, dtype="<f4").reshape(-1, 4), [[1, -2, 3, 0.5], [4, 5, 6, 0.7]])


def test_media_zip_accepts_reproducible_dataset_epoch_timestamps(tmp_path):
    from PIL import Image

    cloud, camera = tmp_path / "cloud.bin", tmp_path / "camera.jpg"
    np.asarray([[1, 2, 3, 0.5, 4]], dtype="<f4").tofile(cloud)
    Image.new("RGB", (4, 4)).save(camera)
    os.utime(camera, (0, 0))
    manifest = {"files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (cloud, camera)}}
    output = tmp_path / "media.zip"
    prepare_media(
        Settings(_env_file=None, lidar3d_dataset_root=tmp_path),
        manifest,
        [{"lidar": "cloud.bin", "cameras": {"CAM_FRONT": "camera.jpg"}}],
        output,
    )
    with ZipFile(output) as archive:
        assert "pointcloud/frame_000000.pcd" in archive.namelist()
        assert archive.read("related_images/frame_000000_pcd/CAM_FRONT.jpg") == camera.read_bytes()
