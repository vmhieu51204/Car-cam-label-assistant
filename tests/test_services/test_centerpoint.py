import json
import math

import pytest

from scripts.precompute import precompute_model
from src.config import Settings
from src.services.detection.centerpoint import load_centerpoint_output
from src.services.geometry.transforms import lidar_box_to_global


def test_lidar_box_to_global_composes_calibration_and_ego_pose():
    half_turn_z = [math.sqrt(0.5), 0, 0, math.sqrt(0.5)]
    center, yaw = lidar_box_to_global(
        (1, 0, 0),
        0,
        {"translation": [10, 0, 0], "rotation": half_turn_z},
        {"translation": [1, 0, 0], "rotation": [1, 0, 0, 0]},
    )
    assert center == pytest.approx((10, 2, 0))
    assert yaw == pytest.approx(math.pi / 2)


@pytest.fixture
def centerpoint_fixture(tmp_path):
    version = tmp_path / "v1.0-mini"
    version.mkdir()
    tables = {
        "scene": [{"name": "scene-fixture", "first_sample_token": "sample-0"}],
        "sample": [{"token": "sample-0", "timestamp": 2_000_000, "next": ""}],
        "sensor": [{"token": "sensor-lidar", "channel": "LIDAR_TOP"}],
        "calibrated_sensor": [
            {
                "token": "calibration-0",
                "sensor_token": "sensor-lidar",
                "translation": [1, 0, 0],
                "rotation": [1, 0, 0, 0],
            }
        ],
        "ego_pose": [{"token": "ego-0", "translation": [10, 20, 0], "rotation": [1, 0, 0, 0]}],
        "sample_data": [
            {
                "token": "lidar-0",
                "sample_token": "sample-0",
                "calibrated_sensor_token": "calibration-0",
                "ego_pose_token": "ego-0",
                "is_key_frame": True,
            }
        ],
    }
    for name, rows in tables.items():
        (version / f"{name}.json").write_text(json.dumps(rows), encoding="utf-8")
    settings = Settings(
        _env_file=None,
        data_root=tmp_path,
        db_path=tmp_path / "state.sqlite3",
        vlm_cache_path=tmp_path / "vlm.sqlite3",
    )
    raw = tmp_path / "centerpoint.json"
    raw.write_text(
        json.dumps(
            {
                "schema_version": "autolabel3d-centerpoint/v1",
                "data_source": "real",
                "dataset": "v1.0-mini",
                "scene": "scene-fixture",
                "coordinate_frame": "nuscenes_lidar",
                "size_order": "lwh",
                "model": {
                    "name": "CenterPoint",
                    "framework": "MMDetection3D",
                    "config": "centerpoint-pillar02",
                    "checkpoint": "sha256:fixture",
                    "device": "cuda:0",
                    "score_threshold": 0.3,
                    "multi_sweep": False,
                },
                "predictions": [
                    {
                        "sample_token": "sample-0",
                        "frame_index": 0,
                        "label": "car",
                        "score": 0.9,
                        "center_lidar_xyz_m": [2, 3, 1],
                        "size_lwh_m": [4, 2, 1.5],
                        "yaw_lidar_rad": 0.2,
                    },
                    {
                        "sample_token": "sample-0",
                        "frame_index": 0,
                        "label": "barrier",
                        "score": 0.8,
                        "center_lidar_xyz_m": [0, 0, 0.5],
                        "size_lwh_m": [1, 1, 1],
                        "yaw_lidar_rad": 0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return settings, raw


def test_centerpoint_import_is_real_global_and_filters_unsupported(centerpoint_fixture):
    settings, raw = centerpoint_fixture
    result = load_centerpoint_output(settings, raw, "scene-fixture")
    assert result.stats["raw_boxes"] == 2
    assert result.stats["predicted_boxes"] == 1
    assert result.stats["skipped_unsupported"] == {"barrier": 1}
    detection = result.frames[0].detections[0]
    assert detection.box.center == pytest.approx((13, 23, 1))
    assert detection.box.coordinate_frame == "nuscenes_global"
    assert detection.confidence == 0.9


def test_model_precompute_tracks_real_predictions(centerpoint_fixture, tmp_path):
    settings, raw = centerpoint_fixture
    output = tmp_path / "predictions"
    metadata = precompute_model(settings, "scene-fixture", raw, output)
    payload = json.loads((output / "scene-fixture.json").read_text(encoding="utf-8"))
    assert metadata["data_source"] == "real"
    assert metadata["tracks"] == 1
    assert payload["predictions"][0]["data_source"] == "real"
    assert payload["predictions"][0]["review_status"] == "pending"
    assert not (output / "mock-audit/scene-fixture.json").exists()


def test_centerpoint_import_rejects_frame_mismatch(centerpoint_fixture):
    settings, raw = centerpoint_fixture
    payload = json.loads(raw.read_text(encoding="utf-8"))
    payload["predictions"][0]["frame_index"] = 4
    raw.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Frame index mismatch"):
        load_centerpoint_output(settings, raw, "scene-fixture")
