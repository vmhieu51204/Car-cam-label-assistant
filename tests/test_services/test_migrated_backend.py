import json
import math
import sqlite3

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.config import Settings
from src.main import create_app
from src.models.perception import Box3D, Detection, Frame, QueueQuery, Track
from src.services.detection.mock import mock_frames
from src.services.geometry.boxes import giou3d
from src.services.storage.repository import Store
from src.services.tracking import Tracker


def box(x=0, z=0, yaw=0, size=(4, 2, 2)):
    return Box3D(center=(x, 0, z), size_lwh=size, yaw=yaw)


def frame(index, x=None, label="car", scene="fixture-scene"):
    detections = [] if x is None else [Detection(label=label, box=box(x), confidence=0.8)]
    return Frame(
        scene=scene, sample_token=f"sample-{index}", frame_index=index, timestamp_s=index * 0.5, detections=detections
    )


def test_giou_identity_overlap_rotation_and_vertical_separation():
    assert giou3d(box(), box()) == pytest.approx(1)
    assert giou3d(box(), box(2)) == pytest.approx(1 / 3)
    assert giou3d(box(), box(8)) == pytest.approx(-1 / 3)
    assert giou3d(box(), box(z=4)) == pytest.approx(-1 / 3)
    assert giou3d(box(), box(yaw=math.pi)) == pytest.approx(1)
    rotated = giou3d(box(), box(yaw=math.pi / 2))
    assert -1 < rotated < 1 / 3
    assert rotated == pytest.approx(giou3d(box(yaw=math.pi / 2), box()))


def test_tracker_preserves_identity_across_one_missing_frame():
    tracker = Tracker(max_missed=1)
    for item in [frame(0, 0), frame(1, 0.5), frame(2), frame(3, 1.5)]:
        tracker.step(item)
    assert len(tracker.tracks) == 1
    assert [o.frame_index for o in tracker.tracks[0].observations] == [0, 1, 3]
    assert tracker.active[0].mean[3] > 0.5
    assert tracker.tracks[0].review_status == "pending"
    assert tracker.tracks[0].p_correct is None


def test_tracker_expires_old_tracks_and_rejects_cross_class_matches():
    tracker = Tracker(max_missed=0)
    for item in [frame(0, 0), frame(1), frame(2, 0), frame(3, 0, "pedestrian")]:
        tracker.step(item)
    assert len(tracker.tracks) == 3


def test_tracker_gating_does_not_force_far_matches():
    tracker = Tracker()
    tracker.step(frame(0, 0))
    tracker.step(frame(1, 100))
    assert len(tracker.tracks) == 2


def test_tracker_uses_motion_when_detection_order_changes():
    tracker = Tracker()
    for index in range(4):
        item = frame(index)
        positions = [index * 0.5, 12 - index * 0.5]
        if index % 2:
            positions.reverse()
        item.detections = [Detection(label="car", box=box(x), confidence=0.8) for x in positions]
        tracker.step(item)
    assert len(tracker.tracks) == 2
    assert tracker.tracks[0].observations[-1].box.center[0] == 1.5
    assert tracker.tracks[1].observations[-1].box.center[0] == 10.5


def test_tracker_rejects_scene_and_time_mixups():
    tracker = Tracker()
    tracker.step(frame(0, 0))
    with pytest.raises(ValueError, match="scene"):
        tracker.step(frame(1, 0, scene="other"))
    with pytest.raises(ValueError, match="increasing"):
        tracker.step(frame(0, 0))


@pytest.mark.parametrize("changes", [{"center": (float("nan"), 0, 0)}, {"size_lwh": (0, 2, 2)}])
def test_invalid_geometry_is_rejected(changes):
    with pytest.raises(ValidationError):
        Box3D(**(box().model_dump() | changes))


@pytest.fixture
def settings(tmp_path):
    root = tmp_path / "v1.0-mini"
    root.mkdir()
    tables = {
        "scene": [{"name": "fixture-scene", "first_sample_token": "s0"}],
        "sample": [
            {"token": "s0", "timestamp": 1_000_000, "next": "s1"},
            {"token": "s1", "timestamp": 1_500_000, "next": ""},
        ],
        "instance": [{"token": "instance", "category_token": "category"}],
        "category": [{"token": "category", "name": "vehicle.car"}],
        "sample_annotation": [
            {
                "token": f"a{i}",
                "sample_token": f"s{i}",
                "instance_token": "instance",
                "rotation": [1, 0, 0, 0],
                "translation": [i * 0.2, 0, 0],
                "size": [2, 4, 2],
            }
            for i in range(2)
        ],
    }
    for name, rows in tables.items():
        (root / f"{name}.json").write_text(json.dumps(rows), encoding="utf-8")
    return Settings(
        _env_file=None,
        data_root=tmp_path,
        app_env="test",
        db_path=tmp_path / "test.sqlite3",
        vlm_cache_path=tmp_path / "vlm.sqlite3",
        mock_miss_rate=0,
        mock_position_std=0,
        propagation_provider="kaggle_sam3",
    )


def test_mock_is_reproducible_and_does_not_expose_gt_identity(settings):
    first = mock_frames(settings, "fixture-scene", 2)
    assert first == mock_frames(settings, "fixture-scene", 2)
    assert len(first) == 2
    assert "instance_token" not in first[0].model_dump_json()
    assert first[0].detections[0].box.coordinate_frame == "nuscenes_global"


def test_api_persists_tracks_and_idempotent_telemetry(settings):
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").json()["training_export_enabled"] is False
        result = client.post("/detect", json={"scene": "fixture-scene", "max_frames": 2})
        assert result.status_code == 200
        run = result.json()
        assert run["data_source"] == "mock"
        assert run["track_count"] == 1
        run_id = run["run_id"]
        queue = client.get("/queue", params={"run_id": run_id}).json()
        assert queue["ranking"] == "hcl"
        assert queue["calibrated"] is False
        track = queue["tracks"][0]
        assert track["review_status"] == "pending"
        assert track["source"] == "auto"
        assert track["p_correct"] is None
        assert track["discrepancy"]["status"] == "insufficient_evidence"
        event = dict(
            event_id="event-1",
            run_id=run_id,
            track_id=track["track_id"],
            event_type="edit_recorded",
            duration_ms=100,
            edit_count=1,
        )
        assert client.post("/telemetry", json=event).json()["created"] is True
        assert client.post("/telemetry", json=event).json()["created"] is False
        assert client.post("/telemetry", json=event | {"edit_count": 2}).status_code == 409
        assert client.post("/telemetry", json=event | {"event_id": "bad", "track_id": "absent"}).status_code == 404
        assert client.post("/telemetry", json=event | {"duration_ms": -1}).status_code == 422
        assert client.post("/telemetry", json=event | {"event_type": "accept"}).status_code == 422
        assert client.post("/validate").status_code == 422
        assert client.get("/queue", params={"run_id": "absent"}).status_code == 404
        assert client.post("/detect", json={"scene": "absent"}).status_code == 422
    with TestClient(create_app(settings)) as client:
        assert client.get("/queue", params={"run_id": run_id}).json()["total"] == 1
    with sqlite3.connect(settings.db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM telemetry").fetchone()[0] == 1


def test_discrepancy_api_is_prefixed_and_does_not_accept_for_a_human(settings):
    evidence = {
        "track_id": "fixture-scene:track-1",
        "label": "car",
        "data_source": "real",
        "frames": [
            {
                "frame_index": 0,
                "timestamp_s": 0,
                "label": "car",
                "box": {"center": [0, 0, 1], "size_lwh": [4, 2, 2], "yaw": 0},
                "detector": {"raw_score": 0.99, "probability": 0.99, "calibration_id": "fixture"},
                "points": {"robust_z": 0, "calibration_id": "fixture"},
                "projection": {"iou": 0, "calibration_id": "fixture", "calibration_valid": True},
            }
        ],
    }
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/v1/discrepancy/evaluate", json=evidence)
    assert response.status_code == 200
    assert response.json()["status"] == "discrepancy"
    assert response.json()["suggested_action"] == "review"
    assert response.json()["review_status"] == "pending"
    assert response.json()["p_correct"] is None


def test_queue_only_returns_tracks_pending_human_review(tmp_path):
    store = Store(tmp_path / "queue.sqlite3")
    pending = Track(
        track_id="scene:pending",
        scene="scene",
        label="car",
        confidence=0.8,
        observations=[],
    )
    accepted = pending.model_copy(update={"track_id": "scene:accepted", "review_status": "accepted"})
    rejected = pending.model_copy(update={"track_id": "scene:rejected", "review_status": "rejected"})
    run_id = store.save_run("scene", {"data_source": "mock"}, [pending, accepted, rejected])
    queue = store.queue(QueueQuery(run_id=run_id))
    assert queue["total"] == 1
    assert [track["track_id"] for track in queue["tracks"]] == ["scene:pending"]


def test_mock_error_quotas_and_quality_correlated_scores(settings):
    from src.services.detection.mock import generate_mock
    from src.services.geometry.boxes import iou3d

    root = settings.data_root / settings.data_version
    rows = [
        {
            "token": f"a{i:03d}",
            "sample_token": "s0",
            "instance_token": "instance",
            "rotation": [1, 0, 0, 0],
            "translation": [i * 6, 0, 0],
            "size": [2, 4, 2],
        }
        for i in range(100)
    ]
    (root / "sample_annotation.json").write_text(json.dumps(rows), encoding="utf-8")
    config = settings.model_copy(update={"mock_miss_rate": 0.2, "mock_false_rate": 0.1, "mock_class_error_rate": 0.05})
    result = generate_mock(config, "fixture-scene")
    assert result.stats["eligible_gt"] == 100
    assert result.stats["dropped_boxes"] == 20
    assert result.stats["spurious_boxes"] == 10
    assert result.stats["class_changed_boxes"] == 4
    assert result.stats["predicted_boxes"] == 90
    good_scores, bad_scores = [], []
    for audit in result.audit:
        detection = result.frames[audit["frame_index"]].detections[audit["detection_index"]]
        if audit["spurious"]:
            assert all(iou3d(detection.box, box(i * 6)) == 0 for i in range(100))
        group = bad_scores if audit["spurious"] or audit["class_changed"] else good_scores
        group.append(detection.confidence)
    assert min(good_scores) > max(bad_scores)
    assert result == generate_mock(config, "fixture-scene")


def test_precompute_writes_common_schema_without_oracle_fields(settings, tmp_path):
    from scripts.precompute import precompute

    output = tmp_path / "predictions"
    stats = precompute(settings, "fixture-scene", output)
    payload = json.loads((output / "fixture-scene.json").read_text())
    assert stats["frames"] == 2
    assert all(
        p["source"] == "auto" and p["track_id"] and p["review_status"] == "pending" for p in payload["predictions"]
    )
    assert "oracle_iou" not in json.dumps(payload)
    assert (output / "mock-audit/fixture-scene.json").exists()
