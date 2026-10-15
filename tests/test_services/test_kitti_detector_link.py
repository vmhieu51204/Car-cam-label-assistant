import json

import pytest

from scripts.import_kitti_detector import detector_draft
from scripts.run_sequence import run_complete_sequence
from scripts.run_video_2d import _source_hash
from src.services.detection.kitti_lidar import LidarBox, associate, sha256, validate_output
from src.services.geometry.kitti_projection import project_lidar_box
from src.services.geometry.kitti_tracking import load_sequence
from tests.test_scripts.test_lift_sequence_3d import _native_sequence


def setup(tmp_path):
    _native_sequence(tmp_path)
    sequence = load_sequence(tmp_path, "0013", source_fps=10, fps_source="synthetic")
    box = LidarBox(label="car", center_lidar_xyz_m=(10, 0, 0), size_lwh_m=(4, 2, 2), yaw_lidar_rad=0, score=0.8)
    bbox = project_lidar_box(
        box.center_lidar_xyz_m, box.size_lwh_m, 0, calibration=sequence.calibration, image_size=sequence.image_size
    )
    tracks = tmp_path / "tracks.json"
    source_hash = _source_hash([f.image for f in sequence.frames], source_fps=10, target_fps=10)
    tracks.write_text(
        json.dumps(
            {
                "schema": "autolabel-stitched-tracks/v1",
                "source_sha256": source_hash,
                "tracks": [
                    {
                        "track_id": "sam:1",
                        "label": "car",
                        "observations": [{"frame": i, "bbox": bbox} for i in range(2)],
                    }
                ],
            }
        )
    )
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps({"source_fps": 10, "selected_source_frames": [0, 1], "source_sha256": source_hash}))
    detector = tmp_path / "detector.json"
    raw = {
        "schema": "autolabel-kitti-detector/v1",
        "sequence_id": "0013",
        "coordinate_frame": "kitti_velodyne_x_forward_y_left_z_up",
        "model_name": "synthetic",
        "checkpoint_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "framework_version": "fixture",
        "frames": [
            {"source_frame_index": i, "lidar_sha256": sha256(f.lidar), "boxes": [box.model_dump()]}
            for i, f in enumerate(sequence.frames)
        ],
    }
    detector.write_text(json.dumps(raw))
    return sequence, box, bbox, tracks, mapping, detector, raw


def test_detector_link_preserves_geometry_id_and_review(tmp_path):
    sequence, box, _, tracks, mapping, detector, _ = setup(tmp_path)
    result = detector_draft(sequence, detector, tracks, mapping, tmp_path / "draft.json")
    assert result["tracks"][0]["track_id"] == "sam:1"
    assert result["review_status"] == "pending"
    assert result["tracks"][0]["observations"][0]["box_lidar"] == box.model_dump(exclude={"label", "score"})
    complete = run_complete_sequence(
        tmp_path,
        "0013",
        tmp_path / "complete",
        source_fps=10,
        fps_source="synthetic",
        tracks_path=tracks,
        source_map_path=mapping,
        detector_output=detector,
    )
    assert complete["geometry_source"] == "pretrained_lidar_detector"
    assert complete["kitti_rows"] == 2
    assert float((tmp_path / "complete/3d/0013.txt").read_text().splitlines()[0].split()[-1]) == 0.8
    assert complete["cvat_3d_published"] is False


@pytest.mark.parametrize("change", ["hash", "missing", "partial", "sequence"])
def test_invalid_detector_provenance_rejected(tmp_path, change):
    sequence, _, _, _, _, detector, raw = setup(tmp_path)
    if change == "hash":
        raw["frames"][0]["lidar_sha256"] = "f" * 64
    elif change == "missing":
        raw["frames"].pop()
    elif change == "partial":
        raw["status"] = "partial"
    else:
        raw["sequence_id"] = "0020"
    detector.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        validate_output(detector, sequence)


def test_ambiguity_wrong_class_and_unmatched_keep_cuboid(tmp_path):
    sequence, box, bbox, _, _, _, _ = setup(tmp_path)
    observations = [{"track_id": str(i), "label": "car", "bbox": bbox} for i in range(2)]
    link = associate([box], observations, sequence)[0]
    assert link["track_id_2d"] is None and link["review_reasons"] == ["ambiguous_2d_3d"]
    observations[0]["label"] = "person"
    assert associate([box], observations[:1], sequence)[0]["track_id_2d"] is None
    assert associate([box], [], sequence)[0]["review_reasons"] == ["unmatched_2d"]


def test_detector_refuses_sampled_source_mapping(tmp_path):
    sequence, _, _, tracks, mapping, detector, _ = setup(tmp_path)
    mapping.write_text(json.dumps({"selected_source_frames": [0]}))
    with pytest.raises(ValueError, match="full-frame"):
        detector_draft(sequence, detector, tracks, mapping, tmp_path / "bad.json")
    assert not (tmp_path / "bad.json").exists()


@pytest.mark.parametrize("change", ["image", "duplicate_id", "duplicate_frame"])
def test_cross_sequence_or_duplicate_2d_rejected(tmp_path, change):
    sequence, _, _, tracks, mapping, detector, _ = setup(tmp_path)
    payload = json.loads(tracks.read_text())
    if change == "image":
        payload["source_sha256"] = "f" * 64
    elif change == "duplicate_id":
        payload["tracks"].append(payload["tracks"][0])
    else:
        payload["tracks"][0]["observations"].append(payload["tracks"][0]["observations"][0])
    tracks.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        detector_draft(sequence, detector, tracks, mapping, tmp_path / "bad.json")


def test_missing_detector_frame_has_outside_and_no_invented_box(tmp_path):
    sequence, _, _, tracks, mapping, detector, raw = setup(tmp_path)
    raw["frames"][1]["boxes"] = []
    detector.write_text(json.dumps(raw))
    result = detector_draft(sequence, detector, tracks, mapping, tmp_path / "draft.json")
    rows = result["tracks"][0]["observations"]
    assert rows[1]["source_frame_index"] == 1 and rows[1]["box_lidar"] is None


def test_pinned_detector_checkpoint_mismatch_stops(tmp_path):
    _, _, _, tracks, mapping, detector, _ = setup(tmp_path)
    with pytest.raises(ValueError, match="checkpoint hash"):
        run_complete_sequence(
            tmp_path,
            "0013",
            tmp_path / "bad",
            source_fps=10,
            fps_source="synthetic",
            tracks_path=tracks,
            source_map_path=mapping,
            detector_output=detector,
            detector_checkpoint_sha256="f" * 64,
        )
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("fail_second", [False, True])
def test_isolated_detector_worker_loads_once_and_checkpoints(tmp_path, monkeypatch, fail_second):
    import sys
    from types import ModuleType, SimpleNamespace

    import numpy as np

    from inference.kitti_detector import run

    _native_sequence(tmp_path)
    calls = []

    class Tensor:
        def __init__(self, values):
            self.values = np.asarray(values)

        def cpu(self):
            return self

        def tolist(self):
            return self.values.tolist()

    class LiDARInstance3DBoxes:
        gravity_center = Tensor([[10, 0, 0]])
        dims = Tensor([[4, 2, 2]])
        yaw = Tensor([0])

    apis = ModuleType("mmdet3d.apis")

    def load(*args, **kwargs):
        calls.append("load")
        return SimpleNamespace(dataset_meta={"classes": ["Pedestrian", "Cyclist", "Car"]})

    def infer(model, filename):
        calls.append(filename)
        if fail_second and filename.endswith("000001.bin"):
            raise RuntimeError("GPU failure")
        return SimpleNamespace(
            pred_instances_3d=SimpleNamespace(
                bboxes_3d=LiDARInstance3DBoxes(), labels_3d=Tensor([2]), scores_3d=Tensor([0.9])
            )
        ), None

    apis.init_model, apis.inference_detector = load, infer
    module = ModuleType("mmdet3d")
    module.__version__ = "fixture"
    monkeypatch.setitem(sys.modules, "mmdet3d", module)
    monkeypatch.setitem(sys.modules, "mmdet3d.apis", apis)
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    config, checkpoint = tmp_path / "config.py", tmp_path / "weights.pth"
    config.write_text("fixture")
    checkpoint.write_text("fixture")
    output = tmp_path / "worker.json"
    kwargs = dict(source_fps=10, fps_source="synthetic", device="cpu")
    if fail_second:
        with pytest.raises(RuntimeError, match="GPU failure"):
            run(tmp_path, "0013", config, checkpoint, output, **kwargs)
        assert json.loads(output.read_text())["status"] == "partial"
    else:
        result = run(tmp_path, "0013", config, checkpoint, output, **kwargs)
        assert result["status"] == "completed"
        assert result["runtime"]["excluded_native_classes"] == ["Cyclist"]
        assert result["frames"][0]["boxes"][0]["center_lidar_xyz_m"] == (10.0, 0.0, 0.0)
    assert calls.count("load") == 1
