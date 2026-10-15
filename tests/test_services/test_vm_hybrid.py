"""CPU protocol fixtures; never evidence of YOLO/SAM quality or GPU speed."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from scripts.run_vm_hybrid import HybridBackend, observation, relative_box, validate_detector_metadata
from src.config import Settings
from src.services.vm.contracts import VmProfile, file_sha256, preflight
from src.services.vm.hybrid import DetectionJournal, TrackerPool, prompt_plan
from src.services.vm.hybrid_contracts import HybridProfile, hybrid_preflight
from src.services.vm.storage import encode


def profile(**changes):
    tracker = {"tracker_type": "bytetrack", "track_buffer": 30}
    return HybridProfile(
        checkpoint="yolo26s.pt",
        checkpoint_sha256="a" * 64,
        classes_manifest_sha256="b" * 64,
        class_mapping={"car": 2, "person": 0},
        detector_names={0: "person", 2: "car"},
        tracker_config=tracker,
        tracker_sha256=hashlib.sha256(encode(tracker)).hexdigest(),
        **changes,
    )


def frame(i, cut=False):
    return {
        "frame_index": i,
        "source_frame_index": i,
        "scene_cut_before": cut,
        "width": 8,
        "height": 8,
        "sha256": "c" * 64,
        "relative_path": f"cache/frame_{i:06d}.jpg",
    }


class FixtureTracker:
    def update(self, boxes):
        return np.asarray([list(row[:4]) + [i + 1, row[4], row[5], i] for i, row in enumerate(boxes)], dtype=np.float32)


def pool(p):
    return TrackerPool(p, FixtureTracker, lambda boxes, _shape: boxes)


DETECTIONS = [{"box": [1, 1, 6, 6], "score": 0.8, "class_id": 2}]


def test_old_job_defaults_baseline_and_mode_requires_pins():
    old = VmProfile(checkpoint="sam3.pt", checkpoint_sha256="a" * 64, source_commit="b" * 40)
    assert old.pipeline == "sam" and old.hybrid is None
    with pytest.raises(ValueError, match="instance"):
        old.model_copy(update={"pipeline": "hybrid_frame"})  # copy isn't validation
        VmProfile.model_validate({**old.model_dump(), "pipeline": "hybrid_frame"})
    with pytest.raises(ValueError, match="instance"):
        preflight(
            Settings(_env_file=None, vm_workflow_enabled=True, local_sam_enabled=True),
            "sam3.1",
            "hybrid_frame",
            ["car"],
        )


def test_preflight_hash_class_manifest_and_explicit_mapping(tmp_path):
    checkpoint = tmp_path / "yolo26s.pt"
    checkpoint.write_bytes(b"fixture")
    manifest = tmp_path / "names.json"
    manifest.write_text(
        json.dumps(
            {"model": "yolo26s", "checkpoint_sha256": file_sha256(checkpoint), "names": {"0": "person", "2": "car"}}
        )
    )
    settings = Settings(
        _env_file=None,
        hybrid_enabled=True,
        hybrid_yolo_checkpoint=checkpoint,
        hybrid_yolo_sha256=file_sha256(checkpoint),
        hybrid_classes_manifest=manifest,
        hybrid_classes_sha256=file_sha256(manifest),
    )
    frozen = hybrid_preflight(settings, ["car"])
    assert frozen.class_mapping == {"car": 2} and frozen.tracker == "bytetrack"
    with pytest.raises(ValueError, match="Unsupported"):
        hybrid_preflight(settings, ["van"])
    settings.hybrid_class_mapping = '{"van":80}'
    with pytest.raises(ValueError, match="support"):
        hybrid_preflight(settings, ["van"])
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        hybrid_preflight(settings, ["car"])


def test_journal_overlap_resume_cpu_replay_and_scene_cut(tmp_path):
    p = profile()
    calls = []

    def detector(f):
        calls.append(f["frame_index"])
        return DETECTIONS

    journal = DetectionJournal(tmp_path, p, pool(p), "hybrid_frame")
    first = journal.detect(frame(0), detector, 32)
    assert first["tracks"][0]["track_id"] == "byte:0:2:1"
    journal.detect(frame(1), detector, 32)
    assert journal.detect(frame(0), detector, 32)["reused"]
    retry = DetectionJournal(tmp_path, p, pool(p), "hybrid_frame")
    third = retry.detect(frame(2, True), detector, 32)
    assert third["tracks"][0]["track_id"] == "byte:1:2:1" and calls == [0, 1, 2]
    assert retry.detect(frame(1), detector, 32)["tracks"][0]["track_id"] == first["tracks"][0]["track_id"]
    with pytest.raises(ValueError, match="identity"):
        DetectionJournal(tmp_path, p, pool(p), "hybrid_propagate").detect(frame(0), detector, 32)
    with pytest.raises(ValueError, match="changed"):
        retry.detect({**frame(0), "sha256": "d" * 64}, detector, 32)


def test_journal_corruption_hole_cap_and_empty_frames(tmp_path):
    p = profile()
    journal = DetectionJournal(tmp_path, p, pool(p))
    with pytest.raises(ValueError, match="cap exceeded"):
        journal.detect(frame(0), lambda _: DETECTIONS * 33, 32)
    assert not list(tmp_path.glob("*.json"))
    assert journal.detect(frame(0), lambda _: [], 32)["tracks"] == []
    journal.detect(frame(1), lambda _: DETECTIONS, 32)
    path = tmp_path / "000000.json"
    data = json.loads(path.read_text())
    data["payload"]["tracks"] = ["changed"]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="checksum"):
        DetectionJournal(tmp_path, p, pool(p)).replay_to(1)
    path.unlink()
    with pytest.raises(ValueError, match="prefix"):
        DetectionJournal(tmp_path, p, pool(p))


def test_tracker_mapping_class_and_geometry_are_strict():
    p = profile()
    tracking = pool(p)
    assert tracking.update(frame(0), DETECTIONS)[0]["detector_bbox"] == [1, 1, 6, 6]
    bad = TrackerPool(p, lambda: SimpleNamespace(update=lambda _: np.ones((1, 7))), lambda b, _: b)
    with pytest.raises(ValueError, match="signature"):
        bad.update(frame(1), DETECTIONS)


def test_late_objects_prompted_before_propagation_and_cap():
    records = [frame(i) for i in range(5)]
    car = {"track_id": "car", "prompt_bbox": [1, 1, 6, 6], "label": "car"}
    person = {**car, "track_id": "person", "label": "person"}
    data = [{"tracks": [car]}, {"tracks": []}, {"tracks": [car]}, {"tracks": [person]}, {"tracks": []}]
    plan = prompt_plan(records, data, 32, 2, 1)
    assert plan["person"]["first"] == 3 and plan["person"]["anchors"] == [(3, [1, 1, 6, 6])]
    assert plan["car"]["end"] == 3 and len(plan["car"]["anchors"]) == 2
    with pytest.raises(ValueError, match="cap exceeded"):
        prompt_plan(records, data, 1, 2, 1)


def test_frame_masks_batch_reuses_one_embedding_and_retains_detector_box(tmp_path):
    backend = HybridBackend.__new__(HybridBackend)
    backend.absences = {}
    backend.hybrid = profile(object_batch=1)
    backend.torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))
    path = tmp_path / "redacted.jpg"
    Image.new("RGB", (8, 8)).save(path)
    backend.image_path = lambda _: path
    embeddings, batches = [], []
    backend.processor = SimpleNamespace(set_image_batch=lambda images: embeddings.append(len(images)) or {})
    mask = np.zeros((1, 1, 8, 8))
    mask[:, :, 2:4, 2:4] = 1

    def predict(state, _points, _labels, **kwargs):
        batches.append(kwargs["box_batch"][0].tolist())
        return [mask], [np.ones((1, 1))], None

    backend.model = SimpleNamespace(predict_inst_batch=predict)
    row = pool(backend.hybrid).update(frame(0), DETECTIONS)[0]
    second = {**row, "track_id": "byte:0:2:2"}
    result = list(backend.masks_by_frame([frame(0), frame(1)], [{"tracks": [row, second]}, {"tracks": []}]))
    assert embeddings == [1] and len(batches) == 2
    assert result[0][1][0]["detector_bbox"] == [1, 1, 6, 6]
    assert result[0][1][0]["bbox"] != result[0][1][0]["detector_bbox"]
    assert result[1][1] == []
    assert observation(row, np.zeros((8, 8)), frame(0)) is None


def test_propagate_preseeds_late_ids_and_suppresses_before_entry(tmp_path):
    backend = HybridBackend.__new__(HybridBackend)
    backend.root = tmp_path
    backend.absences = {}
    (tmp_path / "private").mkdir()
    backend.hybrid = profile(prompt_stride=2, lost_mask_frames=1)
    backend.profile = VmProfile(
        variant="sam3",
        pipeline="hybrid_propagate",
        hybrid=backend.hybrid,
        checkpoint="sam3.pt",
        checkpoint_sha256="a" * 64,
        source_commit="b" * 40,
    )
    backend.torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))
    path = tmp_path / "redacted.jpg"
    Image.new("RGB", (8, 8)).save(path)
    backend.image_path = lambda _: path
    seeded = []

    def add(state, **kw):
        assert kw["rel_coordinates"] is True
        assert np.all((kw["box"] >= 0) & (kw["box"] <= 1))
        seeded.append((kw["frame_idx"], kw["obj_id"]))

    def propagate(state, **kw):
        assert (3, 2) in seeded and kw["propagate_preflight"] is True
        for i in range(5):
            masks = np.ones((2, 1, 8, 8))
            yield i, [1, 2], None, masks, None

    backend.tracker = SimpleNamespace(
        init_state=lambda **kw: {}, add_new_points_or_box=add, propagate_in_video=propagate
    )
    records = [frame(i) for i in range(5)]
    car = pool(backend.hybrid).update(frame(0), DETECTIONS)[0]
    person = {**car, "track_id": "byte:0:0:1", "label": "person", "detector_source_frame": 3}
    data = [
        {"tracks": [car]},
        {"tracks": [{**car, "detector_source_frame": 1}]},
        {"tracks": []},
        {"tracks": [person]},
        {"tracks": [person, {**car, "detector_source_frame": 4}]},
    ]
    results = list(backend.masks_propagated(records, data))
    assert len(results) == 5
    assert all(row["label"] != "person" for _, rows, _ in results[:3] for row in rows)
    assert [row["label"] for row in results[3][1]] == ["person"]
    assert results[2][1][0]["detector_bbox"] is None
    assert results[2][1][0]["score_source"] == "last_detector_observation"
    assert backend.absences[3][0]["reason"] == "detector_gap_limit"
    assert {row["label"] for row in results[4][1]} == {"car", "person"}


def test_nonsquare_sam_prompt_uses_explicit_relative_geometry():
    value = relative_box([32, 16, 160, 64], {"width": 320, "height": 80})
    assert np.allclose(value, [0.1, 0.2, 0.5, 0.8])
    # Actual official API: relative coordinates are multiplied by image_size.
    assert np.allclose(value * 1008, [100.8, 201.6, 504, 806.4])
    validate_detector_metadata({"yaml_file": "ultralytics/cfg/models/26/yolo26.yaml", "scale": "s"}, "detect")
    with pytest.raises(ValueError, match="architecture"):
        validate_detector_metadata({"yaml_file": "yolo11s.yaml", "scale": "s"}, "detect")


def test_empty_sam_mask_keeps_frame_coverage_and_absence_evidence(tmp_path):
    backend = HybridBackend.__new__(HybridBackend)
    backend.hybrid, backend.absences = profile(), {}
    backend.torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))
    path = tmp_path / "redacted.jpg"
    Image.new("RGB", (8, 8)).save(path)
    backend.image_path = lambda _: path
    backend.processor = SimpleNamespace(set_image_batch=lambda _: {})
    backend.model = SimpleNamespace(
        predict_inst_batch=lambda *a, **kw: ([np.zeros((1, 1, 8, 8))], [np.ones((1, 1))], None)
    )
    track = pool(backend.hybrid).update(frame(0), DETECTIONS)[0]
    result = list(backend.masks_by_frame([frame(0)], [{"tracks": [track]}]))
    assert len(result) == 1 and result[0][1] == []
    assert backend.absences[0][0]["reason"] == "empty_sam_mask"


def test_three_mode_comparison_requires_same_redacted_profile_and_masks(tmp_path):
    from dataclasses import asdict

    from scripts.compare_vm_hybrid import compare, read_run
    from src.services.video.stitching import encode_mask
    from src.services.vm.streaming import StreamingEngine
    from tests.test_services.test_vm_streaming import fixture

    roots = []
    mask = asdict(encode_mask(np.ones((8, 8), dtype=bool)))
    for mode in ("sam", "hybrid_frame", "hybrid_propagate"):
        directory = tmp_path / mode
        directory.mkdir()
        store, job, source = fixture(directory, count=2)
        frozen = VmProfile(
            variant="sam3",
            pipeline=mode,
            hybrid=None if mode == "sam" else profile(),
            checkpoint="sam3.pt",
            checkpoint_sha256="a" * 64,
            source_commit="b" * 40,
        )
        job = store.update(job["job_id"], classes=["car"], profile=frozen.model_dump(mode="json", by_alias=True))

        class GPU:
            def __init__(self, *args):
                pass

            def run(self, command):
                for f in command["frames"]:
                    row = {
                        "track_id": "byte:0:2:1",
                        "label": "car",
                        "bbox": [0, 0, 8, 8],
                        "mask": mask,
                        "score": 0.8,
                        "detector_bbox": [1, 1, 6, 6],
                        "prompt_bbox": [1, 1, 6, 6],
                        "detector_source_frame": f["frame_index"],
                    }
                    yield {
                        "type": "frame",
                        "frame": f["frame_index"],
                        "window": command["window"],
                        "class": "car",
                        "pipeline": mode,
                        "image_sha256": f["sha256"],
                        "observations": [row],
                    }
                yield {
                    "type": "complete",
                    "identity": command["identity"],
                    "frames": [f["frame_index"] for f in command["frames"]],
                }

            def close(self):
                pass

        StreamingEngine(store, Settings(_env_file=None), source, lambda *a: None, GPU).run(job)
        roots.append(store.directory(job["job_id"]))
    gt = tmp_path / "gt.json"
    gt.write_text(
        json.dumps(
            {
                "schema": "autolabel-manual-gt/v1",
                "frames": [{"video_id": "fixture", "frame": i, "width": 8, "height": 8} for i in range(2)],
                "annotations": [
                    {"video_id": "fixture", "frame": 0, "class": "car", "track_id": "car-1", "box": [0, 0, 8, 8]}
                ],
            }
        )
    )
    masks = tmp_path / "masks.json"
    masks.write_text(
        json.dumps(
            {
                "schema": "autolabel-manual-masks/v1",
                "images_sha256": read_run(roots[0])["images_sha256"],
                "frames": [{"frame": 0, "class": "car", "masks": [mask]}, {"frame": 1, "class": "car", "masks": []}],
            }
        )
    )
    report = compare(roots, gt, masks)
    assert set(report["modes"]) == {"sam", "hybrid_frame", "hybrid_propagate"}
    for result in report["modes"].values():
        assert result["mask_metrics"]["tp"] == 1 and result["mask_metrics"]["fp"] == 1
        assert result["mask_metrics"]["matched_mask_mean_iou_at_0_5"] == 1.0
    frozen_file = roots[1] / "profile.json"
    document = json.loads(frozen_file.read_text())
    document["window"] = 49
    frozen_file.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="profile"):
        compare(roots, gt)


@pytest.mark.parametrize("mode", ["hybrid_frame", "hybrid_propagate"])
def test_incremental_hybrid_retry_keeps_global_id_across_overlap(tmp_path, mode):
    from src.services.video.stitching import encode_mask
    from src.services.vm.streaming import StreamingEngine
    from tests.test_services.test_vm_streaming import fixture

    store, job, source = fixture(tmp_path, count=60)
    frozen = VmProfile(
        variant="sam3",
        pipeline=mode,
        hybrid=profile(),
        checkpoint="sam3.pt",
        checkpoint_sha256="a" * 64,
        source_commit="b" * 40,
    )
    job = store.update(job["job_id"], profile=frozen.model_dump(mode="json", by_alias=True), classes=["car"])
    generations, published = [], []
    mask = encode_mask(np.ones((8, 8), dtype=bool))

    class GPU:
        def __init__(self, *args):
            self.generation = len(generations)
            generations.append(self.generation)

        def run(self, command):
            for f in command["frames"]:
                if self.generation == 0 and f["frame_index"] == 51:
                    raise ValueError("fixture interruption")
                row = {
                    "track_id": "byte:0:2:1",
                    "label": "car",
                    "bbox": [0, 0, 8, 8],
                    "score": 0.8,
                    "mask": {"width": mask.width, "height": mask.height, "counts": mask.counts, "sha256": mask.sha256},
                    "detector_bbox": [1, 1, 6, 6],
                    "prompt_bbox": [1, 1, 6, 6],
                    "detector_source_frame": f["frame_index"],
                }
                yield {
                    "type": "frame",
                    "window": command["window"],
                    "frame": f["frame_index"],
                    "class": "car",
                    "pipeline": mode,
                    "image_sha256": f["sha256"],
                    "observations": [row],
                }
            yield {
                "type": "complete",
                "identity": command["identity"],
                "frames": [f["frame_index"] for f in command["frames"]],
            }

        def close(self):
            pass

    def publish(event, mapping):
        if store.event(job["job_id"], f"published:{event['frame']}:car", {"frame": event["frame"], "class": "car"}):
            published.append((event["frame"], mapping[event["observations"][0]["track_id"]]))

    engine = StreamingEngine(store, Settings(_env_file=None, local_frame_cache_max_bytes=1048576), source, publish, GPU)
    with pytest.raises(ValueError, match="interruption"):
        engine.run(job)
    engine.run(store.get(job["job_id"]))
    assert sorted(f for f, _ in published) == list(range(60))
    assert len({identity for _, identity in published}) == 1
    assert len(generations) == 2
    assert store.get(job["job_id"])["profile"]["pipeline"] == mode
