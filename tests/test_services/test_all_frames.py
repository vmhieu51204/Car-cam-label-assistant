import asyncio
import json
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from scripts.run_video_2d import run_sequence
from src.models.video import CvatSamplingRequest, VideoJob, VideoJobStatus
from src.services.video.cvat_import import import_cvat_frames
from src.services.video.sampling import choose_keyframes
from src.services.video.sequence_kaggle import sequence_payload
from src.services.video.workspace import VideoWorkspace
from tests.test_scripts.test_run_video_2d import FakeTextModel
from tests.test_services.test_uniform_sequence_optimization import Frames


@pytest.mark.parametrize("count,fps", [(1, 29.97), (2, 30), (301, 30.0003)])
def test_all_frames_import_once_with_real_timebase(tmp_path, count, fps):
    identity = str(uuid4())
    workspace = VideoWorkspace(tmp_path)
    request = CvatSamplingRequest(source_fps=fps, auto_start_sequence=True, sequence_classes=["car"])
    assert request.strategy == "all_frames"
    workspace.write_sampling_request(identity, request)
    now = datetime.now(UTC)
    job = VideoJob(
        job_id=identity,
        status=VideoJobStatus.QUEUED,
        source_name="full",
        source_sha256="0" * 64,
        source_size_bytes=1,
        created_at=now,
        updated_at=now,
        stage="queued",
        cvat_task_id=3,
        cvat_job_id=7,
    )
    client = Frames(count)
    manifest = asyncio.run(import_cvat_frames(job, client=client, workspace=workspace, progress=lambda *_: None))
    assert client.requested == list(range(count))
    assert [r.source_frame_index for r in manifest.frames] == list(range(count))
    assert manifest.frames[-1].timestamp_s == (count - 1) / fps
    assert manifest.target_fps == fps
    _, config = sequence_payload(manifest, workspace, ("car",))
    assert config["sampling_strategy"] == "all_frames"
    assert choose_keyframes([0.0] * count, request)[0] == list(range(count))


def test_cannot_silently_reduce_full_frame_request():
    with pytest.raises(ValueError, match="cannot reduce"):
        CvatSamplingRequest(source_fps=30, target_fps=5)
    with pytest.raises(ValueError, match="requires source_fps"):
        CvatSamplingRequest()


def test_legacy_missing_sampling_file_still_reads_budget(tmp_path):
    assert VideoWorkspace(tmp_path).read_sampling_request(str(uuid4())).strategy == "budget"


def test_offline_default_full_frames_cannot_resume_sampled(tmp_path):
    from PIL import Image

    images = tmp_path / "images"
    images.mkdir()
    for i in range(7):
        Image.new("RGB", (30, 20), "gray").save(images / f"{i:06d}.png")
    backend = FakeTextModel()
    result = run_sequence(
        images, tmp_path / "full", source_fps=30, labels=("car",), backend=backend, window_size=3, overlap=1
    )
    assert result.source_frames == list(range(7))
    assert backend.loads == 1
    manifest = json.loads((tmp_path / "full/sequence-manifest.json").read_text())
    assert manifest["sampling_strategy"] == "all_frames"
    with pytest.raises(ValueError, match="inputs changed"):
        run_sequence(images, tmp_path / "full", source_fps=30, target_fps=5, labels=("car",), backend=backend)


def test_full_frame_scene_cut_separates_windows_and_ids(tmp_path):
    from PIL import Image

    images = tmp_path / "images"
    images.mkdir()
    for i in range(4):
        Image.new("RGB", (30, 20), "black" if i < 2 else "white").save(images / f"{i:06d}.png")
    backend = FakeTextModel()
    result = run_sequence(images, tmp_path / "cut", source_fps=30, labels=("car",), backend=backend)
    manifest = json.loads((tmp_path / "cut/sequence-manifest.json").read_text())
    assert manifest["scene_cut_frames"] == [2]
    raw = json.loads(result.stitched.read_text())
    assert len(raw["tracks"]) == 2
    assert [o["frame"] for o in raw["tracks"][0]["observations"]] == [0, 1]
    assert [o["frame"] for o in raw["tracks"][1]["observations"]] == [2, 3]


def test_processing_ledger_distinguishes_overlap_and_failed_frame(tmp_path):
    from types import SimpleNamespace

    from src.services.video.chunks import plan_chunks
    from src.services.video.sequence_checkpoints import SequenceCheckpoints

    ledger = SequenceCheckpoints.__new__(SequenceCheckpoints)
    ledger.directory = tmp_path / "sequence-checkpoints"
    ledger.directory.mkdir()
    ledger.job_id, ledger.identity = "test", "a" * 64
    ledger.frames = [{"source_frame_index": i, "original_source_index": i} for i in range(4)]
    ledger.plans = plan_chunks(4, window_size=3, overlap=1)
    ledger.store = SimpleNamespace(
        list=lambda _: [SimpleNamespace(index=0, status="done"), SimpleNamespace(index=1, status="failed")]
    )
    ledger.write_frame_status()
    rows = json.loads((tmp_path / "frame-processing.json").read_text())["frames"]
    assert [r["status"] for r in rows] == ["processed", "processed", "failed", "failed"]
    assert rows[2]["window_ids"] == [0, 1]


@pytest.mark.parametrize("include_checkpoint", [True, False])
def test_full_frame_worker_requires_verified_coverage(tmp_path, include_checkpoint):
    import hashlib

    from kaggle.checkpoint_contract import checkpoint_identity
    from tests.test_services.test_sequence_kaggle_worker import _queued_worker

    store, worker, _, identity = _queued_worker(tmp_path)
    manifest = worker.workspace.read_manifest(identity)
    frames = [
        f.model_copy(update={"source_frame_index": i, "timestamp_s": i / 10, "source_timestamp_s": i / 10})
        for i, f in enumerate(manifest.frames)
    ]
    manifest = manifest.model_copy(
        update={"frames": frames, "source_frame_count": 3, "sampling_strategy": "all_frames", "target_fps": 10}
    )
    digest = hashlib.sha256(worker.workspace.write_manifest(manifest)).hexdigest()
    with store.connect() as connection:
        connection.execute("UPDATE video_jobs SET manifest_sha256=? WHERE job_id=?", (digest, identity))
    worker.workspace.write_sampling_request(
        identity, CvatSamplingRequest(source_fps=10, auto_start_sequence=True, sequence_classes=["car"])
    )

    class Runner:
        def run(self, **kwargs):
            files = kwargs["files"]
            config = json.loads(files["benchmark-config.json"])
            row = config["configurations"][0]
            inputs = json.loads(files["frames.json"])["frames"]
            assert [f["original_source_index"] for f in inputs] == [0, 1, 2]
            predictions = [
                {
                    "window_id": 0,
                    "frame_index": i,
                    "track_id": "car:1",
                    "label": "car",
                    "bbox": [1, 1, 12, 12],
                    "score": 0.8,
                }
                for i in range(3)
            ]
            entry = {
                "schema": "autolabel-sam3-window/v1",
                "config_id": row["id"],
                "window_id": 0,
                "input_sha256": checkpoint_identity(inputs, row, config["code_sha256"]),
                "frame_ids": [0, 1, 2],
                "predictions": predictions,
                "model": {"sam3_commit": "fake", "checkpoint_sha256": "a" * 64},
            }
            entry["output_sha256"] = hashlib.sha256(
                json.dumps(entry, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            return {
                "schema": "autolabel-sam3-benchmark/v1",
                "configurations": [
                    {
                        "id": "sequence-text",
                        "config": row,
                        "status": "completed",
                        "selected_source_frames": [0, 1, 2],
                        "evaluated_source_frames": [0, 1, 2],
                        "window_predictions": predictions,
                        "window_checkpoints": [entry] if include_checkpoint else [],
                    }
                ],
            }

    worker.runner = Runner()
    result = worker.process_next()
    assert result.status == (VideoJobStatus.AWAITING_REVIEW if include_checkpoint else VideoJobStatus.FAILED)
    rows = json.loads((worker.workspace.job_dir(identity) / "frame-processing.json").read_text())["frames"]
    assert {r["status"] for r in rows} == ({"processed"} if include_checkpoint else {"failed"})


@pytest.mark.parametrize("failure_stage", ["load", "close", "corrupt_resume"])
def test_local_coverage_reports_model_or_artifact_failure(tmp_path, failure_stage):
    from PIL import Image

    images = tmp_path / "images"
    images.mkdir()
    for i in range(2):
        Image.new("RGB", (30, 20), "gray").save(images / f"{i:06d}.png")
    backend = FakeTextModel()
    output = tmp_path / "output"
    if failure_stage == "corrupt_resume":
        good = run_sequence(images, output, source_fps=30, labels=("car",), backend=backend)
        chunk = good.stitched.parent / "chunk-0000.json"
        chunk.write_text("corrupted")
    else:

        def fail():
            raise RuntimeError(f"model {failure_stage} failed")

        setattr(backend, failure_stage, fail)
    with pytest.raises((RuntimeError, ValueError)):
        run_sequence(images, output, source_fps=30, labels=("car",), backend=backend)
    coverage = json.loads((output / "frame-processing.json").read_text())
    assert coverage["run_error"]
    assert {row["status"] for row in coverage["frames"]} == {"failed"}
    assert [row["source_frame_index"] for row in coverage["frames"]] == [0, 1]
