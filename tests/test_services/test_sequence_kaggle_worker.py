"""A seed-free CVAT job reaches review without a GPU, VLM, or manual seed."""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from PIL import Image

from src.models.video import CvatSamplingRequest, FrameManifest, FrameRecord, VideoJob, VideoJobStatus
from src.services.storage.repository import Store
from src.services.video.sequence_worker import SequenceWorker
from src.services.video.service import VideoJobService
from src.services.video.workspace import VideoWorkspace


class _Cvat:
    async def get_job_labels(self, job_id: int) -> list[dict]:
        assert job_id == 7
        return [{"id": 11, "name": "car"}]


class _Kaggle:
    def __init__(self) -> None:
        self.calls = 0
        self.last_variant = None
        self.last_config = None

    def run(self, *, stage: str, job_id: str, files: dict, code_file: Path, model_variant: str = "sam3") -> dict:
        assert stage == "sam3-sequence"
        assert code_file.is_file()
        self.calls += 1
        self.last_variant = model_variant
        config = json.loads(files["benchmark-config.json"])["configurations"][0]
        self.last_config = config
        frames = json.loads(files["frames.json"])["frames"]
        assert [row["original_source_index"] for row in frames] == [0, 2, 4]
        assert json.loads(files["seed.json"]) == {"seeds": []}
        return {
            "schema": "autolabel-sam3-benchmark/v1",
            "completed": True,
            "model_runtime": {
                "environment": {"versions": {"sam3_commit": "test-commit"}},
                "model_variant": model_variant,
                "checkpoint_sha256": "a" * 64,
            },
            "configurations": [
                {
                    "id": "sequence-text",
                    "config": config,
                    "status": "completed",
                    "selected_source_frames": [0, 1, 2],
                    "evaluated_source_frames": [0, 1, 2],
                    "window_predictions": [
                        {
                            "window_id": 0,
                            "frame_index": index,
                            "track_id": "car:1",
                            "label": "car",
                            "bbox": [2 + index, 3, 12 + index, 13],
                            "score": 0.9,
                            "mask_rle": None,
                        }
                        for index in (0, 1)
                    ],
                }
            ],
        }


def _queued_worker(
    tmp_path: Path, model_variant: str = "sam3", *, vlm_box_rescue: bool = False
) -> tuple[Store, SequenceWorker, _Kaggle, str]:
    store = Store(tmp_path / "jobs.sqlite3")
    workspace = VideoWorkspace(tmp_path / "workspace")
    job_id = str(uuid4())
    frame_dir = workspace.frames_dir(job_id)
    frame_dir.mkdir(parents=True)
    records = []
    for index, source_index in enumerate((0, 2, 4)):
        buffer = BytesIO()
        Image.new("RGB", (32, 24), (index, 0, 0)).save(buffer, format="JPEG")
        encoded = buffer.getvalue()
        (frame_dir / f"frame_{index:06d}.jpg").write_bytes(encoded)
        records.append(
            FrameRecord(
                frame_index=index,
                source_frame_index=source_index,
                timestamp_s=source_index / 10,
                source_timestamp_s=source_index / 10,
                relative_path=f"frames/frame_{index:06d}.jpg",
                sha256=hashlib.sha256(encoded).hexdigest(),
                width=32,
                height=24,
            )
        )
    manifest = FrameManifest(
        job_id=job_id,
        source_name="cvat-job-7.frames",
        source_sha256="a" * 64,
        source_frame_count=5,
        sampling_strategy="uniform",
        source_fps=10,
        target_fps=5,
        fps_source="operator",
        codec="jpeg",
        width=32,
        height=24,
        duration_s=0.5,
        average_fps=5,
        variable_frame_rate=False,
        ffmpeg_version="CVAT test",
        frames=records,
    )
    manifest_sha = hashlib.sha256(workspace.write_manifest(manifest)).hexdigest()
    workspace.write_sampling_request(
        job_id,
        CvatSamplingRequest(
            strategy="uniform",
            source_fps=10,
            target_fps=5,
            auto_start_sequence=True,
            sam3_model_variant=model_variant,
            vlm_box_rescue=vlm_box_rescue,
            sequence_classes=["car"],
        ),
    )
    now = datetime.now(UTC)
    store.create_video_job(
        VideoJob(
            job_id=job_id,
            status=VideoJobStatus.AWAITING_SEED,
            source_name="cvat-job-7.frames",
            source_sha256="a" * 64,
            source_size_bytes=1,
            created_at=now,
            updated_at=now,
            stage="awaiting_seed",
            manifest_sha256=manifest_sha,
            cvat_job_id=7,
            cvat_task_id=3,
        )
    )
    store.queue_video_sequence(job_id, frame_count=3)
    runner = _Kaggle()
    worker = SequenceWorker(
        store=store,
        workspace=workspace,
        cvat=_Cvat(),
        runner=runner,
        notebook=Path(__file__).resolve().parents[2] / "kaggle/notebooks/autolabel_sam3_benchmark.ipynb",
    )
    return store, worker, runner, job_id


def test_sequence_worker_preserves_source_frame_mapping_and_review(tmp_path: Path) -> None:
    store, worker, runner, job_id = _queued_worker(tmp_path)

    completed = worker.process_next()

    assert completed is not None and completed.status is VideoJobStatus.AWAITING_REVIEW
    assert runner.calls == 1
    assert runner.last_variant == "sam3"
    tracks = json.loads(store.read_video_track_set(job_id))["tracks"]
    assert [item["frame_index"] for item in tracks[0]["observations"]] == [0, 1, 2]
    assert [item["outside"] for item in tracks[0]["observations"]] == [False, False, True]
    queue = json.loads(store.read_video_review_queue(job_id))
    assert queue["audit_sha256"] is None
    assert len(queue["units"]) == 3
    assert worker.process_next() is None


def test_sequence_worker_queues_separate_vlm_proposals_without_changing_sam_tracks(tmp_path: Path) -> None:
    from src.models.video import SequenceAuditUsage
    from src.services.video.box_rescue_worker import BoxRescueWorker
    from src.services.video.rescue_matrix import RescueFrameVerdict, RescueProposal

    store, sequence, runner, job_id = _queued_worker(tmp_path, vlm_box_rescue=True)
    completed = sequence.process_next()
    assert completed is not None and completed.status is VideoJobStatus.AWAITING_REVIEW
    assert store.claim_next_video_box_rescue(lease_seconds=3) is None
    _mark_draft_exported(store, job_id)
    sam_tracks = store.read_video_track_set(job_id)
    state = store.get_video_box_rescue_state(job_id)
    assert state["status"] == "queued" and state["proposals"] == []

    class FakeAuditor:
        model = "anthropic/claude-sonnet-5.5"

        def __init__(self):
            self.calls = []

        def propose_missing_boxes(self, **kwargs):
            self.calls.append(kwargs)
            proposal = RescueProposal(
                label="car",
                bbox=(0.5, 0.2, 0.8, 0.7),
                point=(0.6, 0.4),
                occlusion="partial",
                confidence="medium",
                reason="visible vehicle",
            )
            return (
                RescueFrameVerdict(missing="yes", proposals=[proposal], reason="vehicle visible"),
                SequenceAuditUsage(prompt_tokens=10, completion_tokens=8),
                False,
            )

    auditor = FakeAuditor()
    rescue = BoxRescueWorker(store=store, workspace=sequence.workspace, auditor=auditor, lease_seconds=3)
    assert rescue.process_next() == job_id
    state = store.get_video_box_rescue_state(job_id)
    assert state["status"] == "done" and state["proposal_count"] > 0
    assert len(auditor.calls) <= 20
    assert all("car" in item["classes"] for item in auditor.calls)
    proposal = state["proposals"][0]
    assert proposal["status"] == "pending" and proposal["source_frame_index"] in {0, 2, 4}
    assert sequence.workspace.box_rescue_preview_path(job_id, proposal["proposal_id"]).is_file()
    assert store.read_video_track_set(job_id) == sam_tracks
    assert runner.calls == 1


def test_box_rescue_failure_exposes_partial_proposals_and_retry_resumes_remaining_frames(tmp_path: Path) -> None:
    from src.models.video import SequenceAuditUsage
    from src.services.video.box_rescue_worker import BoxRescueWorker
    from src.services.video.rescue_matrix import RescueFrameVerdict, RescueProposal

    store, sequence, runner, job_id = _queued_worker(tmp_path, vlm_box_rescue=True)
    sequence.process_next()
    _mark_draft_exported(store, job_id)
    _mark_draft_exported(store, job_id)
    original_tracks = store.read_video_track_set(job_id)
    proposal = RescueProposal(
        label="car",
        bbox=(0.5, 0.2, 0.8, 0.7),
        point=(0.6, 0.4),
        occlusion="partial",
        confidence="medium",
        reason="visible vehicle",
    )

    class FailingAuditor:
        model = "anthropic/claude-sonnet-5.5"

        def __init__(self, fail_on_call=None):
            self.calls = 0
            self.fail_on_call = fail_on_call

        def propose_missing_boxes(self, **kwargs):
            self.calls += 1
            if self.calls == self.fail_on_call:
                raise RuntimeError("HTTP 429")
            return (
                RescueFrameVerdict(missing="yes", proposals=[proposal], reason="vehicle visible"),
                SequenceAuditUsage(prompt_tokens=10, completion_tokens=8),
                False,
            )

    first_auditor = FailingAuditor(fail_on_call=2)
    worker = BoxRescueWorker(store=store, workspace=sequence.workspace, auditor=first_auditor, lease_seconds=3)
    assert worker.process_next() == job_id
    failed = store.get_video_box_rescue_state(job_id)
    assert failed["status"] == "failed" and "429" in failed["error"]
    assert failed["completed_frame_count"] == 1 and failed["proposal_count"] == 1
    assert failed["proposals"][0]["status"] == "pending"

    store.retry_video_box_rescue(job_id)
    retry_auditor = FailingAuditor()  # The first successful frame is reused from its checkpoint.
    retry_worker = BoxRescueWorker(store=store, workspace=sequence.workspace, auditor=retry_auditor, lease_seconds=3)
    assert retry_worker.process_next() == job_id
    completed = store.get_video_box_rescue_state(job_id)
    assert completed["status"] == "done" and completed["completed_frame_count"] == 3
    assert completed["proposal_count"] == 3 and retry_auditor.calls == 2
    assert store.read_video_track_set(job_id) == original_tracks
    assert runner.calls == 1


def test_sequence_worker_exports_pending_draft_without_second_user_click(tmp_path: Path) -> None:
    store, worker, runner, job_id = _queued_worker(tmp_path)

    async def publish(identity: str, **kwargs: object) -> None:
        assert identity == job_id and kwargs["worker_export"] is True
        tracks = json.loads(store.read_video_track_set(identity))["tracks"]
        assert all(track["review_status"] == "pending" for track in tracks)
        store.record_video_publish(
            identity,
            cvat_job_id=7,
            publish_sha256="f" * 64,
            publish_payload="{}",
            cvat_annotation_revision=1,
            published_track_count=len(tracks),
        )

    service = Mock(spec=VideoJobService)
    service.prepare_sequence_labels = AsyncMock(return_value=[{"id": 11, "name": "car"}])
    service.publish_tracks = AsyncMock(side_effect=publish)
    worker.draft_service = service
    completed = worker.process_next()
    assert completed is not None and completed.stage == "published_to_cvat"
    assert completed.status is VideoJobStatus.AWAITING_REVIEW
    assert completed.published_track_count == 1
    assert runner.calls == 1
    assert worker.process_next() is None
    service.publish_tracks.assert_awaited_once()


def test_sequence_draft_export_resumes_without_gpu_and_failure_keeps_review(tmp_path: Path) -> None:
    store, worker, runner, job_id = _queued_worker(tmp_path)
    worker.process_next()  # Persist inference first, simulating a restart before CVAT export.
    service = Mock(spec=VideoJobService)
    service.publish_tracks = AsyncMock(side_effect=RuntimeError("CVAT unreachable"))
    worker.draft_service = service
    completed = worker.process_next()
    assert completed is not None and completed.stage == "draft_export_failed"
    assert completed.status is VideoJobStatus.AWAITING_REVIEW
    assert completed.track_set_sha256 and completed.review_queue_sha256
    assert runner.calls == 1
    assert worker.process_next() is None  # Unknown write outcomes must not be blindly retried.
    assert store.get_video_job(job_id).error_code == "sequence_draft_export_failed"


def test_bad_cvat_label_setup_stops_before_kaggle_submission(tmp_path: Path) -> None:
    _, worker, runner, _ = _queued_worker(tmp_path)
    service = Mock(spec=VideoJobService)
    service.prepare_sequence_labels = AsyncMock(side_effect=ValueError("incompatible review_status"))
    worker.draft_service = service
    completed = worker.process_next()
    assert completed is not None and completed.status is VideoJobStatus.FAILED
    assert completed.error_code == "sequence_failed"
    assert runner.calls == 0
    service.publish_tracks.assert_not_called()


def test_incomplete_kaggle_result_is_not_exported(tmp_path: Path) -> None:
    _, worker, _, _ = _queued_worker(tmp_path)

    class IncompleteRunner(_Kaggle):
        def run(self, **kwargs: object) -> dict:
            result = super().run(**kwargs)
            result["configurations"][0]["evaluated_source_frames"] = [0]
            return result

    worker.runner = IncompleteRunner()
    completed = worker.process_next()
    assert completed is not None and completed.status is VideoJobStatus.FAILED
    assert completed.track_set_sha256 is None and completed.publish_sha256 is None


def test_partial_window_results_persist_and_retry_reuses_checkpoint(tmp_path):
    from kaggle.sam3_benchmark.main import checkpoint_identity
    from src.services.video.chunks import ChunkStore

    store, worker, _, job_id = _queued_worker(tmp_path)

    class Partial(_Kaggle):
        def run(self, **kwargs):
            result = super().run(**kwargs)
            frames = json.loads(kwargs["files"]["frames.json"])["frames"]
            top = json.loads(kwargs["files"]["benchmark-config.json"])
            item = result["configurations"][0]
            entry = {
                "schema": "autolabel-sam3-window/v1",
                "config_id": item["id"],
                "input_sha256": checkpoint_identity(frames, item["config"], top["code_sha256"]),
                "window_id": 0,
                "frame_ids": [0, 1, 2],
                "model": {"sam3_commit": "test-commit", "checkpoint_sha256": None},
                "predictions": item["window_predictions"],
            }
            entry["output_sha256"] = hashlib.sha256(
                json.dumps(entry, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            item["window_checkpoints"] = [entry]
            item["status"] = "failed"
            return result

    runner = Partial()
    worker.runner = runner
    failed = worker.process_next()
    assert failed.status is VideoJobStatus.FAILED
    assert failed.track_set_sha256 is None
    assert ChunkStore(store).list(job_id)[0].status == "done"
    assert runner.calls == 1
    store.queue_video_sequence(job_id, frame_count=3)
    completed = worker.process_next()
    assert completed.status is VideoJobStatus.AWAITING_REVIEW
    assert runner.calls == 1  # All windows are verified locally; no second GPU call.
    assert ChunkStore(store).list(job_id)[0].attempts == 1


def _mark_draft_exported(store, job_id):
    store.record_video_publish(
        job_id,
        cvat_job_id=7,
        publish_sha256="f" * 64,
        publish_payload="{}",
        cvat_annotation_revision=1,
        published_track_count=1,
    )


@pytest.mark.asyncio
async def test_accept_source_frame_and_idempotent_reject_do_not_rerun_sam(tmp_path):
    from src.services.cvat.track_client import CvatWriteError
    from src.services.video.seeds import CvatSeedReader

    store, sequence, runner, job_id = _queued_worker(tmp_path, vlm_box_rescue=True)
    await asyncio.to_thread(sequence.process_next)
    _mark_draft_exported(store, job_id)
    claim = store.claim_next_video_box_rescue(lease_seconds=3)
    proposals = [
        dict(
            proposal_id=str(i),
            frame_index=1,
            source_frame_index=2,
            label="car",
            bbox=[0.1, 0.1, 0.5, 0.5],
            status="pending",
        )
        for i in range(2)
    ]
    store.complete_video_box_rescue(job_id, claim_token=claim.claim_token, proposals=proposals)
    client = Mock()
    client.accept_rescue_track = AsyncMock(side_effect=CvatWriteError("changed", status_code=409))
    service = VideoJobService(
        store=store,
        workspace=sequence.workspace,
        seed_reader=CvatSeedReader(_Cvat()),
        task_client=Mock(),
        track_client=client,
        max_upload_bytes=1000,
    )
    tracks_before = store.read_video_track_set(job_id)
    with pytest.raises(CvatWriteError):
        await service.accept_box_rescue(job_id, "0", expected_revision=1)
    assert service.get_box_rescue_state(job_id)["proposals"][0]["status"] == "pending"
    client.accept_rescue_track.side_effect = None
    client.accept_rescue_track.return_value = 2
    await service.accept_box_rescue(job_id, "0", expected_revision=1)
    assert client.accept_rescue_track.call_args.kwargs["frame"] == 2
    assert client.accept_rescue_track.call_args.kwargs["proposal_id"] == "0"
    await service.accept_box_rescue(job_id, "0", expected_revision=1)
    assert client.accept_rescue_track.await_count == 2  # Conflict + one write, no repeated write.
    service.reject_box_rescue(job_id, "1")
    service.reject_box_rescue(job_id, "1")
    assert client.accept_rescue_track.await_count == 2
    assert store.read_video_track_set(job_id) == tracks_before and runner.calls == 1


def test_proposal_limit_reports_omitted_candidates_and_keeps_frame_artifacts(tmp_path):
    store, sequence, runner, job_id = _queued_worker(tmp_path, vlm_box_rescue=True)
    sequence.process_next()
    _mark_draft_exported(store, job_id)
    claim = store.claim_next_video_box_rescue(lease_seconds=3)
    proposals = [dict(proposal_id=str(i), status="pending") for i in range(25)]
    store.save_video_box_rescue_partial(
        job_id, claim_token=claim.claim_token, partial=[dict(frame_index=0, proposals=proposals)], lease_seconds=3
    )
    store.complete_video_box_rescue(job_id, claim_token=claim.claim_token, proposals=proposals)
    result = store.get_video_box_rescue_state(job_id)
    assert len(result["proposals"]) == 20 and result["truncated_proposal_count"] == 5
    assert runner.calls == 1
