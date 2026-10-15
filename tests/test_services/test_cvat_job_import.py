from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import httpx
import pytest
from PIL import Image

from src.models.video import CvatSamplingRequest, VideoJob, VideoJobStatus
from src.services.cvat import CvatReadClient, CvatTaskClient, CvatTrackClient
from src.services.storage.repository import Store
from src.services.video import CvatSeedReader, VideoWorkspace
from src.services.video.service import VideoJobService
from src.services.video.worker import VideoWorker


def _jpeg() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (32, 24), color=(20, 30, 40)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _service(tmp_path: Path, handler: httpx.MockTransport) -> VideoJobService:
    reader = CvatReadClient(base_url="http://cvat.test", token="test", transport=handler)
    return VideoJobService(
        store=Store(tmp_path / "jobs.sqlite3"),
        workspace=VideoWorkspace(tmp_path / "frames"),
        seed_reader=CvatSeedReader(reader),
        task_client=CvatTaskClient(base_url="http://cvat.test", token="test", transport=handler),
        track_client=CvatTrackClient(base_url="http://cvat.test", token="test", read_client=reader, transport=handler),
        max_upload_bytes=1024 * 1024,
    )


def test_import_existing_cvat_job_reuses_frames_and_binding(tmp_path: Path):
    frame = _jpeg()
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/jobs/7":
            return httpx.Response(
                200, json={"id": 7, "task_id": 3, "dimension": "2d", "start_frame": 0, "stop_frame": 1}
            )
        if request.url.path.endswith("/data/meta"):
            return httpx.Response(
                200,
                json={
                    "size": 2,
                    "frame_filter": "",
                    "deleted_frames": [],
                    "included_frames": None,
                    "frames": [{"width": 32, "height": 24}, {"width": 32, "height": 24}],
                },
            )
        assert request.url.params["type"] == "frame"
        assert request.url.params["quality"] in {"compressed", "original"}
        return httpx.Response(200, content=frame, headers={"content-type": "image/jpeg"})

    service = _service(tmp_path, httpx.MockTransport(handler))
    imported = asyncio.run(service.import_cvat_job(7, CvatSamplingRequest(strategy="budget", budget=2, source_fps=10)))
    repeated = asyncio.run(service.import_cvat_job(7))

    assert repeated.job_id == imported.job_id
    assert len(requests) == 2
    assert imported.status is VideoJobStatus.QUEUED
    worker = VideoWorker(
        store=service.store, workspace=service.workspace, extractor=None, cvat_client=service.seed_reader.client
    )  # type: ignore[arg-type]
    completed = worker.process_next()
    assert completed is not None and completed.status is VideoJobStatus.AWAITING_SEED
    assert (imported.cvat_task_id, imported.cvat_job_id) == (3, 7)
    manifest = service.get_manifest(imported.job_id)
    assert manifest.frame_count == 2
    assert manifest.source_frame_count == 2
    assert [frame.source_frame_index for frame in manifest.frames] == [0, 1]
    assert [frame.timestamp_s for frame in manifest.frames] == [0.0, 0.1]
    assert manifest.average_fps == 10
    assert manifest.source_fps == 10
    assert manifest.fps_source == "operator"
    assert manifest.width == 32 and manifest.height == 24
    service.workspace.verify_frame_files(manifest)


def test_budget_import_keeps_original_cvat_indexes(tmp_path: Path):
    images = []
    for index in range(10):
        buffer = BytesIO()
        image = Image.new("RGB", (64, 40), color=(0, 0, 0))
        for x in range(10 + index * 3, 20 + index * 3):
            for y in range(10, 20):
                image.putpixel((x, y), (255, 255, 255))
        image.save(buffer, format="JPEG")
        images.append(buffer.getvalue())

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/jobs/9":
            return httpx.Response(
                200, json={"id": 9, "task_id": 4, "dimension": "2d", "start_frame": 0, "stop_frame": 9}
            )
        if request.url.path.endswith("/data/meta"):
            return httpx.Response(200, json={"size": 10, "frames": [{"width": 64, "height": 40}]})
        return httpx.Response(200, content=images[int(request.url.params["number"])])

    service = _service(tmp_path, httpx.MockTransport(handler))
    queued = asyncio.run(service.import_cvat_job(9, CvatSamplingRequest(strategy="budget", budget=4, source_fps=10)))
    worker = VideoWorker(
        store=service.store, workspace=service.workspace, extractor=None, cvat_client=service.seed_reader.client
    )  # type: ignore[arg-type]
    result = worker.process_next()
    assert result is not None and result.status is VideoJobStatus.AWAITING_SEED
    manifest = service.get_manifest(queued.job_id)
    assert manifest.frame_count == 4 and manifest.source_frame_count == 10
    indexes = [frame.source_frame_index for frame in manifest.frames]
    assert indexes[0] == 0 and indexes[-1] == 9
    assert indexes == sorted(set(indexes))
    service.workspace.verify_frame_files(manifest)


def test_uniform_sequence_import_queues_sam3_without_frame_zero_seed(tmp_path: Path):
    frame = _jpeg()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/labels":
            return httpx.Response(200, json={"results": [{"id": 11, "name": "car"}], "next": None})
        if request.url.path == "/api/jobs/7":
            return httpx.Response(
                200, json={"id": 7, "task_id": 3, "dimension": "2d", "start_frame": 0, "stop_frame": 4}
            )
        if request.url.path.endswith("/data/meta"):
            return httpx.Response(
                200,
                json={
                    "size": 5,
                    "frame_filter": "",
                    "deleted_frames": [],
                    "included_frames": None,
                    "frames": [{"width": 32, "height": 24}] * 5,
                },
            )
        return httpx.Response(200, content=frame, headers={"content-type": "image/jpeg"})

    service = _service(tmp_path, httpx.MockTransport(handler))
    imported = asyncio.run(
        service.import_cvat_job(
            7,
            CvatSamplingRequest(
                strategy="uniform",
                source_fps=10,
                target_fps=5,
                auto_start_sequence=True,
                sequence_classes=["car"],
            ),
        )
    )
    worker = VideoWorker(
        store=service.store, workspace=service.workspace, extractor=None, cvat_client=service.seed_reader.client
    )  # type: ignore[arg-type]
    ready = worker.process_next()
    assert ready is not None and ready.status is VideoJobStatus.PROPAGATING
    assert ready.stage == "queued_for_sequence"
    assert ready.seed_sha256 is None
    manifest = service.get_manifest(imported.job_id)
    assert [item.source_frame_index for item in manifest.frames] == [0, 2, 4]


@pytest.mark.asyncio
async def test_sequence_import_rejects_unknown_class_before_import(tmp_path: Path):
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/api/jobs/7":
            return httpx.Response(
                200, json={"id": 7, "task_id": 3, "dimension": "2d", "start_frame": 0, "stop_frame": 4}
            )
        if request.url.path.endswith("/data/meta"):
            return httpx.Response(200, json={"size": 5, "frames": [{"width": 32, "height": 24}] * 5})
        if request.url.path == "/api/labels":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"id": 11, "name": "car"},
                        {"id": 12, "name": "truck"},
                        {"id": 13, "name": "pedestrian"},
                    ],
                    "next": None,
                },
            )
        raise AssertionError("unknown class must be rejected before frame download")

    service = _service(tmp_path, httpx.MockTransport(handler))
    with pytest.raises(ValueError, match="person.*car.*pedestrian.*truck"):
        await service.import_cvat_job(
            7,
            CvatSamplingRequest(
                strategy="uniform",
                source_fps=30,
                target_fps=5,
                auto_start_sequence=True,
                sequence_classes=["car", "person", "truck"],
            ),
        )
    assert service.find_cvat_job(7) is None
    assert "/api/labels" in paths


def test_failed_sequence_can_correct_classes_without_reimport(tmp_path: Path):
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/api/jobs/7":
            return httpx.Response(
                200, json={"id": 7, "task_id": 3, "dimension": "2d", "start_frame": 0, "stop_frame": 4}
            )
        if request.url.path.endswith("/data/meta"):
            return httpx.Response(200, json={"size": 5, "frames": [{"width": 32, "height": 24}] * 5})
        if request.url.path == "/api/labels":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"id": 11, "name": "car"},
                        {"id": 12, "name": "truck"},
                        {"id": 13, "name": "pedestrian"},
                    ],
                    "next": None,
                },
            )
        if request.url.path.endswith("/data"):
            return httpx.Response(200, content=_jpeg(), headers={"content-type": "image/jpeg"})
        raise AssertionError(request.url)

    service = _service(tmp_path, httpx.MockTransport(handler))
    imported = asyncio.run(
        service.import_cvat_job(
            7,
            CvatSamplingRequest(
                strategy="uniform",
                source_fps=10,
                target_fps=5,
                auto_start_sequence=True,
                sequence_classes=["car"],
            ),
        )
    )
    worker = VideoWorker(
        store=service.store, workspace=service.workspace, extractor=None, cvat_client=service.seed_reader.client
    )  # type: ignore[arg-type]
    assert worker.process_next() is not None
    claim = service.store.claim_next_video_sequence(lease_seconds=60)
    assert claim is not None
    service.store.fail_video_job(
        imported.job_id,
        claim_token=claim.claim_token,
        code="sequence_failed",
        message="CVAT job does not contain every requested class",
    )
    frame_calls = sum(path.endswith("/data") for path in calls)
    with pytest.raises(ValueError, match="person.*car.*pedestrian.*truck"):
        asyncio.run(service.retry_sequence_with_classes(imported.job_id, ["car", "person", "truck"]))
    assert service.get_job(imported.job_id).status is VideoJobStatus.FAILED
    retried = asyncio.run(service.retry_sequence_with_classes(imported.job_id, ["car", "truck", "pedestrian"]))
    assert retried.job_id == imported.job_id and retried.stage == "queued_for_sequence"
    assert service.workspace.read_sampling_request(imported.job_id).sequence_classes == ["car", "truck", "pedestrian"]
    assert sum(path.endswith("/data") for path in calls) == frame_calls


def test_failed_frame_import_retries_same_job_without_reupload(tmp_path: Path):
    frame = _jpeg()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/jobs/7":
            return httpx.Response(
                200, json={"id": 7, "task_id": 3, "dimension": "2d", "start_frame": 0, "stop_frame": 4}
            )
        if request.url.path.endswith("/data/meta"):
            return httpx.Response(200, json={"size": 5, "frames": [{"width": 32, "height": 24}] * 5})
        if request.url.path == "/api/labels":
            return httpx.Response(200, json={"results": [{"id": 11, "name": "car"}], "next": None})
        return httpx.Response(200, content=frame)

    service = _service(tmp_path, httpx.MockTransport(handler))
    request = CvatSamplingRequest(
        strategy="uniform", source_fps=10, target_fps=5, auto_start_sequence=True, sequence_classes=["car"]
    )
    imported = asyncio.run(service.import_cvat_job(7, request))
    claim = service.store.claim_next_video_job()
    assert claim is not None
    service.store.fail_video_job(
        imported.job_id, code="video_extraction_internal_error", message="disk I/O error", claim_token=claim.claim_token
    )
    retried = service.retry_cvat_import(imported.job_id)
    assert retried.job_id == imported.job_id and retried.cvat_job_id == 7
    assert retried.source_sha256 == imported.source_sha256 and retried.error_code is None
    assert service.workspace.read_sampling_request(imported.job_id) == request
    with pytest.raises(ValueError):
        service.retry_cvat_import(imported.job_id)  # Double clicks must not queue twice.
    worker = VideoWorker(
        store=service.store, workspace=service.workspace, extractor=None, cvat_client=service.seed_reader.client
    )  # type: ignore[arg-type]
    ready = worker.process_next()
    assert ready is not None and ready.stage == "queued_for_sequence" and ready.seed_sha256 is None


def test_frame_import_retry_does_not_retry_a_model_failure(tmp_path: Path):
    service = _service(tmp_path, httpx.MockTransport(lambda _: httpx.Response(500)))
    with pytest.raises(KeyError):
        service.retry_cvat_import("missing")
    now = datetime.now(UTC)
    job = VideoJob(
        job_id="b77c7eb2-a015-48d8-89d0-d27a91191250",
        status=VideoJobStatus.FAILED,
        source_name="cvat.frames",
        source_sha256="a" * 64,
        source_size_bytes=1,
        cvat_job_id=7,
        cvat_task_id=3,
        created_at=now,
        updated_at=now,
        stage="failed",
        error_code="sequence_failed",
        error_message="model error",
    )
    service.store.create_video_job(job)
    with pytest.raises(ValueError, match="retryable extraction"):
        service.retry_cvat_import(job.job_id)
    assert service.store.get_video_job(job.job_id).error_code == "sequence_failed"


@pytest.mark.asyncio
async def test_import_requires_declared_fps_when_cvat_does_not_expose_one(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/jobs/7":
            return httpx.Response(
                200, json={"id": 7, "task_id": 3, "dimension": "2d", "start_frame": 0, "stop_frame": 1}
            )
        if request.url.path.endswith("/data/meta"):
            return httpx.Response(200, json={"size": 2, "frames": [{"width": 32, "height": 24}]})
        raise AssertionError("frames must not be downloaded without a known fps")

    service = _service(tmp_path, httpx.MockTransport(handler))
    with pytest.raises(ValueError, match="source_fps"):
        await service.import_cvat_job(7, CvatSamplingRequest(strategy="budget", budget=2))


@pytest.mark.asyncio
async def test_import_rejects_nonzero_start_before_reading_frames(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/jobs/7":
            return httpx.Response(
                200, json={"id": 7, "task_id": 3, "dimension": "2d", "start_frame": 4, "stop_frame": 5}
            )
        if request.url.path.endswith("/data/meta"):
            return httpx.Response(200, json={"size": 2, "frames": [{}, {}]})
        raise AssertionError("unsupported job must not download frames")

    service = _service(tmp_path, httpx.MockTransport(handler))
    with pytest.raises(ValueError, match="starting at frame 0"):
        await service.import_cvat_job(7)
