from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from src.models.video import (
    CreateCvatTaskRequest,
    CvatLabelSpec,
    FrameManifest,
    FrameRecord,
    VideoJob,
    VideoJobStatus,
)
from src.services.cvat import CvatReadClient, CvatTaskClient, CvatTrackClient
from src.services.storage.repository import SCHEMA_VERSION, Store
from src.services.video import (
    CvatSeedReader,
    FFmpegVideoExtractor,
    SeedContractError,
    VideoWorker,
    VideoWorkspace,
)
from src.services.video.service import VideoJobService


def _job(*, job_id: str, source_sha256: str, status: VideoJobStatus = VideoJobStatus.UPLOADED) -> VideoJob:
    now = datetime.now(UTC)
    return VideoJob(
        job_id=job_id,
        status=status,
        source_name="clip.mp4",
        source_sha256=source_sha256,
        source_size_bytes=100,
        created_at=now,
        updated_at=now,
        stage=status.value,
    )


def _write_test_video(path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("FFmpeg is not installed")
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=4:duration=1",
            "-c:v",
            "mpeg4",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_video_job_survives_store_restart_and_claim_is_atomic(tmp_path: Path):
    database = tmp_path / "jobs.sqlite3"
    store = Store(database)
    job = _job(job_id="11111111-1111-4111-8111-111111111111", source_sha256="a" * 64)
    store.create_video_job(job)
    store.transition_video_job(
        job.job_id,
        expected={VideoJobStatus.UPLOADED},
        status=VideoJobStatus.QUEUED,
        stage="queued_for_extraction",
    )

    restarted = Store(database)
    claimed = restarted.claim_next_video_job()

    assert claimed is not None
    assert claimed.job.status is VideoJobStatus.EXTRACTING
    assert claimed.claim_token
    assert restarted.claim_next_video_job() is None
    assert restarted.requeue_expired_video_extractions() == 0
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE video_jobs SET claim_expires_at=? WHERE job_id=?",
            ("2000-01-01T00:00:00+00:00", job.job_id),
        )
    assert restarted.requeue_expired_video_extractions() == 1
    assert restarted.get_video_job(job.job_id).status is VideoJobStatus.QUEUED


def test_schema_v1_database_is_migrated_in_place(tmp_path: Path):
    database = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE legacy_record (id INTEGER PRIMARY KEY)")
        connection.execute("INSERT INTO legacy_record VALUES (1)")
        connection.execute("PRAGMA user_version = 1")

    Store(database)

    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert connection.execute("SELECT id FROM legacy_record").fetchone()[0] == 1
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='video_jobs'"
        ).fetchone() == ("video_jobs",)


def test_schema_migration_is_safe_when_workers_start_together(tmp_path: Path):
    database = tmp_path / "concurrent.sqlite3"
    Store(database)
    with sqlite3.connect(database) as connection:
        connection.execute("ALTER TABLE video_jobs DROP COLUMN audit_sha256")
        connection.execute("ALTER TABLE video_jobs DROP COLUMN audit_payload")
        connection.execute("PRAGMA user_version = 5")

    with ThreadPoolExecutor(max_workers=4) as executor:
        stores = list(executor.map(lambda _index: Store(database), range(4)))

    assert len(stores) == 4
    with sqlite3.connect(database) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(video_jobs)")}
        assert {"audit_sha256", "audit_payload"} <= columns
        assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_intent_audit_v8_migration_is_safe_when_workers_start_together(tmp_path: Path):
    database = tmp_path / "intent-v8.sqlite3"
    Store(database)
    with sqlite3.connect(database) as connection:
        connection.execute("ALTER TABLE video_user_intents DROP COLUMN completed_at")
        connection.execute("ALTER TABLE video_user_intents DROP COLUMN http_status")
        connection.execute("PRAGMA user_version = 8")

    with ThreadPoolExecutor(max_workers=4) as executor:
        stores = list(executor.map(lambda _index: Store(database), range(4)))

    assert len(stores) == 4
    with sqlite3.connect(database) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(video_user_intents)")}
        assert {"completed_at", "http_status"} <= columns
        assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_seed_snapshot_is_optimistic_and_persisted_in_database(tmp_path: Path):
    store = Store(tmp_path / "jobs.sqlite3")
    job = _job(
        job_id="55555555-5555-4555-8555-555555555555",
        source_sha256="a" * 64,
        status=VideoJobStatus.AWAITING_SEED,
    ).model_copy(update={"cvat_task_id": 8, "cvat_job_id": 9, "manifest_sha256": "b" * 64})
    store.create_video_job(job)

    attached = store.attach_video_seed(
        job.job_id,
        cvat_job_id=9,
        seed_revision=3,
        seed_count=1,
        seed_sha256="c" * 64,
        seed_payload='{"schema":"autolabel-seed-set/v1"}',
        expected_updated_at=job.updated_at,
    )

    assert attached.status is VideoJobStatus.SEED_READY
    assert store.read_video_seed(job.job_id) == '{"schema":"autolabel-seed-set/v1"}'
    with pytest.raises(ValueError, match="not ready for seed capture"):
        store.attach_video_seed(
            job.job_id,
            cvat_job_id=9,
            seed_revision=4,
            seed_count=1,
            seed_sha256="d" * 64,
            seed_payload="{}",
            expected_updated_at=job.updated_at,
        )


def test_manifest_and_frame_hashes_fail_closed(tmp_path: Path):
    workspace = VideoWorkspace(tmp_path / "video-jobs")
    job_id = "66666666-6666-4666-8666-666666666666"
    frame_dir = workspace.frames_dir(job_id)
    frame_dir.mkdir(parents=True)
    frame = frame_dir / "frame_000000.jpg"
    frame.write_bytes(b"frame-one")
    manifest = _manifest(job_id).model_copy(
        update={
            "frames": [
                _manifest(job_id)
                .frames[0]
                .model_copy(update={"sha256": hashlib.sha256(frame.read_bytes()).hexdigest()})
            ]
        }
    )
    encoded = workspace.write_manifest(manifest)
    expected_hash = hashlib.sha256(encoded).hexdigest()

    loaded = workspace.read_manifest(job_id, expected_sha256=expected_hash)
    workspace.verify_frame_files(loaded)
    frame.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="frame 0 hash"):
        workspace.verify_frame_files(loaded)


def test_worker_extracts_deterministic_frames_and_completes_job(tmp_path: Path):
    store = Store(tmp_path / "jobs.sqlite3")
    workspace = VideoWorkspace(tmp_path / "video-jobs")
    job = _job(job_id="22222222-2222-4222-8222-222222222222", source_sha256="0" * 64)
    job_dir = workspace.job_dir(job.job_id, create=True)
    source = job_dir / "source.mp4"
    _write_test_video(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    job = job.model_copy(update={"source_sha256": digest, "source_size_bytes": source.stat().st_size})
    store.create_video_job(job)
    store.transition_video_job(
        job.job_id,
        expected={VideoJobStatus.UPLOADED},
        status=VideoJobStatus.QUEUED,
        stage="queued_for_extraction",
    )
    worker = VideoWorker(
        store=store,
        workspace=workspace,
        extractor=FFmpegVideoExtractor(workspace, timeout_s=30),
    )

    completed = worker.process_next()
    manifest = workspace.read_manifest(job.job_id)

    assert completed is not None
    assert completed.status is VideoJobStatus.AWAITING_SEED
    assert completed.progress_current == completed.progress_total == 4
    assert manifest.frame_count == 4
    assert [frame.frame_index for frame in manifest.frames] == [0, 1, 2, 3]
    assert [frame.timestamp_s for frame in manifest.frames] == [0.0, 0.25, 0.5, 0.75]
    assert all((job_dir / frame.relative_path).is_file() for frame in manifest.frames)
    assert len({frame.sha256 for frame in manifest.frames}) == 4

    repeated = FFmpegVideoExtractor(workspace, timeout_s=30).extract(job)
    assert [(frame.timestamp_s, frame.sha256) for frame in repeated.frames] == [
        (frame.timestamp_s, frame.sha256) for frame in manifest.frames
    ]


def _manifest(job_id: str) -> FrameManifest:
    return FrameManifest(
        job_id=job_id,
        source_name="clip.mp4",
        source_sha256="f" * 64,
        codec="h264",
        width=640,
        height=480,
        duration_s=1.0,
        average_fps=2.0,
        variable_frame_rate=False,
        ffmpeg_version="ffmpeg test",
        frames=[
            FrameRecord(
                frame_index=0,
                timestamp_s=0.0,
                source_timestamp_s=0.0,
                duration_s=0.5,
                relative_path="frames/frame_000000.jpg",
                sha256="a" * 64,
                width=640,
                height=480,
            ),
            FrameRecord(
                frame_index=1,
                timestamp_s=0.5,
                source_timestamp_s=0.5,
                duration_s=0.5,
                relative_path="frames/frame_000001.jpg",
                sha256="b" * 64,
                width=640,
                height=480,
            ),
        ],
    )


@pytest.mark.asyncio
async def test_cvat_seed_reader_captures_manual_shape_and_track():
    job_id = "33333333-3333-4333-8333-333333333333"

    def handler(request: httpx.Request) -> httpx.Response:
        payloads = {
            "/api/jobs/7": {"id": 7, "task_id": 3, "dimension": "2d", "start_frame": 0},
            "/api/jobs/7/data/meta": {"size": 2, "frames": [{"width": 640, "height": 480}]},
            "/api/jobs/7/annotations": {
                "version": 4,
                "tags": [],
                "shapes": [
                    {
                        "id": 10,
                        "frame": 0,
                        "label_id": 1,
                        "type": "rectangle",
                        "points": [10, 20, 110, 220],
                        "source": "manual",
                    }
                ],
                "tracks": [
                    {
                        "id": 20,
                        "label_id": 2,
                        "source": "manual",
                        "shapes": [
                            {
                                "id": 21,
                                "frame": 0,
                                "type": "rectangle",
                                "points": [200, 100, 300, 250],
                                "outside": False,
                            }
                        ],
                    }
                ],
            },
            "/api/labels": {
                "count": 2,
                "next": None,
                "results": [
                    {"id": 1, "name": "car", "type": "rectangle"},
                    {"id": 2, "name": "person", "type": "any"},
                ],
            },
        }
        return httpx.Response(200, json=payloads[request.url.path])

    client = CvatReadClient(
        base_url="http://cvat.test",
        token="secret",
        retry_delay_s=0,
        transport=httpx.MockTransport(handler),
    )
    seeds = await CvatSeedReader(client).capture(
        job_id=job_id,
        cvat_job_id=7,
        expected_revision=4,
        manifest=_manifest(job_id),
    )

    assert len(seeds.seeds) == 2
    assert {seed.object_id for seed in seeds.seeds} == {"shape:10", "track:20"}
    assert {seed.label_name for seed in seeds.seeds} == {"car", "person"}
    assert all(seed.source == "human_seed" for seed in seeds.seeds)


@pytest.mark.asyncio
async def test_cvat_seed_reader_rejects_stale_revision():
    def handler(request: httpx.Request) -> httpx.Response:
        payloads = {
            "/api/jobs/7": {"id": 7, "dimension": "2d", "start_frame": 0},
            "/api/jobs/7/data/meta": {"size": 2},
            "/api/jobs/7/annotations": {"version": 5, "shapes": [], "tracks": []},
            "/api/labels": {"count": 1, "next": None, "results": [{"id": 1, "name": "car"}]},
        }
        return httpx.Response(200, json=payloads[request.url.path])

    client = CvatReadClient(
        base_url="http://cvat.test",
        token="secret",
        retry_delay_s=0,
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(SeedContractError, match="stale annotation revision"):
        await CvatSeedReader(client).capture(
            job_id="44444444-4444-4444-8444-444444444444",
            cvat_job_id=7,
            expected_revision=4,
            manifest=_manifest("44444444-4444-4444-8444-444444444444"),
        )


@pytest.mark.asyncio
async def test_service_creates_bound_cvat_task_then_persists_seed_snapshot(tmp_path: Path):
    job_id = "77777777-7777-4777-8777-777777777777"
    store = Store(tmp_path / "jobs.sqlite3")
    workspace = VideoWorkspace(tmp_path / "video-jobs")
    frame_dir = workspace.frames_dir(job_id)
    frame_dir.mkdir(parents=True)
    frame = frame_dir / "frame_000000.jpg"
    frame.write_bytes(b"frame")
    manifest = _manifest(job_id).model_copy(
        update={
            "frames": [
                _manifest(job_id)
                .frames[0]
                .model_copy(update={"sha256": hashlib.sha256(frame.read_bytes()).hexdigest()})
            ]
        }
    )
    manifest_bytes = workspace.write_manifest(manifest)
    job = _job(job_id=job_id, source_sha256="f" * 64, status=VideoJobStatus.AWAITING_SEED).model_copy(
        update={"manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()}
    )
    store.create_video_job(job)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/api/tasks":
            payload = json.loads(request.content)
            attributes = {item["name"] for item in payload["labels"][0]["attributes"]}
            assert attributes == {
                "confidence",
                "routing",
                "source",
                "review_status",
                "autolabel_track_id",
                "review_priority",
                "review_score",
                "box_extent",
            }
            return httpx.Response(201, json={"id": 11})
        if request.method == "POST" and request.url.path == "/api/tasks/11/data":
            return httpx.Response(202, json={"rq_id": "upload"})
        payloads = {
            "/api/jobs": {"count": 1, "results": [{"id": 22}]},
            "/api/jobs/22": {"id": 22, "task_id": 11, "dimension": "2d", "start_frame": 0},
            "/api/jobs/22/data/meta": {"size": 1, "frames": [{"width": 640, "height": 480}]},
            "/api/jobs/22/annotations": {
                "version": 1,
                "shapes": [
                    {
                        "id": 5,
                        "frame": 0,
                        "label_id": 1,
                        "type": "rectangle",
                        "points": [10, 20, 110, 220],
                        "source": "manual",
                    }
                ],
                "tracks": [],
            },
            "/api/labels": {
                "count": 1,
                "next": None,
                "results": [{"id": 1, "name": "car", "type": "rectangle"}],
            },
        }
        return httpx.Response(200, json=payloads[request.url.path])

    transport = httpx.MockTransport(handler)
    read_client = CvatReadClient(base_url="http://cvat.test", token="secret", transport=transport)
    task_client = CvatTaskClient(base_url="http://cvat.test", token="secret", retry_delay_s=0, transport=transport)
    track_client = CvatTrackClient(
        base_url="http://cvat.test", token="secret", read_client=read_client, transport=transport
    )
    service = VideoJobService(
        store=store,
        workspace=workspace,
        seed_reader=CvatSeedReader(read_client),
        task_client=task_client,
        track_client=track_client,
        max_upload_bytes=1000,
    )

    binding = await service.create_cvat_task(
        job_id,
        CreateCvatTaskRequest(labels=[CvatLabelSpec(name="car")]),
    )
    seeds = await service.attach_seed(job_id, cvat_job_id=binding.cvat_job_id, annotation_revision=1)

    persisted = store.get_video_job(job_id)
    assert (binding.cvat_task_id, binding.cvat_job_id) == (11, 22)
    assert persisted.status is VideoJobStatus.SEED_READY
    assert persisted.seed_count == 1
    assert seeds.seeds[0].object_id == "shape:5"
    assert workspace.seed_path(job_id).read_text(encoding="utf-8") == store.read_video_seed(job_id)
