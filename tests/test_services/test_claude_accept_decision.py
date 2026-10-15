import asyncio
import hashlib
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest

from src.models.video import FrameManifest, FrameRecord, VideoJob, VideoJobStatus
from src.services.cvat import CvatReadError, CvatWriteError
from src.services.storage.repository import Store
from src.services.video.service import VideoJobService
from src.services.video.workspace import VideoWorkspace


def make_service(tmp_path, write, *, status="done"):
    store = Store(tmp_path / "db.sqlite3")
    workspace = VideoWorkspace(tmp_path / "jobs")
    job_id = str(uuid4())
    workspace.job_dir(job_id, create=True)
    manifest = FrameManifest(
        job_id=job_id,
        source_name="test.mp4",
        source_sha256="a" * 64,
        source_frame_count=8,
        width=100,
        height=80,
        duration_s=1,
        codec="h264",
        average_fps=10,
        variable_frame_rate=False,
        ffmpeg_version="fixture",
        frames=[
            FrameRecord(
                frame_index=i,
                source_frame_index=f,
                timestamp_s=f / 10,
                source_timestamp_s=f / 10,
                width=100,
                height=80,
                relative_path=f"frames/frame_{i:06}.jpg",
                sha256="b" * 64,
            )
            for i, f in enumerate((0, 6))
        ],
    )
    encoded = workspace.write_manifest(manifest)
    now = datetime.now(UTC)
    store.create_video_job(
        VideoJob(
            job_id=job_id,
            status=VideoJobStatus.AWAITING_REVIEW,
            source_name="test.mp4",
            source_sha256="a" * 64,
            source_size_bytes=1,
            manifest_sha256=hashlib.sha256(encoded).hexdigest(),
            created_at=now,
            updated_at=now,
            stage="awaiting_review",
            cvat_job_id=1,
        )
    )
    proposal = dict(
        proposal_id="p", frame_index=1, source_frame_index=6, label="car", bbox=[0.1, 0.1, 0.3, 0.3], status="pending"
    )
    with store.connect() as db:
        db.execute(
            "INSERT INTO video_box_rescues(job_id,status,proposals_payload,partial_payload,updated_at) VALUES(?,?,?,?,?)",
            (job_id, status, json.dumps([proposal]), json.dumps([{"proposals": [proposal]}]), now.isoformat()),
        )

    async def labels(job_id):
        return [{"name": "car", "id": 1}]

    service = VideoJobService(
        store=store,
        workspace=workspace,
        seed_reader=SimpleNamespace(client=SimpleNamespace(get_job_labels=labels)),
        track_client=SimpleNamespace(accept_rescue_track=write),
        task_client=None,
        max_upload_bytes=100,
    )
    return service, job_id


@pytest.mark.asyncio
async def test_accept_reserves_decision_before_write_and_serializes_retry(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()
    writes = []

    async def write(**kwargs):
        entered.set()
        await release.wait()
        writes.append(kwargs)
        return 42

    service, job_id = make_service(tmp_path, write)
    first = asyncio.create_task(service.accept_box_rescue(job_id, "p", expected_revision=1))
    await entered.wait()
    with pytest.raises(ValueError, match="busy"):
        service.reject_box_rescue(job_id, "p")
    second = asyncio.create_task(service.accept_box_rescue(job_id, "p", expected_revision=1))
    release.set()
    assert all(r["status"] == "accepted" for r in await asyncio.gather(first, second))
    assert len(writes) == 1 and writes[0]["frame"] == 6


@pytest.mark.asyncio
async def test_unknown_accept_outcome_stays_reserved_until_idempotent_retry(tmp_path):
    calls = []

    async def write(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise CvatWriteError("response lost after possible create")
        return 42

    service, job_id = make_service(tmp_path, write)
    with pytest.raises(CvatWriteError):
        await service.accept_box_rescue(job_id, "p", expected_revision=1)
    assert service.get_box_rescue_state(job_id)["proposals"][0]["status"] == "accepting"
    with pytest.raises(ValueError, match="busy"):
        service.reject_box_rescue(job_id, "p")
    assert (await service.accept_box_rescue(job_id, "p", expected_revision=1))["status"] == "accepted"


@pytest.mark.asyncio
async def test_partial_results_cannot_be_accepted(tmp_path):
    async def forbidden(**kwargs):
        pytest.fail("unfinished audit must not write CVAT")

    service, job_id = make_service(tmp_path, forbidden, status="failed")
    with pytest.raises(ValueError, match="not ready"):
        await service.accept_box_rescue(job_id, "p", expected_revision=1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "retry_error", [CvatReadError("read unavailable"), CvatWriteError("conflict", status_code=409)]
)
async def test_inconclusive_retry_never_releases_unknown_accept(tmp_path, retry_error):
    calls = 0

    async def write(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise CvatWriteError("create may have committed; response lost")
        if calls == 2:
            raise retry_error
        return 42  # Server reconciles the stable proposal identity on retry.

    service, job_id = make_service(tmp_path, write)
    for error_type in (CvatWriteError, type(retry_error)):
        with pytest.raises(error_type):
            await service.accept_box_rescue(job_id, "p", expected_revision=1)
        assert service.get_box_rescue_state(job_id)["proposals"][0]["status"] == "accepting"
        with pytest.raises(ValueError, match="busy"):
            service.reject_box_rescue(job_id, "p")
    assert (await service.accept_box_rescue(job_id, "p", expected_revision=1))["status"] == "accepted"
