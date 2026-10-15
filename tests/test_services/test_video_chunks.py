"""Durable chunk planning and independent retries for a whole sequence."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from src.models.video import VideoJob, VideoJobStatus
from src.services.storage.repository import Store
from src.services.video.chunks import ChunkStore, plan_chunks


def _job() -> VideoJob:
    now = datetime.now(UTC)
    return VideoJob(
        job_id=str(uuid4()),
        status=VideoJobStatus.QUEUED,
        source_name="sequence",
        source_sha256="a" * 64,
        source_size_bytes=1,
        created_at=now,
        updated_at=now,
        stage="queued",
    )


def test_plan_preserves_all_frames_and_both_sides_of_overlap():
    chunks = plan_chunks(120, window_size=50, overlap=5)
    assert [(chunk.start, chunk.stop) for chunk in chunks] == [(0, 50), (45, 95), (90, 120)]
    assert set(range(120)) == set().union(*(set(range(c.start, c.stop)) for c in chunks))
    assert set(range(chunks[0].start, chunks[0].stop)) & set(range(chunks[1].start, chunks[1].stop)) == set(
        range(45, 50)
    )


def test_chunk_failure_retries_only_that_chunk_and_keeps_completed_output(tmp_path):
    store = Store(tmp_path / "jobs.sqlite3")
    job = _job()
    store.create_video_job(job)
    chunks = ChunkStore(store)
    chunks.register(
        job.job_id, plan_chunks(80, window_size=50, overlap=5), model_version="sam3@test", input_sha256="b" * 64
    )
    first = chunks.claim(job.job_id, max_attempts=2)
    assert first is not None and first.index == 0 and first.attempts == 1
    chunks.complete(first, output_sha256="c" * 64)
    second = chunks.claim(job.job_id, max_attempts=2)
    assert second is not None and second.index == 1
    chunks.fail(second, "GPU OOM")
    retry = chunks.claim(job.job_id, max_attempts=2)
    assert retry is not None and retry.index == 1 and retry.attempts == 2
    assert chunks.list(job.job_id)[0].status == "done"
    chunks.complete(retry, output_sha256="d" * 64)
    assert [item.status for item in chunks.list(job.job_id)] == ["done", "done"]
    assert chunks.claim(job.job_id, max_attempts=2) is None


def test_cancelled_chunk_rejects_late_result(tmp_path):
    store = Store(tmp_path / "jobs.sqlite3")
    job = _job()
    store.create_video_job(job)
    chunks = ChunkStore(store)
    chunks.register(
        job.job_id, plan_chunks(10, window_size=50, overlap=5), model_version="sam3@test", input_sha256="b" * 64
    )
    claim = chunks.claim(job.job_id, max_attempts=2)
    assert claim is not None
    store.cancel_video_job(job.job_id)
    assert chunks.list(job.job_id)[0].status == "cancelled"
    with pytest.raises(ValueError, match="claim"):
        chunks.complete(claim, output_sha256="c" * 64)
