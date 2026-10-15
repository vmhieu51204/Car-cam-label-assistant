from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest

from src.api.video import BulkRescueRequest, bulk_box_rescue
from src.models.video import VideoJob, VideoJobStatus
from src.services.cvat.task_client import CvatWriteError
from src.services.storage.repository import Store


def test_review_and_ignore_regions_survive_store_restart(tmp_path):
    path = tmp_path / "review.sqlite"
    store = Store(path)
    job = VideoJob(
        job_id=str(uuid4()),
        status=VideoJobStatus.UPLOADED,
        source_name="fixture.mp4",
        source_sha256="a" * 64,
        source_size_bytes=1,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        stage="uploaded",
    )
    store.create_video_job(job)
    store.set_track_review_state(job.job_id, "t1", "seen")
    store.set_export_options(job.job_id, {"ignore_regions": [[0, 0, 0.2, 0.2]]})
    again = Store(path)
    assert again.track_review_states(job.job_id) == {"t1": "seen"}
    assert again.get_export_options(job.job_id) == {"ignore_regions": [[0, 0, 0.2, 0.2]]}
    assert again.get_video_job(job.job_id).status == VideoJobStatus.UPLOADED
    with pytest.raises(ValueError):
        again.set_track_review_state(job.job_id, "t1", "accepted")


@pytest.mark.asyncio
async def test_bulk_partial_conflict_and_idempotent_duplicate_retry():
    accepted = set()
    revision = 7
    calls = []

    async def accept(job, proposal, expected_revision):
        nonlocal revision
        calls.append((proposal, expected_revision))
        if proposal in accepted:
            return {"revision_after": None}
        if proposal == "b":
            raise CvatWriteError("new edits", status_code=409)
        assert expected_revision == revision
        revision += 1
        accepted.add(proposal)
        return {"revision_after": revision}

    service = SimpleNamespace(get_box_rescue_state=lambda _: {"status": "done"}, accept_box_rescue=accept)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(video_jobs=service)))
    body = BulkRescueRequest(proposal_ids=["a", "a", "b", "c"], accept=True, expected_annotation_revision=7)
    result = await bulk_box_rescue(request, "job", body)
    assert [r["status"] for r in result["results"]] == ["accepted", "conflict", "accepted"]
    assert calls == [("a", 7), ("b", 8), ("c", 8)]
    retry = await bulk_box_rescue(request, "job", body.model_copy(update={"expected_annotation_revision": 9}))
    assert retry["revision"] == 9 and revision == 9 and accepted == {"a", "c"}


@pytest.mark.asyncio
async def test_bulk_reject_has_no_annotation_write():
    rejected = []
    service = SimpleNamespace(
        get_box_rescue_state=lambda _: {"status": "done"},
        reject_box_rescue=lambda job, p: rejected.append(p) or {"status": "rejected"},
    )
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(video_jobs=service)))
    result = await bulk_box_rescue(
        request, "job", BulkRescueRequest(proposal_ids=["a", "b"], accept=False, expected_annotation_revision=5)
    )
    assert rejected == ["a", "b"] and result["revision"] == 5
