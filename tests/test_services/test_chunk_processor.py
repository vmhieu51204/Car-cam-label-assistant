"""One run completes every planned chunk and resumes without redoing done work."""

import json
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from src.models.video import VideoJob, VideoJobStatus
from src.services.storage.repository import Store
from src.services.video.chunk_processor import ChunkProcessor
from src.services.video.chunks import ChunkStore
from src.services.video.stitching import ChunkObservation, ChunkTrack, StitchConfig


class FakeSam3:
    version = "fake-sam3@1"

    def __init__(self):
        self.loads = 0
        self.calls = []

    def load(self):
        self.loads += 1

    def run_chunk(self, *, index, frames, labels):
        self.calls.append(index)
        assert labels == ("car",)
        return [
            ChunkTrack(
                window_id=index,
                local_track_id=f"car-{index}",
                label="car",
                observations=[ChunkObservation(frame=f, bbox=(float(f), 0, float(f + 10), 10)) for f in frames],
            )
        ]

    def close(self):
        pass


def test_processor_loads_once_and_merges_all_chunks(tmp_path):
    store = Store(tmp_path / "jobs.sqlite3")
    now = datetime.now(UTC)
    job = VideoJob(
        job_id=str(uuid4()),
        status=VideoJobStatus.QUEUED,
        source_name="sequence",
        source_sha256="a" * 64,
        source_size_bytes=1,
        created_at=now,
        updated_at=now,
        stage="queued",
    )
    store.create_video_job(job)
    backend = FakeSam3()
    processor = ChunkProcessor(
        store=ChunkStore(store), artifact_root=tmp_path / "artifacts", backend=backend, window_size=6, overlap=2
    )
    result = processor.run(job.job_id, frame_count=10, input_sha256="b" * 64, labels=("car",))
    assert backend.loads == 1
    artifact = tmp_path / "artifacts" / job.job_id / "stitched.json"
    assert json.loads(artifact.read_text())["boundary_logs"]
    processor.stitch = StitchConfig(decision_log_enabled=False)
    without_logs = processor.run(job.job_id, frame_count=10, input_sha256="b" * 64, labels=("car",))
    assert without_logs.tracks == result.tracks and without_logs.decisions == result.decisions
    assert json.loads(artifact.read_text())["boundary_logs"] == []
    assert backend.calls == [0, 1] and backend.loads == 1
    assert backend.calls == [0, 1]
    assert [item.status for item in ChunkStore(store).list(job.job_id)] == ["done", "done"]
    assert len(result.tracks) == 1
    assert [item.frame for item in result.tracks[0].observations] == list(range(10))
    assert result.decisions[0].action == "matched"
    processor.run(job.job_id, frame_count=10, input_sha256="b" * 64, labels=("car",))
    assert backend.calls == [0, 1]
    assert backend.loads == 1
    with pytest.raises(ValueError, match="different inputs"):
        processor.run(job.job_id, frame_count=10, input_sha256="b" * 64, labels=("person",))


def test_processor_preserves_empty_middle_chunk_and_marks_later_track(tmp_path):
    class MissingMiddle(FakeSam3):
        def run_chunk(self, *, index, frames, labels):
            if index == 1:
                self.calls.append(index)
                return []
            return super().run_chunk(index=index, frames=frames, labels=labels)

    store = Store(tmp_path / "jobs.sqlite3")
    now = datetime.now(UTC)
    job = VideoJob(
        job_id=str(uuid4()),
        status=VideoJobStatus.QUEUED,
        source_name="sequence",
        source_sha256="a" * 64,
        source_size_bytes=1,
        created_at=now,
        updated_at=now,
        stage="queued",
    )
    store.create_video_job(job)
    backend = MissingMiddle()
    result = ChunkProcessor(
        store=ChunkStore(store),
        artifact_root=tmp_path / "artifacts",
        backend=backend,
        window_size=6,
        overlap=2,
    ).run(job.job_id, frame_count=14, input_sha256="b" * 64, labels=("car",))
    assert backend.calls == [0, 1, 2]
    assert len(result.tracks) == 2
    assert result.tracks[1].needs_review
    assert result.decisions[0].reason == "empty_chunk_gap"
