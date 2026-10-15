from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from src.models.video import (
    BoundingBox2D,
    ComponentVersion,
    FrameManifest,
    FrameRecord,
    SeedRectangle,
    SeedSet,
    TrackObservation,
    TrackSet,
    VideoJob,
    VideoJobStatus,
    VideoTrack,
)
from src.services.storage.repository import Store
from src.services.video import PropagationWorker, VideoWorkspace

JOB_ID = "88888888-8888-4888-8888-888888888888"


def _seed_set() -> SeedSet:
    return SeedSet(
        job_id=JOB_ID,
        cvat_job_id=22,
        annotation_revision=1,
        seeds=[
            SeedRectangle(
                seed_id="cvat:shape:5:frame:0",
                object_id="shape:5",
                label_id=1,
                label_name="car",
                shape_id=5,
                bbox=BoundingBox2D(xtl=10, ytl=20, xbr=60, ybr=70),
            )
        ],
    )


def _manifest(frame_hashes: list[str]) -> FrameManifest:
    return FrameManifest(
        job_id=JOB_ID,
        source_name="clip.mp4",
        source_sha256="a" * 64,
        codec="h264",
        width=100,
        height=80,
        duration_s=1.5,
        average_fps=2,
        variable_frame_rate=False,
        ffmpeg_version="ffmpeg test",
        frames=[
            FrameRecord(
                frame_index=index,
                timestamp_s=index * 0.5,
                source_timestamp_s=index * 0.5,
                relative_path=f"frames/frame_{index:06d}.jpg",
                sha256=digest,
                width=100,
                height=80,
                scene_change_score=0.8 if index == 2 else None,
                scene_cut_before=index == 2,
            )
            for index, digest in enumerate(frame_hashes)
        ],
    )


def _model_track_set(manifest_sha256: str, seed_sha256: str) -> TrackSet:
    return TrackSet(
        job_id=JOB_ID,
        manifest_sha256=manifest_sha256,
        seed_sha256=seed_sha256,
        component_versions=[ComponentVersion(component="sam2-video", version="facebook/sam2.1-hiera-tiny")],
        tracks=[
            VideoTrack(
                track_id="shape:5",
                label_id=1,
                label_name="car",
                source="auto",
                observations=[
                    TrackObservation(
                        frame_index=0,
                        bbox=BoundingBox2D(xtl=12, ytl=22, xbr=58, ybr=68),
                        score=0.9,
                        mask_area_px=1200,
                    ),
                    TrackObservation(
                        frame_index=1,
                        bbox=BoundingBox2D(xtl=14, ytl=22, xbr=62, ybr=70),
                        score=0.8,
                        mask_area_px=1150,
                    ),
                ],
            )
        ],
    )


def test_propagation_worker_persists_versioned_track_set(tmp_path: Path):
    store = Store(tmp_path / "jobs.sqlite3")
    workspace = VideoWorkspace(tmp_path / "video-jobs")
    frame_dir = workspace.frames_dir(JOB_ID)
    frame_dir.mkdir(parents=True)
    hashes = []
    for index in range(2):
        path = frame_dir / f"frame_{index:06d}.jpg"
        path.write_bytes(f"frame-{index}".encode())
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    manifest = _manifest(hashes)
    manifest_bytes = workspace.write_manifest(manifest)
    now = datetime.now(UTC)
    job = VideoJob(
        job_id=JOB_ID,
        status=VideoJobStatus.AWAITING_SEED,
        source_name="clip.mp4",
        source_sha256="a" * 64,
        source_size_bytes=100,
        created_at=now,
        updated_at=now,
        stage="awaiting_seed",
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        cvat_task_id=11,
        cvat_job_id=22,
    )
    store.create_video_job(job)
    seed_bytes = workspace.encode_seed_set(_seed_set())
    seeded = store.attach_video_seed(
        JOB_ID,
        cvat_job_id=22,
        seed_revision=1,
        seed_count=1,
        seed_sha256=hashlib.sha256(seed_bytes).hexdigest(),
        seed_payload=seed_bytes.decode(),
        expected_updated_at=job.updated_at,
    )
    store.queue_video_propagation(JOB_ID, frame_count=manifest.frame_count)

    class FixturePropagator:
        def propagate(self, *, manifest, manifest_sha256, seed_set, seed_sha256):
            assert seed_set.job_id == JOB_ID
            return _model_track_set(manifest_sha256, seed_sha256)

    completed = PropagationWorker(
        store=store,
        workspace=workspace,
        propagator=FixturePropagator(),
        lease_seconds=3,
    ).process_next()

    assert seeded.status is VideoJobStatus.SEED_READY
    assert completed is not None
    assert completed.status is VideoJobStatus.DISCOVERING
    assert completed.stage == "queued_for_discovery"
    assert completed.track_set_sha256
    persisted = TrackSet.model_validate_json(store.read_video_track_set(JOB_ID))
    assert persisted.component_versions[0].version == "facebook/sam2.1-hiera-tiny"
    assert workspace.track_set_path(JOB_ID).is_file()
