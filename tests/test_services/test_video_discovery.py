from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.models.video import (
    BoundingBox2D,
    ComponentVersion,
    CvatPublishResult,
    Detection2D,
    DetectionSet,
    DiscoveryResult,
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
from src.services.video.cheap_review import CheapReviewConfig
from src.services.video.discovery import VideoDiscoveryError, discover_missing_objects
from src.services.video.discovery_worker import DiscoveryWorker
from src.services.video.review import build_review_queue
from src.services.video.service import VideoJobService
from src.services.video.workspace import VideoWorkspace

JOB_ID = "77777777-7777-4777-8777-777777777777"
MANIFEST_SHA = "b" * 64
SEED_SHA = "c" * 64


def _manifest(hashes: list[str] | None = None) -> FrameManifest:
    hashes = hashes or [str(index + 1) * 64 for index in range(4)]
    return FrameManifest(
        job_id=JOB_ID,
        source_name="clip.mp4",
        source_sha256="a" * 64,
        codec="h264",
        width=100,
        height=80,
        duration_s=2,
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
            )
            for index, digest in enumerate(hashes)
        ],
    )


def _seeds() -> SeedSet:
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
                bbox=BoundingBox2D(xtl=10, ytl=10, xbr=30, ybr=30),
            )
        ],
    )


def _propagated() -> TrackSet:
    return TrackSet(
        job_id=JOB_ID,
        manifest_sha256=MANIFEST_SHA,
        seed_sha256=SEED_SHA,
        component_versions=[ComponentVersion(component="sam2-video", version="sam2-test")],
        tracks=[
            VideoTrack(
                track_id="shape:5",
                label_id=1,
                label_name="car",
                source="human_seed",
                observations=[
                    TrackObservation(
                        frame_index=index,
                        bbox=BoundingBox2D(xtl=10 + index, ytl=10, xbr=30 + index, ybr=30),
                        score=1 if index == 0 else 0.9,
                    )
                    for index in range(4)
                ],
            )
        ],
    )


def _detections() -> DetectionSet:
    return DetectionSet(
        job_id=JOB_ID,
        manifest_sha256=MANIFEST_SHA,
        component_version=ComponentVersion(component="rtdetr-object-detector", version="rtdetr-test"),
        detections=[
            Detection2D(
                frame_index=1,
                label_id=1,
                label_name="car",
                model_label_name="car",
                bbox=BoundingBox2D(xtl=11, ytl=10, xbr=31, ybr=30),
                score=0.95,
            ),
            Detection2D(
                frame_index=2,
                label_id=1,
                label_name="car",
                model_label_name="car",
                bbox=BoundingBox2D(xtl=60, ytl=20, xbr=80, ybr=40),
                score=0.91,
            ),
            Detection2D(
                frame_index=3,
                label_id=1,
                label_name="car",
                model_label_name="car",
                bbox=BoundingBox2D(xtl=62, ytl=20, xbr=82, ybr=40),
                score=0.89,
            ),
        ],
    )


def test_discovery_groups_an_object_entering_after_frame_zero():
    result = discover_missing_objects(
        manifest=_manifest(),
        propagated=_propagated(),
        detections=_detections(),
        propagated_track_sha256="d" * 64,
    )

    assert len(result.proposal_tracks) == 1
    proposal = result.proposal_tracks[0]
    assert proposal.track_id == "auto:missing:000001"
    assert [item.frame_index for item in proposal.observations] == [2, 3]
    assert [(event.event_type, event.frame_index) for event in result.events] == [("possible_missing_object", 2)]


def test_cheap_review_score_and_percentile_buckets_are_deterministic():
    manifest = _manifest()
    propagated = _propagated()
    discovery = discover_missing_objects(
        manifest=manifest,
        propagated=propagated,
        detections=_detections(),
        propagated_track_sha256="d" * 64,
    )
    tracks = propagated.model_copy(update={"tracks": propagated.tracks + discovery.proposal_tracks})
    config = CheapReviewConfig(critical_fraction=0.25, high_fraction=0.25, normal_fraction=0.25)
    kwargs = {
        "tracks": tracks,
        "discovery": discovery,
        "manifest": manifest,
        "track_set_sha256": "f" * 64,
        "discovery_sha256": "e" * 64,
        "scoring_config": config,
    }
    queue = build_review_queue(**kwargs)
    assert queue == build_review_queue(**kwargs)
    assert [unit.priority for unit in queue.units] == ["critical", "high", "normal", "low"]
    assert all(0 <= unit.review_score <= 1 for unit in queue.units)
    assert queue.audit_sha256 is None and queue.scoring_version.endswith("uncalibrated")
    assert any("short_proposal" in unit.reason_codes for unit in queue.units)


def test_cheap_review_rejects_unscaled_weights():
    with pytest.raises(ValueError, match="sum to one"):
        CheapReviewConfig(weights={name: 1 for name in CheapReviewConfig().weights})


def test_discovery_worker_persists_combined_tracks_and_evidence(tmp_path: Path):
    store = Store(tmp_path / "jobs.sqlite3")
    workspace = VideoWorkspace(tmp_path / "video-jobs")
    frame_dir = workspace.frames_dir(JOB_ID)
    frame_dir.mkdir(parents=True)
    hashes: list[str] = []
    for index in range(4):
        frame_path = frame_dir / f"frame_{index:06d}.jpg"
        frame_path.write_bytes(f"frame-{index}".encode())
        hashes.append(hashlib.sha256(frame_path.read_bytes()).hexdigest())
    manifest = _manifest(hashes)
    manifest_bytes = workspace.write_manifest(manifest)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    seeds = _seeds()
    seed_bytes = workspace.encode_seed_set(seeds)
    seed_sha = hashlib.sha256(seed_bytes).hexdigest()
    propagated = _propagated().model_copy(update={"manifest_sha256": manifest_sha, "seed_sha256": seed_sha})
    track_bytes = workspace.encode_track_set(propagated)
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
        manifest_sha256=manifest_sha,
        cvat_task_id=11,
        cvat_job_id=22,
    )
    store.create_video_job(job)
    attached = store.attach_video_seed(
        JOB_ID,
        cvat_job_id=22,
        seed_revision=1,
        seed_count=1,
        seed_sha256=seed_sha,
        seed_payload=seed_bytes.decode(),
        expected_updated_at=job.updated_at,
    )
    workspace.write_seed_payload(JOB_ID, seed_bytes)
    store.queue_video_propagation(JOB_ID, frame_count=manifest.frame_count)
    propagation_claim = store.claim_next_video_propagation()
    assert propagation_claim is not None
    store.complete_video_propagation(
        JOB_ID,
        claim_token=propagation_claim.claim_token,
        track_set_sha256=hashlib.sha256(track_bytes).hexdigest(),
        track_payload=track_bytes.decode(),
        component_versions=propagated.component_versions,
        frame_count=manifest.frame_count,
    )

    class FixtureDetector:
        def detect(self, *, manifest, manifest_sha256, labels):
            assert labels == {1: "car"}
            assert manifest_sha256 == manifest_sha
            return _detections().model_copy(update={"manifest_sha256": manifest_sha})

    class FailingDetector:
        def detect(self, *, manifest, manifest_sha256, labels):
            raise VideoDiscoveryError("temporary detector failure")

    failed = DiscoveryWorker(
        store=store, workspace=workspace, detector=FailingDetector(), lease_seconds=3
    ).process_next()
    assert failed is not None and failed.status is VideoJobStatus.FAILED
    assert failed.error_code == "video_discovery_failed"
    retried = store.queue_video_discovery(JOB_ID, frame_count=manifest.frame_count)
    assert retried.stage == "queued_for_discovery"
    assert retried.track_set_sha256 == hashlib.sha256(track_bytes).hexdigest()
    with pytest.raises(ValueError, match="not ready for discovery retry"):
        store.queue_video_discovery(JOB_ID, frame_count=manifest.frame_count)

    completed = DiscoveryWorker(
        store=store,
        workspace=workspace,
        detector=FixtureDetector(),
        lease_seconds=3,
    ).process_next()

    assert attached.status is VideoJobStatus.SEED_READY
    assert completed is not None
    assert completed.status is VideoJobStatus.AWAITING_REVIEW
    assert completed.stage == "awaiting_review"
    assert completed.discovery_sha256
    persisted = TrackSet.model_validate_json(store.read_video_track_set(JOB_ID))
    assert [track.track_id for track in persisted.tracks] == ["shape:5", "auto:missing:000001"]
    evidence = DiscoveryResult.model_validate_json(store.read_video_discovery(JOB_ID))
    assert evidence.events[0].event_type == "possible_missing_object"
    assert workspace.track_set_path(JOB_ID).is_file()
    assert workspace.discovery_path(JOB_ID).is_file()
    from src.models.video import ReviewQueue

    queue = ReviewQueue.model_validate_json(store.read_video_review_queue(JOB_ID))
    assert queue.audit_sha256 is None and queue.units
    assert all(unit.review_score is not None for unit in queue.units)
    assert not workspace.audit_path(JOB_ID).exists()
    assert store.claim_next_video_audit() is None

    class FakeTrackClient:
        async def publish_tracks(self, **kwargs):
            assert kwargs["review_score_by_track"]
            return CvatPublishResult(
                job_id=JOB_ID,
                cvat_job_id=22,
                revision_before=1,
                revision_after=2,
                published_track_count=2,
                payload_sha256="e" * 64,
            ), "{}"

    service = VideoJobService(
        store=store,
        workspace=workspace,
        seed_reader=None,
        task_client=None,
        track_client=FakeTrackClient(),
        max_upload_bytes=1000,
    )
    published = asyncio.run(service.publish_tracks(JOB_ID, expected_revision=1))
    assert published.published_track_count == 2
    assert store.get_video_job(JOB_ID).publish_sha256 is not None
