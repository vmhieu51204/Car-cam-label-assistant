from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx

from src.models.video import FrameManifest, FrameRecord, VideoJob, VideoJobStatus
from src.services.cvat import CvatReadClient, CvatTrackClient, build_cvat_annotation_payload
from src.services.storage.repository import Store
from src.services.video.auto_seed import AutoSeedWorker, KaggleSam3Seeder
from src.services.video.workspace import VideoWorkspace

JOB_ID = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"


def _setup(tmp_path: Path):
    store = Store(tmp_path / "jobs.sqlite3")
    workspace = VideoWorkspace(tmp_path / "video-jobs")
    frame_path = workspace.job_dir(JOB_ID, create=True) / "frames/frame_000000.jpg"
    frame_path.parent.mkdir()
    frame_path.write_bytes(b"frame-zero")
    manifest = FrameManifest(
        job_id=JOB_ID,
        source_name="clip.mp4",
        source_sha256="a" * 64,
        codec="h264",
        width=100,
        height=80,
        duration_s=0,
        average_fps=1,
        variable_frame_rate=False,
        ffmpeg_version="test",
        frames=[
            FrameRecord(
                frame_index=0,
                timestamp_s=0,
                source_timestamp_s=0,
                relative_path="frames/frame_000000.jpg",
                sha256=hashlib.sha256(frame_path.read_bytes()).hexdigest(),
                width=100,
                height=80,
            )
        ],
    )
    manifest_bytes = workspace.write_manifest(manifest)
    now = datetime.now(UTC)
    store.create_video_job(
        VideoJob(
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
    )
    return store, workspace, manifest


def test_auto_seed_worker_adds_editable_cvat_boxes_and_persists_seed(tmp_path: Path):
    store, workspace, _manifest = _setup(tmp_path)
    annotations = {"version": 1, "tags": [], "shapes": [], "tracks": []}
    writes = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/jobs/22/annotations":
            if request.method == "PATCH":
                payload = json.loads(request.content)
                assert request.url.params["action"] == "create"
                assert payload["version"] == 1
                assert payload["shapes"][0]["source"] == "auto"
                writes.append(payload)
                annotations["version"] = 2
                annotations["shapes"] = [{**payload["shapes"][0], "id": 7}]
            return httpx.Response(200, json=annotations)
        if path == "/api/jobs/22":
            return httpx.Response(200, json={"id": 22, "task_id": 11, "dimension": "2d", "start_frame": 0})
        if path == "/api/jobs/22/data/meta":
            return httpx.Response(200, json={"size": 1, "frames": [{"width": 100, "height": 80}]})
        if path == "/api/jobs/22/labels":
            return httpx.Response(
                200, json={"count": 1, "next": None, "results": [{"id": 1, "name": "car", "type": "rectangle"}]}
            )
        if path == "/api/labels":
            return httpx.Response(
                200, json={"count": 1, "next": None, "results": [{"id": 1, "name": "car", "type": "rectangle"}]}
            )
        raise AssertionError(path)

    class FakeRunner:
        def run(self, *, stage, job_id, files, code_file):
            assert stage == "sam3-seed" and job_id == JOB_ID
            assert json.loads(files["request.json"])["labels"] == [{"id": 1, "prompt": "car"}]
            return {
                "stage": stage,
                "job_id": job_id,
                "detections": [
                    {"label_id": 1, "score": 0.9, "bbox": {"xtl": 10, "ytl": 20, "xbr": 50, "ybr": 60}},
                ],
            }

    transport = httpx.MockTransport(handler)
    reader = CvatReadClient(base_url="http://cvat.test", token="test", transport=transport)
    writer = CvatTrackClient(base_url="http://cvat.test", token="test", read_client=reader, transport=transport)
    worker = AutoSeedWorker(
        store=store,
        workspace=workspace,
        read_client=reader,
        track_client=writer,
        seeder=KaggleSam3Seeder(runner=FakeRunner(), workspace=workspace, code_file=tmp_path / "seed.ipynb"),
    )
    assert store.queue_video_auto_seed(JOB_ID).status is VideoJobStatus.SEEDING
    result = worker.process_next()
    assert result is not None and result.status is VideoJobStatus.SEED_READY
    assert result.seed_count == 1 and len(writes) == 1
    seed = json.loads(store.read_video_seed(JOB_ID))["seeds"][0]
    assert seed["source"] == "auto_seed" and seed["object_id"] == "shape:7"
    assert worker.process_next() is None


def test_auto_seed_failure_can_retry_or_fall_back_to_manual(tmp_path: Path):
    store, workspace, _manifest = _setup(tmp_path)
    store.queue_video_auto_seed(JOB_ID)
    claim = store.claim_next_video_auto_seed()
    assert claim is not None
    failed = store.fail_video_job(JOB_ID, code="auto_seed_failed", message="no boxes", claim_token=claim.claim_token)
    assert failed.status is VideoJobStatus.FAILED
    from src.models.video import BoundingBox2D, SeedRectangle, SeedSet

    manual = SeedSet(
        job_id=JOB_ID,
        cvat_job_id=22,
        annotation_revision=3,
        seeds=[
            SeedRectangle(
                seed_id="cvat:shape:9:frame:0",
                object_id="shape:9",
                label_id=1,
                label_name="car",
                shape_id=9,
                bbox=BoundingBox2D(xtl=10, ytl=20, xbr=50, ybr=60),
            )
        ],
    )
    encoded = workspace.encode_seed_set(manual)
    ready = store.attach_video_seed(
        JOB_ID,
        cvat_job_id=22,
        seed_revision=3,
        seed_count=1,
        seed_sha256=hashlib.sha256(encoded).hexdigest(),
        seed_payload=encoded.decode(),
        expected_updated_at=failed.updated_at,
    )
    assert ready.status is VideoJobStatus.SEED_READY


def test_sam3_seed_origin_is_preserved_after_video_propagation():
    from src.models.video import (
        BoundingBox2D,
        ComponentVersion,
        PropagationFrameInput,
        PropagationModelRequest,
        SeedRectangle,
        SeedSet,
        TrackObservation,
        TrackSet,
        VideoTrack,
    )
    from src.services.video.propagation import _validate_and_preserve_seeds

    box = BoundingBox2D(xtl=10, ytl=20, xbr=50, ybr=60)
    seed = SeedSet(
        job_id=JOB_ID,
        cvat_job_id=22,
        annotation_revision=2,
        seeds=[
            SeedRectangle(
                seed_id="cvat:shape:7:frame:0",
                object_id="shape:7",
                source="auto_seed",
                label_id=1,
                label_name="car",
                shape_id=7,
                bbox=box,
            )
        ],
    )
    request = PropagationModelRequest(
        job_id=JOB_ID,
        manifest_sha256="a" * 64,
        seed_sha256="b" * 64,
        end_frame_index=0,
        frames=[
            PropagationFrameInput(
                frame_index=0,
                relative_path="frames/frame_000000.jpg",
                sha256="c" * 64,
                width=100,
                height=80,
            )
        ],
        seed_set=seed,
    )
    result = TrackSet(
        job_id=JOB_ID,
        manifest_sha256="a" * 64,
        seed_sha256="b" * 64,
        component_versions=[ComponentVersion(component="sam3-video", version="facebook/sam3")],
        tracks=[
            VideoTrack(
                track_id="shape:7",
                label_id=1,
                label_name="car",
                source="human_seed",
                observations=[TrackObservation(frame_index=0, bbox=box)],
            )
        ],
    )
    validated = _validate_and_preserve_seeds(
        result,
        request,
        expected_model_id="facebook/sam3",
        expected_component="sam3-video",
    )
    assert validated.tracks[0].source == "auto"


def test_sam3_seeded_track_does_not_duplicate_frame_zero_shape():
    from src.models.video import BoundingBox2D, ComponentVersion, TrackObservation, TrackSet, VideoTrack

    manifest = FrameManifest(
        job_id=JOB_ID,
        source_name="clip.mp4",
        source_sha256="a" * 64,
        codec="h264",
        width=100,
        height=80,
        duration_s=1,
        average_fps=2,
        variable_frame_rate=False,
        ffmpeg_version="test",
        frames=[
            FrameRecord(
                frame_index=index,
                timestamp_s=index * 0.5,
                source_timestamp_s=index * 0.5,
                relative_path=f"frames/frame_{index:06d}.jpg",
                sha256=str(index + 1) * 64,
                width=100,
                height=80,
            )
            for index in range(2)
        ],
    )
    tracks = TrackSet(
        job_id=JOB_ID,
        manifest_sha256="b" * 64,
        seed_sha256="c" * 64,
        component_versions=[ComponentVersion(component="sam3-video", version="facebook/sam3")],
        tracks=[
            VideoTrack(
                track_id="shape:7",
                label_id=1,
                label_name="car",
                source="auto",
                observations=[
                    TrackObservation(
                        frame_index=index,
                        bbox=BoundingBox2D(xtl=10, ytl=20, xbr=50, ybr=60),
                    )
                    for index in range(2)
                ],
            )
        ],
    )
    payload = build_cvat_annotation_payload(
        existing={"version": 2, "tags": [], "shapes": [{"id": 7, "frame": 0, "source": "auto"}], "tracks": []},
        manifest=manifest,
        tracks=tracks,
    )
    assert payload["tracks"][0]["shapes"][0]["frame"] == 1
