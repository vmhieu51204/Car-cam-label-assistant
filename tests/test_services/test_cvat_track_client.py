from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from src.models.video import (
    BoundingBox2D,
    ComponentVersion,
    FrameManifest,
    FrameRecord,
    TrackObservation,
    TrackSet,
    VideoTrack,
)
from src.services.cvat import CvatReadClient, CvatTrackClient, CvatWriteError, build_cvat_annotation_payload

JOB_ID = "55555555-5555-4555-8555-555555555555"


@pytest.mark.asyncio
async def test_sparse_export_sidecars_are_request_local_and_source_is_unchanged():
    from src.services.video.keyframe_sparsify import SparseConfig

    manifest = _manifest()
    frames = [
        manifest.frames[0].model_copy(
            update={
                "frame_index": i,
                "timestamp_s": i * 0.5,
                "source_timestamp_s": i * 0.5,
                "relative_path": f"frames/frame_{i:06d}.jpg",
            }
        )
        for i in range(31)
    ]
    manifest = manifest.model_copy(update={"frames": frames})
    linear = VideoTrack(
        track_id="auto:linear",
        label_id=1,
        label_name="car",
        source="auto",
        observations=[
            TrackObservation(frame_index=i, bbox=BoundingBox2D(xtl=i, ytl=0, xbr=i + 20, ybr=20)) for i in range(31)
        ],
    )
    tracks = _tracks().model_copy(update={"tracks": [linear]})
    original = tracks.model_dump_json()

    def handler(request):
        if request.url.path == "/api/labels":
            return httpx.Response(200, json={"count": 1, "next": None, "results": [{"id": 1, "attributes": []}]})
        if request.method == "GET":
            return httpx.Response(200, json={"version": 4, "tags": [], "shapes": [], "tracks": []})
        return httpx.Response(200, json={"version": 5})

    transport = httpx.MockTransport(handler)
    reader = CvatReadClient(base_url="http://cvat.test", token="fake", transport=transport)
    client = CvatTrackClient(
        base_url="http://cvat.test",
        token="fake",
        read_client=reader,
        transport=transport,
        sparse_config=SparseConfig(enabled=True),
    )
    a, b = {}, {}
    await asyncio.gather(
        client.publish_tracks(
            job_id=JOB_ID,
            cvat_job_id=11,
            manifest=manifest,
            tracks=tracks,
            expected_revision=4,
            sparse_sidecar=a,
            mandatory_sparse_frames={15},
        ),
        client.publish_tracks(
            job_id=JOB_ID,
            cvat_job_id=22,
            manifest=manifest,
            tracks=tracks,
            expected_revision=4,
            sparse_sidecar=b,
            mandatory_sparse_frames={20},
        ),
    )
    assert a["auto:linear"]["kept"][15] == ["review_or_boundary"]
    assert b["auto:linear"]["kept"][20] == ["review_or_boundary"]
    assert tracks.model_dump_json() == original
    assert a is not b and a != b


def _manifest() -> FrameManifest:
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
                sha256=str(index + 1) * 64,
                width=100,
                height=80,
            )
            for index in range(3)
        ],
    )


def _tracks() -> TrackSet:
    return TrackSet(
        job_id=JOB_ID,
        manifest_sha256="b" * 64,
        seed_sha256="c" * 64,
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
                        score=0.9,
                    )
                    for index in range(3)
                ],
            ),
            VideoTrack(
                track_id="auto:missing:000001",
                label_id=1,
                label_name="car",
                source="auto",
                observations=[
                    TrackObservation(
                        frame_index=1,
                        bbox=BoundingBox2D(xtl=60, ytl=20, xbr=80, ybr=40),
                        score=0.88,
                    )
                ],
            ),
        ],
    )


def _existing(version: int = 4) -> dict[str, object]:
    return {
        "version": version,
        "tags": [{"id": 9, "frame": 0, "label_id": 2}],
        "shapes": [
            {
                "id": 5,
                "frame": 0,
                "label_id": 1,
                "type": "rectangle",
                "points": [10, 10, 30, 30],
                "source": "manual",
            }
        ],
        "tracks": [
            {"id": 7, "label_id": 1, "source": "manual", "shapes": []},
            {"id": 8, "label_id": 1, "source": "auto", "shapes": []},
        ],
    }


def test_payload_preserves_human_annotations_and_replaces_only_auto_tracks():
    names = ["confidence", "routing", "source", "review_status", "autolabel_track_id", "review_priority"]
    attribute_ids = {1: {name: index for index, name in enumerate(names, start=10)}}
    payload = build_cvat_annotation_payload(
        existing=_existing(),
        manifest=_manifest(),
        tracks=_tracks(),
        label_attributes=attribute_ids,
        mutable_attributes={1: {"review_status"}},
        job_id=JOB_ID,
        review_priority_by_track={"auto:missing:000001": "critical"},
    )

    assert payload["tags"] == _existing()["tags"]
    assert payload["shapes"] == _existing()["shapes"]
    assert payload["tracks"][0] == _existing()["tracks"][0]
    generated = payload["tracks"][1:]
    assert len(generated) == 2
    assert all(track["source"] == "auto" for track in generated)
    seeded_track = generated[0]
    assert [shape["frame"] for shape in seeded_track["shapes"]] == [1, 2]
    assert all(shape["keyframe"] is True for shape in seeded_track["shapes"])
    proposal_values = {item["spec_id"]: item["value"] for item in generated[1]["attributes"]}
    shape_values = {item["spec_id"]: item["value"] for item in generated[1]["shapes"][0]["attributes"]}
    assert shape_values[attribute_ids[1]["review_status"]] == "pending"
    assert attribute_ids[1]["review_status"] not in proposal_values
    assert proposal_values[attribute_ids[1]["review_priority"]] == "critical"
    assert proposal_values[attribute_ids[1]["source"]] == "auto"


def test_review_score_is_written_only_when_cvat_label_has_number_attribute():
    names = [
        "confidence",
        "routing",
        "source",
        "review_status",
        "autolabel_track_id",
        "review_priority",
        "review_score",
    ]
    attribute_ids = {1: {name: index for index, name in enumerate(names, start=10)}}
    payload = build_cvat_annotation_payload(
        existing=_existing(),
        manifest=_manifest(),
        tracks=_tracks(),
        label_attributes=attribute_ids,
        review_score_by_track={"auto:missing:000001": 0.8125},
    )
    generated = payload["tracks"][1:]
    values = {item["spec_id"]: item["value"] for item in generated[1]["attributes"]}
    assert values[attribute_ids[1]["review_score"]] == "0.812"


def test_sampled_tracks_publish_at_original_cvat_frame_indexes():
    base = _manifest()
    source_indexes = [0, 8, 19]
    manifest = base.model_copy(
        update={
            "source_frame_count": 20,
            "frames": [
                frame.model_copy(update={"source_frame_index": source_indexes[index]})
                for index, frame in enumerate(base.frames)
            ],
        }
    )
    payload = build_cvat_annotation_payload(existing=_existing(), manifest=manifest, tracks=_tracks())
    generated = payload["tracks"][1:]
    assert [shape["frame"] for shape in generated[0]["shapes"]] == [8, 19]
    assert generated[0]["frame"] == 8
    assert generated[1]["shapes"][0]["frame"] == 8


@pytest.mark.asyncio
async def test_track_client_patches_only_generated_tracks_and_preserves_manual_shapes():
    writes: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Token secret"
        assert request.headers["host"] == "localhost"
        assert request.headers["accept"] == "application/vnd.cvat+json"
        if request.method == "GET" and request.url.path == "/api/labels":
            return httpx.Response(200, json={"count": 1, "next": None, "results": [{"id": 1, "attributes": []}]})
        if request.method == "GET":
            existing = _existing()
            existing["tracks"] = existing["tracks"][:1]
            return httpx.Response(200, json=existing)
        if request.method == "PATCH":
            assert request.url.params["action"] == "create"
            writes.append(json.loads(await request.aread()))
            return httpx.Response(
                200, json={"version": 5, "tags": [], "shapes": _existing()["shapes"], "tracks": writes[-1]["tracks"]}
            )
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    transport = httpx.MockTransport(handler)
    reader = CvatReadClient(base_url="http://cvat.test", token="secret", host_header="localhost", transport=transport)
    client = CvatTrackClient(
        base_url="http://cvat.test",
        token="secret",
        read_client=reader,
        host_header="localhost",
        transport=transport,
    )

    result, encoded = await client.publish_tracks(
        job_id=JOB_ID,
        cvat_job_id=22,
        manifest=_manifest(),
        tracks=_tracks(),
        expected_revision=4,
    )

    assert result.revision_before == 4
    assert result.revision_after == 5
    assert result.published_track_count == 2
    assert json.loads(encoded) == writes[0]
    assert writes[0]["shapes"] == []
    assert writes[0]["tags"] == []
    assert len(writes[0]["tracks"]) == 2


@pytest.mark.asyncio
async def test_track_client_rejects_a_stale_revision_without_writing():
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        if request.url.path == "/api/labels":
            return httpx.Response(200, json={"count": 1, "next": None, "results": [{"id": 1, "attributes": []}]})
        return httpx.Response(200, json=_existing(version=5))

    transport = httpx.MockTransport(handler)
    reader = CvatReadClient(base_url="http://cvat.test", token="secret", transport=transport)
    client = CvatTrackClient(
        base_url="http://cvat.test",
        token="secret",
        read_client=reader,
        transport=transport,
    )

    with pytest.raises(CvatWriteError, match="stale"):
        await client.publish_tracks(
            job_id=JOB_ID,
            cvat_job_id=22,
            manifest=_manifest(),
            tracks=_tracks(),
            expected_revision=4,
        )

    assert methods == ["GET", "GET"]
