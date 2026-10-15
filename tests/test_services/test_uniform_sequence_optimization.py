from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from io import BytesIO
from uuid import uuid4

import pytest
from PIL import Image

from src.models.video import CvatSamplingRequest, VideoJob, VideoJobStatus
from src.services.video.chunks import plan_chunks
from src.services.video.cvat_import import import_cvat_frames
from src.services.video.sampling import choose_keyframes, uniform_indexes
from src.services.video.sequence_kaggle import make_track_set, parse_sequence_result, sequence_payload
from src.services.video.sequence_review import build_sequence_review_queue
from src.services.video.workspace import VideoWorkspace


class Frames:
    def __init__(self, count, colors=None, short=False):
        self.count, self.colors, self.short = count, colors or {}, short
        self.requested = []

    async def get_job(self, job_id):
        return {"task_id": 3, "start_frame": 0, "stop_frame": self.count - 1}

    async def get_job_data_meta(self, job_id):
        return {"size": self.count, "frames": [{"width": 32, "height": 24}]}

    async def iter_job_frames(self, *args, **kwargs):
        raise AssertionError("uniform must never download the full timeline")
        yield

    async def iter_selected_job_frames(self, job_id, indexes):
        self.requested.extend(indexes)
        for index in indexes[:-1] if self.short else indexes:
            out = BytesIO()
            Image.new("RGB", (32, 24), self.colors.get(index, "black")).save(out, format="JPEG")
            yield out.getvalue()


def imported(tmp_path, count=13, colors=None, short=False):
    identity = str(uuid4())
    workspace = VideoWorkspace(tmp_path)
    workspace.write_sampling_request(
        identity,
        CvatSamplingRequest(
            strategy="uniform", source_fps=30, target_fps=5, auto_start_sequence=True, sequence_classes=["car"]
        ),
    )
    now = datetime.now(UTC)
    job = VideoJob(
        job_id=identity,
        status=VideoJobStatus.QUEUED,
        source_name="test",
        source_sha256="0" * 64,
        source_size_bytes=1,
        created_at=now,
        updated_at=now,
        stage="queued",
        cvat_task_id=3,
        cvat_job_id=7,
    )
    client = Frames(count, colors, short)
    progress = []
    manifest = asyncio.run(
        import_cvat_frames(
            job, client=client, workspace=workspace, progress=lambda current, total: progress.append((current, total))
        )
    )
    return manifest, workspace, client, progress


def test_uniform_downloads_only_selected_once_and_preserves_mapping(tmp_path):
    manifest, _, client, progress = imported(tmp_path)
    assert client.requested == [0, 6, 12]
    assert [r.source_frame_index for r in manifest.frames] == [0, 6, 12]
    assert [r.timestamp_s for r in manifest.frames] == [0, 0.2, 0.4]
    assert progress[-1] == (3, 3)
    assert all(r.optical_flow_score == 0 for r in manifest.frames)


@pytest.mark.parametrize("count,fps,target", [(1, 30, 5), (2, 30, 5), (3631, 30.000300003, 5), (101, 29.97, 3)])
def test_uniform_cadence_matches_old_formula(count, fps, target):
    import math

    step = fps / target
    expected = sorted(
        {min(count - 1, round(i * step)) for i in range(math.ceil((count - 1) / step) + 1)} | {0, count - 1}
    )
    assert uniform_indexes(count, fps, target) == expected
    request = CvatSamplingRequest(strategy="uniform", source_fps=fps, target_fps=target)
    assert choose_keyframes([0.0] * count, request)[0] == expected
    if count == 3631:
        assert len(expected) == 606


def test_missing_selected_frame_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="all selected frames"):
        imported(tmp_path, short=True)


def test_scene_cut_splits_windows_resets_id_and_emits_outside(tmp_path):
    manifest, workspace, _, _ = imported(tmp_path, colors={12: "white"})
    assert [r.scene_cut_before for r in manifest.frames] == [False, False, True]
    plans = plan_chunks(3, window_size=50, overlap=5, scene_cut_frames=(2,))
    assert [(p.start, p.stop) for p in plans] == [(0, 2), (2, 3)]
    _, config = sequence_payload(manifest, workspace, ("car",))
    rows = [
        dict(
            window_id=0 if index < 2 else 1,
            frame_index=index,
            track_id="same",
            label="car",
            bbox=[2, 3, 22, 23],
            score=0.9,
            mask_rle=None,
        )
        for index in range(3)
    ]
    result = {
        "schema": "autolabel-sam3-benchmark/v1",
        "configurations": [
            dict(
                id="sequence-text",
                config=config,
                status="completed",
                selected_source_frames=[0, 1, 2],
                evaluated_source_frames=[0, 1, 2],
                window_predictions=rows,
            )
        ],
    }
    stitched, component = parse_sequence_result(result, expected_config=config, frame_count=3)
    assert len(stitched.tracks) == 2
    tracks = make_track_set(
        manifest=manifest,
        manifest_sha256="a" * 64,
        stitched=stitched,
        labels=[{"id": 11, "name": "car"}],
        classes=("car",),
        component=component,
    )
    assert tracks.tracks[0].observations[-1].outside
    assert tracks.tracks[0].observations[-1].frame_index == 2
    assert tracks.tracks[1].review_reasons == ["scene_cut"]
    queue = build_sequence_review_queue(
        tracks=tracks, manifest=manifest, track_set_sha256="a" * 64, discovery_sha256="b" * 64
    )
    assert all(u.review_score and "short_track" in u.reason_codes for u in queue.units)
    assert any("scene_cut" in u.reason_codes for u in queue.units)
    assert all(t.review_status == "pending" for t in tracks.tracks)
    old = tracks.model_dump(by_alias=True)
    for t in old["tracks"]:
        t.pop("needs_review")
        t.pop("review_reasons")
    assert type(tracks).model_validate(old).tracks[0].review_reasons == []


def test_zero_score_frames_stay_low_and_track_flags_reach_queue(tmp_path):
    from src.models.video import BoundingBox2D, ComponentVersion, TrackObservation, TrackSet, VideoTrack

    manifest, _, _, _ = imported(tmp_path, count=25)
    stable = VideoTrack(
        track_id="stable",
        label_id=11,
        label_name="car",
        source="auto",
        observations=[
            TrackObservation(frame_index=i, bbox=BoundingBox2D(xtl=2, ytl=2, xbr=25, ybr=23)) for i in range(3)
        ],
    )
    short = VideoTrack(
        track_id="short",
        label_id=11,
        label_name="car",
        source="auto",
        needs_review=True,
        review_reasons=["ambiguous_overlap"],
        observations=[TrackObservation(frame_index=3, bbox=BoundingBox2D(xtl=2, ytl=2, xbr=25, ybr=23))],
    )
    tracks = TrackSet(
        job_id=manifest.job_id,
        manifest_sha256="a" * 64,
        prompt_sha256="b" * 64,
        component_versions=[ComponentVersion(component="sam3", version="test")],
        tracks=[stable, short],
    )
    queue = build_sequence_review_queue(
        tracks=tracks, manifest=manifest, track_set_sha256="a" * 64, discovery_sha256="b" * 64
    )
    by_frame = {unit.frame_start: unit for unit in queue.units}
    assert all(by_frame[i].review_score == 0 and by_frame[i].priority == "low" for i in (0, 1, 2, 4))
    assert {"short_track", "ambiguous_overlap", "stitch_ambiguity"} <= set(by_frame[3].reason_codes)
    assert by_frame[3].track_ids == ["short"]
