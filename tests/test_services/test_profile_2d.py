from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.services.evaluation.phase0 import collect_profile, runtime_summary
from src.services.video.profile import Profile2D, current_profile


def test_disabled_profile_is_noop_and_scopes_restore(tmp_path: Path) -> None:
    with Profile2D(tmp_path).activate(), current_profile().stage("import"):
        assert not current_profile().enabled
    assert list(tmp_path.iterdir()) == []
    assert not current_profile().enabled


def test_fake_clock_errors_and_secret_allowlist(tmp_path: Path) -> None:
    ticks = iter((10.0, 13.0))
    profile = Profile2D(tmp_path, enabled=True, clock=lambda: next(ticks))
    with pytest.raises(ValueError), profile.stage("fake", api_key="never-log-this", object_count=3):
        raise ValueError("do-not-log-error-message")
    event_path = next((tmp_path / "profile-2d/events").glob("*.json"))
    event = json.loads(event_path.read_text())
    assert event["seconds"] == 3
    assert event["status"] == "failed"
    assert event["object_count"] == 3
    assert "never-log" not in event_path.read_text()
    assert "do-not-log" not in event_path.read_text()
    assert event["peak_gpu_memory_bytes"] is None


def test_old_runtime_missing_memory_is_unavailable_and_not_decode(tmp_path: Path) -> None:
    report = collect_profile(tmp_path)
    assert report["decode_s"] is report["import_total_s"] is report["end_to_end_s"] is None
    assert runtime_summary({}, {})["peak_gpu_memory_bytes"] is None


def test_object_count_bins_and_resumed_windows_are_separate() -> None:
    predictions = [{"window_id": w, "track_id": str(i)} for w, n in enumerate((0, 3, 9, 12)) for i in range(n)]
    item = {
        "window_predictions": predictions,
        "window_checkpoints": [{"window_id": 0}],
        "resumed_windows": [3],
        "runtime": {"timing_s": {"windows": [{"window_id": i, "seconds": 10} for i in range(4)]}},
    }
    result = runtime_summary({}, item)
    assert set(result["by_object_count"]) == {"0", "1-5", "6-10"}


def test_failed_window_does_not_become_zero_object_measurement() -> None:
    item = {
        "window_predictions": [],
        "window_checkpoints": [{"window_id": 0}],
        "runtime": {"timing_s": {"windows": [{"window_id": 0, "seconds": 3}, {"window_id": 1, "seconds": 8}]}},
    }
    assert runtime_summary({}, item)["by_object_count"]["0"] == {"windows": 1, "mean_window_s": 3}


def test_clock_or_sink_failure_does_not_mask_pipeline_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_clock() -> float:
        raise RuntimeError("clock broken")

    profile = Profile2D(tmp_path, enabled=True, clock=broken_clock)
    with pytest.raises(ValueError, match="pipeline error"), profile.stage("worker"):
        raise ValueError("pipeline error")
    profile = Profile2D(tmp_path, enabled=True)
    monkeypatch.setattr(Path, "write_text", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")))
    with profile.stage("worker"):
        result = 42
    assert result == 42


def test_profile_on_off_produces_identical_tracks_and_queue(tmp_path: Path) -> None:
    from tests.test_services.test_sequence_kaggle_worker import _queued_worker

    artifacts = []
    for flag in (False, True):
        store, worker, runner, job_id = _queued_worker(tmp_path / str(flag))
        worker.profile_enabled = flag
        before = worker.workspace.read_manifest(job_id).model_dump()
        result = worker.process_next()
        assert result is not None and result.stage == "sequence_draft_ready"
        assert runner.calls == 1
        tracks = json.loads(store.read_video_track_set(job_id))
        queue = json.loads(store.read_video_review_queue(job_id))
        # UUID/checksum bindings legitimately differ between these independent jobs.
        artifacts.append(
            (tracks["tracks"], [(u["frame_start"], u["review_score"], u["priority"]) for u in queue["units"]])
        )
        assert worker.workspace.read_manifest(job_id).model_dump() == before
        paths = list((worker.workspace.job_dir(job_id) / "profile-2d/events").glob("*.json"))
        assert bool(paths) == flag
    assert artifacts[0] == artifacts[1]


def test_profile_import_keeps_mapping_and_saved_images(tmp_path: Path) -> None:
    import asyncio
    from datetime import UTC, datetime

    from src.models.video import CvatSamplingRequest, VideoJob, VideoJobStatus
    from src.services.video.cvat_import import import_cvat_frames
    from src.services.video.workspace import VideoWorkspace
    from tests.test_services.test_uniform_sequence_optimization import Frames

    now = datetime.now(UTC)
    job = VideoJob(
        job_id="00000000-0000-4000-8000-000000000001",
        status=VideoJobStatus.QUEUED,
        source_name="fake",
        source_sha256="0" * 64,
        source_size_bytes=1,
        created_at=now,
        updated_at=now,
        stage="queued",
        cvat_task_id=3,
        cvat_job_id=7,
    )
    outputs = []
    for flag in (False, True):
        workspace = VideoWorkspace(tmp_path / str(flag))
        workspace.write_sampling_request(
            job.job_id, CvatSamplingRequest(strategy="uniform", source_fps=30, target_fps=5, sequence_classes=["car"])
        )
        client = Frames(13, colors={12: "white"})
        manifest = asyncio.run(
            import_cvat_frames(job, client=client, workspace=workspace, progress=lambda *_: None, profile_enabled=flag)
        )
        outputs.append(
            (
                [x.model_dump() for x in manifest.frames],
                [p.read_bytes() for p in sorted(workspace.job_dir(job.job_id).glob("frames/*"))],
            )
        )
        assert len(outputs[-1][1]) == 3
        assert client.requested == [0, 6, 12]
        assert bool(list(workspace.job_dir(job.job_id).glob("profile-2d/events/*.json"))) == flag
    assert outputs[0] == outputs[1]


def test_fake_claude_profiles_429_backoff_and_cache_without_new_request(tmp_path: Path) -> None:
    import hashlib

    import httpx
    from PIL import Image

    from src.services.video.claude_box_rescue import ClaudeBoxRescueAuditor
    from tests.test_services.test_claude_box_rescue import answer

    image = tmp_path / "frame.png"
    Image.new("RGB", (100, 80)).save(image)
    calls = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "2"})
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": answer()}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    auditor = ClaudeBoxRescueAuditor(
        api_key="fake-key",
        cache_path=tmp_path / "cache.db",
        transport=httpx.MockTransport(respond),
        sleeper=lambda _: None,
    )
    args = dict(
        overlay_path=image,
        overlay_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
        classes=("car",),
        image_width=100,
        image_height=80,
        existing_boxes=[],
    )
    with Profile2D(tmp_path, enabled=True).activate():
        first = auditor.propose_missing_boxes(**args)
        cached = auditor.propose_missing_boxes(**args)
    assert first[0] == cached[0] and cached[2] is True and len(calls) == 2
    events = [json.loads(p.read_text()) for p in (tmp_path / "profile-2d/events").glob("*.json")]
    assert len([x for x in events if x["stage"] == "claude.http"]) == 2
    assert any(x["stage"] == "claude.retry_backoff" for x in events)
    assert any(x.get("http_status") == 429 for x in events)
    assert "fake-key" not in json.dumps(events)
