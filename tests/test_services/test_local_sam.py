import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from kaggle.sam3_benchmark.main import Sam3Backend
from scripts.run_local_sam import MultiplexBridge
from src.config import Settings
from src.models.video import CvatSamplingRequest
from src.services.video.local_sam import LocalSamRunner, local_ready


def configured(tmp_path):
    python = tmp_path / "python"
    python.touch()
    checkpoint = tmp_path / "sam3.pt"
    checkpoint.write_bytes(b"fixture")
    return Settings(
        _env_file=None,
        local_sam_enabled=True,
        local_sam_python=python,
        local_sam_commit="b" * 40,
        local_sam3_checkpoint=checkpoint,
        local_sam3_sha256="a" * 64,
    )


def test_variants_are_separate_and_request_persists(tmp_path):
    settings = configured(tmp_path)
    assert local_ready(settings, "sam3")
    assert not local_ready(settings, "sam3.1")
    original = CvatSamplingRequest(
        source_fps=30, auto_start_sequence=True, sam_execution_provider="local", sam3_model_variant="sam3.1"
    )
    assert CvatSamplingRequest.model_validate_json(original.model_dump_json()) == original
    assert CvatSamplingRequest(source_fps=30).sam_execution_provider == "kaggle"


def test_checkpoint_and_source_changes_invalidate_resume_identity(tmp_path):
    settings = configured(tmp_path)
    runner = LocalSamRunner(settings, tmp_path)
    before = runner.fingerprint()
    settings.local_sam3_sha256 = "c" * 64
    assert runner.fingerprint() != before
    before = runner.fingerprint()
    settings.local_sam_commit = "d" * 40
    assert runner.fingerprint() != before
    assert runner.fingerprint("sam3.1") != runner.fingerprint("sam3")


def test_kaggle_cannot_label_original_checkpoint_as_sam31(tmp_path, monkeypatch):
    (tmp_path / "benchmark-config.json").write_text(json.dumps({"configurations": [{"model_variant": "sam3.1"}]}))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    with pytest.raises(NotImplementedError, match="SAM3 only"):
        Sam3Backend(tmp_path).load(tmp_path)


def test_local_runner_calls_only_local_process(tmp_path, monkeypatch):
    runner = LocalSamRunner(configured(tmp_path), tmp_path / "jobs")
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        output = tmp_path / "jobs/job/local-sam/output/result.json"
        output.write_text(json.dumps({"configurations": [{"status": "completed"}]}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", execute)
    result = runner.run(stage="sam3-sequence", job_id="job", files={"frames.json": b"{}"}, code_file=tmp_path)
    assert result["execution_provider"] == "local"
    assert len(calls) == 1 and "scripts.run_local_sam" in calls[0]
    assert "--sha256" in calls[0]
    with pytest.raises(ValueError, match="not configured"):
        runner.run(stage="sam3-sequence", job_id="job", files={}, code_file=tmp_path, model_variant="sam3.1")


def test_timeout_keeps_completed_window_checkpoint(tmp_path, monkeypatch):
    runner = LocalSamRunner(configured(tmp_path), tmp_path / "jobs")

    def execute(command, **kwargs):
        (tmp_path / "jobs/job/local-sam/output/result.json").write_text(
            json.dumps({"configurations": [{"status": "pending", "window_checkpoints": [{"window_id": 0}]}]})
        )
        raise subprocess.TimeoutExpired(command, 1)

    monkeypatch.setattr(subprocess, "run", execute)
    result = runner.run(stage="sam3-sequence", job_id="job", files={}, code_file=tmp_path)
    assert result["configurations"][0]["status"] == "failed"
    assert result["configurations"][0]["window_checkpoints"] == [{"window_id": 0}]


def test_multiplex_sessions_close_and_map_output():
    calls = []

    class Predictor:
        def handle_request(self, request):
            calls.append(request)
            return {"session_id": "session"}

        def handle_stream_request(self, request):
            yield {"frame_index": 2, "outputs": {"out_obj_ids": [12]}}

    bridge = MultiplexBridge(Predictor())
    state = bridge.init_state(resource_path="frames")
    bridge.add_prompt(inference_state=state, frame_idx=0, text_str="car")
    assert list(bridge.propagate_in_video(state)) == [(2, {"out_obj_ids": [12]})]
    assert [r["type"] for r in calls] == ["start_session", "add_prompt", "close_session"]
