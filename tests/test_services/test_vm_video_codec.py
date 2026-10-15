"""Regression coverage for the system-FFmpeg privacy decoder."""

import json
import os
import shutil
import signal
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from src.services.vm.contracts import file_sha256
from src.services.vm.privacy import probe, redact, redact_video


def test_probe_uses_presentation_intervals_when_packet_durations_disagree(tmp_path, monkeypatch):
    from types import SimpleNamespace

    data = {
        "streams": [{"width": 64, "height": 64, "avg_frame_rate": "30000/1001"}],
        "frames": [
            {"best_effort_timestamp_time": timestamp, "pkt_duration_time": "0.033367"}
            for timestamp in ["2.231000", "2.264367", "2.464567"]
        ],
    }
    monkeypatch.setattr(
        "src.services.vm.privacy.subprocess.run", lambda *a, **kw: SimpleNamespace(stdout=json.dumps(data))
    )
    metadata = probe(tmp_path / "edited.mp4")
    assert metadata["timestamps"] == [2.231, 2.264367, 2.464567]
    assert metadata["durations"] == pytest.approx([0.033367, 0.2002, 0.033367])
    assert metadata["variable_frame_rate"]


def _source(path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("FFmpeg required for video codec integration")
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x64:rate=10",
            "-frames:v",
            "4",
            "-c:v",
            "libx264rgb",
            "-crf",
            "0",
            "-bf",
            "0",
            str(path),
        ],
        check=True,
    )
    return ffmpeg


def _ffmpeg_wrapper(tmp_path, real_ffmpeg, *, mode="normal"):
    """Make a controllable executable stand-in while still decoding real video."""
    marker = tmp_path / "ffmpeg-called.json"
    wrapper = tmp_path / "ffmpeg-wrapper"
    wrapper.write_text(
        textwrap.dedent(
            f"""\
            #!{sys.executable}
            import json, subprocess, sys
            args = sys.argv[1:]
            result = subprocess.run([{real_ffmpeg!r}, *args], capture_output=True)
            with open({str(marker)!r}, "w") as record:
                json.dump(args, record)
            data = result.stdout
            if {mode!r} == "truncate":
                data = data[:-1]
            elif {mode!r} == "extra":
                data += bytes(64 * 64 * 3)
            sys.stdout.buffer.write(data)
            sys.stderr.buffer.write(result.stderr)
            sys.exit(7 if {mode!r} == "nonzero" else result.returncode)
            """
        ),
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    return str(wrapper), marker


class _Detector:
    def __init__(self):
        self.frames = []

    def detect(self, image):
        self.frames.append(image.copy())
        return [[0, 0, 64, 64]]


@pytest.mark.parametrize("failure", ["nonzero", "truncate", "extra"])
def test_decoder_failures_never_finalize_a_receipt(tmp_path, failure):
    source = tmp_path / "source.mp4"
    real_ffmpeg = _source(source)
    source_hash = file_sha256(source)
    ffmpeg, marker = _ffmpeg_wrapper(tmp_path, real_ffmpeg, mode=failure)
    output = tmp_path / "redacted.mp4"

    with pytest.raises(ValueError, match="Privacy .*decoder"):
        redact_video(source, output, _Detector(), expected_sha256=source_hash, ffmpeg=ffmpeg)

    assert marker.exists()
    assert file_sha256(source) == source_hash
    assert not output.exists()
    assert not output.with_suffix(".privacy.json").exists()


def test_supplied_ffmpeg_decodes_every_frame_for_redaction(tmp_path):
    source, output = tmp_path / "source.mp4", tmp_path / "redacted.mp4"
    real_ffmpeg = _source(source)
    source_hash = file_sha256(source)
    ffmpeg, marker = _ffmpeg_wrapper(tmp_path, real_ffmpeg)
    detector = _Detector()

    receipt = redact_video(source, output, detector, expected_sha256=source_hash, ffmpeg=ffmpeg)

    assert len(detector.frames) == receipt["total_frames"] == 4
    assert all(frame.shape == (64, 64, 3) for frame in detector.frames)
    assert file_sha256(source) == source_hash
    args = json.loads(marker.read_text(encoding="utf-8"))
    assert "-vsync" in args and "0" in args
    assert "-noautorotate" in args

    import av

    with av.open(str(output)) as container:
        frames = list(container.decode(video=0))
    assert len(frames) == 4
    for frame, original in zip(frames, detector.frames, strict=True):
        assert np.array_equal(frame.to_ndarray(format="bgr24"), redact(original, [[0, 0, 64, 64]]))


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_cancelled_privacy_stops_decoder_process_group(tmp_path, monkeypatch):
    from src.config import Settings
    from src.services.vm.worker import VmWorker

    worker = VmWorker(Settings(_env_file=None, db_path=tmp_path / "db", vm_work_root=tmp_path / "jobs"))
    job = worker.store.create(1, {}, "fixture")
    job = worker.store.update(
        job["job_id"],
        status="cancelled",
        source_extension=".mp4",
        source_sha256="a" * 64,
        manual_regions=[],
        privacy_profile={
            "privacy_face_checkpoint": "face",
            "privacy_plate_checkpoint": "plate",
            "privacy_face_sha256": "b" * 64,
            "privacy_plate_sha256": "c" * 64,
            "privacy_source_commit": "d" * 40,
        },
    )
    calls = []

    class Process:
        pid = 12345

        def poll(self):
            return None

        def wait(self):
            calls.append("wait")

    def spawn(*args, **kwargs):
        assert kwargs["start_new_session"] is True
        return Process()

    monkeypatch.setattr("src.services.vm.worker.subprocess.Popen", spawn)
    monkeypatch.setattr("src.services.vm.worker.os.killpg", lambda pid, sig: calls.append((pid, sig)))
    with pytest.raises(ValueError, match="cancelled or timed out"):
        worker.privacy(job)
    assert calls == [(Process.pid, signal.SIGKILL), "wait"]
