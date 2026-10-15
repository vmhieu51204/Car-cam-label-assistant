import json
import shutil
import subprocess

import numpy as np
import pytest

from src.services.vm.contracts import file_sha256
from src.services.vm.privacy import redact, redact_video


def test_expanded_blur_and_tiny_opaque_keep_geometry():
    image = np.random.default_rng(42).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    output = redact(image, [[16, 16, 48, 48], [1, 1, 4, 4]])
    assert output.shape == image.shape
    assert np.array_equal(output[60:, 60:], image[60:, 60:])
    assert not np.array_equal(output[16:48, 16:48], image[16:48, 16:48])
    assert not output[1:4, 1:4].any()
    with pytest.raises(ValueError):
        redact(image, [[0, 0, float("nan"), 4]])


def test_retry_recovers_incomplete_old_receipt_without_exposing_output(tmp_path, monkeypatch):
    from src.config import Settings
    from src.services.vm.worker import VmWorker

    worker = VmWorker(Settings(_env_file=None, db_path=tmp_path / "db", vm_work_root=tmp_path / "jobs"))
    job = worker.store.create(1, {"profile": {"variant": "sam3.1"}}, "fixture")
    job = worker.store.update(
        job["job_id"],
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
    directory = worker.store.directory(job["job_id"])
    output = directory / "public/redacted.mp4"
    output.write_bytes(b"unverified")
    output.with_suffix(".privacy.json").write_text('{"status":"complete"}')

    def interrupt(*args, **kwargs):
        assert not output.exists() and not output.with_suffix(".privacy.json").exists()
        assert next((directory / "private").glob("incomplete-redacted-*.mp4")).read_bytes() == b"unverified"
        assert next((directory / "private").glob("incomplete-receipt-*.json")).is_file()
        raise RuntimeError("fixture stops before GPU")

    monkeypatch.setattr("src.services.vm.worker.subprocess.Popen", interrupt)
    with pytest.raises(RuntimeError, match="before GPU"):
        worker.privacy(job)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg required for bounded video integration")
def test_sequential_video_has_all_frames_and_no_source_fallback(tmp_path):
    source, output = tmp_path / "source.mp4", tmp_path / "clean.mp4"
    subprocess.run(
        [
            "ffmpeg",
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
            str(source),
        ],
        check=True,
    )
    source_hash = file_sha256(source)

    class Detector:
        def detect(self, image):
            return [[20, 20, 40, 40]]

    progress = []
    receipt = redact_video(
        source,
        output,
        Detector(),
        expected_sha256=source_hash,
        manual_regions=[{"start": 1, "end": 2, "box": [0, 0, 10, 10]}],
        progress=lambda a, b: progress.append((a, b)),
        model_metadata={"face_sha256": "a" * 64, "plate_sha256": "b" * 64, "source_commit": "c" * 40},
    )
    assert receipt["total_frames"] == 4 and progress[-1] == (4, 4)
    assert file_sha256(source) == source_hash
    assert receipt["output_sha256"] == file_sha256(output)
    assert len(output.with_suffix(".regions.jsonl").read_text().splitlines()) == 4
    assert json.loads(output.with_suffix(".privacy.json").read_text())["status"] == "complete"
    assert json.loads(output.with_suffix(".privacy.json").read_text())["model"]["face_sha256"] == "a" * 64

    class Broken:
        def detect(self, image):
            raise RuntimeError("fixture detector failed")

    failed = tmp_path / "failed.mp4"
    with pytest.raises(RuntimeError):
        redact_video(source, failed, Broken(), expected_sha256=source_hash)
    assert not failed.exists() and not failed.with_suffix(".privacy.json").exists()


def test_vfr_timestamps_durations_and_every_frame_preserved(tmp_path):
    from src.services.vm.privacy import probe

    source, output = tmp_path / "vfr.mp4", tmp_path / "clean.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x64:rate=10",
            "-vf",
            "select='not(eq(n,2))'",
            "-fps_mode",
            "vfr",
            "-frames:v",
            "4",
            "-c:v",
            "libx264rgb",
            "-crf",
            "0",
            "-bf",
            "0",
            str(source),
        ],
        check=True,
    )

    class Detector:
        count = 0

        def detect(self, image):
            self.count += 1
            return []

    detector = Detector()
    original = probe(source)
    assert original["variable_frame_rate"]
    receipt = redact_video(source, output, detector, expected_sha256=file_sha256(source))
    actual = probe(output)
    assert detector.count == 4 and receipt["total_frames"] == 4
    assert original["timestamps"] == actual["timestamps"] and original["durations"] == actual["durations"]
