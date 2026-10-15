from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path

import pytest

from scripts.build_kaggle_notebooks import MAIN_GUARD, ROOT, build_notebook
from src.models.video import BoundingBox2D, FrameManifest, FrameRecord, SeedRectangle, SeedSet
from src.services.video.kaggle_batch import KaggleBatchError, KaggleBatchRunner, _kernel_status
from src.services.video.propagation import KaggleSam3Propagator
from src.services.video.workspace import VideoWorkspace

JOB_ID = "11111111-1111-4111-8111-111111111111"


def test_upload_input_flattens_payload_archive_without_changing_resume_identity(tmp_path: Path) -> None:
    from src.services.video.kaggle_batch import _prepare_upload

    inner = tmp_path / "inner.zip"
    with zipfile.ZipFile(inner, "w") as archive:
        archive.writestr("frames.json", "{}")
        archive.writestr("frames/frame_000000.jpg", b"image")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    with zipfile.ZipFile(dataset / "payload.zip", "w") as archive:
        archive.write(inner, "payload.zip")
    (dataset / "dataset-metadata.json").write_text("{}")
    original = (dataset / "payload.zip").read_bytes()
    upload = _prepare_upload(dataset)
    assert (upload / "payload/frames.json").read_text() == "{}"
    assert (upload / "payload/frames/frame_000000.jpg").read_bytes() == b"image"
    assert not list(upload.rglob("*.zip"))
    assert (dataset / "payload.zip").read_bytes() == original


def test_upload_input_rejects_archive_path_escape(tmp_path: Path) -> None:
    from src.services.video.kaggle_batch import _prepare_upload

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    with zipfile.ZipFile(dataset / "payload.zip", "w") as archive:
        archive.writestr("../escape", b"bad")
    (dataset / "dataset-metadata.json").write_text("{}")
    with pytest.raises(ValueError, match="unsafe"):
        _prepare_upload(dataset)
    assert not (tmp_path / "escape").exists()


def test_kaggle_runner_passes_private_token_to_cli_without_logging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured.update(kwargs)
        return subprocess.CompletedProcess(command, 0, stdout="ready")

    monkeypatch.setattr(subprocess, "run", fake_run)
    runner = KaggleBatchRunner(username="tester", work_root=tmp_path, api_token="private-test-token")
    runner._command("datasets", "status", "tester/private", timeout=1)
    assert captured["env"]["KAGGLE_API_TOKEN"] == "private-test-token"
    assert captured["env"]["KAGGLE_USERNAME"] == "tester"


def test_kaggle_runner_classifies_transient_network_failure_without_echoing_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command, 1, stderr="ConnectTimeoutError: Connection to api.kaggle.com timed out. secret"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    runner = KaggleBatchRunner(username="tester", work_root=tmp_path)
    with pytest.raises(KaggleBatchError, match="network unavailable") as error:
        runner._command("kernels", "status", "tester/kernel", timeout=1)
    assert "secret" not in str(error.value)


def test_kaggle_runner_accepts_valid_result_if_cli_fails_after_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = KaggleBatchRunner(username="tester", work_root=tmp_path)

    def fake_command(*args: str, timeout: float) -> subprocess.CompletedProcess[str]:
        if args[:2] == ("kernels", "status"):
            return subprocess.CompletedProcess(args, 0, stdout="KernelWorkerStatus.COMPLETE")
        output = Path(args[args.index("-p") + 1])
        (output / "result.json").write_text(
            json.dumps(
                {
                    "schema": "autolabel-sam3-benchmark/v1",
                    "configurations": [{"id": "sequence-text"}],
                }
            )
        )
        raise KaggleBatchError("Kaggle CLI log encoding failed")

    monkeypatch.setattr(runner, "_command", fake_command)
    result = runner._collect_result("sam3-sequence", JOB_ID, "tester/kernel", tmp_path / "output")
    assert result["schema"] == "autolabel-sam3-benchmark/v1"


def test_kaggle_runner_uses_private_dataset_and_validates_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code = tmp_path / "kernel.py"
    code.write_text("print('run')", encoding="utf-8")
    commands: list[tuple[str, ...]] = []

    def fake_command(self: KaggleBatchRunner, *args: str, timeout: float) -> subprocess.CompletedProcess[str]:
        commands.append(args)
        if args[:2] == ("datasets", "status"):
            return subprocess.CompletedProcess(args, 0, stdout="ready")
        if args[:2] == ("kernels", "output"):
            output_dir = Path(args[args.index("-p") + 1])
            (output_dir / "result.json").write_text(json.dumps({"job_id": JOB_ID, "stage": "sam3"}))
        return subprocess.CompletedProcess(args, 0, stdout="KernelWorkerStatus.COMPLETE")

    monkeypatch.setattr(KaggleBatchRunner, "_command", fake_command)
    runner = KaggleBatchRunner(
        username="tester",
        work_root=tmp_path,
        sam3_checkpoint_dataset="tester/autolabel-sam3-checkpoint",
    )
    result = runner.run(stage="sam3", job_id=JOB_ID, files={"request.json": b"{}"}, code_file=code)
    assert result["job_id"] == JOB_ID
    run_dir = next((tmp_path / JOB_ID / "kaggle").iterdir())
    metadata = json.loads((run_dir / "kernel/kernel-metadata.json").read_text())
    dataset_metadata = json.loads((run_dir / "dataset/dataset-metadata.json").read_text())
    assert metadata["is_private"] is True
    assert metadata["id"] != dataset_metadata["id"]
    assert metadata["dataset_sources"] == [dataset_metadata["id"], "tester/autolabel-sam3-checkpoint"]
    with zipfile.ZipFile(run_dir / "dataset/payload.zip") as archive:
        assert archive.read("request.json") == b"{}"
    assert [command[:2] for command in commands] == [
        ("datasets", "create"),
        ("datasets", "status"),
        ("kernels", "push"),
        ("kernels", "status"),
        ("kernels", "output"),
    ]
    commands.clear()
    runner.run(stage="sam3", job_id=JOB_ID, files={"request.json": b"{}"}, code_file=code)
    assert [command[:2] for command in commands] == [("kernels", "status"), ("kernels", "output")]
    commands.clear()
    runner.run(stage="sam3", job_id=JOB_ID, files={"request.json": b'{"changed":true}'}, code_file=code)
    assert ("datasets", "create") in [command[:2] for command in commands]


@pytest.mark.parametrize("stage", ["sam3_seed", "sam3", "sam3_benchmark"])
def test_uploadable_notebook_matches_backend_model_script(stage: str) -> None:
    path = ROOT / "kaggle/notebooks" / f"autolabel_{stage}.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook == build_notebook(stage)
    assert notebook["nbformat"] == 4
    assert notebook["cells"][0]["cell_type"] == "markdown"
    script = (ROOT / "kaggle" / stage / "main.py").read_text(encoding="utf-8")
    assert "".join(notebook["cells"][1]["source"]) == script[: -len(MAIN_GUARD)]
    for cell in notebook["cells"][1:]:
        compile("".join(cell["source"]), str(path), "exec")


def test_kaggle_runner_submits_notebook_as_private_kernel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    notebook = tmp_path / "upload.ipynb"
    notebook.write_text(json.dumps(build_notebook("sam3")), encoding="utf-8")

    def fake_command(self: KaggleBatchRunner, *args: str, timeout: float) -> subprocess.CompletedProcess[str]:
        if args[:2] == ("datasets", "status"):
            return subprocess.CompletedProcess(args, 0, stdout="ready")
        if args[:2] == ("kernels", "output"):
            output_dir = Path(args[args.index("-p") + 1])
            (output_dir / "result.json").write_text(json.dumps({"job_id": JOB_ID, "stage": "sam3"}))
        return subprocess.CompletedProcess(args, 0, stdout="KernelWorkerStatus.COMPLETE")

    monkeypatch.setattr(KaggleBatchRunner, "_command", fake_command)
    KaggleBatchRunner(username="tester", work_root=tmp_path, sam3_checkpoint_dataset="tester/checkpoint").run(
        stage="sam3",
        job_id=JOB_ID,
        files={"request.json": b"{}"},
        code_file=notebook,
    )
    run_dir = next((tmp_path / JOB_ID / "kaggle").iterdir())
    metadata = json.loads((run_dir / "kernel/kernel-metadata.json").read_text())
    assert metadata["kernel_type"] == "notebook"
    assert metadata["code_file"] == "main.ipynb"
    assert metadata["is_private"] is True
    assert (run_dir / "kernel/main.ipynb").read_bytes() == notebook.read_bytes()


def test_kaggle_sequence_uses_private_checkpoint_and_accepts_benchmark_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notebook = ROOT / "kaggle/notebooks/autolabel_sam3_benchmark.ipynb"

    def fake_command(self: KaggleBatchRunner, *args: str, timeout: float) -> subprocess.CompletedProcess[str]:
        if args[:2] == ("datasets", "status"):
            return subprocess.CompletedProcess(args, 0, stdout="ready")
        if args[:2] == ("kernels", "output"):
            output_dir = Path(args[args.index("-p") + 1])
            (output_dir / "result.json").write_text(
                json.dumps(
                    {
                        "schema": "autolabel-sam3-benchmark/v1",
                        "configurations": [{"id": "sequence-text", "status": "completed"}],
                    }
                )
            )
        return subprocess.CompletedProcess(args, 0, stdout="KernelWorkerStatus.COMPLETE")

    monkeypatch.setattr(KaggleBatchRunner, "_command", fake_command)
    runner = KaggleBatchRunner(username="tester", work_root=tmp_path, sam3_checkpoint_dataset="tester/sam3-private")
    result = runner.run(stage="sam3-sequence", job_id=JOB_ID, files={"frames.json": b"{}"}, code_file=notebook)
    assert result["schema"] == "autolabel-sam3-benchmark/v1"
    run_dir = next((tmp_path / JOB_ID / "kaggle").iterdir())
    metadata = json.loads((run_dir / "kernel/kernel-metadata.json").read_text())
    assert metadata["dataset_sources"][1] == "tester/sam3-private"
    assert metadata["is_private"] is True


def test_kaggle_runner_accepts_multi_configuration_sam3_benchmark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code = tmp_path / "main.ipynb"
    code.write_text("{}", encoding="utf-8")
    config = {"configurations": [{"id": "baseline"}, {"id": "fast"}]}

    def fake_command(self: KaggleBatchRunner, *args: str, timeout: float) -> subprocess.CompletedProcess[str]:
        if args[:2] == ("datasets", "status"):
            return subprocess.CompletedProcess(args, 0, stdout="ready")
        if args[:2] == ("kernels", "output"):
            output = Path(args[args.index("-p") + 1])
            (output / "result.json").write_text(
                json.dumps(
                    {
                        "schema": "autolabel-sam3-benchmark/v1",
                        "completed": True,
                        "configurations": [
                            {"id": "baseline", "status": "completed"},
                            {"id": "fast", "status": "timed_out"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(args, 0, stdout="KernelWorkerStatus.COMPLETE")

    monkeypatch.setattr(KaggleBatchRunner, "_command", fake_command)
    runner = KaggleBatchRunner(username="tester", work_root=tmp_path, sam3_checkpoint_dataset="tester/sam3-private")
    result = runner.run(
        stage="sam3-benchmark",
        job_id=JOB_ID,
        files={"benchmark-config.json": json.dumps(config).encode()},
        code_file=code,
    )
    assert [row["id"] for row in result["configurations"]] == ["baseline", "fast"]
    metadata = json.loads(next((tmp_path / JOB_ID / "kaggle").glob("*/kernel/kernel-metadata.json")).read_text())
    assert metadata["dataset_sources"][1] == "tester/sam3-private"


def test_sam3_benchmark_retry_never_submits_second_kernel_when_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code = tmp_path / "main.ipynb"
    code.write_text("{}", encoding="utf-8")
    pushes = 0

    def fake_command(self: KaggleBatchRunner, *args: str, timeout: float) -> subprocess.CompletedProcess[str]:
        nonlocal pushes
        if args[:2] == ("datasets", "status"):
            return subprocess.CompletedProcess(args, 0, stdout="ready")
        if args[:2] == ("kernels", "push"):
            pushes += 1
        if args[:2] == ("kernels", "status"):
            return subprocess.CompletedProcess(args, 0, stdout="KernelWorkerStatus.ERROR")
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(KaggleBatchRunner, "_command", fake_command)
    runner = KaggleBatchRunner(username="tester", work_root=tmp_path, sam3_checkpoint_dataset="tester/sam3-private")
    kwargs = {
        "stage": "sam3-benchmark",
        "job_id": JOB_ID,
        "files": {"benchmark-config.json": b"{}"},
        "code_file": code,
        "allow_retry_on_terminal_failure": False,
    }
    with pytest.raises(KaggleBatchError, match="run error"):
        runner.run(**kwargs)
    with pytest.raises(KaggleBatchError, match="run error"):
        runner.run(**kwargs)
    assert pushes == 1


def test_failed_sam3_benchmark_returns_saved_window_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code = tmp_path / "main.ipynb"
    code.write_text("{}", encoding="utf-8")
    pushes = 0

    def fake_command(self: KaggleBatchRunner, *args: str, timeout: float) -> subprocess.CompletedProcess[str]:
        nonlocal pushes
        if args[:2] == ("datasets", "status"):
            return subprocess.CompletedProcess(args, 0, stdout="ready")
        if args[:2] == ("kernels", "push"):
            pushes += 1
        if args[:2] == ("kernels", "status"):
            return subprocess.CompletedProcess(args, 0, stdout="KernelWorkerStatus.ERROR")
        if args[:2] == ("kernels", "output"):
            output = Path(args[args.index("-p") + 1])
            (output / "result.json").write_text(
                json.dumps(
                    {
                        "schema": "autolabel-sam3-benchmark/v1",
                        "completed": False,
                        "configurations": [
                            {"id": "baseline", "status": "completed", "window_checkpoints": [{"window_id": 0}]},
                            {"id": "fast", "status": "pending", "window_checkpoints": []},
                        ],
                    }
                ),
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(KaggleBatchRunner, "_command", fake_command)
    runner = KaggleBatchRunner(username="tester", work_root=tmp_path, sam3_checkpoint_dataset="tester/sam3-private")
    result = runner.run(
        stage="sam3-benchmark",
        job_id=JOB_ID,
        files={"benchmark-config.json": b"{}"},
        code_file=code,
        allow_retry_on_terminal_failure=False,
    )
    assert result["terminal_kernel_status"] == "error"
    assert len(result["configurations"][0]["window_checkpoints"]) == 1
    assert pushes == 1


def test_failed_sam3_seed_kernel_gets_new_private_run_on_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    code = tmp_path / "seed.py"
    code.write_text("print('synthetic')", encoding="utf-8")
    pushes = 0

    def fake_command(self: KaggleBatchRunner, *args: str, timeout: float) -> subprocess.CompletedProcess[str]:
        nonlocal pushes
        if args[:2] == ("datasets", "status"):
            return subprocess.CompletedProcess(args, 0, stdout="ready")
        if args[:2] == ("kernels", "push"):
            pushes += 1
        if args[:2] == ("kernels", "status"):
            return subprocess.CompletedProcess(
                args,
                0,
                stdout="KernelWorkerStatus.ERROR" if pushes == 1 else "KernelWorkerStatus.COMPLETE",
            )
        if args[:2] == ("kernels", "output"):
            output_dir = Path(args[args.index("-p") + 1])
            (output_dir / "result.json").write_text(json.dumps({"job_id": JOB_ID, "stage": "sam3-seed"}))
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(KaggleBatchRunner, "_command", fake_command)
    runner = KaggleBatchRunner(
        username="tester",
        work_root=tmp_path,
        sam3_checkpoint_dataset="tester/sam3-private",
    )
    with pytest.raises(KaggleBatchError, match="run error"):
        runner.run(stage="sam3-seed", job_id=JOB_ID, files={"request.json": b"{}"}, code_file=code)
    assert (
        runner.run(stage="sam3-seed", job_id=JOB_ID, files={"request.json": b"{}"}, code_file=code)["job_id"] == JOB_ID
    )
    assert pushes == 2


def test_kaggle_runner_waits_for_private_dataset_indexing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0

    def fake_command(self: KaggleBatchRunner, *args: str, timeout: float) -> subprocess.CompletedProcess[str]:
        nonlocal attempts
        assert args[:2] == ("datasets", "status")
        attempts += 1
        if attempts == 1:
            raise KaggleBatchError("temporary 403 while indexing")
        return subprocess.CompletedProcess(args, 0, stdout="pending" if attempts == 2 else "ready")

    monkeypatch.setattr(KaggleBatchRunner, "_command", fake_command)
    monkeypatch.setattr("src.services.video.kaggle_batch.time.sleep", lambda _: None)
    KaggleBatchRunner(username="tester", work_root=tmp_path)._wait_for_dataset("tester/new-dataset")
    assert attempts == 3


def test_kaggle_runner_rejects_path_traversal(tmp_path: Path) -> None:
    code = tmp_path / "kernel.py"
    code.write_text("", encoding="utf-8")
    runner = KaggleBatchRunner(
        username="tester", work_root=tmp_path, sam3_checkpoint_dataset="tester/autolabel-sam3-checkpoint"
    )
    with pytest.raises(ValueError, match="unsafe"):
        runner.run(stage="sam3", job_id=JOB_ID, files={"../secret": b"x"}, code_file=code)


def test_kaggle_sam3_requires_checkpoint_dataset(tmp_path: Path) -> None:
    runner = KaggleBatchRunner(username="tester", work_root=tmp_path)
    with pytest.raises(KaggleBatchError, match="KAGGLE_SAM3_CHECKPOINT_DATASET"):
        runner.run(stage="sam3", job_id=JOB_ID, files={}, code_file=tmp_path / "main.py")


def test_kernel_status_requires_recognized_terminal_or_running_state() -> None:
    assert _kernel_status("Status: KernelWorkerStatus.COMPLETE") == "complete"
    assert _kernel_status("Status: RUNNING") == "running"
    with pytest.raises(KaggleBatchError):
        _kernel_status("not a status")


def test_kaggle_sam3_preserves_frame_zero_seed_and_track_identity(tmp_path: Path) -> None:
    workspace = VideoWorkspace(tmp_path / "workspace")
    frames_dir = workspace.prepare_frames_dir(JOB_ID)
    (frames_dir / "frame_000000.jpg").write_bytes(b"frame")
    box = BoundingBox2D(xtl=1, ytl=2, xbr=8, ybr=9)
    manifest = FrameManifest(
        job_id=JOB_ID,
        source_name="clip.mp4",
        source_sha256="a" * 64,
        codec="h264",
        width=20,
        height=20,
        duration_s=1,
        average_fps=1,
        variable_frame_rate=False,
        ffmpeg_version="test",
        frames=[
            FrameRecord(
                frame_index=0,
                timestamp_s=0,
                source_timestamp_s=0,
                relative_path="frames/frame_000000.jpg",
                sha256="b" * 64,
                width=20,
                height=20,
            )
        ],
    )
    seeds = SeedSet(
        job_id=JOB_ID,
        cvat_job_id=1,
        annotation_revision=0,
        seeds=[
            SeedRectangle(
                seed_id="seed:1",
                object_id="shape:1",
                label_id=1,
                label_name="car",
                shape_id=1,
                bbox=box,
            )
        ],
    )

    class FakeRunner:
        def run(self, **kwargs: object) -> dict:
            assert kwargs["stage"] == "sam3"
            assert "frames/frame_000000.jpg" in kwargs["files"]  # type: ignore[operator]
            return {
                "job_id": JOB_ID,
                "stage": "sam3",
                "track_set": {
                    "job_id": JOB_ID,
                    "manifest_sha256": "c" * 64,
                    "seed_sha256": "d" * 64,
                    "component_versions": [{"component": "sam3-video", "version": "facebook/sam3"}],
                    "tracks": [
                        {
                            "track_id": "shape:1",
                            "label_id": 1,
                            "label_name": "car",
                            "source": "human_seed",
                            "observations": [{"frame_index": 0, "bbox": {"xtl": 2, "ytl": 2, "xbr": 9, "ybr": 9}}],
                        }
                    ],
                },
            }

    propagator = KaggleSam3Propagator(
        runner=FakeRunner(),
        workspace=workspace,
        model_id="facebook/sam3",
        code_file=tmp_path / "main.py",
    )  # type: ignore[arg-type]
    tracks = propagator.propagate(manifest=manifest, manifest_sha256="c" * 64, seed_set=seeds, seed_sha256="d" * 64)
    assert tracks.tracks[0].observations[0].bbox == box
