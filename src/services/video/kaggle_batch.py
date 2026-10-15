"""Run private, finite Kaggle GPU notebooks and verify their result artifact."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import time
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import IO
from uuid import uuid4

from src.services.video.batch_runner import VideoBatchError
from src.services.video.profile import current_profile


class KaggleBatchError(VideoBatchError):
    """Kaggle setup, upload, execution, or result failure."""


class KaggleBatchRunner:
    def __init__(
        self,
        *,
        username: str,
        work_root: Path,
        machine_shape: str = "NvidiaTeslaT4",
        timeout_s: float = 21_600,
        poll_seconds: float = 20,
        sam3_checkpoint_dataset: str = "",
        command: str = "kaggle",
        api_token: str = "",
    ) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]{2,50}", username):
            raise KaggleBatchError("KAGGLE_USERNAME is required for Kaggle inference")
        if not re.fullmatch(r"[A-Za-z0-9_-]{3,50}", machine_shape):
            raise KaggleBatchError("Kaggle accelerator name is invalid")
        if timeout_s <= 0 or poll_seconds <= 0:
            raise ValueError("Kaggle timeout and poll interval must be positive")
        self.username = username
        self.work_root = work_root.resolve()
        self.machine_shape = machine_shape
        self.timeout_s = timeout_s
        self.poll_seconds = poll_seconds
        self.command = command
        self.api_token = api_token
        if sam3_checkpoint_dataset and not re.fullmatch(
            r"[A-Za-z0-9_-]{2,50}/[a-z0-9][a-z0-9-]{1,99}", sam3_checkpoint_dataset
        ):
            raise KaggleBatchError("KAGGLE_SAM3_CHECKPOINT_DATASET must be username/dataset-slug")
        self.sam3_checkpoint_dataset = sam3_checkpoint_dataset

    def run(
        self,
        *,
        stage: str,
        job_id: str,
        files: Mapping[str, Path | bytes],
        code_file: Path,
        allow_retry_on_terminal_failure: bool = True,
        model_variant: str = "sam3",
    ) -> dict:
        if stage not in {"sam3", "sam3-seed", "sam3-sequence", "sam3-benchmark", "sam3-rescue"} or not re.fullmatch(
            r"[0-9a-f-]{36}", job_id
        ):
            raise ValueError("invalid Kaggle stage or video job ID")
        if model_variant != "sam3":
            raise ValueError("SAM3 is the only executable model variant")
        checkpoint_dataset = self.sam3_checkpoint_dataset
        model_source = ""
        # Prefer Kaggle Models when configured. Attaching the same weights as a
        # dataset too would make the notebook find two checkpoints and fail closed.
        attached_checkpoint_dataset = checkpoint_dataset if not model_source else ""
        previous = self._matching_run(
            stage=stage,
            job_id=job_id,
            files=files,
            code_file=code_file,
            model_variant=model_variant,
            checkpoint_dataset=attached_checkpoint_dataset,
            model_source=model_source,
        )
        if previous is not None:
            run_dir, kernel_reference = previous
            try:
                return self._collect_result(stage, job_id, kernel_reference, run_dir / "output")
            except KaggleBatchError as error:
                # A failed kernel cannot be resumed. An explicit job retry must get
                # a fresh private run; running and completed kernels remain reusable.
                if f"Kaggle {stage} run error;" not in str(error) and f"Kaggle {stage} run cancelled;" not in str(
                    error
                ):
                    raise
                if not allow_retry_on_terminal_failure:
                    raise
        slug = f"autolabel-{stage}-{job_id[:8]}-{uuid4().hex[:6]}"
        dataset_reference = f"{self.username}/{slug}"
        kernel_reference = f"{self.username}/{slug}-run"
        if stage.startswith("sam3") and not checkpoint_dataset and not model_source:
            raise KaggleBatchError("KAGGLE_SAM3_CHECKPOINT_DATASET is required for SAM3 inference")
        run_dir = self.work_root / job_id / "kaggle" / slug
        dataset_dir = run_dir / "dataset"
        kernel_dir = run_dir / "kernel"
        output_dir = run_dir / "output"
        for directory in (dataset_dir, kernel_dir, output_dir):
            directory.mkdir(parents=True, exist_ok=False)
        with zipfile.ZipFile(dataset_dir / "payload.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, source in files.items():
                if name.startswith("/") or "\\" in name or any(part in {"", ".", ".."} for part in name.split("/")):
                    raise ValueError("Kaggle payload contains an unsafe file path")
                if isinstance(source, Path):
                    if not source.is_file():
                        raise FileNotFoundError(source)
                    archive.write(source, name)
                else:
                    archive.writestr(name, source)
        (dataset_dir / "dataset-metadata.json").write_text(
            json.dumps(
                {
                    "title": f"AutoLabel {stage} {job_id[:8]} {slug[-6:]}",
                    "id": dataset_reference,
                    "licenses": [{"name": "other"}],
                    "description": "Private AutoLabel inference input. Ownership and reuse terms remain with the uploader.",
                }
            ),
            encoding="utf-8",
        )
        if not code_file.is_file():
            raise FileNotFoundError(code_file)
        if code_file.suffix not in {".py", ".ipynb"}:
            raise KaggleBatchError("Kaggle code must be a Python script or notebook")
        code_name = f"main{code_file.suffix}"
        shutil.copyfile(code_file, kernel_dir / code_name)
        (kernel_dir / "kernel-metadata.json").write_text(
            json.dumps(
                {
                    "id": kernel_reference,
                    "title": f"AutoLabel {stage} {job_id[:8]} {slug[-6:]} run",
                    "code_file": code_name,
                    "language": "python",
                    "kernel_type": "notebook" if code_file.suffix == ".ipynb" else "script",
                    "is_private": True,
                    "enable_gpu": True,
                    "enable_internet": True,
                    "machine_shape": self.machine_shape,
                    "dataset_sources": [dataset_reference]
                    + (
                        [attached_checkpoint_dataset]
                        if stage.startswith("sam3") and attached_checkpoint_dataset
                        else []
                    ),
                    "model_sources": [model_source] if model_source and stage.startswith("sam3") else [],
                    "model_variant": model_variant,
                }
            ),
            encoding="utf-8",
        )
        upload_dir = _prepare_upload(dataset_dir)
        self._command("datasets", "create", "-p", str(upload_dir), "--dir-mode", "zip", timeout=900)
        self._wait_for_dataset(dataset_reference)
        self._command("kernels", "push", "-p", str(kernel_dir), timeout=900)
        (run_dir / "kernel-submitted.json").write_text(
            json.dumps({"kernel_reference": kernel_reference}), encoding="utf-8"
        )
        return self._collect_result(stage, job_id, kernel_reference, output_dir)

    def _matching_run(
        self,
        *,
        stage: str,
        job_id: str,
        files: Mapping[str, Path | bytes],
        code_file: Path,
        model_variant: str = "sam3",
        checkpoint_dataset: str = "",
        model_source: str = "",
    ) -> tuple[Path, str] | None:
        root = self.work_root / job_id / "kaggle"
        if not root.is_dir() or not code_file.is_file():
            return None
        for run_dir in sorted(
            root.glob(f"autolabel-{stage}-{job_id[:8]}-*"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        ):
            metadata_path = run_dir / "kernel" / "kernel-metadata.json"
            archive_path = run_dir / "dataset" / "payload.zip"
            try:
                if not (run_dir / "kernel-submitted.json").is_file():
                    continue
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata.get("model_variant", "sam3") != model_variant:
                    continue
                if metadata.get("model_sources", []) != ([model_source] if model_source else []):
                    continue
                sources = metadata.get("dataset_sources")
                if not isinstance(sources, list):
                    continue
                if stage.startswith("sam3") and checkpoint_dataset:
                    if len(sources) != 2 or sources[1] != checkpoint_dataset:
                        continue
                elif len(sources) != 1:
                    continue
                kernel_reference = metadata["id"]
                if not isinstance(kernel_reference, str) or not kernel_reference.startswith(f"{self.username}/"):
                    continue
                copied_code = run_dir / "kernel" / metadata["code_file"]
                if _sha256_file(copied_code) != _sha256_file(code_file):
                    continue
                with zipfile.ZipFile(archive_path) as archive:
                    if set(archive.namelist()) != set(files):
                        continue
                    if any(
                        _sha256_archive_entry(archive, name) != _sha256_input(source) for name, source in files.items()
                    ):
                        continue
            except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile):
                continue
            return run_dir, kernel_reference
        return None

    def _collect_result(self, stage: str, job_id: str, kernel_reference: str, output_dir: Path) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout_s
        while True:
            if time.monotonic() >= deadline:
                raise KaggleBatchError(f"Kaggle {stage} run timed out; kernel {kernel_reference}")
            try:
                result = self._command("kernels", "status", kernel_reference, timeout=60)
            except KaggleBatchError as error:
                if "command timed out" not in str(error) and "network unavailable" not in str(error):
                    raise
                with current_profile().stage("kaggle.poll_wait"):
                    time.sleep(min(self.poll_seconds, max(0, deadline - time.monotonic())))
                continue
            status = _kernel_status(result.stdout)
            if status == "complete":
                break
            if status in {"error", "cancelled"}:
                if stage in {"sam3-sequence", "sam3-benchmark", "sam3-rescue"}:
                    try:
                        self._command(
                            "kernels",
                            "output",
                            kernel_reference,
                            "-p",
                            str(output_dir),
                            "--file-pattern",
                            "result.json",
                            timeout=900,
                        )
                        partial_path = output_dir / "result.json"
                        if partial_path.stat().st_size <= 200_000_000:
                            partial = json.loads(partial_path.read_text(encoding="utf-8"))
                            if (
                                stage == "sam3-rescue"
                                and isinstance(partial, dict)
                                and partial.get("schema") == "point-rescue-result/v1"
                                and partial.get("job_id") == job_id
                                and partial.get("candidates")
                            ):
                                partial["terminal_kernel_status"] = status
                                return partial
                            rows = partial.get("configurations", [])
                            sequence_partial = (
                                stage == "sam3-sequence"
                                and len(rows) == 1
                                and rows[0].get("id") == "sequence-text"
                                and rows[0].get("window_checkpoints")
                            )
                            benchmark_partial = stage == "sam3-benchmark" and any(
                                row.get("status") == "completed" or row.get("window_checkpoints") for row in rows
                            )
                            if partial.get("schema") == "autolabel-sam3-benchmark/v1" and (
                                sequence_partial or benchmark_partial
                            ):
                                if sequence_partial:
                                    rows[0]["status"] = "failed"
                                partial["terminal_kernel_status"] = status
                                return partial
                    except (KaggleBatchError, OSError, ValueError, TypeError, AttributeError):
                        pass
                raise KaggleBatchError(
                    f"Kaggle {stage} run {status}; no verified checkpoint available; "
                    f"inspect private kernel {kernel_reference}"
                )
            with current_profile().stage("kaggle.poll_wait"):
                time.sleep(min(self.poll_seconds, max(0, deadline - time.monotonic())))
        path = output_dir / "result.json"
        try:
            self._command(
                "kernels",
                "output",
                kernel_reference,
                "-p",
                str(output_dir),
                "--file-pattern",
                "result.json",
                timeout=900,
            )
        except KaggleBatchError:
            # Kaggle CLI on Windows can download result.json and then fail while
            # printing a Unicode kernel log. Validate the downloaded artifact below.
            if not path.is_file():
                raise
        max_result_bytes = 200_000_000 if stage in {"sam3-sequence", "sam3-benchmark", "sam3-rescue"} else 30_000_000
        if not path.is_file() or path.stat().st_size > max_result_bytes:
            raise KaggleBatchError(f"Kaggle {stage} result is missing or too large; kernel {kernel_reference}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise KaggleBatchError("Kaggle result is not valid JSON") from error
        if stage == "sam3-sequence":
            rows = payload.get("configurations") if isinstance(payload, dict) else None
            if (
                not isinstance(payload, dict)
                or payload.get("schema") != "autolabel-sam3-benchmark/v1"
                or not isinstance(rows, list)
                or len(rows) != 1
                or rows[0].get("id") != "sequence-text"
            ):
                raise KaggleBatchError("Kaggle result is not a SAM3 sequence result")
        elif stage == "sam3-benchmark":
            rows = payload.get("configurations") if isinstance(payload, dict) else None
            statuses = {"completed", "failed", "timed_out", "unsupported", "pending"}
            if (
                not isinstance(payload, dict)
                or payload.get("schema") != "autolabel-sam3-benchmark/v1"
                or payload.get("completed") is not True
                or not isinstance(rows, list)
                or not rows
                or any(
                    not isinstance(row, dict) or not isinstance(row.get("id"), str) or row.get("status") not in statuses
                    for row in rows
                )
                or len({row["id"] for row in rows}) != len(rows)
            ):
                raise KaggleBatchError("Kaggle result is not a valid SAM3 benchmark result")
        elif stage == "sam3-rescue":
            if (
                not isinstance(payload, dict)
                or payload.get("schema") != "point-rescue-result/v1"
                or payload.get("job_id") != job_id
                or payload.get("stage") != stage
                or not isinstance(payload.get("candidates"), list)
            ):
                raise KaggleBatchError("Kaggle result is not a valid SAM3 rescue result")
        elif not isinstance(payload, dict) or payload.get("job_id") != job_id or payload.get("stage") != stage:
            raise KaggleBatchError("Kaggle result does not match the requested job and stage")
        return payload

    def _wait_for_dataset(self, reference: str) -> None:
        deadline = time.monotonic() + min(self.timeout_s, 1800)
        while True:
            try:
                result = self._command("datasets", "status", reference, timeout=60)
            except KaggleBatchError:
                # Kaggle can return 403 while a newly created private dataset is indexing.
                state = "pending"
            else:
                state = result.stdout.strip().lower()
            if state == "ready":
                return
            if state in {"error", "failed"}:
                raise KaggleBatchError(f"Kaggle input dataset failed to process: {reference}")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise KaggleBatchError(f"Kaggle input dataset did not become ready: {reference}")
            with current_profile().stage("kaggle.poll_wait"):
                time.sleep(min(self.poll_seconds, remaining))

    def _command(self, *args: str, timeout: float) -> subprocess.CompletedProcess[str]:
        stages: dict[tuple[str, ...], str] = {
            ("datasets", "create"): "kaggle.input_upload",
            ("datasets", "status"): "kaggle.input_poll",
            ("kernels", "push"): "kaggle.submit",
            ("kernels", "status"): "kaggle.kernel_poll",
            ("kernels", "output"): "kaggle.output_transfer",
        }
        with current_profile().stage(stages.get(args[:2], "kaggle.cli")):
            return self._command_impl(*args, timeout=timeout)

    def _command_impl(self, *args: str, timeout: float) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["PYTHONIOENCODING"] = "utf-8"
        environment["PYTHONUTF8"] = "1"
        if self.api_token:
            environment["KAGGLE_USERNAME"] = self.username
            environment["KAGGLE_API_TOKEN"] = self.api_token
        try:
            result = subprocess.run(
                [self.command, *args],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                shell=False,
                env=environment,
            )
        except FileNotFoundError as error:
            raise KaggleBatchError("Kaggle CLI is not installed in the worker") from error
        except subprocess.TimeoutExpired as error:
            raise KaggleBatchError(f"Kaggle {args[0]} {args[1]} command timed out") from error
        if result.returncode != 0:
            diagnostic = ((result.stderr or "") + "\n" + (result.stdout or "")).lower()
            reason = ""
            if any(
                item in diagnostic
                for item in (
                    "connection refused",
                    "connection timed out",
                    "connecttimeouterror",
                    "max retries exceeded",
                    "temporary failure in name resolution",
                )
            ):
                reason = "; network unavailable"
            raise KaggleBatchError(f"Kaggle {args[0]} {args[1]} failed (exit {result.returncode}{reason})")
        if args[:2] == ("datasets", "create") and "dataset creation error:" in result.stdout.lower():
            raise KaggleBatchError("Kaggle dataset creation failed; no kernel submitted")
        return result


def _prepare_upload(dataset_dir: Path) -> Path:
    """Upload raw payload files, avoiding Kaggle's failed nested-archive indexing.

    Keep the original wrapper zip unchanged for legacy resume identity checks.
    Kaggle CLI zips the payload directory once; Kaggle then expands one level.
    """
    upload = dataset_dir.parent / "upload"
    upload.mkdir(exist_ok=False)
    shutil.copyfile(dataset_dir / "dataset-metadata.json", upload / "dataset-metadata.json")
    with zipfile.ZipFile(dataset_dir / "payload.zip") as wrapper:
        if wrapper.namelist() == ["payload.zip"]:
            if wrapper.getinfo("payload.zip").file_size > 1_000_000_000:
                raise ValueError("payload archive too large")
            with zipfile.ZipFile(io.BytesIO(wrapper.read("payload.zip"))) as payload:
                _extract_upload(payload, upload / "payload")
        else:
            _extract_upload(wrapper, upload / "payload")
    return upload


def _extract_upload(archive: zipfile.ZipFile, destination: Path) -> None:
    members = archive.infolist()
    if sum(item.file_size for item in members) > 2_000_000_000:
        raise ValueError("expanded payload too large")
    seen: set[str] = set()
    for item in members:
        name = item.filename
        key = name.casefold()
        if (
            name.startswith("/")
            or "\\" in name
            or ":" in name
            or any(part in {"", ".", ".."} for part in name.rstrip("/").split("/"))
            or key in seen
            or (item.external_attr >> 16) & 0o170000 == 0o120000
        ):
            raise ValueError("unsafe upload archive path")
        seen.add(key)
    for item in members:
        target = destination / item.filename
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(item) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def _kernel_status(output: str) -> str:
    match = re.search(
        r"(?:KernelWorkerStatus\.|status\s*:\s*)(complete|error|cancelled|cancel_acknowledged|running|queued|pending|initializing)",
        output,
        re.IGNORECASE,
    )
    if match is None:
        raise KaggleBatchError("Kaggle returned an unrecognized kernel status")
    value = match.group(1).lower()
    return "cancelled" if value == "cancel_acknowledged" else value


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return _sha256_stream(handle)


def _sha256_archive_entry(archive: zipfile.ZipFile, name: str) -> str:
    with archive.open(name) as handle:
        return _sha256_stream(handle)


def _sha256_input(source: Path | bytes) -> str:
    return _sha256_file(source) if isinstance(source, Path) else hashlib.sha256(source).hexdigest()


def _sha256_stream(handle: IO[bytes]) -> str:
    digest = hashlib.sha256()
    while chunk := handle.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()
