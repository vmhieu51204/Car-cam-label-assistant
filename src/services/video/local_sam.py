"""Isolated local CUDA runner using the existing window/checkpoint contract."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT, Settings


def local_ready(settings: Settings, variant: str) -> bool:
    checkpoint = settings.local_sam31_checkpoint if variant == "sam3.1" else settings.local_sam3_checkpoint
    digest = settings.local_sam31_sha256 if variant == "sam3.1" else settings.local_sam3_sha256
    return bool(
        settings.local_sam_enabled
        and settings.local_sam_python.is_file()
        and checkpoint
        and checkpoint.is_file()
        and len(digest) == 64
        and all(c in "0123456789abcdef" for c in digest.lower())
        and len(settings.local_sam_commit) == 40
        and all(c in "0123456789abcdef" for c in settings.local_sam_commit.lower())
    )


class LocalSamRunner:
    def __init__(self, settings: Settings, root: Path):
        self.settings, self.root = settings, root

    def fingerprint(self, variant: str = "sam3") -> str:
        digest = self.settings.local_sam31_sha256 if variant == "sam3.1" else self.settings.local_sam3_sha256
        identity = json.dumps(
            [
                variant,
                digest.lower(),
                self.settings.local_sam_commit.lower(),
                str(self.settings.local_sam_python.resolve()),
            ]
        ).encode()
        return hashlib.sha256(
            identity
            + b"".join(
                (PROJECT_ROOT / p).read_bytes()
                for p in ("scripts/run_local_sam.py", "kaggle/sam3_benchmark/main.py", "kaggle/checkpoint_contract.py")
            )
        ).hexdigest()

    def run(
        self, *, stage: str, job_id: str, files: dict[str, Any], code_file: Path, model_variant: str = "sam3"
    ) -> dict:
        if stage != "sam3-sequence" or not local_ready(self.settings, model_variant):
            raise ValueError("Local SAM is not configured: interpreter, checkpoint and SHA-256 are required")
        checkpoint = (
            self.settings.local_sam31_checkpoint if model_variant == "sam3.1" else self.settings.local_sam3_checkpoint
        )
        digest = self.settings.local_sam31_sha256 if model_variant == "sam3.1" else self.settings.local_sam3_sha256
        directory = self.root / job_id / "local-sam"
        directory.mkdir(parents=True, exist_ok=True)
        for name, value in files.items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("unsafe local payload path")
            target = directory / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(value if isinstance(value, bytes) else Path(value).read_bytes())
        output = directory / "output"
        output.mkdir(exist_ok=True)
        result_path = output / "result.json"
        # No stale successful result can survive an invocation that failed before writing.
        result_path.unlink(missing_ok=True)
        with (directory / "inference.log").open("w", encoding="utf-8") as log:
            try:
                completed = subprocess.run(
                    [
                        str(self.settings.local_sam_python.resolve()),
                        "-m",
                        "scripts.run_local_sam",
                        "--input",
                        str(directory.resolve()),
                        "--output",
                        str(output.resolve()),
                        "--checkpoint",
                        str(checkpoint),
                        "--sha256",
                        digest,
                        "--commit",
                        self.settings.local_sam_commit,
                    ],
                    cwd=PROJECT_ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=self.settings.local_sam_timeout_s,
                )
            except subprocess.TimeoutExpired as error:
                if result_path.is_file():
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    result["execution_provider"] = "local"
                    result["local_error"] = "timeout; only completed window checkpoints are reusable"
                    for row in result.get("configurations", []):
                        if row.get("status") != "completed":
                            row["status"] = "failed"
                    return result
                raise RuntimeError("Local SAM timed out; see local-sam/inference.log") from error
        if not result_path.is_file():
            raise RuntimeError(f"Local SAM exited {completed.returncode}; see local-sam/inference.log")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["execution_provider"] = "local"
        return result
