"""Common finite GPU batch boundary for notebook and queue providers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol


class VideoBatchError(RuntimeError):
    """Sanitized finite model-job failure."""


class VideoBatchRunner(Protocol):
    def run(self, *, stage: str, job_id: str, files: Mapping[str, Path | bytes], code_file: Path) -> dict: ...
