"""Opt-in, best-effort timing sidecars; no annotations, credentials or API data.

Events are immutable files so independently running workers cannot overwrite
each other's timings. Wall timestamps allow spans to overlap; never sum
asynchronous worker totals to claim an end-to-end duration.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import math
import os
import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from typing import Any
from uuid import uuid4

_active: contextvars.ContextVar[Profile2D | None] = contextvars.ContextVar("profile_2d", default=None)


class Profile2D:
    def __init__(self, root: Path, *, enabled: bool = False, clock: Callable[[], float] = time.perf_counter) -> None:
        self.root, self.enabled, self.clock = root, enabled, clock

    def record(self, stage: str, seconds: float, *, status: str = "completed", **metadata: Any) -> None:
        if not self.enabled:
            return
        # Deliberately allowlist scalar counters, never arbitrary request/error text.
        if not math.isfinite(seconds):
            return
        allowed = {
            key: value
            for key, value in metadata.items()
            if key in {"frame_count", "object_count", "frame_index", "attempt", "cache_hit", "http_status"}
            and isinstance(value, (int, float, bool))
            and math.isfinite(value)
        }
        event = {
            "schema": "autolabel-profile-2d/v1",
            "stage": stage,
            "seconds": max(0.0, seconds),
            "status": status,
            "ended_at": datetime.now(UTC).isoformat(),
            "pid": os.getpid(),
            "peak_gpu_memory_bytes": None,
            "memory_measurement": "unavailable",
            **allowed,
        }
        try:
            folder = self.root / "profile-2d" / "events"
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / f"{uuid4().hex}.json"
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(event, sort_keys=True), encoding="utf-8")
            temporary.replace(target)
        except (OSError, ValueError, TypeError):
            logging.warning("Optional 2D profile event could not be saved")

    @contextlib.contextmanager
    def stage(self, name: str, **metadata: Any) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        try:
            started = self.clock()
        except Exception:
            yield
            return
        status = "failed"
        try:
            yield
            status = "completed"
        finally:
            try:
                self.record(name, self.clock() - started, status=status, **metadata)
            except Exception:
                logging.warning("Optional 2D profile measurement unavailable")

    @contextlib.contextmanager
    def activate(self) -> Iterator[None]:
        token = _active.set(self)
        try:
            yield
        finally:
            _active.reset(token)


def current_profile() -> Profile2D:
    return _active.get() or Profile2D(Path("."))


def profiled_worker(stage: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Profile existing sync worker methods with an optional constructor flag."""

    def decorate(method: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(method)
        def wrapped(self: Any, claim_or_id: Any, *args: Any, **kwargs: Any) -> Any:
            if not self.profile_enabled:
                return method(self, claim_or_id, *args, **kwargs)
            job_id = claim_or_id if isinstance(claim_or_id, str) else claim_or_id.job.job_id
            profile = Profile2D(self.workspace.job_dir(job_id), enabled=self.profile_enabled)
            with profile.activate(), profile.stage(stage):
                return method(self, claim_or_id, *args, **kwargs)

        return wrapped

    return decorate
