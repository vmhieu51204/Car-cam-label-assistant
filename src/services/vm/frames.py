"""Verified redacted CVAT frames with byte-bounded cache and pinned ranges."""

from __future__ import annotations

import hashlib
import io
import json
import threading
from collections import OrderedDict
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Protocol

from PIL import Image

from src.services.video.extractor import _histogram_distance
from src.services.vm.storage import VmStore


class FrameSource(Protocol):
    def iter_selected_job_frames(self, job: int, indexes: list[int]) -> AsyncIterator[bytes]: ...


class FrameProvider:
    def __init__(self, store: VmStore, job: str, source: FrameSource, cvat_job: int, capacity: int):
        self.store, self.job, self.source, self.cvat_job, self.capacity = store, job, source, cvat_job, capacity
        self.directory = store.directory(job) / "cache"
        self.directory.mkdir(exist_ok=True)
        self.cache: OrderedDict[int, tuple[Path, int]] = OrderedDict()
        self.pins: dict[int, int] = {}
        self.condition = threading.Condition()
        self.closed = False
        receipt = store.directory(job) / "public/redacted.privacy.json"
        self.media = json.loads(receipt.read_text()) if receipt.exists() else None
        # Old unpinned cache files can be reused only after their journal hash verifies.
        for path in self.directory.glob("frame_*.jpg"):
            index = int(path.stem.split("_")[-1])
            self.cache[index] = (path, path.stat().st_size)
        while self.size > self.capacity:
            _index, (path, _size) = self.cache.popitem(last=False)
            path.unlink()

    @property
    def size(self) -> int:
        return sum(item[1] for item in self.cache.values())

    def record(self, index: int) -> dict | None:
        with self.store.connect() as db:
            row = db.execute(
                "SELECT payload FROM vm_events WHERE job=? AND event=?", (self.job, f"input:{index}")
            ).fetchone()
        if row is None:
            return None
        import json

        return json.loads(row[0])

    def release(self, indexes: list[int]) -> None:
        with self.condition:
            for index in indexes:
                self.pins[index] = max(0, self.pins.get(index, 0) - 1)
            self.condition.notify_all()

    def close(self) -> None:
        with self.condition:
            self.closed = True
            self.condition.notify_all()

    async def acquire(self, indexes: list[int]) -> list[dict]:
        missing = [i for i in indexes if i not in self.cache]
        acquired: list[int] = []
        try:
            with self.condition:
                for i in indexes:
                    self.pins[i] = self.pins.get(i, 0) + 1
                    acquired.append(i)
            cursor = 0

            async def download() -> AsyncIterator[bytes]:
                if missing:
                    async for data in self.source.iter_selected_job_frames(self.cvat_job, missing):
                        yield data

            async for encoded in download():
                if cursor >= len(missing):
                    raise ValueError("Excess CVAT frames")
                index = missing[cursor]
                cursor += 1
                with Image.open(io.BytesIO(encoded)) as image:
                    width, height = image.size
                    histogram = image.convert("L").resize((64, 36)).histogram()
                    if image.format != "JPEG":
                        output = io.BytesIO()
                        image.convert("RGB").save(output, format="JPEG", quality=95)
                        encoded = output.getvalue()
                previous = self.record(index - 1) if index else None
                score = _histogram_distance(previous["histogram"] if previous else None, histogram)
                record = {
                    "source_frame_index": index,
                    "frame_index": index,
                    "width": width,
                    "height": height,
                    "sha256": hashlib.sha256(encoded).hexdigest(),
                    "histogram": histogram,
                    "scene_cut_before": bool(score is not None and score >= 0.35),
                    "relative_path": f"cache/frame_{index:06d}.jpg",
                    "source_timestamp": self.media["timestamps"][index] if self.media else None,
                    "source_duration": self.media["durations"][index] if self.media else None,
                }
                self.store.event(self.job, f"input:{index}", record)
                path = self.directory / f"frame_{index:06d}.jpg"
                with self.condition:
                    wanted = sum(self.cache[i][1] for i in indexes if i in self.cache) + len(encoded)
                    if wanted > self.capacity:
                        raise ValueError(
                            "One active SAM window exceeds JPEG cache capacity; increase LOCAL_FRAME_CACHE_MAX_BYTES"
                        )
                    while self.size + len(encoded) > self.capacity:
                        victim = next((i for i in self.cache if not self.pins.get(i)), None)
                        if victim is not None:
                            victim_path, _size = self.cache.pop(victim)
                            victim_path.unlink(missing_ok=True)
                        elif self.closed:
                            raise ValueError("Frame provider closed")
                        else:
                            self.condition.wait(timeout=1)
                    path.write_bytes(encoded)
                    self.cache[index] = (path, len(encoded))
            if cursor != len(missing):
                raise ValueError("Missing CVAT frames")
            result = []
            for index in indexes:
                cached_record = self.record(index)
                if cached_record is None:
                    raise ValueError("Cache has no verified input journal")
                with self.condition:
                    path, _size = self.cache[index]
                    if hashlib.sha256(path.read_bytes()).hexdigest() != cached_record["sha256"]:
                        raise ValueError("Cached frame differs from input journal")
                    self.cache.move_to_end(index)
                result.append(cached_record)
            return result
        except Exception:
            self.release(acquired)
            raise


def plan_window(records: list[dict], total: int, size: int, overlap: int) -> tuple[list[dict], int | None]:
    """One-frame lookahead prevents a redundant overlap window at scene endpoints."""
    if not 0 < overlap < size or not records:
        raise ValueError("Invalid online window")
    start = records[0]["source_frame_index"]
    cut = next((i for i, r in enumerate(records[1:], 1) if r["scene_cut_before"]), len(records))
    selected = records[: min(size, cut)]
    end = start + len(selected)
    if end >= total:
        return selected, None
    if cut <= size:
        return selected, end
    return selected, end - overlap
