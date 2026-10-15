"""SAM, previews and VLM consume the same verified redacted JPEG representation."""

import hashlib
import io
import json
from collections.abc import AsyncIterator
from contextlib import closing
from pathlib import Path
from typing import Any

from PIL import Image

from src.services.vm.media import verified_media
from src.services.vm.storage import VmStore


class VerifiedFrameReader:
    def __init__(self, store: VmStore, job: dict, client: Any):
        verified_media(store, job)
        self.store, self.job, self.client = store, job, client

    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)

    async def iter_selected_job_frames(self, cvat_job: int, indexes: list[int]) -> AsyncIterator[bytes]:
        if cvat_job != self.job["cvat_job_id"]:
            raise ValueError("Frame provider is bound to another redacted job")
        records = {}
        paths = {}
        for index in indexes:
            with self.store.connect() as db:
                row = db.execute(
                    "SELECT payload FROM vm_events WHERE job=? AND event=?", (self.job["job_id"], f"input:{index}")
                ).fetchone()
            records[index] = json.loads(row[0]) if row else None
            paths[index] = self.store.directory(self.job["job_id"]) / "cache" / f"frame_{index:06d}.jpg"
        missing = [i for i in indexes if records[i] is None or not paths[i].exists()]
        if self.job.get("in_place"):
            video, _receipt = verified_media(self.store, self.job)
            for index in indexes:
                if not 0 <= index < self.job["total_frames"]:
                    raise ValueError("Frame outside verified redacted video")
            decoded = self._redacted_frames(video, missing)
            for index in indexes:
                record, path = records[index], paths[index]
                try:
                    data = path.read_bytes() if record else None
                except FileNotFoundError:
                    data = None
                if data is None:
                    # A concurrent cache eviction may remove a previously present frame.
                    if index not in decoded:
                        decoded.update(self._redacted_frames(video, [index]))
                    data = decoded[index]
                if record and hashlib.sha256(data).hexdigest() != record["sha256"]:
                    raise ValueError("Redacted preview/VLM frame differs from SAM input")
                yield data
            return
        incoming = self.client.iter_selected_job_frames(cvat_job, missing) if missing else None
        for index in indexes:
            if not 0 <= index < self.job["total_frames"]:
                raise ValueError("Frame outside verified redacted video")
            record = records[index]
            path = paths[index]
            try:
                data = path.read_bytes() if record else None
            except FileNotFoundError:
                data = None
            if data is None:
                if index in missing:
                    assert incoming is not None
                    try:
                        data = await anext(incoming)
                    except StopAsyncIteration:
                        raise ValueError("Redacted source frame coverage differs") from None
                else:
                    items = [blob async for blob in self.client.iter_selected_job_frames(cvat_job, [index])]
                    if len(items) != 1:
                        raise ValueError("Redacted source frame coverage differs")
                    data = items[0]
                assert data is not None
                with Image.open(io.BytesIO(data)) as image:
                    if image.format != "JPEG":
                        stream = io.BytesIO()
                        image.convert("RGB").save(stream, format="JPEG", quality=95)
                        data = stream.getvalue()
            assert data is not None
            if record and hashlib.sha256(data).hexdigest() != record["sha256"]:
                raise ValueError("Redacted preview/VLM frame differs from SAM input")
            yield data

    @staticmethod
    def _redacted_frames(video: Path, indexes: list[int]) -> dict[int, bytes]:
        """Decode requested frames in one pass, retaining only selected JPEGs."""
        import av

        wanted = set(indexes)
        if not wanted:
            return {}
        decoded = {}
        with closing(av.open(str(video))) as container:
            for frame_index, frame in enumerate(container.decode(video=0)):
                if frame_index in wanted:
                    image = Image.fromarray(frame.to_ndarray(format="rgb24"))
                    stream = io.BytesIO()
                    image.save(stream, format="JPEG", quality=95)
                    decoded[frame_index] = stream.getvalue()
                    if len(decoded) == len(wanted):
                        break
        if len(decoded) != len(wanted):
            raise ValueError("Redacted source frame coverage differs")
        return decoded
