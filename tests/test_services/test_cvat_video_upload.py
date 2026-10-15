from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx

from src.models.video import CvatLabelSpec
from src.services.cvat import CvatTaskClient


def test_video_upload_creates_one_contiguous_cvat_job(tmp_path: Path):
    video = tmp_path / "videotest.mp4"
    video.write_bytes(b"synthetic video bytes")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/tasks":
            body = json.loads(request.content)
            assert body["segment_size"] == 100_000
            assert body["labels"][0]["name"] == "car"
            return httpx.Response(201, json={"id": 19})
        if request.url.path == "/api/tasks/19/data":
            assert request.headers["Upload-Start"] == ""
            assert request.headers["Upload-Finish"] == ""
            assert b"videotest.mp4" in request.content
            assert b"synthetic video bytes" in request.content
            return httpx.Response(202, json={"rq_id": "upload-19"})
        if request.url.path == "/api/jobs":
            return httpx.Response(200, json={"count": 1, "results": [{"id": 23}]})
        raise AssertionError(request.url.path)

    client = CvatTaskClient(
        base_url="http://cvat.test",
        token="test",
        transport=httpx.MockTransport(handler),
    )
    task_id, job_id = asyncio.run(
        client.create_video_task(
            name="AutoLabel test",
            labels=[CvatLabelSpec(name="car")],
            video_path=video,
        )
    )
    assert (task_id, job_id) == (19, 23)
    assert len(requests) == 3
