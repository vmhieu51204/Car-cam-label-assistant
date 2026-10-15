from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from src.models.video import CvatAttributeSpec, CvatLabelSpec
from src.services.cvat import CvatTaskClient, CvatWriteError


@pytest.mark.asyncio
async def test_prepare_review_attributes_adds_only_missing_names():
    required = CvatAttributeSpec(
        name="review_status",
        mutable=True,
        input_type="select",
        default_value="pending",
        values=["pending", "accepted", "rejected"],
    )
    observed = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "PATCH" and request.url.path == "/api/labels/11"
        observed.append(json.loads(request.content))
        return httpx.Response(200, json={"id": 11})

    client = CvatTaskClient(base_url="http://cvat.test", token="secret", transport=httpx.MockTransport(handler))
    labels = [
        {
            "id": 11,
            "name": "car",
            "type": "rectangle",
            "attributes": [
                {"id": 91, "name": "color", "input_type": "text", "mutable": False, "values": []},
            ],
        }
    ]
    await client.ensure_label_attributes(labels, [required])
    assert observed == [{"attributes": [required.model_dump()]}]
    labels[0]["attributes"].append({"id": 92, **required.model_dump()})
    await client.ensure_label_attributes(labels, [required])
    assert len(observed) == 1  # Repeating setup makes no write.
    labels[0]["attributes"][-1]["mutable"] = False
    with pytest.raises(ValueError, match="incompatible attribute"):
        await client.ensure_label_attributes(labels, [required])
    assert len(observed) == 1


@pytest.mark.asyncio
async def test_prepare_review_attributes_accepts_cvat_empty_text_normalization():
    required = CvatAttributeSpec(
        name="autolabel_track_id", mutable=False, input_type="text", default_value="", values=[]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("normalized existing attribute must not be patched")

    client = CvatTaskClient(base_url="http://cvat.test", token="secret", transport=httpx.MockTransport(handler))
    labels = [
        {
            "id": 11,
            "name": "car",
            "type": "any",
            "attributes": [
                {
                    "id": 92,
                    "name": "autolabel_track_id",
                    "input_type": "text",
                    "mutable": False,
                    "default_value": "",
                    "values": [""],
                },
            ],
        }
    ]
    await client.ensure_label_attributes(labels, [required])


@pytest.mark.asyncio
async def test_cvat_task_client_creates_one_job_from_ordered_frames(tmp_path: Path):
    frames = [tmp_path / "frame_000000.jpg", tmp_path / "frame_000001.jpg"]
    for index, frame in enumerate(frames):
        frame.write_bytes(f"jpeg-{index}".encode())
    job_reads = 0
    observed_upload = b""

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal job_reads, observed_upload
        assert request.headers["authorization"] == "Token secret"
        assert request.headers["host"] == "localhost"
        assert request.headers["accept"] == "application/vnd.cvat+json"
        if request.method == "POST" and request.url.path == "/api/tasks":
            payload = json.loads(await request.aread())
            assert payload["name"] == "Demo task"
            assert payload["segment_size"] == 2
            assert payload["labels"] == [{"name": "car", "type": "rectangle"}]
            return httpx.Response(201, json={"id": 41})
        if request.method == "POST" and request.url.path == "/api/tasks/41/data":
            assert "Upload-Start" in request.headers
            assert "Upload-Finish" in request.headers
            observed_upload = await request.aread()
            return httpx.Response(202, json={"rq_id": "upload-1"})
        if request.method == "GET" and request.url.path == "/api/jobs":
            job_reads += 1
            results = [] if job_reads == 1 else [{"id": 73}]
            return httpx.Response(200, json={"count": len(results), "results": results})
        raise AssertionError(f"unexpected CVAT request {request.method} {request.url}")

    client = CvatTaskClient(
        base_url="http://cvat.test",
        token="secret",
        host_header="localhost",
        ready_timeout_s=1,
        retry_delay_s=0,
        transport=httpx.MockTransport(handler),
    )

    task_id, job_id = await client.create_image_task(
        name="Demo task",
        labels=[CvatLabelSpec(name="car")],
        frame_paths=frames,
    )

    assert (task_id, job_id) == (41, 73)
    assert b'name="client_files[0]"; filename="frame_000000.jpg"' in observed_upload
    assert b'name="client_files[1]"; filename="frame_000001.jpg"' in observed_upload
    assert b"lexicographical" in observed_upload


@pytest.mark.asyncio
async def test_cvat_task_client_reports_possible_orphan_after_upload_failure(tmp_path: Path):
    frame = tmp_path / "frame_000000.jpg"
    frame.write_bytes(b"jpeg")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tasks":
            return httpx.Response(201, json={"id": 41})
        return httpx.Response(500)

    client = CvatTaskClient(
        base_url="http://cvat.test",
        token="secret",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(CvatWriteError, match="HTTP 500") as captured:
        await client.create_image_task(
            name="Demo task",
            labels=[CvatLabelSpec(name="car")],
            frame_paths=[frame],
        )

    assert captured.value.task_id == 41
