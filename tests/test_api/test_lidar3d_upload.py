import hashlib
import io
from copy import deepcopy
from zipfile import ZipFile

import httpx
import pytest

from src.services.vm.lidar3d_review import revision
from tests.test_api.test_vm_routes import vm_client as vm_client


@pytest.mark.asyncio
async def test_dataset_upload_is_chunked_idempotent_bounded_and_owner_scoped(vm_client, monkeypatch):
    client, app = vm_client
    app.state.settings.lidar3d_enabled = True
    monkeypatch.setattr("src.api.lidar3d_upload.CHUNK_BYTES", 16)
    data = io.BytesIO()
    with ZipFile(data, "w") as archive:
        archive.writestr("file.txt", "native fixture")
    content = data.getvalue()
    body = {"filename": "kitti.zip", "size": len(content), "dataset_format": "kitti", "mode": "import"}
    assert (
        await client.post("/api/v1/vm/3d/uploads", json=body, headers={"Origin": "https://other.test"})
    ).status_code == 403
    response = await client.post("/api/v1/vm/3d/uploads", json=body)
    assert response.status_code == 200
    identity = response.json()["job_id"]
    path = f"/api/v1/vm/3d/uploads/{identity}"
    assert (await client.post(path + "/complete")).status_code == 409
    assert (await client.put(path + "?offset=0", content=b"a" * 17)).status_code == 413
    client.cookies.set("sessionid", "2")
    assert (await client.put(path + "?offset=0", content=content[:16])).status_code == 404
    client.cookies.set("sessionid", "1")
    assert (await client.put(path + "?offset=1", content=content[:16])).status_code == 409
    for offset in range(0, len(content), 16):
        chunk = content[offset : offset + 16]
        assert (await client.put(path + f"?offset={offset}", content=chunk)).status_code == 200
        assert (await client.put(path + f"?offset={offset}", content=chunk)).status_code == 200
    assert (await client.put(path + "?offset=0", content=b"different")).status_code == 409
    completed = await client.post(path + "/complete")
    assert completed.status_code == 200 and completed.json()["status"] == "queued"
    job = app.state.vm_store.get(identity)
    assert job["source_sha256"] == hashlib.sha256(content).hexdigest()
    assert (app.state.vm_store.directory(identity) / "private/dataset.archive").read_bytes() == content
    assert (await client.put(path + "?offset=0", content=content[:16])).status_code == 409
    assert (await client.post(path + "/complete")).status_code == 200


@pytest.mark.asyncio
async def test_kitti_does_not_silently_run_nuscenes_checkpoint(vm_client):
    client, app = vm_client
    app.state.settings.lidar3d_enabled = True
    response = await client.post(
        "/api/v1/vm/3d/uploads",
        json={"filename": "kitti.zip", "size": 100, "dataset_format": "kitti", "mode": "detect"},
    )
    assert response.status_code == 422 and "KITTI-trained" in response.json()["detail"]
    with app.state.vm_store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM vm_jobs").fetchone()[0] == 0


@pytest.mark.asyncio
async def test_3d_review_is_owner_scoped_and_requires_atomic_content_revision(vm_client, monkeypatch):
    client, app = vm_client
    job = app.state.vm_store.create(1, {"workflow": "lidar3d", "profile": {"variant": "bevfusion"}}, "sealed")
    app.state.vm_store.update(job["job_id"], status="completed", cvat_job_id=10, cvat_task_id=10)
    data = {
        "version": 0,
        "tags": [],
        "tracks": [],
        "shapes": [
            {
                "id": 20,
                "label_id": 1,
                "frame": 0,
                "type": "cuboid",
                "points": [1, 2, 3, 0, 0, 0, 4, 2, 2, *([0] * 7)],
                "attributes": [{"spec_id": 5, "value": "pending"}],
            }
        ],
    }
    labels = [{"id": 1, "name": "car", "attributes": [{"id": 5, "name": "review_status"}]}]
    posts = []

    def respond(request):
        if request.url.path == "/api/labels":
            return httpx.Response(200, json={"results": labels})
        if request.method == "GET":
            return httpx.Response(200, json=data)
        import json

        body = json.loads(request.content)
        posts.append(body)
        assert body["require_revision"] is True
        assert body["expected_revision"] == revision(data)
        assert body["base"] == data
        assert body["local"]["shapes"][0]["points"] == data["shapes"][0]["points"]
        assert body["local"]["shapes"][0]["attributes"] == [{"spec_id": 5, "value": "accepted"}]
        # Model a save arriving after this API's GET but before CVAT takes the row lock.
        return httpx.Response(409, json={"detail": "CVAT annotations changed"})

    monkeypatch.setattr(
        "src.api.lidar3d_upload.cvat_client",
        lambda request: httpx.AsyncClient(base_url="http://cvat", transport=httpx.MockTransport(respond)),
    )
    path = f"/api/v1/vm/3d/jobs/{job['job_id']}/review"
    listed = await client.get(path)
    assert listed.json()["pending"] == 1 and listed.json()["objects"][0]["object_id"] == "shapes:20"
    body = {"object_id": "shapes:20", "status": "accepted", "expected_revision": listed.json()["revision"]}
    assert (await client.post(path, json=body)).status_code == 409
    assert len(posts) == 1
    previous = deepcopy(data)
    data["shapes"][0]["points"][0] = 99
    assert (await client.post(path, json=body)).status_code == 409 and len(posts) == 1
    assert previous["shapes"][0]["attributes"] == data["shapes"][0]["attributes"]
    client.cookies.set("sessionid", "2")
    assert (await client.get(path)).status_code == 404
