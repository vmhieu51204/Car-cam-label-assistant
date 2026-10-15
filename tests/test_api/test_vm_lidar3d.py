import hashlib
import json

import httpx
import pytest

from src.services.vm.lidar3d_contracts import BEVFUSION_CHECKPOINT_SHA
from tests.test_api.test_vm_routes import vm_client as vm_client


@pytest.mark.asyncio
async def test_real_3d_job_requires_session_origin_and_ownership(vm_client, tmp_path, monkeypatch):
    client, app = vm_client
    settings = app.state.settings
    settings.lidar3d_enabled = True
    settings.bevfusion_enabled = True
    settings.lidar3d_dataset_root = tmp_path
    content = json.dumps(
        {
            "schema": "autolabel-nuscenes-vm/v1",
            "version": "v1.0-mini",
            "scenes": {"scene-0061": {"frames": [{"frame": 0}, {"frame": 1}]}},
            "files": {},
        }
    )
    (tmp_path / "dataset-manifest.json").write_text(content)
    settings.lidar3d_manifest_sha256 = hashlib.sha256(content.encode()).hexdigest()
    assert (await client.get("/api/v1/vm/3d/capabilities")).json()["enabled"]
    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original_client(
            **kwargs,
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200,
                    json={
                        "bevfusion_checkpoint_sha256": BEVFUSION_CHECKPOINT_SHA,
                        "dataset_manifest_sha256": settings.lidar3d_manifest_sha256,
                    },
                )
            ),
        ),
    )
    body = {"scene": "scene-0061", "max_frames": 2}
    assert (
        await client.post("/api/v1/vm/3d/jobs", json=body, headers={"Origin": "https://other.test"})
    ).status_code == 403
    response = await client.post("/api/v1/vm/3d/jobs", json=body)
    assert response.status_code == 200
    job = response.json()
    assert job["pipeline"] == "lidar3d" and job["total_frames"] == 2
    stored = app.state.vm_store.get(job["job_id"])
    assert stored["workflow"] == "lidar3d" and stored["profile"]["checkpoint_sha256"] == BEVFUSION_CHECKPOINT_SHA
    client.cookies.set("sessionid", "2")
    assert (await client.get(f"/api/v1/vm/3d/jobs/{job['job_id']}")).status_code == 404
    client.cookies.clear()
    assert (await client.get("/api/v1/vm/3d/capabilities")).status_code == 401
