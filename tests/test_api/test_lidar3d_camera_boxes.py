import json

import httpx
import pytest

from tests.test_api.test_vm_routes import vm_client as vm_client


@pytest.mark.asyncio
async def test_camera_boxes_use_saved_native_detections_and_cvat_read_permission(vm_client, monkeypatch):
    client, app = vm_client
    store = app.state.vm_store
    job = store.create(99, {"workflow": "lidar3d"}, "private-credentials")
    store.update(job["job_id"], status="completed", cvat_job_id=5, total_frames=2)
    directory = store.directory(job["job_id"]) / "private/lidar3d"
    directory.mkdir()
    observation = {
        "detector_bbox": [10, 20, 100, 200],
        "prompt_bbox": [11, 22, 99, 199],
        "label": "car",
        "score": 0.9,
        "track_id": "camera:1",
    }
    payload = {
        "camera_tracks": [
            {"frame": 0, "camera": "CAM_FRONT", "observations": [observation]},
            {"frame": 0, "camera": "CAM_BACK", "observations": [observation]},
            {"frame": 1, "camera": "CAM_FRONT", "observations": [{**observation, "label": "truck"}]},
        ],
        "links": [
            {"frame": 0, "camera": "CAM_FRONT", "status": "matched", "track_2d_id": "camera:1", "track_3d_id": "3d:7"}
        ],
    }
    (directory / "camera-links.json").write_text(json.dumps(payload))
    allowed = True
    calls = []

    def respond(request):
        calls.append(request.url.path)
        return httpx.Response(200 if allowed else 403, json={"id": 5})

    monkeypatch.setattr(
        "src.api.lidar3d_upload.cvat_client",
        lambda request: httpx.AsyncClient(base_url="http://cvat", transport=httpx.MockTransport(respond)),
    )
    path = "/api/v1/vm/3d/cvat-jobs/5/camera-boxes"
    response = await client.get(path + "?frame=0")
    assert response.status_code == 200  # CVAT reviewer need not be the VM upload owner.
    data = response.json()
    assert data["detector"] == "YOLO26s" and data["coordinates"] == "native_image_pixels"
    front, back = data["cameras"]
    assert front["boxes"][0]["bbox"] == observation["detector_bbox"]
    assert front["boxes"][0]["linked_3d_track_ids"] == ["3d:7"]
    assert back["boxes"][0]["linked_3d_track_ids"] == []
    assert (await client.get(path + "?frame=1")).json()["cameras"][0]["boxes"][0]["label"] == "truck"
    assert (await client.get(path + "?frame=2")).status_code == 404
    assert (await client.get(path + "?frame=-1")).status_code == 422
    allowed = False
    assert (await client.get(path + "?frame=0")).status_code == 404
    assert set(calls) == {"/api/jobs/5"}
    client.cookies.clear()
    assert (await client.get(path + "?frame=0")).status_code == 401
