import json
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio
from cryptography.fernet import Fernet

from src.config import Settings
from src.main import create_app
from src.services.vm.contracts import VmProfile


@pytest_asyncio.fixture
async def vm_client(tmp_path):
    settings = Settings(
        _env_file=None,
        app_env="production",
        vm_workflow_enabled=True,
        db_path=tmp_path / "db.sqlite3",
        vm_work_root=tmp_path / "vm",
        video_work_root=tmp_path / "old",
        vlm_cache_path=tmp_path / "vlm.sqlite3",
        vlm_settings_encryption_key=Fernet.generate_key().decode(),
    )
    app = create_app(settings)
    async with app.router.lifespan_context(app):

        async def identity(session):
            return int(session)

        async def authorize(session, job, **kwargs):
            return int(session)

        app.state.cvat_authorizer = SimpleNamespace(identity=identity, authorize=authorize)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            cookies={"sessionid": "1", "csrftoken": "csrf"},
            headers={"Origin": "http://localhost:8080", "X-CSRFToken": "csrf"},
        ) as client:
            yield client, app


@pytest.mark.asyncio
async def test_settings_auth_origin_csrf_and_masked_keys(vm_client):
    client, app = vm_client
    assert (await client.get("/api/v1/settings/vlm")).status_code == 200
    body = {"base_url": "https://example.com/v1", "model": "vision", "api_key": "private-test-key"}
    assert (
        await client.put("/api/v1/settings/vlm", json=body, headers={"Origin": "https://other.example"})
    ).status_code == 403
    assert (await client.put("/api/v1/settings/vlm", json=body, headers={"X-CSRFToken": "wrong"})).status_code == 403
    saved = await client.put("/api/v1/settings/vlm", json=body)
    assert saved.status_code == 200 and "private-test-key" not in saved.text and saved.json()["api_key"] == "********"
    snapshot = app.state.vlm_profiles.snapshot(1)
    await client.delete("/api/v1/settings/vlm")
    assert app.state.vlm_profiles.auditor(snapshot).model == "vision"
    assert b"private-test-key" not in app.state.settings.db_path.read_bytes()
    client.cookies.clear()
    assert (await client.get("/api/v1/settings/vlm")).status_code == 401
    assert (await client.get("/api/v1/vm/capabilities")).json()["enabled"]


@pytest.mark.asyncio
async def test_preflight_blocks_upload_and_raw_legacy_import(vm_client):
    client, app = vm_client
    response = await client.post(
        "/api/v1/vm/jobs",
        data={"options": json.dumps({"classes": ["car"]})},
        files={"video": ("test.mp4", b"fixture", "video/mp4")},
    )
    assert response.status_code == 422
    with app.state.vm_store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM vm_jobs").fetchone()[0] == 0
    assert (await client.post("/api/v1/video/jobs/cvat-jobs/1/import", json={})).status_code == 409


@pytest.mark.asyncio
async def test_jobs_owner_isolation_no_raw_route_and_retry_credentials(vm_client, monkeypatch):
    client, app = vm_client
    profile = VmProfile(checkpoint="sam3.1_multiplex.pt", checkpoint_sha256="a" * 64, source_commit="b" * 40)
    monkeypatch.setattr("src.api.vm.preflight", lambda *a: profile)
    response = await client.post(
        "/api/v1/vm/jobs",
        data={"options": json.dumps({"classes": ["car"]})},
        files={"video": ("test.mp4", b"private-raw-fixture", "video/mp4")},
    )
    assert response.status_code == 200
    job = response.json()["job_id"]
    assert "source" not in response.text and "snapshot" not in response.text
    assert (await client.get(f"/api/v1/vm/jobs/{job}/video")).status_code == 409
    assert (await client.get(f"/api/v1/vm/jobs/{job}/source")).status_code == 404
    client.cookies.set("sessionid", "2")
    assert (await client.get(f"/api/v1/vm/jobs/{job}/processing")).status_code == 404
    client.cookies.set("sessionid", "1")
    app.state.vm_store.update(job, status="failed")
    previous = app.state.vm_store.credentials(job)
    assert (await client.post(f"/api/v1/vm/jobs/{job}/retry")).status_code == 200
    assert app.state.vm_store.credentials(job) != previous
    processing = await client.get(f"/api/v1/video/jobs/{job}/processing")
    assert processing.status_code == 200
    app.state.vm_store.update(job, status="failed", total_frames=3, failed_frames=[0, 1, 2])
    app.state.vm_store.event(job, "result:0:car", {"frame": 0, "class": "car"})
    processing = await client.get(f"/api/v1/video/jobs/{job}/processing")
    assert processing.json()["frame_states"] == [2, 3, 3]
    source = app.state.vm_store.directory(job) / "private/source.mp4"
    assert source.read_bytes() == b"private-raw-fixture"


@pytest.mark.asyncio
async def test_completed_video_can_explicitly_bind_current_private_vlm_profile(vm_client, monkeypatch):
    client, app = vm_client
    store = app.state.vm_store
    job = store.create(
        1, {"profile": {"variant": "sam3"}, "vlm_snapshot": app.state.vlm_profiles.snapshot(1)}, "sealed"
    )
    path = f"/api/v1/vm/jobs/{job['job_id']}/vlm/profile"
    assert (await client.post(path)).status_code == 409
    store.update(job["job_id"], status="completed")
    monkeypatch.setattr("src.api.vm.clean_receipt", lambda *args: None)
    assert (await client.post(path)).status_code == 409  # No accidental system/other-user key.
    await client.put(
        "/api/v1/settings/vlm",
        json={
            "base_url": "https://openrouter.ai/api/v1",
            "model": "anthropic/claude-sonnet-5.5",
            "api_key": "private-test-key",
        },
    )
    assert (await client.post(path, headers={"X-CSRFToken": "wrong"})).status_code == 403
    response = await client.post(path)
    assert response.status_code == 200 and "private-test-key" not in response.text
    snapshot = store.get(job["job_id"])["vlm_snapshot"]
    assert app.state.vlm_profiles.auditor(snapshot).endpoint == "https://openrouter.ai/api/v1/chat/completions"
    client.cookies.set("sessionid", "2")
    assert (await client.post(path)).status_code == 404
    client.cookies.set("sessionid", "1")
    store.update(job["job_id"], workflow="lidar3d")
    assert (await client.post(path)).status_code == 409


def cvat_source_client(app, status=200, body=b"cvat-stored-video", filename="clip.MP4", labels=None, start_frame=0):
    from src.services.cvat.session_read_client import CvatSessionReadClient

    def handler(request):
        if request.url.path == "/api/labels":
            names = ["car", "person"] if labels is None else labels
            return httpx.Response(200, json={"results": [{"id": i + 1, "name": name} for i, name in enumerate(names)], "next": None})
        if request.url.path == "/api/tasks/19":
            return httpx.Response(200, json={"size": 3})
        if request.url.path == "/api/jobs/7/data/meta":
            return httpx.Response(200, json={"size": 3, "start_frame": start_frame, "deleted_frames": []})
        if request.url.path == "/api/jobs/7":
            return httpx.Response(200, json={"id": 7, "task_id": 19, "start_frame": start_frame, "stop_frame": 2})
        assert request.url.path == "/api/jobs/7/autolabel-source"
        assert "sessionid=1" in request.headers["cookie"]
        if status != 200:
            return httpx.Response(status, json={"detail": "internal CVAT detail"})
        return httpx.Response(200, content=body, headers={"Content-Disposition": f'attachment; filename="{filename}"'})

    return lambda request: CvatSessionReadClient(
        app.state.settings, request.cookies["sessionid"], transport=httpx.MockTransport(handler)
    )


@pytest.mark.asyncio
async def test_run_from_cvat_job_reuses_stored_video_without_upload(vm_client, monkeypatch):
    client, app = vm_client
    profile = VmProfile(checkpoint="sam3.1_multiplex.pt", checkpoint_sha256="a" * 64, source_commit="b" * 40)
    monkeypatch.setattr("src.api.vm.preflight", lambda *a: profile)
    monkeypatch.setattr("src.api.vm.session_read", cvat_source_client(app))
    response = await client.post("/api/v1/vm/cvat-jobs/7/runs", json={"classes": ["car", "person"]})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "queued" and payload["origin_cvat_job_id"] == 7
    assert "source" not in response.text
    job = app.state.vm_store.get(payload["job_id"])
    assert job["stage"] == "queued_privacy" and job["source_extension"] == ".mp4"
    assert job["in_place"] is True
    assert (job["cvat_task_id"], job["cvat_job_id"]) == (19, 7)
    source = app.state.vm_store.directory(payload["job_id"]) / "private/source.mp4"
    assert source.read_bytes() == b"cvat-stored-video"
    found = await client.get("/api/v1/vm/cvat-jobs/7")
    assert found.json()["job"]["job_id"] == payload["job_id"]
    client.cookies.set("sessionid", "2")
    assert (await client.get("/api/v1/vm/cvat-jobs/7")).json()["job"] is None


@pytest.mark.asyncio
async def test_run_from_cvat_job_without_stored_video_fails_cleanly(vm_client, monkeypatch):
    client, app = vm_client
    profile = VmProfile(checkpoint="sam3.1_multiplex.pt", checkpoint_sha256="a" * 64, source_commit="b" * 40)
    monkeypatch.setattr("src.api.vm.preflight", lambda *a: profile)
    monkeypatch.setattr("src.api.vm.session_read", cvat_source_client(app, status=409))
    response = await client.post("/api/v1/vm/cvat-jobs/7/runs", json={"classes": ["car"]})
    assert response.status_code == 409
    assert "internal CVAT detail" not in response.text and "no stored source video" in response.text
    with app.state.vm_store.connect() as db:
        (job,) = db.execute("SELECT id FROM vm_jobs").fetchone()
    assert app.state.vm_store.get(job)["stage"] == "upload_failed"
    assert not list((app.state.vm_store.directory(job) / "private").glob("source*"))
    assert (await client.post(f"/api/v1/vm/jobs/{job}/retry")).status_code == 409
    monkeypatch.setattr("src.api.vm.session_read", cvat_source_client(app, filename="clip.exe"))
    assert (await client.post("/api/v1/vm/cvat-jobs/7/runs", json={"classes": ["car"]})).status_code == 422


@pytest.mark.asyncio
async def test_refine_route_checks_owner_box_and_hint_before_any_vlm_call(vm_client):
    client, app = vm_client
    job = app.state.vm_store.create(1, {"profile": {"variant": "sam3"}}, "sealed")["job_id"]
    path = f"/api/v1/vm/jobs/{job}/vlm/refine"
    body = {"source_frame": 0, "label": "car", "bbox": [0.1, 0.1, 0.4, 0.2]}
    assert (await client.post("/api/v1/vm/jobs/missing/vlm/refine", json=body)).status_code == 404
    assert (await client.post(path, json={**body, "bbox": [0.5, 0.1, 0.4, 0.2]})).status_code == 422
    assert (await client.post(path, json={**body, "bbox": [0.1, 0.1, 1.4, 0.2]})).status_code == 422
    assert (await client.post(path, json={**body, "instruction": "x" * 501})).status_code == 422
    # Without a verified redacted video the frame is never read or sent anywhere.
    assert (await client.post(path, json=body)).status_code == 409


@pytest.mark.asyncio
async def test_run_from_cvat_job_requires_write_permission(vm_client):
    from unittest.mock import AsyncMock

    from src.services.cvat.session_authorizer import CvatAuthorizationError

    client, app = vm_client
    authorize = AsyncMock(side_effect=CvatAuthorizationError(403, "Read only"))
    app.state.cvat_authorizer.authorize = authorize
    response = await client.post("/api/v1/vm/cvat-jobs/7/runs", json={"classes": ["car"]})
    assert response.status_code == 403
    authorize.assert_awaited_once_with("1", 7, write=True)
    with app.state.vm_store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM vm_jobs").fetchone()[0] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("labels", "expected_pipeline"),
    [
        (["carr", "truck", "container", "person"], "sam"),
        ([f"class {i}" for i in range(10)], "sam"),
        (["car", "person"], "hybrid_frame"),
    ],
)
async def test_current_job_uses_every_cvat_label_not_request_defaults(vm_client, monkeypatch, labels, expected_pipeline):
    client, app = vm_client
    calls = []

    def preflight(settings, variant, pipeline, classes):
        calls.append((variant, pipeline, classes))
        return VmProfile(checkpoint="sam3.pt", checkpoint_sha256="a" * 64, source_commit="b" * 40)

    monkeypatch.setattr("src.api.vm.preflight", preflight)
    monkeypatch.setattr("src.api.vm.session_read", cvat_source_client(app, labels=labels))
    response = await client.post("/api/v1/vm/cvat-jobs/7/runs", json={"classes": ["car", "person"]})
    assert response.status_code == 200
    assert response.json()["classes"] == labels
    assert calls == [("sam3", expected_pipeline, labels)]
    stored = app.state.vm_store.get(response.json()["job_id"])
    assert stored["classes"] == labels
    assert (stored["cvat_task_id"], stored["cvat_job_id"]) == (19, 7)


@pytest.mark.asyncio
async def test_current_job_with_no_labels_is_rejected_before_creating_run(vm_client, monkeypatch):
    client, app = vm_client
    monkeypatch.setattr("src.api.vm.session_read", cvat_source_client(app, labels=[]))
    response = await client.post("/api/v1/vm/cvat-jobs/7/runs", json={})
    assert response.status_code == 422
    assert "at least one label" in response.text
    with app.state.vm_store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM vm_jobs").fetchone()[0] == 0


@pytest.mark.asyncio
async def test_privacy_rerun_preserves_existing_cvat_target(vm_client, monkeypatch):
    client, app = vm_client
    store = app.state.vm_store
    job = store.create(1, {
        "in_place": True, "cvat_task_id": 19, "cvat_job_id": 7, "origin_cvat_job_id": 7,
        "classes": ["car"], "profile": {"variant": "sam3"}, "vlm_snapshot": {}, "vlm_enabled": False,
        "source_extension": ".mp4", "privacy_profile": {}, "source_sha256": "a" * 64, "manual_regions": [],
    }, "sealed")
    source = store.directory(job["job_id"]) / "private/source.mp4"
    source.write_bytes(b"source video")
    monkeypatch.setattr("src.api.vm.verified_media", lambda *args: (None, {
        "timestamps": [0.0, 0.1], "width": 64, "height": 64, "total_frames": 2,
    }))
    response = await client.post(f"/api/v1/vm/jobs/{job['job_id']}/privacy-regions", json={
        "regions": [{"start": 0, "end": 1, "box": [1, 1, 10, 10]}],
    })
    assert response.status_code == 200
    fresh = store.get(response.json()["job_id"])
    assert fresh["in_place"] is True
    assert (fresh["cvat_task_id"], fresh["cvat_job_id"]) == (19, 7)
    assert fresh["origin_cvat_job_id"] == 7
    assert (store.directory(fresh["job_id"]) / "private/source.mp4").read_bytes() == source.read_bytes()


@pytest.mark.asyncio
async def test_segmented_job_is_rejected_before_queuing_or_downloading(vm_client, monkeypatch):
    client, app = vm_client
    monkeypatch.setattr("src.api.vm.session_read", cvat_source_client(app, start_frame=1))
    response = await client.post("/api/v1/vm/cvat-jobs/7/runs", json={})
    assert response.status_code == 409
    assert "full-video job" in response.text
    with app.state.vm_store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM vm_jobs").fetchone()[0] == 0
