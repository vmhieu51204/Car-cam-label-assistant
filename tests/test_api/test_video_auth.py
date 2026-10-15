"""Production video routes must honor the active CVAT user's job permissions."""

from datetime import UTC, datetime
from uuid import uuid4

import httpx
import pytest

from src.config import Settings
from src.main import create_app
from src.models.video import VideoJob, VideoJobStatus
from src.services.cvat.session_authorizer import CvatSessionAuthorizer


def _job(*, cvat_job_id: int | None = 7) -> VideoJob:
    now = datetime.now(UTC)
    return VideoJob(
        job_id=str(uuid4()),
        status=VideoJobStatus.QUEUED,
        source_name="clip.mp4",
        source_sha256="a" * 64,
        source_size_bytes=10,
        created_at=now,
        updated_at=now,
        stage="queued_for_extraction",
        cvat_job_id=cvat_job_id,
    )


@pytest.mark.asyncio
async def test_production_import_retry_requires_write_scope_and_is_not_duplicate(tmp_path):
    allowed = [False]

    def cvat_response(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/users/self":
            return httpx.Response(200, json={"id": 42})
        if request.url.path.endswith("/autolabel-permission"):
            return httpx.Response(200, json={"allowed": True}) if allowed[0] else httpx.Response(403)
        return httpx.Response(404)

    settings = Settings(
        _env_file=None,
        app_env="production",
        db_path=tmp_path / "app.sqlite3",
        video_work_root=tmp_path / "jobs",
        vlm_cache_path=tmp_path / "vlm.sqlite3",
        cvat_url="http://cvat.test",
        cvat_public_url="http://localhost:8080",
    )
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        app.state.cvat_authorizer = CvatSessionAuthorizer(settings, transport=httpx.MockTransport(cvat_response))
        failed = _job()
        failed.status = VideoJobStatus.FAILED
        failed.error_code = "video_extraction_internal_error"
        app.state.store.create_video_job(failed)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            url = f"/api/v1/video/jobs/{failed.job_id}/retry-import"
            assert (await client.post(url)).status_code == 401
            client.cookies.update({"sessionid": "valid-session", "csrftoken": "csrf"})
            headers = {"Origin": "http://localhost:8080", "X-CSRFToken": "csrf"}
            assert (await client.post(url, headers=headers)).status_code == 403
            allowed[0] = True
            retried = await client.post(url, headers=headers)
            assert retried.status_code == 202
            assert retried.json()["job_id"] == failed.job_id and retried.json()["status"] == "queued"
            assert (await client.post(url, headers=headers)).status_code == 409


@pytest.mark.asyncio
async def test_production_video_routes_use_cvat_session_and_job_scope(tmp_path):
    seen: list[httpx.Request] = []
    write_allowed = [False]

    def cvat_response(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert request.headers.get("authorization") is None
        assert "sessionid=valid-session" in request.headers.get("cookie", "")
        if request.url.path == "/api/users/self":
            return httpx.Response(200, json={"id": 42})
        if request.url.path == "/api/jobs/7":
            return httpx.Response(200, json={"id": 7})
        if request.url.path == "/api/jobs/7/autolabel-permission":
            return httpx.Response(200, json={"allowed": True}) if write_allowed[0] else httpx.Response(403)
        return httpx.Response(404)

    settings = Settings(
        _env_file=None,
        app_env="production",
        db_path=tmp_path / "app.sqlite3",
        video_work_root=tmp_path / "jobs",
        vlm_cache_path=tmp_path / "vlm.sqlite3",
        cvat_url="http://cvat_server:8080",
        cvat_public_url="http://localhost:8080",
    )
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        app.state.cvat_authorizer = CvatSessionAuthorizer(settings, transport=httpx.MockTransport(cvat_response))
        bound = _job()
        app.state.store.create_video_job(bound)
        unbound = _job(cvat_job_id=None)
        app.state.store.create_video_job(unbound)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            base = f"/api/v1/video/jobs/{bound.job_id}"
            assert (await client.get("/healthz")).status_code == 200
            assert (await client.get("/docs")).status_code == 404
            assert (await client.post("/api/v1/detect", json={})).status_code == 404
            assert (await client.get("/api/v1/demo/config")).status_code == 200
            assert (await client.get(base)).status_code == 401
            assert not seen

            client.cookies.update({"sessionid": "valid-session", "csrftoken": "csrf-value"})
            viewed = await client.get(base)
            assert viewed.status_code == 200
            assert viewed.json()["cvat_job_id"] == 7
            assert [request.url.path for request in seen] == ["/api/users/self", "/api/jobs/7"]
            assert seen[0].headers["host"] == "localhost"

            imported_lookup = await client.get("/api/v1/video/jobs/cvat-jobs/7")
            assert imported_lookup.status_code == 200
            capabilities = await client.get("/api/v1/video/jobs/cvat-jobs/7/capabilities")
            assert capabilities.status_code == 200
            assert capabilities.json()["sam3_1_experimental"] is False

            denied_origin = await client.post(base + "/cancel")
            assert denied_origin.status_code == 403
            denied_csrf = await client.post(base + "/cancel", headers={"Origin": "http://localhost:8080"})
            assert denied_csrf.status_code == 403
            denied_write = await client.post(
                base + "/cancel",
                headers={"Origin": "http://localhost:8080", "X-CSRFToken": "csrf-value"},
            )
            assert denied_write.status_code == 403
            assert seen[-1].url.path == "/api/jobs/7/autolabel-permission"
            assert app.state.store.get_video_job(bound.job_id).status == VideoJobStatus.QUEUED
            with app.state.store.connect() as connection:
                assert connection.execute("SELECT COUNT(*) FROM video_user_intents").fetchone()[0] == 0

            write_allowed[0] = True
            cancelled = await client.post(
                base + "/cancel",
                headers={"Origin": "http://localhost:8080", "X-CSRFToken": "csrf-value"},
            )
            assert cancelled.status_code == 200
            assert cancelled.json()["status"] == "cancelled"
            with app.state.store.connect() as connection:
                intent = connection.execute(
                    "SELECT cvat_user_id, cvat_job_id, video_job_id, method, http_status, completed_at "
                    "FROM video_user_intents"
                ).fetchone()
            assert intent[:5] == (42, 7, bound.job_id, "POST", 200)
            assert intent[5] is not None

            assert (await client.get(f"/api/v1/video/jobs/{unbound.job_id}")).status_code == 403
            assert (await client.post("/api/v1/video/jobs")).status_code == 403
