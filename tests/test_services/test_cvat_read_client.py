import json

import httpx
import pytest
from pydantic import SecretStr

from src.config import Settings
from src.services.cvat import CvatReadClient, CvatReadError


@pytest.mark.asyncio
async def test_capture_snapshot_uses_get_only_and_keeps_token_out_of_output():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        payloads = {
            "/api/jobs/7": {"id": 7, "task_id": 2},
            "/api/jobs/7/data/meta": {"size": 39},
            "/api/jobs/7/annotations": {"version": 1, "shapes": [], "tracks": [], "tags": []},
        }
        return httpx.Response(200, json=payloads[request.url.path])

    client = CvatReadClient(
        base_url="http://cvat.test",
        token=SecretStr("top-secret"),
        transport=httpx.MockTransport(handler),
        retry_delay_s=0,
    )
    snapshot = await client.capture_job_snapshot(7)

    assert [request.method for request in requests] == ["GET", "GET", "GET"]
    assert [request.url.path for request in requests] == [
        "/api/jobs/7",
        "/api/jobs/7/data/meta",
        "/api/jobs/7/annotations",
    ]
    assert all(request.headers["Authorization"] == "Token top-secret" for request in requests)
    assert snapshot["write_operations_performed"] is False
    assert snapshot["data_meta"]["size"] == 39
    assert "top-secret" not in json.dumps(snapshot)


@pytest.mark.asyncio
async def test_docker_transport_preserves_public_cvat_host():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "host.docker.internal"
        assert request.headers["host"] == "localhost"
        assert request.headers["accept"] == "application/vnd.cvat+json"
        return httpx.Response(200, json={"id": 7})

    settings = Settings(
        _env_file=None,
        cvat_url="http://host.docker.internal:8080",
        cvat_public_url="http://localhost:8080",
        cvat_token=SecretStr("demo-token"),
    )
    client = CvatReadClient.from_settings(settings, transport=httpx.MockTransport(handler))
    assert await client.get_job(7) == {"id": 7}


@pytest.mark.asyncio
async def test_retryable_response_is_retried_with_a_bound():
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(503 if attempts < 3 else 200, json={"version": 0})

    client = CvatReadClient(
        base_url="http://cvat.test",
        token="abc",
        retry_attempts=3,
        retry_delay_s=0,
        transport=httpx.MockTransport(handler),
    )

    assert await client.get_job_annotations(1) == {"version": 0}
    assert attempts == 3


@pytest.mark.asyncio
async def test_auth_failure_is_sanitized_and_not_retried():
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(401, text="server body with implementation details")

    client = CvatReadClient(
        base_url="http://cvat.test",
        token="do-not-leak",
        retry_attempts=5,
        retry_delay_s=0,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(CvatReadError) as captured:
        await client.get_job_annotations(1)

    assert captured.value.status_code == 401
    assert attempts == 1
    assert "do-not-leak" not in str(captured.value)
    assert "implementation details" not in str(captured.value)


@pytest.mark.asyncio
async def test_missing_token_fails_before_network_access():
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("Network must not be called without a token")

    client = CvatReadClient(
        base_url="http://cvat.test", token="", transport=httpx.MockTransport(handler), retry_delay_s=0
    )
    with pytest.raises(CvatReadError, match="not configured"):
        await client.get_job_annotations(1)


@pytest.mark.parametrize("url", ["file:///tmp/cvat", "http://user:pass@cvat.test", "cvat.test"])
def test_base_url_rejects_unsupported_or_embedded_credentials(url):
    with pytest.raises(ValueError, match="base URL"):
        CvatReadClient(base_url=url, token="abc")


@pytest.mark.asyncio
async def test_job_labels_reads_every_page_without_following_server_urls():
    pages: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        page = request.url.params["page"]
        pages.append(page)
        if page == "1":
            return httpx.Response(
                200,
                json={
                    "count": 2,
                    "next": "http://untrusted.example/api/labels?page=2",
                    "results": [{"id": 1, "name": "car"}],
                },
            )
        return httpx.Response(200, json={"count": 2, "next": None, "results": [{"id": 2, "name": "person"}]})

    client = CvatReadClient(
        base_url="http://cvat.test",
        token="abc",
        retry_delay_s=0,
        transport=httpx.MockTransport(handler),
    )

    labels = await client.get_job_labels(7)

    assert pages == ["1", "2"]
    assert [label["id"] for label in labels] == [1, 2]
