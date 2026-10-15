"""VM reads must follow the caller's identity, including expired sessions."""

import httpx
import pytest

from src.config import Settings
from src.services.cvat.read_client import CvatReadError
from src.services.cvat.session_read_client import CvatSessionReadClient


@pytest.mark.asyncio
async def test_session_reads_are_isolated_and_preserve_public_host():
    seen = []

    def handler(request):
        assert request.headers["host"] == "public.example"
        assert "authorization" not in request.headers
        seen.append(request.headers["cookie"])
        return httpx.Response(200, json={"id": 7})

    settings = Settings(_env_file=None, cvat_url="http://internal:8080", cvat_public_url="https://public.example")
    for session in ("caller-one", "caller-two"):
        client = CvatSessionReadClient(settings, session, transport=httpx.MockTransport(handler))
        assert await client.get_job(7) == {"id": 7}
    assert seen == ["sessionid=caller-one", "sessionid=caller-two"]


@pytest.mark.asyncio
async def test_expired_session_does_not_fall_back_to_service_token():
    client = CvatSessionReadClient(
        Settings(_env_file=None),
        "expired",
        transport=httpx.MockTransport(lambda request: httpx.Response(401)),
    )
    with pytest.raises(CvatReadError) as error:
        await client.get_job(7)
    assert error.value.status_code == 401


@pytest.mark.parametrize("session", ["", "x" * 513])
def test_missing_or_oversized_session_is_rejected(session):
    with pytest.raises(CvatReadError) as error:
        CvatSessionReadClient(Settings(_env_file=None), session)
    assert error.value.status_code == 401
