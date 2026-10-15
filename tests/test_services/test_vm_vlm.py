import json
import socket

import httpx
import pytest
from cryptography.fernet import Fernet
from pydantic import SecretStr

from src.config import Settings
from src.services.vm.storage import VmStore
from src.services.vm.vlm import PinnedTransport, VlmProfiles, endpoint


def profiles(tmp_path):
    settings = Settings(
        _env_file=None,
        vlm_settings_encryption_key=Fernet.generate_key().decode(),
        vlm_cache_path=tmp_path / "cache.sqlite3",
        openrouter_api_key="system-secret",
    )
    return VlmProfiles(VmStore(tmp_path / "state.sqlite3", tmp_path / "jobs"), settings)


def test_keys_are_encrypted_isolated_and_snapshotted(tmp_path):
    repo = profiles(tmp_path)
    saved = repo.save(1, model="custom-vision", base_url="https://example.com/v1", key="private-api-secret")
    assert saved["api_key"] == "********"
    assert repo.current(2)["mode"] == "system"
    assert not repo.public(repo.current(2))["key_configured"]
    with pytest.raises(ValueError, match="own VLM API key"):
        repo.auditor(repo.snapshot(2), transport=httpx.MockTransport(lambda _r: pytest.fail("No request permitted")))
    with pytest.raises(ValueError):
        repo.resolve(2, saved["version"])
    snapshot = repo.snapshot(1)
    repo.save(1, model="new-model", base_url="https://example.com/v1", key="new-key")
    assert (
        repo.auditor(snapshot, transport=httpx.MockTransport(lambda _r: httpx.Response(200))).model == "custom-vision"
    )
    assert b"private-api-secret" not in repo.store.path.read_bytes()
    repo.delete(1)
    assert repo.current(1)["mode"] == "system"
    assert repo.resolve(1, saved["version"])["model"] == "custom-vision"


@pytest.mark.parametrize(
    "url", ["http://example.com/v1", "https://key@example.com/v1", "https://example.com/v1?key=secret"]
)
def test_urls_with_leaky_credentials_rejected(url):
    with pytest.raises(ValueError):
        endpoint(url)


def test_dns_private_and_mixed_addresses_blocked(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **kw: [(2, 1, 6, "", ("127.0.0.1", 443))])
    transport = PinnedTransport()
    with pytest.raises(ValueError, match="forbidden"):
        transport.handle_request(httpx.Request("POST", "https://example.com/v1/chat/completions"))
    transport.close()


def test_compatible_api_uses_model_and_strict_vision_output(tmp_path):
    from PIL import Image

    from src.services.vm.contracts import file_sha256

    repo = profiles(tmp_path)
    repo.save(1, model="other-vlm", base_url="https://example.com/v1", key="test-key")

    def respond(request):
        payload = json.loads(request.content)
        assert payload["model"] == "other-vlm" and "provider" not in payload
        assert payload["messages"][0]["content"][-1]["type"] == "image_url"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": json.dumps({"missing": "no", "proposals": [], "reason": "No object"})}}
                ],
                "usage": {},
            },
        )

    image = tmp_path / "clean.jpg"
    Image.new("RGB", (64, 64)).save(image)
    auditor = repo.auditor(repo.snapshot(1), transport=httpx.MockTransport(respond))
    answer, _usage, _cached = auditor.propose_missing_boxes(
        overlay_path=image,
        overlay_sha256=file_sha256(image),
        classes=("car",),
        image_width=64,
        image_height=64,
        existing_boxes=[],
    )
    assert not answer.proposals


def test_default_vlm_uses_private_operator_key_without_storing_it(tmp_path):
    settings = Settings(
        _env_file=None,
        vlm_settings_encryption_key=Fernet.generate_key().decode(),
        vlm_cache_path=tmp_path / "cache.sqlite3",
        vlm_default_api_key="default-operator-secret",
    )
    repo = VlmProfiles(VmStore(tmp_path / "state.sqlite3", tmp_path / "jobs"), settings)
    public = repo.public(repo.current(7))
    assert public["mode"] == "system" and public["model"] == "anthropic/claude-sonnet-5.5"
    assert public["key_configured"] and public["default_available"] and public["api_key"] == ""
    assert repo.has_key(7)
    snapshot = repo.snapshot(7)
    seen = []

    def respond(request):
        seen.append(request.headers["authorization"])
        return httpx.Response(200)

    auditor = repo.auditor(snapshot, transport=httpx.MockTransport(respond))
    assert auditor.api_key == "default-operator-secret" and auditor.endpoint.startswith("https://openrouter.ai/")
    # Never persisted: rotating the server key applies to existing snapshots immediately.
    assert b"default-operator-secret" not in repo.store.path.read_bytes()
    settings.vlm_default_api_key = SecretStr("rotated-secret")
    assert repo.auditor(snapshot, transport=httpx.MockTransport(respond)).api_key == "rotated-secret"
    # A private key always wins, and the default key can never be copied into a custom profile.
    with pytest.raises(ValueError, match="own API key"):
        repo.save(7, model="vision", base_url="https://example.com/v1", key=None)
    repo.save(7, model="vision", base_url="https://example.com/v1", key="private-key")
    assert repo.auditor(repo.snapshot(7), transport=httpx.MockTransport(respond)).api_key == "private-key"
    repo.delete(7)
    assert repo.public(repo.current(7))["key_configured"]


def test_legacy_audit_key_is_not_shared_as_default(tmp_path):
    repo = profiles(tmp_path)  # Only OPENROUTER_API_KEY is set.
    assert not repo.has_key(3)
    assert not repo.public(repo.current(3))["default_available"]
