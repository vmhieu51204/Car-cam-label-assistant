"""Versioned user VLM credentials and a pinned, SSRF-resistant vision transport."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import socket
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
from cryptography.fernet import Fernet, InvalidToken

from src.config import Settings
from src.services.video.claude_box_rescue import URL, ClaudeBoxRescueAuditor
from src.services.vm.storage import VmStore, encode


class Secrets:
    def __init__(self, settings: Settings):
        try:
            self.cipher = Fernet(settings.vlm_settings_encryption_key.get_secret_value().encode())
        except ValueError:
            raise ValueError("Configure VLM_SETTINGS_ENCRYPTION_KEY before saving credentials") from None

    def seal(self, value: dict) -> str:
        return self.cipher.encrypt(encode(value)).decode()

    def open(self, value: str) -> dict:
        try:
            return json.loads(self.cipher.decrypt(value.encode()))
        except (InvalidToken, ValueError):
            raise ValueError("Credential unavailable; restore the deployment encryption key") from None


def endpoint(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("VLM base URL must use HTTPS without credentials, query or fragment")
    if parsed.port is not None and not 1 <= parsed.port <= 65535:
        raise ValueError("Invalid HTTPS port")
    return base_url.rstrip("/") + "/chat/completions"


class PinnedTransport(httpx.BaseTransport):
    """Connect to the inspected address while retaining TLS SNI and Host."""

    def __init__(self, allowed_hosts: set[str] | None = None):
        self.allowed = allowed_hosts or set()
        self.inner = httpx.HTTPTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        answers = socket.getaddrinfo(host, request.url.port or 443, type=socket.SOCK_STREAM)
        addresses = [item[4][0] for item in answers]
        if not addresses or (
            host not in self.allowed and any(not ipaddress.ip_address(a).is_global for a in addresses)
        ):
            raise ValueError("VLM endpoint resolves to a forbidden address")
        request.headers["Host"] = request.url.netloc.decode()
        request.extensions["sni_hostname"] = host
        request.url = request.url.copy_with(host=addresses[0])
        return self.inner.handle_request(request)

    def close(self) -> None:
        self.inner.close()


class VlmProfiles:
    def __init__(self, store: VmStore, settings: Settings):
        self.store, self.settings = store, settings

    def current(self, owner: int) -> dict:
        with self.store.connect() as db:
            row = db.execute("SELECT version FROM vm_vlm_profiles WHERE owner=? AND current=1", (owner,)).fetchone()
        return self.resolve(owner, row[0] if row else "system")

    def resolve(self, owner: int, version: str) -> dict:
        if version == "system":
            return {
                "version": "system",
                "mode": "system",
                "model": self.settings.vlm_box_rescue_model,
                "endpoint": URL,
                "key": "",
                "owner": owner,
            }
        with self.store.connect() as db:
            row = db.execute(
                "SELECT payload FROM vm_vlm_profiles WHERE owner=? AND version=?", (owner, version)
            ).fetchone()
        if row is None:
            raise ValueError("VLM credential version not found for this account")
        return {**Secrets(self.settings).open(row[0]), "version": version, "owner": owner}

    def default_key(self) -> str:
        return self.settings.vlm_default_api_key.get_secret_value().strip()

    def usable_key(self, profile: dict) -> str:
        """The account's own key, else the operator's default key for the system profile."""
        if profile["key"]:
            return profile["key"]
        return self.default_key() if profile["mode"] == "system" else ""

    def has_key(self, owner: int) -> bool:
        return bool(self.usable_key(self.current(owner)))

    def public(self, profile: dict) -> dict:
        return {key: profile[key] for key in ("mode", "model", "endpoint", "version")} | {
            "key_configured": bool(self.usable_key(profile)),
            "default_available": bool(self.default_key()),
            # Only a private key is ever masked back; the shared default key is never shown.
            "api_key": "********" if profile["key"] else "",
        }

    def save(self, owner: int, *, model: str, base_url: str, key: str | None) -> dict:
        target = endpoint(base_url)
        version = str(uuid4())
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            previous = self.current(owner)
            if key is None:
                if previous["mode"] != "custom":
                    raise ValueError("Enter your own API key; the system key cannot be copied")
                key = previous["key"]
            if not key.strip() or not model.strip() or len(model) > 200 or len(key) > 4096:
                raise ValueError("Model and API key are required")
            profile = {"mode": "custom", "model": model, "endpoint": target, "key": key}
            sealed = Secrets(self.settings).seal(profile)
            db.execute("UPDATE vm_vlm_profiles SET current=0 WHERE owner=?", (owner,))
            db.execute("INSERT INTO vm_vlm_profiles VALUES(?,?,?,1)", (owner, version, sealed))
        return self.public({**profile, "version": version})

    def delete(self, owner: int) -> dict:
        with self.store.connect() as db:
            db.execute("UPDATE vm_vlm_profiles SET current=0 WHERE owner=?", (owner,))
        return self.public(self.current(owner))

    def snapshot(self, owner: int) -> dict:
        profile = self.current(owner)
        # System settings are snapshotted too; later env changes cannot alter a job.
        return {"owner": owner, "profile": Secrets(self.settings).seal(profile)}

    def auditor(self, snapshot: dict, *, transport: httpx.BaseTransport | None = None) -> ClaudeBoxRescueAuditor:
        profile = Secrets(self.settings).open(snapshot["profile"])
        if profile["owner"] != snapshot["owner"]:
            raise ValueError("VLM snapshot owner mismatch")
        key = self.usable_key(profile)
        if not key:
            raise ValueError("Configure your own VLM API key in Settings before requesting VLM")
        namespace = hashlib.sha256(encode({k: v for k, v in profile.items() if k != "key"})).hexdigest()
        cache = self.settings.vlm_cache_path.parent / "user-vlm" / str(snapshot["owner"]) / f"{namespace}.sqlite3"
        return ClaudeBoxRescueAuditor(
            api_key=key,
            model=profile["model"],
            endpoint=profile["endpoint"],
            compatible_api=True,
            cache_namespace=namespace,
            cache_path=cache,
            transport=transport
            or PinnedTransport(set(filter(None, self.settings.vlm_allowed_internal_hosts.split(",")))),
            # Legacy examples have no privacy receipts; never transmit them from a VM job.
            fewshot_manifest=None,
            timeout_s=self.settings.vlm_timeout_s,
        )

    def test(self, owner: int) -> dict:
        from PIL import Image

        from src.services.vm.contracts import file_sha256

        image = self.store.root / "connection-test.jpg"
        if not image.exists():
            Image.new("RGB", (64, 64), "white").save(image)
        auditor = self.auditor(self.snapshot(owner))
        _answer, usage, _cache = auditor.propose_missing_boxes(
            overlay_path=image,
            overlay_sha256=file_sha256(image),
            classes=("test_object",),
            image_width=64,
            image_height=64,
            existing_boxes=[],
        )
        return {"ok": True, "model": auditor.model, "usage": usage.model_dump()}
