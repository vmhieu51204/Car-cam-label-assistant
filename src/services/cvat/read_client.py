"""Bounded, read-only access to the supported CVAT REST API."""

from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit

import httpx
from pydantic import SecretStr

if TYPE_CHECKING:
    from src.config import Settings

RETRYABLE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})


def cvat_route_host(settings: Settings) -> str | None:
    """Preserve CVAT's public Host when Docker uses a different transport hostname."""
    internal = urlsplit(settings.cvat_url).hostname
    public = urlsplit(settings.cvat_public_url).hostname
    return public if public and public != internal else None


class CvatReadError(RuntimeError):
    """A sanitized CVAT read failure that never includes credentials or response bodies."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class CvatReadClient:
    """Read job metadata and annotations; this class deliberately exposes no write methods."""

    def __init__(
        self,
        *,
        base_url: str,
        token: SecretStr | str,
        timeout_s: float = 30,
        retry_attempts: int = 3,
        retry_delay_s: float = 0.25,
        host_header: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("CVAT base URL must be an HTTP(S) URL without embedded credentials")
        if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("CVAT timeout must be a finite positive number")
        if isinstance(retry_attempts, bool) or not isinstance(retry_attempts, int) or retry_attempts < 1:
            raise ValueError("CVAT retry attempts must be a positive integer")
        if isinstance(retry_delay_s, bool) or not math.isfinite(retry_delay_s) or retry_delay_s < 0:
            raise ValueError("CVAT retry delay must be a finite non-negative number")
        secret = token.get_secret_value() if isinstance(token, SecretStr) else token
        self.base_url = base_url.rstrip("/")
        self.authorization = secret if secret.lower().startswith(("token ", "bearer ")) else f"Token {secret}"
        self.timeout_s = timeout_s
        self.retry_attempts = retry_attempts
        self.retry_delay_s = retry_delay_s
        self.host_header = host_header
        self.transport = transport

    @classmethod
    def from_settings(cls, settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None) -> CvatReadClient:
        return cls(
            base_url=settings.cvat_url,
            token=settings.cvat_token,
            timeout_s=settings.cvat_timeout_s,
            retry_attempts=settings.cvat_retry_attempts,
            retry_delay_s=settings.cvat_retry_delay_s,
            host_header=cvat_route_host(settings),
            transport=transport,
        )

    def _headers(self) -> dict[str, str]:
        if self.authorization == "Token ":
            raise CvatReadError("CVAT token is not configured")
        headers = {"Authorization": self.authorization, "Accept": "application/vnd.cvat+json"}
        if self.host_header:
            headers["Host"] = self.host_header
        return headers

    async def _get_json(self, client: httpx.AsyncClient, path: str) -> dict[str, Any]:
        for attempt in range(self.retry_attempts):
            try:
                response = await client.get(path)
            except httpx.RequestError as error:
                if attempt + 1 == self.retry_attempts:
                    raise CvatReadError(f"CVAT GET {path} failed after {self.retry_attempts} attempt(s)") from error
            else:
                if response.is_success:
                    try:
                        payload = response.json()
                    except ValueError as error:
                        raise CvatReadError(f"CVAT GET {path} returned invalid JSON") from error
                    if not isinstance(payload, dict):
                        raise CvatReadError(f"CVAT GET {path} returned a non-object JSON payload")
                    return payload
                if response.status_code not in RETRYABLE_STATUS_CODES or attempt + 1 == self.retry_attempts:
                    raise CvatReadError(
                        f"CVAT GET {path} failed with HTTP {response.status_code}",
                        status_code=response.status_code,
                    )
            if self.retry_delay_s:
                await asyncio.sleep(self.retry_delay_s * (2**attempt))
        raise AssertionError("Retry loop must return or raise")

    async def _request(self, path: str) -> dict[str, Any]:
        async with httpx.AsyncClient(
            base_url=self.base_url,
            headers=self._headers(),
            timeout=self.timeout_s,
            transport=self.transport,
        ) as client:
            return await self._get_json(client, path)

    async def get_job(self, job_id: int) -> dict[str, Any]:
        return await self._request(f"/api/jobs/{valid_job_id(job_id)}")

    async def get_job_data_meta(self, job_id: int) -> dict[str, Any]:
        return await self._request(f"/api/jobs/{valid_job_id(job_id)}/data/meta")

    async def iter_job_frames(
        self, job_id: int, frame_count: int, *, quality: Literal["compressed", "original"] = "original"
    ) -> AsyncIterator[bytes]:
        """Download frames in bounded parallel batches while yielding source order."""
        normalized_job_id = valid_job_id(job_id)
        if isinstance(frame_count, bool) or frame_count < 1:
            raise ValueError("CVAT frame count must be positive")
        async for encoded in self._iter_frames(normalized_job_id, range(frame_count), quality=quality):
            yield encoded

    async def iter_selected_job_frames(self, job_id: int, indexes: list[int]) -> AsyncIterator[bytes]:
        """Fetch only selected source frames after the motion-analysis pass."""
        normalized_job_id = valid_job_id(job_id)
        if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in indexes):
            raise ValueError("CVAT frame indexes must be non-negative integers")
        async for encoded in self._iter_frames(normalized_job_id, indexes, quality="original"):
            yield encoded

    async def _iter_frames(
        self,
        job_id: int,
        indexes: range | list[int],
        *,
        quality: Literal["compressed", "original"],
    ) -> AsyncIterator[bytes]:
        batch_size = 8
        async with httpx.AsyncClient(
            base_url=self.base_url,
            headers=self._headers(),
            timeout=self.timeout_s,
            transport=self.transport,
            limits=httpx.Limits(max_connections=batch_size, max_keepalive_connections=batch_size),
        ) as client:
            for offset in range(0, len(indexes), batch_size):
                batch = indexes[offset : offset + batch_size]
                frames = await asyncio.gather(
                    *(self._get_frame(client, job_id, index, quality=quality) for index in batch)
                )
                for encoded in frames:
                    yield encoded

    @staticmethod
    async def _get_frame(
        client: httpx.AsyncClient,
        job_id: int,
        index: int,
        *,
        quality: Literal["compressed", "original"],
    ) -> bytes:
        try:
            response = await client.get(
                f"/api/jobs/{job_id}/data",
                params={"type": "frame", "number": index, "quality": quality},
            )
        except httpx.RequestError as error:
            raise CvatReadError(f"CVAT frame {index} download failed") from error
        if not response.is_success:
            raise CvatReadError(
                f"CVAT frame {index} download failed with HTTP {response.status_code}",
                status_code=response.status_code,
            )
        return response.content

    async def get_job_annotations(self, job_id: int) -> dict[str, Any]:
        return await self._request(f"/api/jobs/{valid_job_id(job_id)}/annotations")

    async def get_guarded_job_annotations(self, job_id: int) -> dict[str, Any]:
        """Require the atomic proposal extension; native CVAT version is not a revision."""
        async with httpx.AsyncClient(
            base_url=self.base_url,
            headers={**self._headers(), "X-Autolabel-Revision": "1"},
            timeout=self.timeout_s,
            transport=self.transport,
        ) as client:
            payload = await self._get_json(client, f"/api/jobs/{valid_job_id(job_id)}/annotations")
        revision = payload.get("autolabel_revision")
        if isinstance(revision, bool) or not isinstance(revision, int) or not 0 <= revision < 2**52:
            raise CvatReadError(
                "CVAT atomic proposal revision extension is missing; apply and rebuild the server overlay"
            )
        return payload

    async def get_job_labels(self, job_id: int) -> list[dict[str, Any]]:
        """Read every label visible to a job without trusting server-provided next URLs."""
        normalized_job_id = valid_job_id(job_id)
        results: list[dict[str, Any]] = []
        async with httpx.AsyncClient(
            base_url=self.base_url,
            headers=self._headers(),
            timeout=self.timeout_s,
            transport=self.transport,
        ) as client:
            for page in range(1, 101):
                payload = await self._get_json(client, f"/api/labels?job_id={normalized_job_id}&page={page}")
                raw_results = payload.get("results")
                if not isinstance(raw_results, list) or any(not isinstance(item, dict) for item in raw_results):
                    raise CvatReadError("CVAT label list returned an invalid payload")
                results.extend(raw_results)
                if not payload.get("next"):
                    return results
        raise CvatReadError("CVAT label list exceeded 100 pages")

    async def capture_job_snapshot(self, job_id: int) -> dict[str, Any]:
        """Capture a reproducible read-only snapshot for Gate 1/3 evidence."""
        normalized_job_id = valid_job_id(job_id)
        async with httpx.AsyncClient(
            base_url=self.base_url,
            headers=self._headers(),
            timeout=self.timeout_s,
            transport=self.transport,
        ) as client:
            job = await self._get_json(client, f"/api/jobs/{normalized_job_id}")
            data_meta = await self._get_json(client, f"/api/jobs/{normalized_job_id}/data/meta")
            annotations = await self._get_json(client, f"/api/jobs/{normalized_job_id}/annotations")
        return {
            "schema": "cvat-read-snapshot/v1",
            "captured_at_utc": datetime.now(UTC).isoformat(),
            "write_operations_performed": False,
            "cvat_url": self.base_url,
            "job_id": normalized_job_id,
            "job": job,
            "data_meta": data_meta,
            "annotations": annotations,
        }


def valid_job_id(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("CVAT job ID must be a positive integer")
    return value
