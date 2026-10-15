"""Create one CVAT 2D image task from an integrity-checked frame manifest."""

from __future__ import annotations

import asyncio
import math
import time
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import httpx
from pydantic import SecretStr

from src.models.video import CvatAttributeSpec, CvatLabelSpec
from src.services.cvat.read_client import RETRYABLE_STATUS_CODES, cvat_route_host

if TYPE_CHECKING:
    from src.config import Settings


class CvatWriteError(RuntimeError):
    """A sanitized CVAT write failure; task_id identifies a possible orphan."""

    def __init__(self, message: str, *, task_id: int | None = None, status_code: int | None = None) -> None:
        super().__init__(message)
        self.task_id = task_id
        self.status_code = status_code


class CvatTaskClient:
    def __init__(
        self,
        *,
        base_url: str,
        token: SecretStr | str,
        timeout_s: float = 30,
        ready_timeout_s: float = 120,
        retry_delay_s: float = 0.25,
        host_header: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("CVAT base URL must be an HTTP(S) URL without embedded credentials")
        if not _positive_finite(timeout_s) or not _positive_finite(ready_timeout_s):
            raise ValueError("CVAT timeouts must be finite positive numbers")
        if isinstance(retry_delay_s, bool) or not math.isfinite(retry_delay_s) or retry_delay_s < 0:
            raise ValueError("CVAT retry delay must be a finite non-negative number")
        secret = token.get_secret_value() if isinstance(token, SecretStr) else token
        self.base_url = base_url.rstrip("/")
        self.authorization = secret if secret.lower().startswith(("token ", "bearer ")) else f"Token {secret}"
        self.timeout_s = timeout_s
        self.ready_timeout_s = ready_timeout_s
        self.retry_delay_s = retry_delay_s
        self.host_header = host_header
        self.transport = transport

    @classmethod
    def from_settings(cls, settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None) -> CvatTaskClient:
        return cls(
            base_url=settings.cvat_url,
            token=settings.cvat_token,
            timeout_s=settings.cvat_timeout_s,
            ready_timeout_s=settings.cvat_task_ready_timeout_s,
            retry_delay_s=settings.cvat_retry_delay_s,
            host_header=cvat_route_host(settings),
            transport=transport,
        )

    async def create_image_task(
        self, *, name: str, labels: list[CvatLabelSpec], frame_paths: list[Path]
    ) -> tuple[int, int]:
        if not frame_paths or any(not path.is_file() for path in frame_paths):
            raise ValueError("CVAT task requires existing frame files")
        async with httpx.AsyncClient(
            base_url=self.base_url,
            headers=self._headers(),
            timeout=self.timeout_s,
            transport=self.transport,
        ) as client:
            task_id = await self._create_task(client, name, labels, len(frame_paths))
            await self._upload_frames(client, task_id, frame_paths)
            job_id = await self._wait_for_single_job(client, task_id)
        return task_id, job_id

    async def create_video_task(
        self,
        *,
        name: str,
        labels: list[CvatLabelSpec],
        video_path: Path,
    ) -> tuple[int, int]:
        """Upload one video as a single contiguous CVAT job for later keyframe selection."""
        if not video_path.is_file() or video_path.suffix.lower() not in {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
            raise ValueError("CVAT video task requires an existing supported video file")
        async with httpx.AsyncClient(
            base_url=self.base_url,
            headers=self._headers(),
            timeout=self.timeout_s,
            transport=self.transport,
        ) as client:
            task_id = await self._create_task(client, name, labels, 100_000)
            with video_path.open("rb") as handle:
                await self._post_once(
                    client,
                    f"/api/tasks/{task_id}/data",
                    data={"image_quality": "95", "sorting_method": "lexicographical"},
                    files=[("client_files[0]", (video_path.name, handle, "video/mp4"))],
                    headers={"Upload-Start": "", "Upload-Finish": ""},
                    task_id=task_id,
                )
            job_id = await self._wait_for_single_job(client, task_id)
        return task_id, job_id

    def _headers(self) -> dict[str, str]:
        if self.authorization == "Token ":
            raise CvatWriteError("CVAT token is not configured")
        headers = {"Authorization": self.authorization, "Accept": "application/vnd.cvat+json"}
        if self.host_header:
            headers["Host"] = self.host_header
        return headers

    async def ensure_label_attributes(
        self,
        labels: list[dict[str, Any]],
        required: list[CvatAttributeSpec],
    ) -> None:
        """Add missing review attributes; never remove or silently redefine existing ones.

        The CVAT fork's LabelSerializer updates attributes by ID/name and deletes
        only with deleted=true. Sending only missing names preserves existing data.
        """
        plans = []
        for label in labels:
            label_id = _positive_int(label.get("id"), "CVAT label ID")
            if label.get("type", "any") not in ("any", "rectangle"):
                raise ValueError(f"CVAT label {label.get('name')} does not support rectangles")
            existing = {item["name"]: item for item in label.get("attributes", [])}
            missing = []
            for attribute in required:
                current = existing.get(attribute.name)
                if current is None:
                    missing.append(attribute.model_dump(mode="json"))
                else:
                    values = current.get("values")
                    # CVAT serializes an empty text attribute's [] as [""].
                    values_match = values == attribute.values or (
                        attribute.input_type == "text" and attribute.values == [] and values == [""]
                    )
                    if (
                        current.get("mutable") == attribute.mutable
                        and current.get("input_type") == attribute.input_type
                        and values_match
                    ):
                        continue
                    raise ValueError(
                        f"CVAT label {label.get('name')} has incompatible attribute {attribute.name}; "
                        "correct it before running SAM3"
                    )
            if missing:
                plans.append((label_id, missing))
        async with httpx.AsyncClient(
            base_url=self.base_url,
            headers=self._headers(),
            timeout=self.timeout_s,
            transport=self.transport,
        ) as client:
            for label_id, missing in plans:
                try:
                    response = await client.patch(f"/api/labels/{label_id}", json={"attributes": missing})
                except httpx.RequestError as error:
                    raise CvatWriteError(
                        "CVAT label attribute update had an unknown outcome; retry after checking CVAT"
                    ) from error
                if not response.is_success:
                    raise CvatWriteError(
                        f"CVAT label attribute update failed with HTTP {response.status_code}",
                        status_code=response.status_code,
                    )

    async def _create_task(
        self, client: httpx.AsyncClient, name: str, labels: list[CvatLabelSpec], frame_count: int
    ) -> int:
        payload: dict[str, object] = {
            "name": name,
            "labels": [label.model_dump(exclude_none=True) for label in labels],
            "segment_size": frame_count,
            "overlap": 0,
        }
        response = await self._post_once(client, "/api/tasks", json=payload)
        body = _json_object(response, "CVAT task creation")
        return _positive_int(body.get("id"), "CVAT task ID")

    async def _upload_frames(self, client: httpx.AsyncClient, task_id: int, frame_paths: list[Path]) -> None:
        with ExitStack() as stack:
            files = [
                (f"client_files[{index}]", (path.name, stack.enter_context(path.open("rb")), "image/jpeg"))
                for index, path in enumerate(frame_paths)
            ]
            response = await self._post_once(
                client,
                f"/api/tasks/{task_id}/data",
                data={"image_quality": "95", "sorting_method": "lexicographical"},
                files=files,
                headers={"Upload-Start": "", "Upload-Finish": ""},
                task_id=task_id,
            )
        if response.status_code not in {200, 201, 202}:
            raise CvatWriteError(
                f"CVAT frame upload failed with HTTP {response.status_code}",
                task_id=task_id,
                status_code=response.status_code,
            )

    async def _wait_for_single_job(self, client: httpx.AsyncClient, task_id: int) -> int:
        deadline = time.monotonic() + self.ready_timeout_s
        while time.monotonic() < deadline:
            payload = await self._get_json(client, f"/api/jobs?task_id={task_id}&page_size=100", task_id)
            raw_results = payload.get("results")
            if isinstance(raw_results, list) and raw_results:
                if len(raw_results) != 1 or not isinstance(raw_results[0], dict):
                    raise CvatWriteError("CVAT task must contain exactly one annotation job", task_id=task_id)
                return _positive_int(raw_results[0].get("id"), "CVAT job ID")
            await asyncio.sleep(self.retry_delay_s or 0.05)
        raise CvatWriteError("CVAT task did not become ready before the configured timeout", task_id=task_id)

    async def _post_once(
        self,
        client: httpx.AsyncClient,
        path: str,
        *,
        json: dict[str, object] | None = None,
        data: dict[str, str] | None = None,
        files: list[tuple[str, tuple[str, Any, str]]] | None = None,
        headers: dict[str, str] | None = None,
        task_id: int | None = None,
    ) -> httpx.Response:
        try:
            response = await client.post(path, json=json, data=data, files=files, headers=headers)
        except httpx.RequestError as error:
            raise CvatWriteError(
                f"CVAT POST {path} had an unknown outcome; inspect CVAT before retrying",
                task_id=task_id,
            ) from error
        if not response.is_success:
            raise CvatWriteError(
                f"CVAT POST {path} failed with HTTP {response.status_code}",
                task_id=task_id,
                status_code=response.status_code,
            )
        return response

    async def _get_json(self, client: httpx.AsyncClient, path: str, task_id: int) -> dict[str, Any]:
        try:
            response = await client.get(path)
        except httpx.RequestError as error:
            raise CvatWriteError(f"CVAT GET {path} failed while preparing task", task_id=task_id) from error
        if not response.is_success:
            if response.status_code in RETRYABLE_STATUS_CODES:
                return {}
            raise CvatWriteError(
                f"CVAT GET {path} failed with HTTP {response.status_code}",
                task_id=task_id,
                status_code=response.status_code,
            )
        return _json_object(response, f"CVAT GET {path}", task_id=task_id)


def _json_object(response: httpx.Response, operation: str, *, task_id: int | None = None) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as error:
        raise CvatWriteError(f"{operation} returned invalid JSON", task_id=task_id) from error
    if not isinstance(payload, dict):
        raise CvatWriteError(f"{operation} returned a non-object JSON payload", task_id=task_id)
    return payload


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CvatWriteError(f"{name} is missing or invalid")
    return value


def _positive_finite(value: float) -> bool:
    return not isinstance(value, bool) and math.isfinite(value) and value > 0
