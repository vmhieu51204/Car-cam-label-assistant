"""Read CVAT with the authenticated caller's session in the VM workflow."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import httpx

from src.config import Settings
from src.services.cvat.read_client import CvatReadClient, CvatReadError, cvat_route_host, valid_job_id

SOURCE_VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm"})
_FILENAME = re.compile(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', re.IGNORECASE)
# Fixed messages: CVAT response bodies are never forwarded to the browser.
_SOURCE_ERRORS = {
    401: "CVAT sign-in required",
    403: "No permission to read this CVAT job's media",
    404: "CVAT job not found",
    409: "This CVAT job has no stored source video (image or cloud-storage task); upload the video instead",
}


class CvatSessionReadClient(CvatReadClient):
    def __init__(
        self, settings: Settings, session_id: str, *, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        if not session_id or len(session_id) > 512:
            raise CvatReadError("CVAT sign-in required", status_code=401)
        super().__init__(
            base_url=settings.cvat_url,
            token="",
            timeout_s=settings.cvat_timeout_s,
            retry_attempts=settings.cvat_retry_attempts,
            retry_delay_s=settings.cvat_retry_delay_s,
            host_header=cvat_route_host(settings),
            transport=transport,
        )
        self.session_id = session_id

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/vnd.cvat+json", "Cookie": f"sessionid={self.session_id}"}
        if self.host_header:
            headers["Host"] = self.host_header
        return headers

    async def download_job_source(self, job_id: int, directory: Path, max_bytes: int) -> tuple[Path, str, int]:
        """Stream the task's original video into ``directory``; returns path, SHA-256 and size.

        One download of the stored file keeps frame rate and timestamps exact, unlike
        re-encoding frames read one by one.
        """
        path = f"/api/jobs/{valid_job_id(job_id)}/autolabel-source"
        target: Path | None = None
        try:
            async with (
                httpx.AsyncClient(
                    base_url=self.base_url,
                    headers=self._headers(),
                    timeout=self.timeout_s,
                    transport=self.transport,
                    follow_redirects=False,
                ) as client,
                client.stream("GET", path) as response,
            ):
                if not response.is_success:
                    raise CvatReadError(
                        _SOURCE_ERRORS.get(
                            response.status_code, f"CVAT source video failed with HTTP {response.status_code}"
                        ),
                        status_code=response.status_code,
                    )
                match = _FILENAME.search(response.headers.get("content-disposition", ""))
                extension = Path(match.group(1)).suffix.lower() if match else ""
                if extension not in SOURCE_VIDEO_EXTENSIONS:
                    raise CvatReadError("The CVAT task video format is not supported", status_code=422)
                target = directory / ("source" + extension)
                digest, size = hashlib.sha256(), 0
                with target.open("xb") as handle:
                    async for data in response.aiter_bytes(1048576):
                        size += len(data)
                        if size > max_bytes:
                            raise CvatReadError("The CVAT task video exceeds the upload limit", status_code=413)
                        handle.write(data)
                        digest.update(data)
        except httpx.RequestError as error:
            if target is not None:
                target.unlink(missing_ok=True)
            raise CvatReadError("CVAT source video download failed", status_code=502) from error
        except CvatReadError:
            if target is not None:
                target.unlink(missing_ok=True)
            raise
        if not size:
            target.unlink(missing_ok=True)
            raise CvatReadError("The CVAT task video is empty", status_code=422)
        return target, digest.hexdigest(), size
