"""Authorize AutoLabel requests with the caller's CVAT session, never the service PAT."""

from __future__ import annotations

import httpx

from src.config import Settings
from src.services.cvat.read_client import cvat_route_host


class CvatAuthorizationError(RuntimeError):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code


class CvatSessionAuthorizer:
    def __init__(self, settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.base_url = settings.cvat_url.rstrip("/")
        self.timeout_s = settings.cvat_timeout_s
        self.host_header = cvat_route_host(settings)
        self.transport = transport

    async def authorize(self, session_id: str, cvat_job_id: int, *, write: bool) -> int:
        """Return CVAT user ID only after both identity and job scope pass."""
        if not session_id or len(session_id) > 512:
            raise CvatAuthorizationError(401, "CVAT sign-in required")
        if cvat_job_id < 1:
            raise CvatAuthorizationError(404, "CVAT job not found")

        headers = {"Accept": "application/vnd.cvat+json"}
        if self.host_header:
            headers["Host"] = self.host_header
        async with httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout_s,
            transport=self.transport,
            follow_redirects=False,
            headers=headers,
            cookies={"sessionid": session_id},
        ) as client:
            try:
                identity = await client.get("/api/users/self")
                if identity.status_code in (401, 403) or identity.is_redirect:
                    raise CvatAuthorizationError(401, "CVAT sign-in required")
                if not identity.is_success:
                    raise CvatAuthorizationError(502, "CVAT identity check unavailable")
                user_id = identity.json().get("id")
                if isinstance(user_id, bool) or not isinstance(user_id, int) or user_id < 1:
                    raise CvatAuthorizationError(502, "CVAT identity response invalid")

                path = f"/api/jobs/{cvat_job_id}"
                if write:
                    path += "/autolabel-permission"
                permission = await client.get(path)
            except httpx.RequestError as error:
                raise CvatAuthorizationError(502, "CVAT authorization unavailable") from error
            except (ValueError, AttributeError) as error:
                raise CvatAuthorizationError(502, "CVAT identity response invalid") from error

        if permission.status_code in (401, 403) or permission.is_redirect:
            raise CvatAuthorizationError(403, "CVAT job permission denied")
        if permission.status_code == 404:
            raise CvatAuthorizationError(404, "CVAT job not found or permission endpoint unavailable")
        if not permission.is_success:
            raise CvatAuthorizationError(502, "CVAT permission check unavailable")
        if write:
            try:
                allowed = permission.json().get("allowed") is True
            except (ValueError, AttributeError):
                allowed = False
            if not allowed:
                raise CvatAuthorizationError(502, "CVAT permission response invalid")
        return user_id

    async def identity(self, session_id: str) -> int:
        """Authenticate settings/upload before a CVAT job exists."""
        headers = {"Accept": "application/vnd.cvat+json"}
        if self.host_header:
            headers["Host"] = self.host_header
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                cookies={"sessionid": session_id},
                timeout=self.timeout_s,
                transport=self.transport,
                follow_redirects=False,
            ) as client:
                response = await client.get("/api/users/self")
            if not response.is_success:
                raise CvatAuthorizationError(401, "CVAT sign-in required")
            value = response.json().get("id")
            if type(value) is not int or value < 1:
                raise CvatAuthorizationError(502, "CVAT identity response invalid")
            return value
        except httpx.RequestError:
            raise CvatAuthorizationError(502, "CVAT authorization unavailable") from None
        except (ValueError, AttributeError):
            raise CvatAuthorizationError(502, "CVAT identity response invalid") from None
