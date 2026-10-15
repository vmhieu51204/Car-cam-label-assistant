"""Job-scoped CVAT session gate for the production video API."""

from __future__ import annotations

import sqlite3

from fastapi import HTTPException, Request

from src.services.cvat.session_authorizer import CvatAuthorizationError


async def require_cvat_job_permission(request: Request) -> None:
    if request.app.state.settings.vm_workflow_enabled:
        route = request.url.path
        if request.method not in ("GET", "HEAD", "OPTIONS") and (
            route.endswith(("/import", "/retry-import", "/manual-claude", "/box-rescue/retry"))
            or "job_id" not in request.path_params
            and "cvat_job_id" not in request.path_params
        ):
            raise HTTPException(
                409, "Ubuntu VM jobs must start with privacy upload; legacy raw-media processing is disabled"
            )
    if request.app.state.settings.app_env != "production":
        return

    path = request.path_params
    if "cvat_job_id" in path:
        try:
            cvat_job_id = int(path["cvat_job_id"])
        except (TypeError, ValueError) as error:
            raise HTTPException(404, "CVAT job not found") from error
    elif "job_id" in path:
        # Production jobs must originate from a CVAT job. Standalone upload and
        # task creation use a global PAT and cannot be scoped to a CVAT user.
        if request.url.path.endswith("/cvat-task"):
            raise HTTPException(403, "Standalone CVAT task creation is disabled")
        try:
            job = request.app.state.video_jobs.get_job(path["job_id"])
        except (KeyError, ValueError) as error:
            raise HTTPException(404, "video job not found") from error
        cvat_job_id = job.cvat_job_id
        if cvat_job_id is None:
            raise HTTPException(403, "Video job is not bound to CVAT")
    else:
        raise HTTPException(403, "Standalone video upload is disabled")

    if cvat_job_id < 1:
        raise HTTPException(404, "CVAT job not found")

    session_id = request.cookies.get("sessionid", "")
    if not session_id:
        raise HTTPException(401, "CVAT sign-in required")

    write = request.method not in ("GET", "HEAD", "OPTIONS")
    if write:
        expected_origin = request.app.state.settings.cvat_public_url.rstrip("/")
        if request.headers.get("origin") != expected_origin:
            raise HTTPException(403, "CVAT origin required")
        csrf_cookie = request.cookies.get("csrftoken")
        csrf_header = request.headers.get("x-csrftoken")
        if not csrf_cookie or not csrf_header or csrf_cookie != csrf_header:
            raise HTTPException(403, "CVAT CSRF token required")

    try:
        user_id = await request.app.state.cvat_authorizer.authorize(session_id, cvat_job_id, write=write)
    except CvatAuthorizationError as error:
        raise HTTPException(error.status_code, str(error)) from error
    request.state.cvat_user_id = user_id
    if write:
        try:
            request.state.video_intent_id = request.app.state.store.record_video_user_intent(
                cvat_user_id=user_id,
                cvat_job_id=cvat_job_id,
                video_job_id=path.get("job_id"),
                method=request.method,
                route=request.url.path,
            )
        except (sqlite3.Error, ValueError) as error:
            raise HTTPException(503, "AutoLabel audit storage unavailable") from error
