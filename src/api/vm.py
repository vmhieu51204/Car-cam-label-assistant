"""Authenticated settings and privacy-first VM runs; no raw-media download route."""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import Field, FiniteFloat, SecretStr

from src.models.video import StrictContract
from src.services.cvat import CvatReadError, CvatWriteError
from src.services.cvat.session_authorizer import CvatAuthorizationError
from src.services.cvat.session_read_client import SOURCE_VIDEO_EXTENSIONS, CvatSessionReadClient
from src.services.video.audit import VideoAuditError
from src.services.video.manual_frame_rescue import decide_frame, propose_frame, refine_frame_box
from src.services.vm.contracts import preflight
from src.services.vm.hybrid_contracts import Pipeline
from src.services.vm.media import verified_media
from src.services.vm.reader import VerifiedFrameReader
from src.services.vm.storage import VmStore, atomic_json
from src.services.vm.vlm import Secrets, VlmProfiles


async def identity(request: Request) -> int:
    if request.method == "GET" and request.url.path.endswith("/vm/capabilities"):
        return 0  # Only non-sensitive feature flags; jobs and settings always require identity.
    session = request.cookies.get("sessionid", "")
    if not session or len(session) > 512:
        raise HTTPException(401, "CVAT sign-in required")
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        if request.headers.get("origin") != request.app.state.settings.cvat_public_url.rstrip("/"):
            raise HTTPException(403, "CVAT origin required")
        csrf = request.cookies.get("csrftoken")
        if not csrf or request.headers.get("x-csrftoken") != csrf:
            raise HTTPException(403, "CVAT CSRF token required")
    try:
        user = await request.app.state.cvat_authorizer.identity(session)
    except CvatAuthorizationError as error:
        raise HTTPException(error.status_code, str(error)) from None
    request.state.cvat_user_id = user
    job_id = request.path_params.get("job_id")
    if job_id:
        try:
            job = request.app.state.vm_store.get(job_id, user)
            if job.get("cvat_job_id"):
                scoped = await request.app.state.cvat_authorizer.authorize(
                    session, job["cvat_job_id"], write=request.method not in ("GET", "HEAD", "OPTIONS")
                )
                if scoped != user:
                    raise HTTPException(403, "CVAT session identity changed")
        except KeyError:
            raise HTTPException(404, "VM job not found") from None
        except CvatAuthorizationError as error:
            raise HTTPException(error.status_code, str(error)) from None
    return user


router = APIRouter(dependencies=[Depends(identity)])


def profiles(request: Request) -> VlmProfiles:
    return request.app.state.vlm_profiles


class VlmSettingsRequest(StrictContract):
    base_url: str = Field(max_length=2048)
    model: str = Field(min_length=1, max_length=200)
    api_key: SecretStr | None = None


@router.get("/settings/vlm")
def get_vlm(request: Request) -> dict:
    return profiles(request).public(profiles(request).current(request.state.cvat_user_id))


@router.put("/settings/vlm")
def put_vlm(request: Request, body: VlmSettingsRequest) -> dict:
    try:
        return profiles(request).save(
            request.state.cvat_user_id,
            model=body.model,
            base_url=body.base_url,
            key=body.api_key.get_secret_value() if body.api_key is not None else None,
        )
    except ValueError as error:
        raise HTTPException(422, str(error)) from None


@router.delete("/settings/vlm")
def delete_vlm(request: Request) -> dict:
    return profiles(request).delete(request.state.cvat_user_id)


@router.post("/settings/vlm/test")
def test_vlm(request: Request) -> dict:
    try:
        return profiles(request).test(request.state.cvat_user_id)
    except (ValueError, VideoAuditError):
        raise HTTPException(502, "Vision API test failed; check configuration and model compatibility") from None


class PrivacyRegion(StrictContract):
    start: int = Field(ge=0, strict=True)
    end: int = Field(ge=0, strict=True)
    box: list[FiniteFloat] = Field(min_length=4, max_length=4)


class UploadOptions(StrictContract):
    classes: list[str] = Field(min_length=1, max_length=8)
    variant: str = "sam3"
    pipeline: Pipeline = "hybrid_frame"
    vlm_enabled: bool = False
    regions: list[PrivacyRegion] = Field(default_factory=list, max_length=1000)


class CvatRunOptions(UploadOptions):
    # Existing-job classes are authoritative on the server, with no upload-only cap.
    classes: list[str] = Field(default_factory=list)


@router.get("/vm/capabilities")
def capabilities(request: Request) -> dict:
    settings = request.app.state.settings
    return {
        "enabled": settings.vm_workflow_enabled,
        "operating_system": "Ubuntu 24.04",
        "input_kind": "redacted",
        "cache_max_bytes": settings.local_frame_cache_max_bytes,
        "pipelines": ["sam", "hybrid_frame", "hybrid_propagate"] if settings.hybrid_enabled else ["sam"],
        "hybrid_variant": "sam3",
    }


async def job_payload(request: Request, body: UploadOptions) -> tuple[dict, str]:
    """Validate run options and freeze model/VLM profiles; raises ValueError for the user."""
    settings = request.app.state.settings
    if (
        body.variant not in ("sam3", "sam3.1")
        or any(not c.strip() or len(c) > 128 for c in body.classes)
        or len(set(body.classes)) != len(body.classes)
    ):
        raise ValueError("Choose exact nonempty, unique class names and a local SAM variant")
    profile = await asyncio.to_thread(preflight, settings, body.variant, body.pipeline, body.classes)
    credentials = Secrets(settings).seal(
        {"sessionid": request.cookies["sessionid"], "csrftoken": request.cookies["csrftoken"]}
    )
    snapshot = profiles(request).snapshot(request.state.cvat_user_id)
    if body.vlm_enabled and not profiles(request).has_key(request.state.cvat_user_id):
        raise ValueError("Configure your VLM key before starting an audit")
    payload = {
        "classes": body.classes,
        "profile": profile.model_dump(mode="json", by_alias=True),
        "vlm_snapshot": snapshot,
        "vlm_enabled": body.vlm_enabled,
        "manual_regions": [r.model_dump() for r in body.regions],
        "privacy_profile": {
            key: str(getattr(settings, key))
            for key in (
                "privacy_face_checkpoint",
                "privacy_plate_checkpoint",
                "privacy_face_sha256",
                "privacy_plate_sha256",
                "privacy_source_commit",
            )
        },
    }
    return payload, credentials


@router.post("/vm/jobs")
async def upload_vm(request: Request, options: str = Form(...), video: UploadFile = File(...)) -> dict:
    settings, store = request.app.state.settings, request.app.state.vm_store
    try:
        payload, credentials = await job_payload(request, UploadOptions.model_validate_json(options))
    except ValueError as error:
        await video.close()
        raise HTTPException(422, str(error)) from None
    extension = Path(video.filename or "").suffix.lower()
    if extension not in SOURCE_VIDEO_EXTENSIONS:
        await video.close()
        raise HTTPException(422, "Upload a supported video")
    job = store.create(request.state.cvat_user_id, {**payload, "source_extension": extension}, credentials)
    source = store.directory(job["job_id"]) / "private" / ("source" + extension)
    digest, size = hashlib.sha256(), 0
    try:
        with source.open("xb") as handle:
            while data := await video.read(1048576):
                size += len(data)
                if size > settings.video_max_upload_bytes:
                    raise ValueError("Video exceeds configured upload limit")
                handle.write(data)
                digest.update(data)
        if not size:
            raise ValueError("Video is empty")
        store.update(job["job_id"], status="queued", stage="queued_privacy", source_sha256=digest.hexdigest())
    except Exception:
        store.update(job["job_id"], status="failed", stage="upload_failed", error="Upload failed; retry with a new job")
        raise HTTPException(422, "Upload failed or limit exceeded") from None
    finally:
        await video.close()
    return public_job(store.get(job["job_id"]))


@router.post("/vm/cvat-jobs/{cvat_job_id}/runs")
async def run_from_cvat_job(request: Request, cvat_job_id: int, body: CvatRunOptions) -> dict:
    """Redact the stored video and publish annotations into this same CVAT job."""
    settings, store = request.app.state.settings, request.app.state.vm_store
    try:
        user = await request.app.state.cvat_authorizer.authorize(
            request.cookies.get("sessionid", ""), cvat_job_id, write=True
        )
    except CvatAuthorizationError as error:
        raise HTTPException(error.status_code, str(error)) from None
    if user != request.state.cvat_user_id:
        raise HTTPException(403, "CVAT session identity changed")
    try:
        read = session_read(request)
        scope = await read.get_job(cvat_job_id)
        labels = await read.get_job_labels(cvat_job_id)
        meta = await read.get_job_data_meta(cvat_job_id)
        task = await read._request(f"/api/tasks/{scope['task_id']}")
    except CvatReadError as error:
        raise HTTPException(error.status_code, str(error)) from None
    if (
        scope.get("start_frame", 0) != 0
        or scope.get("stop_frame") != task["size"] - 1
        or meta.get("start_frame", 0) != 0
        or meta.get("frame_filter")
        or meta.get("deleted_frames")
        or meta.get("included_frames") is not None
    ):
        raise HTTPException(409, "AutoLabel currently requires a full-video job without frame filtering or deleted frames")
    classes = [label["name"] for label in labels]
    if not classes:
        raise HTTPException(422, "This CVAT job must have at least one label")
    # Never let a stale UI or caller silently omit task labels. Text-prompted SAM
    # covers the entire label set when the hybrid detector cannot represent it.
    body = body.model_copy(update={"classes": classes})
    try:
        if body.pipeline != "sam":
            mapping = json.loads(settings.hybrid_class_mapping)
            if isinstance(mapping, dict) and any(name not in mapping for name in classes):
                body = body.model_copy(update={"pipeline": "sam"})
        payload, credentials = await job_payload(request, body)
    except ValueError as error:
        raise HTTPException(422, str(error)) from None
    job = store.create(
        request.state.cvat_user_id,
        {
            **payload,
            "source_extension": "",
            "origin_cvat_job_id": cvat_job_id,
            "cvat_job_id": cvat_job_id,
            "cvat_task_id": scope["task_id"],
            "in_place": True,
        },
        credentials,
    )
    try:
        source, digest, _size = await session_read(request).download_job_source(
            cvat_job_id, store.directory(job["job_id"]) / "private", settings.video_max_upload_bytes
        )
    except CvatReadError as error:
        store.update(job["job_id"], status="failed", stage="upload_failed", error=str(error))
        status = error.status_code if error.status_code in (401, 403, 404, 409, 413, 422) else 502
        detail = "This job has no stored source video; AutoLabel cannot run on this job" if status == 409 else str(error)
        raise HTTPException(status, detail) from None
    store.update(
        job["job_id"],
        status="queued",
        stage="queued_privacy",
        source_extension=source.suffix,
        source_sha256=digest,
    )
    return public_job(store.get(job["job_id"]))


def public_job(job: dict) -> dict:
    names = (
        "job_id",
        "status",
        "stage",
        "classes",
        "privacy_frames",
        "delivered_frames",
        "processed_frames",
        "complete_frames",
        "published_frames",
        "total_frames",
        "cvat_job_id",
        "cvat_task_id",
        "error",
        "current_window",
        "completed_windows",
        "coverage",
        "class_coverage",
        "cache_bytes",
        "cap_warning",
        "mask_absence_count",
        "cuboid_count",
        "scene",
        "dataset_format",
        "mode",
        "upload_name",
        "upload_size",
        "uploaded_bytes",
        "origin_cvat_job_id",
    )
    return {k: v for k, v in job.items() if k in (*names, "publication_version")} | {
        "variant": job["profile"]["variant"],
        "pipeline": job["profile"].get("pipeline", "sam"),
    }


@router.get("/vm/jobs/{job_id}")
@router.get("/vm/jobs/{job_id}/processing")
def get_vm(request: Request, job_id: str) -> dict:
    try:
        store = request.app.state.vm_store
        job = store.get(job_id, request.state.cvat_user_id)
        states = [0] * job["total_frames"]
        counts: dict[int, int] = {}
        with store.connect() as db:
            rows = db.execute(
                "SELECT payload FROM vm_events WHERE job=? AND event LIKE 'result:%'", (job_id,)
            ).fetchall()
        for row in rows:
            event = json.loads(row[0])
            counts[event["frame"]] = counts.get(event["frame"], 0) + 1
        for frame, count in counts.items():
            states[frame] = 2 if count == len(job["classes"]) else 1
        if job["status"] == "failed":
            for frame in job.get("failed_frames", []):
                if 0 <= frame < len(states) and states[frame] != 2:
                    states[frame] = 3
        return public_job(job) | {"frame_states": states, "vlm_status": job.get("vlm_status", "idle")}
    except KeyError:
        raise HTTPException(404, "VM job not found") from None


@router.get("/vm/cvat-jobs/{cvat_job_id}")
def find_vm(request: Request, cvat_job_id: int) -> dict:
    store = request.app.state.vm_store
    with store.connect() as db:
        rows = db.execute(
            "SELECT id FROM vm_jobs WHERE owner=? ORDER BY rowid DESC", (request.state.cvat_user_id,)
        ).fetchall()
    # A CVAT job maps to the run that produced it, or else to the latest run started from it.
    jobs = [store.get(row[0]) for row in rows]
    found = next((job for job in jobs if job.get("cvat_job_id") == cvat_job_id), None) or next(
        (job for job in jobs if job.get("origin_cvat_job_id") == cvat_job_id), None
    )
    return {"job": public_job(found) if found else None}


@router.post("/vm/jobs/{job_id}/cancel")
def cancel_vm(request: Request, job_id: str) -> dict:
    store = request.app.state.vm_store
    try:
        current = store.get(job_id, request.state.cvat_user_id)
        if current["status"] == "cancelled":
            return public_job(current)
        atomic_json(store.directory(job_id) / "private" / "cancelled.json", {"cancelled": True})
        return public_job(store.update(job_id, status="cancelled", stage="cancelled"))
    except KeyError:
        raise HTTPException(404, "VM job not found") from None


@router.post("/vm/jobs/{job_id}/retry")
def retry_vm(request: Request, job_id: str) -> dict:
    store = request.app.state.vm_store
    try:
        job = store.get(job_id, request.state.cvat_user_id)
        if job["status"] not in ("failed", "cancelled"):
            raise ValueError("Only failed/cancelled jobs can be retried")
        if job.get("stage") == "upload_failed":
            # No source file was ever stored, so the worker would have nothing to process.
            raise ValueError("The video never reached AutoLabel; start a new run")
        store.renew_credentials(
            job_id,
            request.state.cvat_user_id,
            Secrets(request.app.state.settings).seal(
                {"sessionid": request.cookies["sessionid"], "csrftoken": request.cookies["csrftoken"]}
            ),
        )
        (store.directory(job_id) / "private" / "cancelled.json").unlink(missing_ok=True)
        return public_job(store.update(job_id, status="queued", stage="retry", error=None))
    except KeyError:
        raise HTTPException(404, "VM job not found") from None
    except ValueError as error:
        raise HTTPException(409, str(error)) from None


def clean_receipt(store: VmStore, job: dict) -> Path:
    return verified_media(store, job)[0]


@router.get("/vm/jobs/{job_id}/video")
def get_clean_video(request: Request, job_id: str) -> FileResponse:
    store = request.app.state.vm_store
    try:
        job = store.get(job_id, request.state.cvat_user_id)
        return FileResponse(clean_receipt(store, job), media_type="video/mp4", filename="redacted.mp4")
    except KeyError:
        raise HTTPException(404, "VM job not found") from None
    except (ValueError, FileNotFoundError):
        raise HTTPException(409, "Redacted video is not ready") from None


class ManualRequest(StrictContract):
    source_frame: int = Field(ge=0)


def session_read(request: Request) -> CvatSessionReadClient:
    return CvatSessionReadClient(request.app.state.settings, request.cookies.get("sessionid", ""))


def manual_service(request: Request, job_id: str) -> SimpleNamespace:
    store = request.app.state.vm_store
    job = store.get(job_id, request.state.cvat_user_id)
    clean_receipt(store, job)
    if not job.get("cvat_job_id"):
        raise ValueError("CVAT job not ready")
    if not job.get("incremental_ready"):
        raise ValueError("Progressive annotation protection is not ready")

    async def labels(cvat_job_id: int, _classes: tuple) -> list[dict]:
        return await session_read(request).get_job_labels(cvat_job_id)

    return SimpleNamespace(
        box_rescue_enabled=True,
        prepare_sequence_labels=labels,
        store=SimpleNamespace(get_video_job=lambda _id: SimpleNamespace(cvat_job_id=job["cvat_job_id"], job_id=job_id)),
        workspace=SimpleNamespace(job_dir=lambda _id: store.directory(job_id)),
        seed_reader=SimpleNamespace(client=VerifiedFrameReader(store, job, session_read(request))),
        track_client=request.app.state.cvat_tracks,
    )


@router.post("/vm/jobs/{job_id}/vlm")
async def manual_vlm(request: Request, job_id: str, body: ManualRequest) -> dict:
    try:
        service = manual_service(request, job_id)
        job = request.app.state.vm_store.get(job_id, request.state.cvat_user_id)
        auditor = profiles(request).auditor(job["vlm_snapshot"])
        return await propose_frame(service, request.app.state.settings, job_id, body.source_frame, auditor=auditor)
    except KeyError:
        raise HTTPException(404, "VM job not found") from None
    except (ValueError, FileNotFoundError):
        raise HTTPException(409, "Verified redacted input or VLM configuration unavailable") from None
    except VideoAuditError:
        raise HTTPException(502, "VLM request failed; frame remains retryable") from None
    except CvatReadError:
        raise HTTPException(502, "CVAT redacted frame unavailable") from None


class RefineRequest(StrictContract):
    source_frame: int = Field(ge=0)
    label: str = Field(min_length=1, max_length=128)
    # Normalized xyxy on the source frame.
    bbox: list[FiniteFloat] = Field(min_length=4, max_length=4)
    instruction: str = Field(default="", max_length=500)


@router.post("/vm/jobs/{job_id}/vlm/refine")
async def refine_box(request: Request, job_id: str, body: RefineRequest) -> dict:
    """Fit a drawn box to its object with the owner's VLM; the editor applies and saves the result."""
    x1, y1, x2, y2 = body.bbox
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise HTTPException(422, "Box must be normalized xyxy inside the frame")
    try:
        service = manual_service(request, job_id)
        job = request.app.state.vm_store.get(job_id, request.state.cvat_user_id)
        auditor = profiles(request).auditor(job["vlm_snapshot"])
        return await refine_frame_box(
            service, job_id, body.source_frame, body.label, list(body.bbox), body.instruction, auditor=auditor
        )
    except KeyError:
        raise HTTPException(404, "VM job not found") from None
    except FileNotFoundError:
        raise HTTPException(409, "Verified redacted input unavailable") from None
    except ValueError as error:
        # Messages raised here are fixed, user-facing strings (missing key, class, frame range).
        raise HTTPException(409, str(error)) from None
    except VideoAuditError:
        raise HTTPException(502, "VLM request failed; try again or adjust the box") from None
    except CvatReadError:
        raise HTTPException(502, "CVAT redacted frame unavailable") from None


@router.post("/vm/jobs/{job_id}/vlm/profile")
def bind_vlm_profile(request: Request, job_id: str) -> dict:
    """Explicitly bind the owner's current profile to an already completed video."""
    store = request.app.state.vm_store
    job = store.get(job_id, request.state.cvat_user_id)
    if job.get("workflow") == "lidar3d" or job["status"] != "completed":
        raise HTTPException(409, "Apply VLM settings only to a completed redacted video job")
    try:
        clean_receipt(store, job)
        if not profiles(request).has_key(request.state.cvat_user_id):
            raise ValueError("VLM key missing")
        snapshot = profiles(request).snapshot(request.state.cvat_user_id)
    except (ValueError, FileNotFoundError):
        raise HTTPException(409, "Save your VLM API key and verify the redacted video first") from None
    return public_job(store.update(job_id, vlm_snapshot=snapshot)) | {"vlm_profile_applied": True}


class DecisionRequest(StrictContract):
    index: int = Field(ge=0)
    accept: bool


@router.post("/vm/jobs/{job_id}/vlm/{result_id}/decision")
async def decision(request: Request, job_id: str, result_id: str, body: DecisionRequest) -> dict:
    try:
        return await decide_frame(manual_service(request, job_id), job_id, result_id, body.index, body.accept)
    except KeyError:
        raise HTTPException(404, "VM job not found") from None
    except (ValueError, FileNotFoundError):
        raise HTTPException(409, "Save and refresh annotations before accepting") from None
    except (CvatReadError, CvatWriteError) as error:
        raise HTTPException(
            409 if error.status_code == 409 else 502, "CVAT annotation decision failed; refresh before retrying"
        ) from None


@router.get("/vm/jobs/{job_id}/vlm")
def list_vlm(request: Request, job_id: str) -> dict:
    store = request.app.state.vm_store
    store.get(job_id, request.state.cvat_user_id)
    paths = sorted(
        (store.directory(job_id) / "manual-claude").glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return {"results": [json.loads(p.read_text()) for p in paths[:20]]}


class PrivacyKeyframe(StrictContract):
    frame: int = Field(ge=0, strict=True)
    box: list[FiniteFloat] = Field(min_length=4, max_length=4)


class PrivacyTrack(StrictContract):
    track_id: str = Field(min_length=1, max_length=128)
    keyframes: list[PrivacyKeyframe] = Field(min_length=1, max_length=1000)


class RegionRequest(StrictContract):
    regions: list[PrivacyRegion] = Field(default_factory=list, max_length=1000)
    tracks: list[PrivacyTrack] = Field(default_factory=list, max_length=100)


@router.post("/vm/jobs/{job_id}/privacy-regions")
async def privacy_regions(request: Request, job_id: str, body: RegionRequest) -> dict:
    """Create a new private redaction run, preserving an existing CVAT target."""
    store = request.app.state.vm_store
    job = store.get(job_id, request.state.cvat_user_id)
    _video, receipt = verified_media(store, job)
    regions = [r.model_dump() for r in body.regions]
    from src.services.vm.regions import expand_tracks

    try:
        regions += expand_tracks(
            [r.model_dump() for r in body.tracks], receipt["timestamps"], receipt["width"], receipt["height"]
        )
    except ValueError as error:
        raise HTTPException(422, str(error)) from None
    if not regions:
        raise HTTPException(422, "Add a privacy region or track")
    for region in regions:
        if not 0 <= region["start"] <= region["end"] < receipt["total_frames"]:
            raise HTTPException(422, "Invalid privacy region frame range")
        import numpy as np

        from src.services.vm.privacy import redact

        try:
            redact(np.zeros((receipt["height"], receipt["width"], 3), dtype=np.uint8), [region["box"]])
        except (ValueError, TypeError):
            raise HTTPException(422, "Invalid pixel privacy rectangle") from None
    payload = {
        k: job[k]
        for k in (
            "classes",
            "profile",
            "vlm_snapshot",
            "vlm_enabled",
            "source_extension",
            "privacy_profile",
            "source_sha256",
        )
    }
    if job.get("in_place"):
        payload.update({key: job[key] for key in ("in_place", "cvat_task_id", "cvat_job_id", "origin_cvat_job_id")})
    payload["manual_regions"] = job["manual_regions"] + regions
    credentials = Secrets(request.app.state.settings).seal(
        {"sessionid": request.cookies["sessionid"], "csrftoken": request.cookies["csrftoken"]}
    )
    fresh = store.create(request.state.cvat_user_id, payload, credentials)
    await asyncio.to_thread(
        shutil.copyfile,
        store.directory(job_id) / "private" / ("source" + job["source_extension"]),
        store.directory(fresh["job_id"]) / "private" / ("source" + job["source_extension"]),
    )
    return public_job(store.update(fresh["job_id"], status="queued", stage="queued_privacy", parent_job=job_id))


@router.get("/video/jobs/{job_id}/processing")
def processing(request: Request, job_id: str) -> dict:
    # Same owner and CVAT permissions as the VM endpoint, including in development.
    return get_vm(request, job_id)


@router.get("/vm/jobs/{job_id}/review")
async def review(request: Request, job_id: str, page: int = 1, class_name: str = "", viewed: str = "") -> dict:
    store = request.app.state.vm_store
    job = store.get(job_id, request.state.cvat_user_id)
    if not job.get("cvat_job_id"):
        return {"tracks": [], "total": 0, "page": 1}
    annotations = await session_read(request).get_guarded_job_annotations(job["cvat_job_id"])
    labels = await session_read(request).get_job_labels(job["cvat_job_id"])
    names = {r["id"]: r["name"] for r in labels}
    specs = {a["id"]: a["name"] for r in labels for a in r["attributes"]}
    states = job.get("review_states", {})
    rows = []
    for track in annotations["tracks"]:
        attrs = {specs.get(a["spec_id"]): a["value"] for a in track["attributes"]}
        identity = attrs.get("autolabel_track_id")
        visible = next((s for s in track["shapes"] if not s["outside"]), None)
        state = states.get(identity, "unseen")
        if (
            not identity
            or not visible
            or class_name
            and names[track["label_id"]] != class_name
            or viewed
            and state != viewed
        ):
            continue
        rows.append(
            {
                "track_id": identity,
                "class": names[track["label_id"]],
                "frame": visible["frame"],
                "state": state,
                "reason": attrs.get("review_reason", "pending human review"),
            }
        )
    page = max(1, page)
    return {"tracks": rows[(page - 1) * 50 : page * 50], "total": len(rows), "page": page}


class ReviewStateRequest(StrictContract):
    track_id: str = Field(min_length=1, max_length=200)
    state: str


@router.post("/vm/jobs/{job_id}/review/state")
def review_state(request: Request, job_id: str, body: ReviewStateRequest) -> dict:
    if body.state not in ("unseen", "seen", "resolved"):
        raise HTTPException(422, "Invalid review progress state")
    store = request.app.state.vm_store
    # This records review progress only; it never changes annotation acceptance.
    store.review(job_id, request.state.cvat_user_id, body.track_id, body.state)
    return {"track_id": body.track_id, "state": body.state}
