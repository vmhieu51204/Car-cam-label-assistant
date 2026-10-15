"""Owner-scoped, bounded dataset uploads and saved-annotation downloads."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Literal
from uuid import uuid4

import httpx
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import Field

from src.api.vm import public_job
from src.models.video import StrictContract
from src.services.vm.contracts import file_sha256
from src.services.vm.lidar3d_contracts import BEVFUSION_CHECKPOINT_SHA, CHUNK_BYTES, CLASSES, KITTI_CLASSES
from src.services.vm.lidar3d_exchange import export_annotations
from src.services.vm.lidar3d_review import review_objects, reviewed_annotations, revision
from src.services.vm.storage import atomic_json, worker_lock
from src.services.vm.vlm import Secrets
from src.services.vm.worker import UserTaskClient

router = APIRouter()


def published_job(request: Request, job_id: str) -> dict:
    job = request.app.state.vm_store.get(job_id, request.state.cvat_user_id)
    if job.get("workflow") != "lidar3d" or job["status"] != "completed" or not job.get("cvat_job_id"):
        raise HTTPException(409, "A completed 3D job is required")
    return job


def cvat_client(request: Request) -> httpx.AsyncClient:
    settings = request.app.state.settings
    credentials = {"sessionid": request.cookies["sessionid"], "csrftoken": request.cookies.get("csrftoken", "")}
    return httpx.AsyncClient(
        base_url=settings.cvat_url, headers=UserTaskClient(settings, credentials)._headers(), timeout=120
    )


async def saved_annotations(client: httpx.AsyncClient, job: dict) -> tuple[dict, list[dict]]:
    annotations, labels = await asyncio.gather(
        client.get(f"/api/jobs/{job['cvat_job_id']}/annotations"),
        client.get(f"/api/labels?task_id={job['cvat_task_id']}&page_size=100"),
    )
    if annotations.is_error or labels.is_error:
        raise HTTPException(403, "CVAT annotations are not authorized")
    return annotations.json(), labels.json()["results"]


@router.get("/jobs/{job_id}/review")
async def get_review(request: Request, job_id: str, page: int = Query(default=0, ge=0)) -> dict:
    job = published_job(request, job_id)
    async with cvat_client(request) as client:
        annotations, labels = await saved_annotations(client, job)
    objects = review_objects(annotations, labels)
    return {
        "revision": revision(annotations),
        "total": len(objects),
        "page": page,
        "pending": sum(item["review_status"] == "pending" for item in objects),
        "objects": objects[page * 50 : (page + 1) * 50],
    }


class ReviewDecision(StrictContract):
    object_id: str = Field(pattern=r"^(shapes|tracks):[1-9][0-9]*$")
    status: Literal["pending", "accepted", "rejected"]
    expected_revision: int = Field(ge=0, strict=True)


@router.get("/jobs/{job_id}/camera-links")
def camera_links(request: Request, job_id: str, frame: int = Query(ge=0)) -> dict:
    published_job(request, job_id)
    path = request.app.state.vm_store.directory(job_id) / "private/lidar3d/camera-links.json"
    if not path.is_file():
        raise HTTPException(409, "This job has no camera association evidence; create a new 3D run")
    payload = json.loads(path.read_bytes())
    return {
        "frame": frame,
        "links": [item for item in payload["links"] if item["frame"] == frame],
        "cameras": [item for item in payload["camera_tracks"] if item["frame"] == frame],
    }


@router.post("/jobs/{job_id}/review")
async def decide_review(request: Request, job_id: str, body: ReviewDecision) -> dict:
    job = published_job(request, job_id)
    async with cvat_client(request) as client:
        annotations, labels = await saved_annotations(client, job)
        if revision(annotations) != body.expected_revision:
            raise HTTPException(409, "Annotations changed; refresh the review list")
        try:
            local = reviewed_annotations(annotations, labels, body.object_id, body.status)
        except ValueError as error:
            raise HTTPException(409, str(error)) from None
        response = await client.post(
            f"/api/jobs/{job['cvat_job_id']}/autolabel-stream",
            json={
                "mode": "save",
                "operation_id": f"3d-review:{uuid4()}",
                "base": annotations,
                "local": local,
                "require_revision": True,
                "expected_revision": body.expected_revision,
            },
        )
        if response.is_error:
            raise HTTPException(
                409 if response.status_code == 409 else 502, "Review was not saved; refresh annotations before retrying"
            )
    return {"saved": True, "revision": response.json()["revision"]}


@router.get("/cvat-jobs/{cvat_job_id}/camera-boxes")
async def camera_boxes(request: Request, cvat_job_id: int, frame: int = Query(ge=0)) -> dict:
    # The camera viewer is usable by every reviewer who can read this CVAT job,
    # while private VM job APIs continue to be restricted to the upload owner.
    async with cvat_client(request) as client:
        try:
            response = await client.get(f"/api/jobs/{cvat_job_id}")
        except httpx.HTTPError:
            raise HTTPException(503, "CVAT authorization is unavailable") from None
    if response.is_error:
        raise HTTPException(404, "CVAT job not found")
    store = request.app.state.vm_store
    with store.connect() as db:
        row = db.execute(
            "SELECT id FROM vm_jobs WHERE status='completed' "
            "AND json_extract(payload, '$.workflow')='lidar3d' "
            "AND json_extract(payload, '$.cvat_job_id')=? ORDER BY updated DESC LIMIT 1",
            (cvat_job_id,),
        ).fetchone()
    if row is None:
        raise HTTPException(404, "No saved camera detections for this job")
    job = store.get(row["id"])
    if frame >= job["total_frames"]:
        raise HTTPException(404, "Camera frame not found")
    path = store.directory(job["job_id"]) / "private/lidar3d/camera-links.json"
    if not path.is_file():
        raise HTTPException(404, "No saved camera detections for this job")
    payload = await asyncio.to_thread(lambda: json.loads(path.read_bytes()))
    cameras = []
    for camera in payload["camera_tracks"]:
        if camera["frame"] != frame:
            continue
        boxes = []
        for observation in camera["observations"]:
            matches = [
                link["track_3d_id"]
                for link in payload["links"]
                if link["frame"] == frame
                and link["camera"] == camera["camera"]
                and link["status"] == "matched"
                and link["track_2d_id"] == observation["track_id"]
            ]
            boxes.append(
                {
                    "bbox": observation["detector_bbox"],
                    "label": observation["label"],
                    "score": observation["score"],
                    "track_id": observation["track_id"],
                    "linked_3d_track_ids": matches,
                }
            )
        cameras.append({"camera": camera["camera"], "boxes": boxes})
    return {
        "frame": frame,
        "detector": "YOLO26s",
        "tracker": "ByteTrack",
        "coordinates": "native_image_pixels",
        "cameras": cameras,
    }


class DatasetUpload(StrictContract):
    filename: str = Field(min_length=1, max_length=180)
    size: int = Field(gt=0, strict=True)
    dataset_format: Literal["nuscenes", "kitti"]
    mode: Literal["detect", "import"] = "detect"
    scene: str = Field(default="", max_length=40, pattern=r"^[a-zA-Z0-9_-]*$")
    max_frames: int = Field(default=200, ge=1, le=200, strict=True)


@router.post("/uploads")
async def start_upload(request: Request, body: DatasetUpload) -> dict:
    settings, store = request.app.state.settings, request.app.state.vm_store
    if not settings.lidar3d_enabled:
        raise HTTPException(503, "3D workflow is not enabled")
    if body.size > settings.lidar3d_max_upload_bytes:
        raise HTTPException(413, "Dataset exceeds the configured upload limit")
    if (
        Path(body.filename).name != body.filename
        or "/" in body.filename
        or "\\" in body.filename
        or not body.filename.lower().endswith((".zip", ".tar.gz", ".tgz"))
    ):
        raise HTTPException(422, "Upload a ZIP or TAR.GZ native dataset")
    if body.mode == "detect":
        if body.dataset_format == "kitti":
            raise HTTPException(
                422,
                "The installed BEVFusion checkpoint is for nuScenes. KITTI supports annotation import/export; detection needs a KITTI-trained BEVFusion config/checkpoint.",
            )
        if not settings.bevfusion_enabled:
            raise HTTPException(503, "BEVFusion is not ready")
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(settings.lidar3d_service_url + "/health")
                response.raise_for_status()
                if response.json().get("bevfusion_checkpoint_sha256") != BEVFUSION_CHECKPOINT_SHA:
                    raise ValueError("BEVFusion checkpoint differs")
        except (httpx.HTTPError, ValueError):
            raise HTTPException(503, "BEVFusion detector is not ready") from None
    secrets = Secrets(settings)
    credentials = secrets.seal({"sessionid": request.cookies["sessionid"], "csrftoken": request.cookies["csrftoken"]})
    job = store.create(
        request.state.cvat_user_id,
        {
            "workflow": "lidar3d",
            "input_kind": "dataset_upload",
            "dataset_format": body.dataset_format,
            "mode": body.mode,
            "scene": body.scene,
            "max_frames": body.max_frames,
            "upload_name": body.filename,
            "upload_size": body.size,
            "uploaded_bytes": 0,
            "classes": CLASSES if body.dataset_format == "nuscenes" else KITTI_CLASSES,
            "profile": {
                "variant": "bevfusion" if body.mode == "detect" else "imported",
                "pipeline": "lidar3d",
                "checkpoint_sha256": BEVFUSION_CHECKPOINT_SHA if body.mode == "detect" else "",
            },
        },
        credentials,
    )
    return public_job(job) | {"chunk_bytes": CHUNK_BYTES}


@router.put("/uploads/{job_id}")
async def upload_chunk(request: Request, job_id: str, offset: int = Query(ge=0)) -> dict:
    store = request.app.state.vm_store
    data = bytearray()
    async for block in request.stream():
        if len(data) + len(block) > CHUNK_BYTES:
            raise HTTPException(413, "Dataset chunk exceeds the upload chunk limit")
        data.extend(block)
    if not data:
        raise HTTPException(422, "Empty dataset chunk")
    try:
        with worker_lock(store.directory(job_id) / "private/upload.lock"):
            job = store.get(job_id, request.state.cvat_user_id)
            if job.get("input_kind") != "dataset_upload" or job["status"] != "uploading":
                raise HTTPException(409, "Dataset upload is no longer writable")
            path = store.directory(job_id) / "private/dataset.archive"
            size = path.stat().st_size if path.exists() else 0
            if offset > size or offset + len(data) > job["upload_size"]:
                raise HTTPException(409, "Dataset chunk offset or size differs")
            if offset < size:
                with path.open("rb") as stream:
                    stream.seek(offset)
                    if stream.read(len(data)) != data:
                        raise HTTPException(409, "Retried dataset chunk differs from existing bytes")
            else:
                with path.open("ab") as stream:
                    stream.write(data)
                    stream.flush()
            return public_job(store.update(job_id, uploaded_bytes=path.stat().st_size))
    except BlockingIOError:
        raise HTTPException(409, "Another dataset upload operation is in progress") from None


@router.post("/uploads/{job_id}/complete")
async def finish_upload(request: Request, job_id: str) -> dict:
    store = request.app.state.vm_store
    try:
        with worker_lock(store.directory(job_id) / "private/upload.lock"):
            job = store.get(job_id, request.state.cvat_user_id)
            if job.get("input_kind") != "dataset_upload":
                raise HTTPException(409, "Not a dataset upload")
            if job["status"] in ("queued", "running", "completed") and job.get("source_sha256"):
                return public_job(job)
            if job["status"] != "uploading":
                raise HTTPException(409, "Dataset upload is no longer writable")
            path = store.directory(job_id) / "private/dataset.archive"
            if not path.is_file() or path.stat().st_size != job["upload_size"]:
                raise HTTPException(409, "Dataset upload is incomplete")
            digest = await asyncio.to_thread(file_sha256, path)
            return public_job(store.update(job_id, status="queued", stage="lidar3d_validate", source_sha256=digest))
    except BlockingIOError:
        raise HTTPException(409, "Another dataset upload operation is in progress") from None


@router.get("/jobs/{job_id}/export")
async def download_annotations(
    request: Request,
    job_id: str,
    purpose: Literal["training", "draft"] = "training",
    annotation_type: Literal["detection", "tracking"] = "detection",
) -> FileResponse:
    settings, store = request.app.state.settings, request.app.state.vm_store
    job = store.get(job_id, request.state.cvat_user_id)
    directory = store.directory(job_id) / "private/lidar3d"
    manifest_path = directory / "dataset.json"
    if job.get("workflow") != "lidar3d" or not job.get("cvat_job_id") or not manifest_path.is_file():
        raise HTTPException(409, "This job has no native dataset export mapping")
    manifest = json.loads(manifest_path.read_bytes())
    credentials = {"sessionid": request.cookies["sessionid"], "csrftoken": request.cookies.get("csrftoken", "")}
    async with httpx.AsyncClient(
        base_url=settings.cvat_url, headers=UserTaskClient(settings, credentials)._headers(), timeout=120
    ) as client:
        annotations = await client.get(f"/api/jobs/{job['cvat_job_id']}/annotations")
        labels = await client.get(f"/api/labels?task_id={job['cvat_task_id']}&page_size=100")
    if annotations.is_error or labels.is_error:
        raise HTTPException(403, "CVAT annotation export is not authorized")
    extension = "json" if manifest["format"] == "nuscenes" else "zip"
    target = directory / f"export-{uuid4().hex}.{extension}"
    try:
        receipt = await asyncio.to_thread(
            export_annotations,
            manifest,
            annotations.json(),
            labels.json()["results"],
            target,
            purpose=purpose,
            annotation_type=annotation_type,
        )
    except (ValueError, KeyError) as error:
        raise HTTPException(422, str(error)) from None
    atomic_json(
        target.with_suffix(".receipt.json"), receipt | {"cvat_annotation_version": annotations.json().get("version")}
    )
    return FileResponse(
        target,
        media_type="application/json" if extension == "json" else "application/zip",
        filename=f"{manifest['format']}-{manifest['scene']}-{annotation_type}-{purpose}.{extension}",
    )
