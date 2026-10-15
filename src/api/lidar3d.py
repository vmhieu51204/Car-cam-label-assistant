"""Authenticated scene selection for the provisioned real LiDAR workflow."""

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import Field

from src.api.lidar3d_upload import router as upload_router
from src.api.vm import get_vm, identity, public_job
from src.models.video import StrictContract
from src.services.vm.lidar3d import CLASSES, dataset
from src.services.vm.lidar3d_contracts import BEVFUSION_CHECKPOINT_SHA, CHUNK_BYTES
from src.services.vm.vlm import Secrets

router = APIRouter(dependencies=[Depends(identity)])
router.include_router(upload_router)


class Lidar3DRequest(StrictContract):
    scene: str = Field(pattern=r"^scene-\d{4}$")
    max_frames: int = Field(default=10, ge=1, le=200, strict=True)


@router.get("/capabilities")
def capabilities(request: Request) -> dict:
    settings = request.app.state.settings
    try:
        manifest = dataset(settings)
    except (ValueError, OSError):
        manifest = {"scenes": {}}
    return {
        "enabled": settings.lidar3d_enabled,
        "model": "BEVFusion",
        "detection_enabled": settings.bevfusion_enabled,
        "formats": {
            "nuscenes": {"import": True, "export": True, "detect": settings.bevfusion_enabled},
            "kitti": {"import": True, "export": True, "detect": False},
        },
        "max_upload_bytes": settings.lidar3d_max_upload_bytes,
        "chunk_bytes": CHUNK_BYTES,
        "scenes": [{"name": name, "frames": len(scene["frames"])} for name, scene in manifest["scenes"].items()],
    }


@router.post("/jobs")
async def create_job(request: Request, body: Lidar3DRequest) -> dict:
    settings, store = request.app.state.settings, request.app.state.vm_store
    try:
        if not settings.bevfusion_enabled:
            raise ValueError("BEVFusion is not enabled")
        manifest = dataset(settings)
        if body.scene not in manifest["scenes"]:
            raise ValueError("Choose a provisioned 3D scene")
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(settings.lidar3d_service_url + "/health")
            response.raise_for_status()
        if (
            response.json().get("bevfusion_checkpoint_sha256") != BEVFUSION_CHECKPOINT_SHA
            or response.json().get("dataset_manifest_sha256") != settings.lidar3d_manifest_sha256
        ):
            raise ValueError("3D detector/dataset pin differs")
    except (ValueError, OSError, httpx.HTTPError):
        raise HTTPException(503, "3D detector or dataset is not ready") from None
    credentials = Secrets(settings).seal(
        {"sessionid": request.cookies["sessionid"], "csrftoken": request.cookies["csrftoken"]}
    )
    job = store.create(
        request.state.cvat_user_id,
        {
            "workflow": "lidar3d",
            "scene": body.scene,
            "classes": CLASSES,
            "profile": {
                "variant": "bevfusion",
                "pipeline": "lidar3d",
                "manifest_sha256": settings.lidar3d_manifest_sha256,
                "checkpoint_sha256": BEVFUSION_CHECKPOINT_SHA,
            },
        },
        credentials,
    )
    return public_job(
        store.update(
            job["job_id"],
            status="queued",
            stage="queued_lidar3d",
            total_frames=min(body.max_frames, len(manifest["scenes"][body.scene]["frames"])),
        )
    )


@router.get("/jobs/{job_id}")
def processing(request: Request, job_id: str) -> dict:
    return get_vm(request, job_id)
