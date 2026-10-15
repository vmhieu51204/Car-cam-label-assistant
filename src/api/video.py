"""HTTP surface for the video-first Modules 0-2 workflow."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.api.video_auth import require_cvat_job_permission
from src.models.video import (
    AcceptBoxRescueRequest,
    AttachSeedRequest,
    CreateCvatTaskRequest,
    CvatPublishResult,
    CvatSamplingRequest,
    CvatTaskBinding,
    DiscoveryResult,
    FrameManifest,
    PublishCvatTracksRequest,
    ReviewQueue,
    SeedSet,
    StrictContract,
    TrackSet,
    VideoJob,
)
from src.services.cvat import CvatReadError, CvatWriteError
from src.services.video import SeedContractError, VideoJobService, VideoUploadError
from src.services.video.audit import VideoAuditError
from src.services.video.claude_box_rescue import claude_rescue_is_ready
from src.services.video.manual_frame_rescue import decide_frame, propose_frame

router = APIRouter(prefix="/video/jobs", dependencies=[Depends(require_cvat_job_permission)])


def _service(request: Request) -> VideoJobService:
    return request.app.state.video_jobs


class ManualFrameRequest(BaseModel):
    source_frame: int = Field(ge=0, strict=True)


class ManualDecisionRequest(BaseModel):
    index: int = Field(ge=0, strict=True)
    accept: bool


@router.get("/{job_id}/manual-claude/latest")
def latest_manual_claude(request: Request, job_id: str, source_frame: int | None = Query(default=None, ge=0)) -> dict:
    import json

    service = _service(request)
    try:
        service.store.get_video_job(job_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="video job not found") from error
    directory = service.workspace.job_dir(job_id) / "manual-claude"
    for path in sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime_ns, reverse=True):
        result = json.loads(path.read_text(encoding="utf-8"))
        if source_frame is None or result["source_frame"] == source_frame:
            return {"result": result}
    return {"result": None}


@router.post("/{job_id}/manual-claude")
async def manual_claude(request: Request, job_id: str, body: ManualFrameRequest) -> dict:
    try:
        return await propose_frame(_service(request), request.app.state.settings, job_id, body.source_frame)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="video job not found") from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except (CvatReadError, VideoAuditError) as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@router.post("/{job_id}/manual-claude/{result_id}/decision")
async def manual_claude_decision(request: Request, job_id: str, result_id: str, body: ManualDecisionRequest) -> dict:
    try:
        return await decide_frame(_service(request), job_id, result_id, body.index, body.accept)
    except (KeyError, FileNotFoundError) as error:
        raise HTTPException(status_code=404, detail="proposal not found") from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except (CvatWriteError, CvatReadError) as error:
        raise HTTPException(status_code=409 if error.status_code == 409 else 502, detail=str(error)) from error


@router.get("/capabilities")
@router.get("/cvat-jobs/{cvat_job_id}/capabilities")
def video_capabilities(request: Request) -> dict[str, bool]:
    """Expose choices enabled for the current CVAT workflow."""
    settings = request.app.state.settings
    from src.services.video.local_sam import local_ready

    return {
        "docked_panel": settings.docked_panel_enabled,
        "canvas_overlay": settings.claude_canvas_overlay_enabled,
        "track_review": settings.track_review_enabled,
        "track_filter": settings.track_filter_enabled,
        "sam3_1_experimental": local_ready(settings, "sam3.1"),
        "local_sam3": local_ready(settings, "sam3"),
        "local_sam31": local_ready(settings, "sam3.1"),
        "vlm_box_rescue": claude_rescue_is_ready(
            enabled=settings.vlm_box_rescue_enabled,
            api_key=settings.openrouter_api_key.get_secret_value(),
            model=settings.vlm_box_rescue_model,
        ),
    }


@router.get("/{job_id}/box-rescue")
def get_box_rescue_state(request: Request, job_id: str) -> dict[str, object]:
    try:
        return _service(request).get_box_rescue_state(job_id)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="video job not found") from error


@router.post("/{job_id}/box-rescue/retry", status_code=status.HTTP_202_ACCEPTED)
def retry_box_rescue(request: Request, job_id: str) -> dict[str, object]:
    try:
        return _service(request).retry_box_rescue(job_id)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="video job not found") from error
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error


@router.post("/{job_id}/box-rescue/{proposal_id}/accept")
async def accept_box_rescue_proposal(
    request: Request, job_id: str, proposal_id: str, body: AcceptBoxRescueRequest
) -> dict[str, object]:
    try:
        return await _service(request).accept_box_rescue(
            job_id,
            proposal_id,
            expected_revision=body.expected_annotation_revision,
        )
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="proposal or video job not found") from error
    except (CvatWriteError, CvatReadError) as error:
        code = status.HTTP_409_CONFLICT if error.status_code == 409 else status.HTTP_502_BAD_GATEWAY
        raise HTTPException(status_code=code, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error


@router.get("/{job_id}/box-rescue/revision")
async def get_box_rescue_revision(request: Request, job_id: str) -> dict[str, int]:
    try:
        service = _service(request)
        job = service.store.get_video_job(job_id)
        if job.cvat_job_id is None:
            raise ValueError("video job has no CVAT target")
        annotations = await service.seed_reader.client.get_guarded_job_annotations(job.cvat_job_id)
        return {"revision": annotations["autolabel_revision"]}
    except KeyError as error:
        raise HTTPException(status_code=404, detail="video job not found") from error
    except (CvatReadError, ValueError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.post("/{job_id}/box-rescue/{proposal_id}/reject")
def reject_box_rescue_proposal(request: Request, job_id: str, proposal_id: str) -> dict[str, object]:
    try:
        return _service(request).reject_box_rescue(job_id, proposal_id)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="proposal or video job not found") from error
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error


@router.get("/{job_id}/box-rescue/{proposal_id}/preview")
def get_box_rescue_preview(request: Request, job_id: str, proposal_id: str) -> FileResponse:
    try:
        state = _service(request).get_box_rescue_state(job_id)
        if not any(row.get("proposal_id") == proposal_id for row in state.get("proposals", [])):
            raise KeyError(proposal_id)
        path = request.app.state.video_workspace.box_rescue_preview_path(job_id, proposal_id)
        if not path.is_file():
            raise ValueError("proposal preview is unavailable")
        return FileResponse(path, media_type="image/png")
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="proposal or video job not found") from error
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error


@router.get("/cvat-jobs/{cvat_job_id}", response_model=VideoJob)
def get_imported_cvat_job(request: Request, cvat_job_id: int) -> VideoJob:
    try:
        job = _service(request).find_cvat_job(cvat_job_id)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)) from error
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="CVAT job is not linked to AutoLabel")
    return job


@router.get("/cvat-jobs/{cvat_job_id}/labels", response_model=list[str])
async def get_cvat_job_label_names(request: Request, cvat_job_id: int) -> list[str]:
    try:
        return await _service(request).get_sequence_class_names(cvat_job_id)
    except CvatReadError as error:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error


@router.post("/cvat-jobs/{cvat_job_id}/import", response_model=VideoJob, status_code=status.HTTP_202_ACCEPTED)
async def import_cvat_job(request: Request, cvat_job_id: int, body: CvatSamplingRequest | None = None) -> VideoJob:
    settings = request.app.state.settings
    from src.services.video.local_sam import local_ready

    if body and body.sam_execution_provider == "local" and not local_ready(settings, body.sam3_model_variant):
        raise HTTPException(status_code=409, detail="Local SAM interpreter/checkpoint/SHA-256 are not configured")
    if body and body.sam3_model_variant == "sam3.1" and body.sam_execution_provider != "local":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="SAM 3.1 is temporarily unavailable for new jobs; use SAM3",
        )
    if (
        body
        and body.vlm_box_rescue
        and not (
            claude_rescue_is_ready(
                enabled=settings.vlm_box_rescue_enabled,
                api_key=settings.openrouter_api_key.get_secret_value(),
                model=settings.vlm_box_rescue_model,
            )
        )
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Claude box suggestions are disabled or require an OpenRouter key and Sonnet 5.5 configuration",
        )
    try:
        return await _service(request).import_cvat_job(cvat_job_id, body)
    except CvatReadError as error:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error
    except VideoUploadError as error:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)) from error


@router.post("", response_model=VideoJob, status_code=status.HTTP_202_ACCEPTED)
async def create_video_job(request: Request, file: Annotated[UploadFile, File(...)]) -> VideoJob:
    """Stream a bounded video into the job workspace and queue frame extraction."""
    try:
        return await _service(request).create_job(file)
    except VideoUploadError as error:
        code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE if error.too_large else status.HTTP_422_UNPROCESSABLE_CONTENT
        raise HTTPException(status_code=code, detail=str(error)) from error


@router.get("/{job_id}", response_model=VideoJob)
def get_video_job(request: Request, job_id: str) -> VideoJob:
    try:
        return _service(request).get_job(job_id)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="video job not found") from error


@router.get("/{job_id}/manifest", response_model=FrameManifest)
def get_frame_manifest(request: Request, job_id: str) -> FrameManifest:
    try:
        return _service(request).get_manifest(job_id)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="video job not found") from error
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error


@router.get("/{job_id}/seed", response_model=SeedSet)
def get_video_seed(request: Request, job_id: str) -> SeedSet:
    try:
        return _service(request).get_seed(job_id)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="video job not found") from error
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error


@router.post("/{job_id}/cancel", response_model=VideoJob)
def cancel_video_job(request: Request, job_id: str) -> VideoJob:
    try:
        return _service(request).cancel_job(job_id)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="video job not found") from error
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error


@router.post("/{job_id}/propagate", response_model=VideoJob, status_code=status.HTTP_202_ACCEPTED)
def start_video_propagation(request: Request, job_id: str) -> VideoJob:
    try:
        return _service(request).start_propagation(job_id)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="video job not found") from error
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error


@router.post("/{job_id}/retry-import", response_model=VideoJob, status_code=status.HTTP_202_ACCEPTED)
def retry_cvat_frame_import(request: Request, job_id: str) -> VideoJob:
    try:
        return _service(request).retry_cvat_import(job_id)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="video job not found") from error
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error


@router.post("/{job_id}/sequence", response_model=VideoJob, status_code=status.HTTP_202_ACCEPTED)
def start_seed_free_sequence(request: Request, job_id: str) -> VideoJob:
    try:
        return _service(request).start_sequence(job_id)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="video job not found") from error
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error


@router.post("/{job_id}/sequence/classes", response_model=VideoJob, status_code=status.HTTP_202_ACCEPTED)
async def retry_sequence_with_classes(request: Request, job_id: str, body: list[str]) -> VideoJob:
    try:
        return await _service(request).retry_sequence_with_classes(job_id, body)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="video job not found") from error
    except CvatReadError as error:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error


@router.get("/{job_id}/tracks", response_model=TrackSet)
def get_video_tracks(request: Request, job_id: str) -> TrackSet:
    try:
        return _service(request).get_tracks(job_id)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="video job not found") from error
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error


@router.post("/{job_id}/discover", response_model=VideoJob, status_code=status.HTTP_202_ACCEPTED)
def retry_video_discovery(request: Request, job_id: str) -> VideoJob:
    try:
        return _service(request).retry_discovery(job_id)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="video job not found") from error
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error


@router.get("/{job_id}/discovery", response_model=DiscoveryResult)
def get_video_discovery(request: Request, job_id: str) -> DiscoveryResult:
    try:
        return _service(request).get_discovery(job_id)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="video job not found") from error
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error


@router.get("/{job_id}/audit-state")
def get_video_audit_state(request: Request, job_id: str) -> dict[str, object]:
    try:
        return _service(request).get_sequence_audit_state(job_id) or {"status": "disabled"}
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="video job not found") from error


class ReviewStateRequest(StrictContract):
    track_id: str = Field(min_length=1, max_length=128)
    state: str = Field(pattern=r"^(unseen|seen|resolved)$")


def _track_review(request: Request, job_id: str) -> list[dict]:
    import json
    from dataclasses import replace

    from src.services.video.sequence_review import SequenceReviewConfig
    from src.services.video.track_triage import track_triage

    settings = request.app.state.settings
    if not settings.track_review_enabled:
        raise HTTPException(status_code=409, detail="Track review is disabled")
    service = _service(request)
    config = SequenceReviewConfig(class_review_policy={})
    overrides = json.loads(settings.track_review_weights)
    if set(overrides) - config.weights.keys():
        raise HTTPException(status_code=409, detail="Unknown review weight")
    weights = config.weights | overrides
    total = sum(weights.values())
    if total <= 0:
        raise HTTPException(status_code=409, detail="Review weights must be positive")
    config = replace(config, weights={k: v / total for k, v in weights.items()})
    rows = track_triage(service.get_tracks(job_id), service.get_manifest(job_id), config)
    states = service.store.track_review_states(job_id)
    for row in rows:
        row["state"] = states.get(row["track_id"], "unseen")
    return rows


@router.get("/{job_id}/review")
def get_track_review(
    request: Request,
    job_id: str,
    level: str | None = None,
    class_name: str | None = None,
    reason: str | None = None,
    viewed: bool | None = None,
) -> dict:
    try:
        rows = _track_review(request, job_id)
        filtered = [
            r
            for r in rows
            if (level is None or r["level"] == level)
            and (class_name is None or r["class"] == class_name)
            and (reason is None or reason in r["reasons"])
            and (viewed is None or (r["state"] != "unseen") == viewed)
        ]
        return {"tracks": filtered, "total": len(rows), "seen": sum(r["state"] != "unseen" for r in rows)}
    except KeyError as error:
        raise HTTPException(status_code=404, detail="video job not found") from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.post("/{job_id}/review/state")
def set_track_review(request: Request, job_id: str, body: ReviewStateRequest) -> dict:
    try:
        rows = _track_review(request, job_id)
        if body.track_id not in {r["track_id"] for r in rows}:
            raise HTTPException(status_code=404, detail="track not found")
        _service(request).store.set_track_review_state(job_id, body.track_id, body.state)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="video job not found") from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return {"track_id": body.track_id, "state": body.state}


@router.get("/{job_id}/review/navigation")
def navigate_track_review(
    request: Request,
    job_id: str,
    current_track_id: str | None = None,
    direction: str = Query(default="next", pattern="^(next|prev)$"),
    level: str | None = None,
    class_name: str | None = None,
    reason: str | None = None,
    viewed: bool | None = None,
) -> dict:
    rows = get_track_review(request, job_id, level, class_name, reason, viewed)["tracks"]
    if not rows:
        return {"track": None}
    index = next((i for i, r in enumerate(rows) if r["track_id"] == current_track_id), None)
    target = (index + (1 if direction == "next" else -1)) % len(rows) if index is not None else 0
    return {"track": rows[target]}


class BulkRescueRequest(StrictContract):
    proposal_ids: list[str] = Field(min_length=1, max_length=20)
    accept: bool
    expected_annotation_revision: int = Field(ge=0, strict=True)


@router.post("/{job_id}/box-rescue/bulk")
async def bulk_box_rescue(request: Request, job_id: str, body: BulkRescueRequest) -> dict:
    service = _service(request)
    state = service.get_box_rescue_state(job_id)
    if state["status"] != "done":
        raise HTTPException(status_code=409, detail="Wait for Claude to complete")
    results = []
    revision = body.expected_annotation_revision
    for proposal_id in dict.fromkeys(body.proposal_ids):
        try:
            if body.accept:
                result = await service.accept_box_rescue(job_id, proposal_id, expected_revision=revision)
                # Chain only our successful writes; an external edit still conflicts on the next proposal.
                if result.get("revision_after") is not None:
                    revision = int(result["revision_after"])
            else:
                result = service.reject_box_rescue(job_id, proposal_id)
            results.append(
                {"proposal_id": proposal_id, "status": "accepted" if body.accept else "rejected", "result": result}
            )
        except (ValueError, KeyError, CvatWriteError, CvatReadError) as error:
            results.append(
                {
                    "proposal_id": proposal_id,
                    "status": "conflict" if getattr(error, "status_code", None) == 409 else "failed",
                    "error": str(error),
                }
            )
    return {"results": results, "revision": revision}


class IgnoreRegionsRequest(StrictContract):
    regions: list[tuple[float, float, float, float]] = Field(max_length=20)


@router.post("/{job_id}/suspect-tracks/regions")
def set_ignore_regions(request: Request, job_id: str, body: IgnoreRegionsRequest) -> dict:
    from src.services.video.track_filter import TrackFilterConfig

    if not request.app.state.settings.track_filter_enabled:
        raise HTTPException(status_code=409, detail="Track filter is disabled")
    try:
        config = TrackFilterConfig(ignore_regions=body.regions)
        _service(request).store.set_export_options(job_id, {"ignore_regions": config.ignore_regions})
        return {"ignore_regions": config.ignore_regions, "annotations_changed": False}
    except KeyError as error:
        raise HTTPException(status_code=404, detail="video job not found") from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.get("/{job_id}/suspect-tracks")
def get_suspect_tracks(request: Request, job_id: str) -> dict:
    import json

    from src.services.video.track_filter import TrackFilterConfig, classify_tracks

    service = _service(request)
    settings = request.app.state.settings
    config = TrackFilterConfig.model_validate(json.loads(settings.track_filter_rules))
    try:
        options = service.store.get_export_options(job_id)
        config = config.model_copy(
            update={
                "enabled": settings.track_filter_enabled,
                "ignore_regions": options.get("ignore_regions", config.ignore_regions),
            }
        )
        tracks, manifest = service.get_tracks(job_id), service.get_manifest(job_id)
        return {
            "suspects": classify_tracks(tracks.tracks, config, width=manifest.width, height=manifest.height),
            "annotations_hidden": False,
            "uncalibrated": True,
        }
    except KeyError as error:
        raise HTTPException(status_code=404, detail="video job not found") from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.get("/{job_id}/review-queue", response_model=ReviewQueue)
def get_video_review_queue(request: Request, job_id: str) -> ReviewQueue:
    try:
        return _service(request).get_review_queue(job_id)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="video job not found") from error
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error


@router.get("/{job_id}/publish", response_model=CvatPublishResult)
def get_video_publish_result(request: Request, job_id: str) -> CvatPublishResult:
    try:
        return _service(request).get_publish_result(job_id)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="video job not found") from error
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error


@router.post("/{job_id}/publish", response_model=CvatPublishResult)
async def publish_video_tracks(request: Request, job_id: str, body: PublishCvatTracksRequest) -> CvatPublishResult:
    try:
        return await _service(request).publish_tracks(
            job_id,
            expected_revision=body.expected_annotation_revision,
        )
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="video job not found") from error
    except CvatWriteError as error:
        code = status.HTTP_409_CONFLICT if error.status_code == 409 else status.HTTP_502_BAD_GATEWAY
        raise HTTPException(status_code=code, detail=str(error)) from error
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error


@router.post("/{job_id}/seed", response_model=SeedSet)
async def attach_video_seed(request: Request, job_id: str, body: AttachSeedRequest) -> SeedSet:
    try:
        return await _service(request).attach_seed(
            job_id,
            cvat_job_id=body.cvat_job_id,
            annotation_revision=body.annotation_revision,
        )
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="video job not found") from error
    except SeedContractError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    except CvatReadError as error:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error


@router.post("/{job_id}/seed-current", response_model=SeedSet)
async def capture_current_video_seed(request: Request, job_id: str) -> SeedSet:
    try:
        return await _service(request).capture_current_seed(job_id)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="video job not found") from error
    except SeedContractError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    except CvatReadError as error:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error


@router.post("/{job_id}/seed-sam3", response_model=VideoJob, status_code=status.HTTP_202_ACCEPTED)
def queue_sam3_frame_zero(request: Request, job_id: str) -> VideoJob:
    raise HTTPException(status_code=410, detail="Frame-zero execution retired; create a whole-video SAM3 job")


@router.post("/{job_id}/cvat-task", response_model=CvatTaskBinding, status_code=status.HTTP_201_CREATED)
async def create_cvat_task(request: Request, job_id: str, body: CreateCvatTaskRequest) -> CvatTaskBinding:
    try:
        return await _service(request).create_cvat_task(job_id, body)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="video job not found") from error
    except CvatWriteError as error:
        detail = str(error)
        if error.task_id is not None:
            detail = f"{detail}; inspect possible CVAT task {error.task_id} before retrying"
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=detail) from error
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
