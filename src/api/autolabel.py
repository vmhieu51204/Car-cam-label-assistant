"""AutoLabel endpoints and compatibility aliases for the previous backend."""

import sqlite3
from typing import Annotated

import httpx
from fastapi import APIRouter, HTTPException, Query, Request

from src.agents.autolabel_graph import autolabel_graph
from src.models.discrepancy import DiscrepancyResult, TrackEvidence
from src.models.perception import DetectRequest, QueueQuery, TelemetryRequest, ValidateRequest
from src.services.discrepancy import DiscrepancyFilter

router = APIRouter()


@router.get("/health")
def health(request: Request) -> dict:
    """Report actual local backend capabilities without claiming CVAT verification."""
    with request.app.state.store.connect() as connection:
        connection.execute("SELECT 1")
    return {
        "status": "ok",
        "data_source": "mock",
        "cvat_gates": "unverified",
        "training_export_enabled": False,
        "env": request.app.state.settings.app_env,
    }


@router.post("/detect")
def detect(payload: DetectRequest, request: Request) -> dict:
    """Persist a new mock run and return its provenance."""
    try:
        result = autolabel_graph.invoke(
            {"settings": request.app.state.settings, "request": payload}, config={"callbacks": []}
        )
        metadata, tracks = result["metadata"], result["tracks"]
    except FileNotFoundError as error:
        raise HTTPException(503, "nuScenes metadata missing; check DATA_DIR") from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    run_id = request.app.state.store.save_run(payload.scene, metadata, tracks)
    return {"run_id": run_id, "track_count": len(tracks), **metadata}


@router.post("/discrepancy/evaluate", response_model=DiscrepancyResult)
def evaluate_discrepancy(payload: TrackEvidence, request: Request) -> DiscrepancyResult:
    """Evaluate supplied evidence; this endpoint does not persist or approve labels."""
    return DiscrepancyFilter(request.app.state.settings.hcl_config).evaluate(payload)


@router.get("/queue")
def queue(request: Request, query: Annotated[QueueQuery, Query()]) -> dict:
    """Read persisted pending tracks with stable pagination."""
    try:
        result = request.app.state.store.queue(query)
    except KeyError as error:
        raise HTTPException(404, "Unknown run_id") from error
    return {
        "run_id": query.run_id,
        "ranking": query.order,
        "calibrated": False,
        "limitations": "HCL review priority only; diversity/quota AL ranking not implemented",
        **result,
    }


@router.post("/validate")
async def validate(payload: ValidateRequest, request: Request) -> dict:
    """Validate semantics in the configured mode; never fall back to mock on error."""
    try:
        results = await request.app.state.vlm.batch(payload.items)
    except httpx.HTTPError as error:
        raise HTTPException(502, "VLM server unavailable; no mock fallback") from error
    except (ValueError, KeyError, IndexError, TypeError) as error:
        raise HTTPException(422, "Missing images or invalid VLM JSON response") from error
    return {
        "results": results,
        "warning": "VLM verdicts are not calibrated probabilities; mock does not inspect images",
    }


@router.post("/telemetry")
def telemetry(payload: TelemetryRequest, request: Request) -> dict:
    """Record client-supplied timing/edits; no automatic human review transition."""
    try:
        created = request.app.state.store.telemetry(payload)
    except sqlite3.IntegrityError as error:
        raise HTTPException(404, "Unknown run/track pair") from error
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    return {"event_id": payload.event_id, "created": created, "data_source": "mock", "review_status_changed": False}
