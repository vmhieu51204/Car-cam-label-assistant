from types import SimpleNamespace

import pytest

from src.api.video import video_capabilities
from src.config import Settings


@pytest.mark.asyncio
async def test_upload_video_creates_persistent_queued_job(client):
    response = await client.post(
        "/api/v1/video/jobs",
        files={"file": ("clip.mp4", b"small-video-fixture", "video/mp4")},
    )

    assert response.status_code == 202
    payload = response.json()
    assert payload["status"] == "queued"
    assert payload["source_name"] == "clip.mp4"
    assert payload["source_size_bytes"] == len(b"small-video-fixture")

    fetched = await client.get(f"/api/v1/video/jobs/{payload['job_id']}")
    assert fetched.status_code == 200
    assert fetched.json() == payload


@pytest.mark.asyncio
async def test_upload_video_rejects_unsupported_extension(client):
    response = await client.post(
        "/api/v1/video/jobs",
        files={"file": ("payload.txt", b"not-a-video", "text/plain")},
    )

    assert response.status_code == 422
    assert "filename" in response.json()["detail"]


@pytest.mark.asyncio
async def test_manifest_is_conflict_until_worker_finishes(client):
    created = await client.post(
        "/api/v1/video/jobs",
        files={"file": ("clip.mp4", b"queued", "video/mp4")},
    )
    job_id = created.json()["job_id"]

    response = await client.get(f"/api/v1/video/jobs/{job_id}/manifest")

    assert response.status_code == 409


@pytest.mark.asyncio
async def test_cvat_import_rejects_budget_over_model_limit_before_contacting_cvat(client):
    response = await client.post(
        "/api/v1/video/jobs/cvat-jobs/7/import",
        json={"strategy": "budget", "budget": 301},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_sam31_experimental_capability_is_hidden_by_default(client):
    response = await client.get("/api/v1/video/jobs/capabilities")
    assert response.status_code == 200
    assert response.json() == {
        "sam3_1_experimental": False,
        "vlm_box_rescue": False,
        "local_sam3": False,
        "local_sam31": False,
        "docked_panel": False,
        "canvas_overlay": False,
        "track_review": False,
        "track_filter": False,
    }


def test_sam31_capability_stays_hidden_even_if_legacy_checkpoint_settings_exist():
    settings = Settings(
        _env_file=None,
        sam31_experimental_enabled=True,
        kaggle_sam31_checkpoint_dataset="private/sam31",
        kaggle_sam31_model_source="owner/model/pyTorch/default/1",
    )
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(settings=settings)))
    assert video_capabilities(request)["sam3_1_experimental"] is False


@pytest.mark.asyncio
async def test_sam31_request_is_temporarily_rejected_for_new_jobs(client):
    response = await client.post(
        "/api/v1/video/jobs/cvat-jobs/7/import",
        json={
            "strategy": "uniform",
            "source_fps": 30,
            "target_fps": 5,
            "auto_start_sequence": True,
            "sequence_classes": ["car"],
            "sam3_model_variant": "sam3.1",
        },
    )
    assert response.status_code == 409
    assert "temporarily unavailable" in response.json()["detail"]


@pytest.mark.asyncio
async def test_vlm_box_rescue_request_is_rejected_when_global_gate_is_off(client):
    response = await client.post(
        "/api/v1/video/jobs/cvat-jobs/7/import",
        json={
            "strategy": "uniform",
            "source_fps": 30,
            "target_fps": 5,
            "auto_start_sequence": True,
            "sequence_classes": ["car"],
            "vlm_box_rescue": True,
        },
    )
    assert response.status_code == 409
    assert "disabled" in response.json()["detail"]
