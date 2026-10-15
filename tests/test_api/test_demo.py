import pytest


@pytest.mark.asyncio
async def test_demo_page_exposes_video_workflow(client):
    response = await client.get("/demo")

    assert response.status_code == 200
    assert "Run AI labeling" in response.text
    assert "/api/v1/video/jobs" in response.text


@pytest.mark.asyncio
async def test_demo_config_exposes_browser_safe_cvat_url(client):
    response = await client.get("/api/v1/demo/config")

    assert response.status_code == 200
    assert response.json() == {
        "cvat_public_url": "http://localhost:8080",
        "propagation_provider": "kaggle_sam3",
    }
