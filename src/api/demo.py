"""Legacy standalone diagnostic UI; the CVAT fork has the integrated panel."""

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter()
DEMO_HTML = Path(__file__).resolve().parents[1] / "web/demo.html"


@router.get("/demo", response_class=HTMLResponse, include_in_schema=False)
def demo_page() -> HTMLResponse:
    return HTMLResponse(DEMO_HTML.read_text(encoding="utf-8"))


@router.get("/api/v1/demo/config")
def demo_config(request: Request) -> dict[str, str]:
    settings = request.app.state.settings
    return {
        "cvat_public_url": settings.cvat_public_url.rstrip("/"),
        "propagation_provider": settings.propagation_provider,
    }
