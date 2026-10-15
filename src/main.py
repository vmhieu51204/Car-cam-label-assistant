"""AutoLabel 3D application entry point, retaining the AI20K template layout."""

import logging
import sqlite3
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from src.api.autolabel import router as autolabel_router
from src.api.demo import router as demo_router
from src.api.lidar3d import router as lidar3d_router
from src.api.video import router as video_router
from src.api.vm import router as vm_router
from src.config import Settings, get_settings
from src.services.cvat import CvatReadClient, CvatTaskClient, CvatTrackClient
from src.services.cvat.session_authorizer import CvatSessionAuthorizer
from src.services.storage.repository import Store
from src.services.video import CvatSeedReader, VideoJobService, VideoWorkspace
from src.services.video.claude_box_rescue import claude_rescue_is_ready
from src.services.vlm.semantic import VLMClient
from src.services.vm.storage import VmStore
from src.services.vm.vlm import VlmProfiles

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create an isolated application; initialize local storage only during startup."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Initialize per-process local services during application startup."""
        app.state.settings = settings
        app.state.store = Store(settings.db_path)
        app.state.vm_store = VmStore(settings.db_path, settings.vm_work_root)
        app.state.vlm_profiles = VlmProfiles(app.state.vm_store, settings)
        app.state.vlm = VLMClient(settings)
        app.state.video_workspace = VideoWorkspace(settings.video_work_root)
        app.state.cvat_read = CvatReadClient.from_settings(settings)
        app.state.cvat_authorizer = CvatSessionAuthorizer(settings)
        app.state.cvat_tasks = CvatTaskClient.from_settings(settings)
        app.state.cvat_tracks = CvatTrackClient.from_settings(settings, read_client=app.state.cvat_read)
        import json

        from src.services.video.track_filter import TrackFilterConfig

        filter_config = TrackFilterConfig.model_validate(json.loads(settings.track_filter_rules))
        filter_config = filter_config.model_copy(update={"enabled": settings.track_filter_enabled})
        app.state.video_jobs = VideoJobService(
            store=app.state.store,
            workspace=app.state.video_workspace,
            seed_reader=CvatSeedReader(app.state.cvat_read),
            task_client=app.state.cvat_tasks,
            track_client=app.state.cvat_tracks,
            max_upload_bytes=settings.video_max_upload_bytes,
            track_filter_config=filter_config,
            box_rescue_enabled=(
                settings.vlm_box_rescue_enabled
                and claude_rescue_is_ready(
                    enabled=True,
                    api_key=settings.openrouter_api_key.get_secret_value(),
                    model=settings.vlm_box_rescue_model,
                )
            ),
        )
        yield

    production = settings.app_env == "production"
    app = FastAPI(
        title=settings.app_name,
        version="0.2.0",
        lifespan=lifespan,
        docs_url=None if production else "/docs",
        redoc_url=None if production else "/redoc",
        openapi_url=None if production else "/openapi.json",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[] if production else settings.cors_origins.split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def record_video_action_outcome(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            intent_id = getattr(request.state, "video_intent_id", None)
            if intent_id is not None:
                try:
                    request.app.state.store.finish_video_user_intent(intent_id, status)
                except (sqlite3.Error, ValueError):
                    logger.exception("Could not record CVAT action outcome")

    app.include_router(video_router, prefix="/api/v1", tags=["Video workflow"])
    app.include_router(vm_router, prefix="/api/v1", tags=["Privacy-first VM"])
    app.include_router(lidar3d_router, prefix="/api/v1/vm/3d", tags=["LiDAR 3D"])
    if production:

        @app.get("/api/v1/demo/config", include_in_schema=False)
        def video_ui_config() -> dict[str, str]:
            return {
                "cvat_public_url": settings.cvat_public_url.rstrip("/"),
                "propagation_provider": settings.propagation_provider,
            }
    else:
        app.include_router(demo_router, tags=["Demo"])
        app.include_router(autolabel_router, prefix="/api/v1", tags=["AutoLabel"])
        app.include_router(autolabel_router, include_in_schema=False)
        app.add_api_route("/", lambda: RedirectResponse("/docs"), include_in_schema=False)

    @app.get("/healthz", include_in_schema=False)
    def healthz() -> dict[str, str]:
        with app.state.store.connect() as connection:
            connection.execute("SELECT 1")
        return {"status": "ok"}

    return app


app = create_app()
