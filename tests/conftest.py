from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.config import Settings
from src.main import create_app


@pytest_asyncio.fixture
async def client(tmp_path):
    """Async HTTP client for testing API endpoints."""
    app = create_app(
        Settings(
            _env_file=None,
            app_env="test",
            db_path=tmp_path / "app.sqlite3",
            vlm_cache_path=tmp_path / "vlm.sqlite3",
            video_work_root=tmp_path / "video-jobs",
            propagation_provider="kaggle_sam3",
            vlm_box_rescue_enabled=False,
            local_sam_enabled=False,
        )
    )
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


@pytest.fixture
def mock_llm():
    """Mock LLM to avoid calling OpenAI during tests.

    Usage in test:
        def test_something(mock_llm):
            # LLM calls will return mock response instead of hitting OpenAI
            ...
    """
    mock = AsyncMock()
    mock.ainvoke.return_value = AsyncMock(content="Mocked LLM response")
    return mock
