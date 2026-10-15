"""Production must not silently run a demonstration validator."""

import pytest
from pydantic import ValidationError

from src.config import Settings


def test_production_accepts_default_settings():
    assert Settings(_env_file=None, app_env="production").publish_requires_review is True


def test_production_cannot_disable_human_review():
    with pytest.raises(ValidationError, match="human review"):
        Settings(_env_file=None, app_env="production", publish_requires_review=False)
