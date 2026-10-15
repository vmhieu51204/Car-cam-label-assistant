"""Retired 3D semantic provider. Stored artifacts remain readable by their data contracts."""

from src.models.perception import ValidationItem


class VLMClient:
    def __init__(self, settings: object, transport: object = None) -> None:
        pass

    async def batch(self, items: list[ValidationItem]) -> list[dict[str, object]]:
        raise ValueError("Legacy semantic VLM execution is retired; use Claude video box proposals")
