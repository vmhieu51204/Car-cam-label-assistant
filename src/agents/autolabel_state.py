"""Typed state for deterministic local mock detection and HCL assessment."""

from typing import TypedDict

from src.config import Settings
from src.models.perception import DetectRequest, Track


class AutoLabelState(TypedDict, total=False):
    settings: Settings
    request: DetectRequest
    metadata: dict
    tracks: list[Track]
