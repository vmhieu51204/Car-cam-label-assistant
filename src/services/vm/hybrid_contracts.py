"""Frozen, local-only detector/tracker identity for experimental hybrid jobs."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from src.config import PROJECT_ROOT, Settings
from src.models.video import StrictContract

Pipeline = Literal["sam", "hybrid_frame", "hybrid_propagate"]
ULTRALYTICS_VERSION: Literal["8.4.63"] = "8.4.63"
TRACKER_PATH = PROJECT_ROOT / "config/bytetrack-pilot.json"


class HybridProfile(StrictContract):
    model_config = ConfigDict(frozen=True)
    detector: Literal["yolo26s"] = "yolo26s"
    checkpoint: str
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    classes_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ultralytics_version: Literal["8.4.63"] = ULTRALYTICS_VERSION
    class_mapping: dict[str, int]
    detector_names: dict[int, str]
    tracker: Literal["bytetrack"] = "bytetrack"
    tracker_config: dict
    tracker_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    object_batch: int = Field(default=8, ge=1, le=32)
    image_size: int = Field(default=640, ge=32)
    detector_conf: float = Field(default=0.1, gt=0, lt=1)
    detector_iou: float = Field(default=0.7, gt=0, le=1)
    prompt_stride: int = Field(default=15, ge=1)
    lost_mask_frames: int = Field(default=15, ge=0, le=30)

    @model_validator(mode="after")
    def valid(self) -> HybridProfile:
        if not self.class_mapping or len(set(self.class_mapping.values())) != len(self.class_mapping):
            raise ValueError("Hybrid classes must map one-to-one to detector classes")
        if any(type(i) is not int or i not in self.detector_names for i in self.class_mapping.values()):
            raise ValueError("Detector does not support the requested class")
        from src.services.vm.storage import encode

        if (
            self.tracker_config.get("tracker_type") != "bytetrack"
            or hashlib.sha256(encode(self.tracker_config)).hexdigest() != self.tracker_sha256
        ):
            raise ValueError("ByteTrack config pin differs")
        return self


def hybrid_preflight(settings: Settings, classes: list[str]) -> HybridProfile:
    from src.services.vm.contracts import file_sha256
    from src.services.vm.storage import encode

    if not settings.hybrid_enabled:
        raise ValueError("Hybrid is disabled; enable it explicitly for a pilot")
    checkpoint, manifest = settings.hybrid_yolo_checkpoint, settings.hybrid_classes_manifest
    if checkpoint is None or checkpoint.name != "yolo26s.pt" or not checkpoint.is_file():
        raise ValueError("Configure a local yolo26s.pt checkpoint; no automatic download")
    if file_sha256(checkpoint) != settings.hybrid_yolo_sha256:
        raise ValueError("YOLO checkpoint SHA-256 mismatch")
    if manifest is None or not manifest.is_file() or file_sha256(manifest) != settings.hybrid_classes_sha256:
        raise ValueError("Pin the detector class manifest SHA-256")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("checkpoint_sha256") != settings.hybrid_yolo_sha256 or data.get("model") != "yolo26s":
        raise ValueError("Detector class manifest belongs to another checkpoint/model")
    try:
        names = {int(k): str(v) for k, v in data["names"].items()}
        mapping = json.loads(settings.hybrid_class_mapping)
        if not classes or not isinstance(mapping, dict) or any(type(v) is not int for v in mapping.values()):
            raise ValueError("Hybrid requires explicit integer detector class IDs")
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Invalid detector class manifest or mapping") from error
    if any(c not in mapping for c in classes):
        raise ValueError("Unsupported hybrid class; configure an explicit detector class mapping")
    tracker = json.loads(TRACKER_PATH.read_text(encoding="utf-8"))
    return HybridProfile(
        checkpoint=str(checkpoint.resolve()),
        checkpoint_sha256=settings.hybrid_yolo_sha256,
        classes_manifest_sha256=settings.hybrid_classes_sha256,
        class_mapping={c: mapping[c] for c in classes},
        detector_names=names,
        tracker_config=tracker,
        tracker_sha256=hashlib.sha256(encode(tracker)).hexdigest(),
    )
