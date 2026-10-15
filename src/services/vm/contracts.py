"""Immutable execution profiles and early readiness checks for VM jobs."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from src.config import Settings
from src.models.video import StrictContract
from src.services.vm.hybrid_contracts import HybridProfile, Pipeline, hybrid_preflight


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


class VmProfile(StrictContract):
    model_config = ConfigDict(frozen=True)
    schema_id: Literal["autolabel-vm-profile/v1"] = Field(default="autolabel-vm-profile/v1", alias="schema")
    variant: Literal["sam3", "sam3.1"] = "sam3.1"
    pipeline: Pipeline = "sam"
    hybrid: HybridProfile | None = None
    window: int = Field(default=48, ge=2)
    overlap: int = Field(default=4, ge=1)
    max_objects: int = Field(default=32, ge=1)
    grounding_batch_size: int = Field(default=2, ge=1)
    compile: Literal[False] = False
    flash_attention3: Literal[False] = False
    precision: Literal["bf16"] = "bf16"
    tf32: Literal[True] = True
    vram_fraction: float = Field(default=0.9, gt=0, le=1)
    detection_threshold: float = Field(default=0.5, ge=0, le=1)
    new_detection_threshold: float = Field(default=0.7, ge=0, le=1)
    checkpoint: str
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    input_kind: Literal["redacted"] = "redacted"

    @model_validator(mode="after")
    def valid_window(self) -> VmProfile:
        if self.overlap >= self.window:
            raise ValueError("Overlap must be smaller than the window")
        if self.pipeline != "sam" and (self.variant != "sam3" or self.hybrid is None):
            raise ValueError("Hybrid requires SAM3 instance APIs and a pinned detector profile")
        if self.pipeline == "sam" and self.hybrid is not None:
            raise ValueError("Baseline cannot contain a hybrid detector profile")
        return self


def preflight(
    settings: Settings, variant: str = "sam3.1", pipeline: Pipeline = "sam", classes: list[str] | None = None
) -> VmProfile:
    """Check local assets, not GPU quality; no credential or provider requests."""
    if not settings.vm_workflow_enabled or not settings.local_sam_enabled:
        raise ValueError("VM workflow and local SAM must be enabled")
    if pipeline != "sam" and variant != "sam3":
        raise ValueError("Hybrid requires SAM3 instance APIs; choose SAM3 before uploading")
    for name, path, digest in (
        ("face", settings.privacy_face_checkpoint, settings.privacy_face_sha256),
        ("plate", settings.privacy_plate_checkpoint, settings.privacy_plate_sha256),
        (
            "SAM",
            settings.local_sam31_checkpoint if variant == "sam3.1" else settings.local_sam3_checkpoint,
            settings.local_sam31_sha256 if variant == "sam3.1" else settings.local_sam3_sha256,
        ),
    ):
        if path is None or not path.is_file() or len(digest) != 64 or file_sha256(path) != digest:
            raise ValueError(f"{name} checkpoint missing or SHA-256 mismatch")
    if not settings.local_sam_python.is_file() or not settings.privacy_python.is_file():
        raise ValueError("Configure both local GPU interpreters before uploading")
    if len(settings.privacy_source_commit) != 40 or any(
        c not in "0123456789abcdef" for c in settings.privacy_source_commit
    ):
        raise ValueError("Pin the full EgoBlur source commit")
    checkpoint = settings.local_sam31_checkpoint if variant == "sam3.1" else settings.local_sam3_checkpoint
    assert checkpoint is not None
    if checkpoint.name != ("sam3.1_multiplex.pt" if variant == "sam3.1" else "sam3.pt"):
        raise ValueError("Use the checkpoint filename expected by the local SAM builder")
    return VmProfile(
        variant=variant,
        pipeline=pipeline,
        hybrid=hybrid_preflight(settings, classes or []) if pipeline != "sam" else None,
        checkpoint=str(checkpoint.resolve()),
        checkpoint_sha256=settings.local_sam31_sha256 if variant == "sam3.1" else settings.local_sam3_sha256,
        source_commit=settings.local_sam_commit,
    )
