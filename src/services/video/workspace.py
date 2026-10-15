"""Own all runtime paths for a video job and enforce workspace containment."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from uuid import uuid4

from src.models.video import (
    AuditResult,
    CvatPublishResult,
    CvatSamplingRequest,
    DiscoveryResult,
    FrameManifest,
    ReviewQueue,
    SeedSet,
    SequenceAuditResult,
    TrackSet,
    VideoJob,
)


class VideoWorkspace:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def job_dir(self, job_id: str, *, create: bool = False) -> Path:
        path = (self.root / job_id).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("video job path escaped the configured workspace")
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def source_path(self, job: VideoJob) -> Path:
        suffix = Path(job.source_name).suffix.lower()
        return self.job_dir(job.job_id) / f"source{suffix}"

    def frames_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "frames"

    def prepare_frames_dir(self, job_id: str) -> Path:
        """Clear only regular frame files inside the verified job directory."""
        path = self.frames_dir(job_id).resolve()
        job_dir = self.job_dir(job_id).resolve()
        if path.parent != job_dir or not path.is_relative_to(self.root):
            raise ValueError("frames path escaped the video job workspace")
        if path.exists():
            for child in path.iterdir():
                if not child.is_file():
                    raise ValueError("unexpected directory inside frames workspace")
                child.unlink()
        else:
            path.mkdir(parents=True)
        return path

    def manifest_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "manifest.json"

    def sampling_request_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "sampling-request.json"

    def write_sampling_request(self, job_id: str, request: CvatSamplingRequest) -> None:
        self.job_dir(job_id, create=True)
        self._atomic_bytes_write(self.sampling_request_path(job_id), request.model_dump_json().encode("utf-8"))

    def read_sampling_request(self, job_id: str) -> CvatSamplingRequest:
        path = self.sampling_request_path(job_id)
        if not path.is_file():
            return CvatSamplingRequest(strategy="budget")
        return CvatSamplingRequest.model_validate_json(path.read_bytes())

    def seed_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "seeds.json"

    def track_set_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "tracks.json"

    def discovery_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "discovery.json"

    def audit_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "audit.json"

    def review_queue_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "review-queue.json"

    def publish_result_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "cvat-publish.json"

    def overlays_dir(self, job_id: str) -> Path:
        path = self.job_dir(job_id, create=True) / "overlays"
        path.mkdir(exist_ok=True)
        return path

    def overlay_path(self, job_id: str, frame_index: int) -> Path:
        return self.overlays_dir(job_id) / f"frame_{frame_index:06d}.png"

    def box_rescue_preview_path(self, job_id: str, proposal_id: str, *, create: bool = False) -> Path:
        if not re.fullmatch(r"[0-9a-f]{64}", proposal_id):
            raise ValueError("invalid box-rescue proposal ID")
        directory = self.job_dir(job_id, create=create) / "box-rescue"
        if create:
            directory.mkdir(exist_ok=True)
        path = (directory / f"{proposal_id}.png").resolve()
        if not path.is_relative_to(self.job_dir(job_id)):
            raise ValueError("box-rescue preview path escaped the video workspace")
        return path

    def read_manifest(self, job_id: str, *, expected_sha256: str | None = None) -> FrameManifest:
        path = self.manifest_path(job_id)
        if not path.is_file():
            raise FileNotFoundError(f"frame manifest does not exist for job {job_id}")
        encoded = path.read_bytes()
        if expected_sha256 is not None and hashlib.sha256(encoded).hexdigest() != expected_sha256:
            raise ValueError(f"frame manifest hash does not match video job {job_id}")
        return FrameManifest.model_validate_json(encoded)

    def verify_frame_files(self, manifest: FrameManifest) -> None:
        job_dir = self.job_dir(manifest.job_id).resolve()
        for frame in manifest.frames:
            path = (job_dir / frame.relative_path).resolve()
            if not path.is_relative_to(job_dir) or not path.is_file():
                raise ValueError(f"frame {frame.frame_index} is missing from the video workspace")
            if _sha256_file(path) != frame.sha256:
                raise ValueError(f"frame {frame.frame_index} hash does not match the manifest")

    def frame_paths(self, manifest: FrameManifest) -> list[Path]:
        self.verify_frame_files(manifest)
        job_dir = self.job_dir(manifest.job_id)
        return [(job_dir / frame.relative_path).resolve() for frame in manifest.frames]

    def write_manifest(self, manifest: FrameManifest) -> bytes:
        return self._atomic_model_write(
            self.manifest_path(manifest.job_id), manifest.model_dump(mode="json", by_alias=True)
        )

    def write_seed_set(self, seed_set: SeedSet) -> bytes:
        encoded = self.encode_seed_set(seed_set)
        self.write_seed_payload(seed_set.job_id, encoded)
        return encoded

    @staticmethod
    def encode_seed_set(seed_set: SeedSet) -> bytes:
        payload = seed_set.model_dump(mode="json", by_alias=True)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    def write_seed_payload(self, job_id: str, encoded: bytes) -> None:
        self._atomic_bytes_write(self.seed_path(job_id), encoded)

    @staticmethod
    def encode_track_set(track_set: TrackSet) -> bytes:
        payload = track_set.model_dump(mode="json", by_alias=True)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    def write_track_set_payload(self, job_id: str, encoded: bytes) -> None:
        self._atomic_bytes_write(self.track_set_path(job_id), encoded)

    @staticmethod
    def encode_discovery(result: DiscoveryResult) -> bytes:
        payload = result.model_dump(mode="json", by_alias=True)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    def write_discovery_payload(self, job_id: str, encoded: bytes) -> None:
        self._atomic_bytes_write(self.discovery_path(job_id), encoded)

    @staticmethod
    def encode_audit(result: AuditResult | SequenceAuditResult) -> bytes:
        payload = result.model_dump(mode="json", by_alias=True)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    def write_audit_payload(self, job_id: str, encoded: bytes) -> None:
        self._atomic_bytes_write(self.audit_path(job_id), encoded)

    @staticmethod
    def encode_review_queue(queue: ReviewQueue) -> bytes:
        payload = queue.model_dump(mode="json", by_alias=True)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    def write_review_queue_payload(self, job_id: str, encoded: bytes) -> None:
        self._atomic_bytes_write(self.review_queue_path(job_id), encoded)

    @staticmethod
    def encode_publish_result(result: CvatPublishResult) -> bytes:
        payload = result.model_dump(mode="json", by_alias=True)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    def write_publish_result_payload(self, job_id: str, encoded: bytes) -> None:
        self._atomic_bytes_write(self.publish_result_path(job_id), encoded)

    @staticmethod
    def _atomic_model_write(path: Path, payload: dict[str, object]) -> bytes:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        VideoWorkspace._atomic_bytes_write(path, encoded)
        return encoded

    @staticmethod
    def _atomic_bytes_write(path: Path, encoded: bytes) -> None:
        temporary = path.with_suffix(path.suffix + f".{uuid4().hex}.tmp")
        temporary.write_bytes(encoded)
        temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
