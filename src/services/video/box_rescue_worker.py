"""Generate review-only VLM box proposals after a completed SAM sequence."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Protocol

from PIL import Image, ImageDraw

from src.models.video import DiscoveryResult, SequenceAuditUsage, TrackSet
from src.services.storage.repository import Store
from src.services.video.audit import render_overlay
from src.services.video.profile import current_profile, profiled_worker
from src.services.video.rescue_matrix import BoxRescueSuggestion, RescueFrameVerdict, select_box_rescue_frames
from src.services.video.workspace import VideoWorkspace


class BoxRescueAuditor(Protocol):
    model: str
    model_identity: str

    def propose_missing_boxes(
        self,
        *,
        overlay_path: Path,
        overlay_sha256: str,
        classes: tuple[str, ...],
        image_width: int,
        image_height: int,
        existing_boxes: list[dict],
    ) -> tuple[RescueFrameVerdict, SequenceAuditUsage, bool]: ...


class BoxRescueWorker:
    def __init__(
        self,
        *,
        profile_enabled: bool = False,
        store: Store,
        workspace: VideoWorkspace,
        auditor: BoxRescueAuditor,
        lease_seconds: float = 180,
        max_frames: int = 20,
    ) -> None:
        if lease_seconds < 3 or not 1 <= max_frames <= 20:
            raise ValueError("invalid VLM box-rescue worker limits")
        self.profile_enabled = profile_enabled
        self.store = store
        self.workspace = workspace
        self.auditor = auditor
        self.lease_seconds = lease_seconds
        self.max_frames = max_frames

    def process_next(self) -> str | None:
        claim = self.store.claim_next_video_box_rescue(lease_seconds=self.lease_seconds)
        if claim is None:
            return None
        job_id = claim.job.job_id
        try:
            with _LeaseHeartbeat(self.store, job_id, claim.claim_token, self.lease_seconds):
                self._process(job_id, claim.claim_token)
            return job_id
        except Exception as error:
            logging.exception("VLM box rescue for %s failed", job_id)
            try:
                self.store.fail_video_box_rescue(
                    job_id,
                    claim_token=claim.claim_token,
                    error=f"{type(error).__name__}: {str(error)[:450]}",
                )
            except ValueError:
                pass
            return job_id

    @profiled_worker("claude.worker")
    def _process(self, job_id: str, claim_token: str) -> None:
        job = self.store.get_video_job(job_id)
        if job.manifest_sha256 is None or job.track_set_sha256 is None or job.discovery_sha256 is None:
            raise ValueError("SAM sequence artifacts are incomplete")
        manifest = self.workspace.read_manifest(job_id, expected_sha256=job.manifest_sha256)
        self.workspace.verify_frame_files(manifest)
        tracks = TrackSet.model_validate_json(self.store.read_video_track_set(job_id))
        discovery = DiscoveryResult.model_validate_json(self.store.read_video_discovery(job_id))
        if tracks.manifest_sha256 != job.manifest_sha256:
            raise ValueError("SAM track set does not match the frame manifest")
        request = self.workspace.read_sampling_request(job_id)
        if not request.vlm_box_rescue:
            raise ValueError("job did not opt into VLM box suggestions")
        frames = select_box_rescue_frames(manifest, tracks, discovery, max_frames=self.max_frames)
        partial = self.store.read_video_box_rescue_partial(job_id, claim_token=claim_token)
        completed_by_frame = {row["frame_index"]: row for row in partial}
        proposals: list[dict] = []
        paths = self.workspace.frame_paths(manifest)

        for frame_index, reasons in frames:
            record = manifest.frames[frame_index]
            overlay_path, overlay_sha, _ = render_overlay(
                job_id=job_id,
                frame_index=frame_index,
                raw_path=paths[frame_index],
                tracks=tracks,
                workspace=self.workspace,
            )
            existing = completed_by_frame.get(frame_index)
            model_identity = getattr(self.auditor, "model_identity", self.auditor.model)
            if (
                existing is not None
                and existing.get("image_sha256") == overlay_sha
                and existing.get("model_identity", existing.get("model_name")) == model_identity
            ):
                frame_result = existing
            else:
                with Image.open(paths[frame_index]) as image:
                    width, height = image.size
                boxes = []
                for track in tracks.tracks:
                    observation = next(
                        (item for item in track.observations if item.frame_index == frame_index and not item.outside),
                        None,
                    )
                    if observation is not None:
                        box = observation.bbox
                        boxes.append({"label": track.label_name, "bbox": [box.xtl, box.ytl, box.xbr, box.ybr]})
                started = time.monotonic()
                with current_profile().stage("claude.request_with_retries"):
                    answer, usage, cache_hit = self.auditor.propose_missing_boxes(
                        overlay_path=overlay_path,
                        overlay_sha256=overlay_sha,
                        classes=tuple(request.sequence_classes),
                        image_width=width,
                        image_height=height,
                        existing_boxes=boxes,
                    )
                source_index = record.source_frame_index if record.source_frame_index is not None else frame_index
                suggestions = []
                for index, suggestion in enumerate(answer.proposals):
                    proposal_material = json.dumps(
                        [job_id, frame_index, index, suggestion.model_dump(mode="json"), overlay_sha],
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                    proposal_id = hashlib.sha256(proposal_material).hexdigest()
                    saved = BoxRescueSuggestion(
                        proposal_id=proposal_id,
                        frame_index=frame_index,
                        source_frame_index=source_index,
                        label=suggestion.label,
                        bbox=suggestion.bbox,
                        point=suggestion.point,
                        occlusion=suggestion.occlusion,
                        confidence=suggestion.confidence,
                        reason=suggestion.reason,
                        image_sha256=overlay_sha,
                        box_extent=suggestion.box_extent,
                        visible_bbox=suggestion.visible_bbox,
                    ).model_dump(mode="json")
                    self._render_preview(job_id, overlay_path, saved)
                    suggestions.append(saved)
                frame_result = {
                    "frame_index": frame_index,
                    "source_frame_index": source_index,
                    "image_sha256": overlay_sha,
                    "model_name": self.auditor.model,
                    "model_identity": model_identity,
                    "selected_reasons": list(reasons),
                    "missing": answer.missing,
                    "reason": answer.reason,
                    "usage": usage.model_dump(mode="json"),
                    "cache_hit": cache_hit,
                    "proposals": suggestions,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "latency_s": time.monotonic() - started,
                }
                partial = [row for row in partial if row.get("frame_index") != frame_index]
                partial.append(frame_result)
                self.store.save_video_box_rescue_partial(
                    job_id,
                    claim_token=claim_token,
                    partial=partial,
                    lease_seconds=self.lease_seconds,
                )
            proposals.extend(frame_result.get("proposals", []))

        self.store.complete_video_box_rescue(job_id, claim_token=claim_token, proposals=proposals)

    def _render_preview(self, job_id: str, overlay_path: Path, proposal: dict) -> None:
        with Image.open(overlay_path) as source:
            image = source.convert("RGB")
        width, height = image.size
        draw = ImageDraw.Draw(image)
        x1, y1, x2, y2 = proposal["bbox"]
        box = (round(x1 * width), round(y1 * height), round(x2 * width), round(y2 * height))
        draw.rectangle(box, outline=(255, 45, 45), width=max(2, min(width, height) // 250))
        draw.text(
            (box[0], max(0, box[1] - 12)),
            f"Claude estimate? {proposal['label']} {proposal['confidence']}",
            fill=(255, 45, 45),
        )
        path = self.workspace.box_rescue_preview_path(job_id, proposal["proposal_id"], create=True)
        image.save(path, format="PNG", optimize=False)


class _LeaseHeartbeat:
    def __init__(self, store: Store, job_id: str, token: str, seconds: float) -> None:
        self.store, self.job_id, self.token, self.seconds = store, job_id, token, seconds
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> _LeaseHeartbeat:
        self.thread.start()
        return self

    def __exit__(self, kind: type[BaseException] | None, error: BaseException | None, tb: TracebackType | None) -> None:
        self.stop.set()
        self.thread.join(timeout=1)

    def _run(self) -> None:
        while not self.stop.wait(max(1.0, self.seconds / 3)):
            try:
                self.store.extend_video_box_rescue_lease(
                    self.job_id,
                    claim_token=self.token,
                    lease_seconds=self.seconds,
                )
            except Exception:
                return
