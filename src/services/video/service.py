"""Small application interface for video ingestion and seed attachment."""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import UploadFile

from src.models.video import (
    CreateCvatTaskRequest,
    CvatAttributeSpec,
    CvatLabelSpec,
    CvatPublishResult,
    CvatSamplingRequest,
    CvatTaskBinding,
    DiscoveryResult,
    FrameManifest,
    ReviewQueue,
    SeedSet,
    TrackSet,
    VideoJob,
    VideoJobStatus,
)
from src.services.cvat import CvatReadError, CvatTaskClient, CvatTrackClient, CvatWriteError
from src.services.storage.repository import Store
from src.services.video.seeds import CvatSeedReader
from src.services.video.workspace import VideoWorkspace

ALLOWED_VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"})


class VideoUploadError(ValueError):
    def __init__(self, message: str, *, too_large: bool = False) -> None:
        super().__init__(message)
        self.too_large = too_large


class VideoJobService:
    def __init__(
        self,
        *,
        store: Store,
        workspace: VideoWorkspace,
        seed_reader: CvatSeedReader,
        task_client: CvatTaskClient,
        track_client: CvatTrackClient,
        max_upload_bytes: int,
        box_rescue_enabled: bool = False,
        track_filter_config: object = None,
    ) -> None:
        self.store = store
        self.workspace = workspace
        self.seed_reader = seed_reader
        self.task_client = task_client
        self.track_client = track_client
        self.max_upload_bytes = max_upload_bytes
        self.box_rescue_enabled = box_rescue_enabled
        from src.services.video.track_filter import TrackFilterConfig

        self.track_filter_config = (
            track_filter_config if isinstance(track_filter_config, TrackFilterConfig) else TrackFilterConfig()
        )
        self._task_locks: dict[str, asyncio.Lock] = {}
        self._proposal_locks: dict[str, asyncio.Lock] = {}

    async def create_job(self, upload: UploadFile) -> VideoJob:
        filename = (upload.filename or "").replace("\\", "/").rsplit("/", maxsplit=1)[-1]
        suffix = Path(filename).suffix.lower()
        if not filename or len(filename) > 255 or suffix not in ALLOWED_VIDEO_SUFFIXES:
            raise VideoUploadError("video filename must use MP4, MOV, MKV, AVI, WEBM, or M4V")
        job_id = str(uuid4())
        job_dir = self.workspace.job_dir(job_id, create=True)
        destination = job_dir / f"source{suffix}"
        temporary = job_dir / f"source{suffix}.uploading"
        digest = hashlib.sha256()
        size = 0
        try:
            with temporary.open("xb") as handle:
                while chunk := await upload.read(1024 * 1024):
                    size += len(chunk)
                    if size > self.max_upload_bytes:
                        raise VideoUploadError(
                            f"video exceeds the {self.max_upload_bytes}-byte upload limit", too_large=True
                        )
                    digest.update(chunk)
                    handle.write(chunk)
            if size == 0:
                raise VideoUploadError("uploaded video is empty")
            temporary.replace(destination)
        except Exception:
            if temporary.exists():
                temporary.unlink()
            raise
        finally:
            await upload.close()

        now = datetime.now(UTC)
        job = VideoJob(
            job_id=job_id,
            status=VideoJobStatus.UPLOADED,
            source_name=filename,
            source_sha256=digest.hexdigest(),
            source_size_bytes=size,
            created_at=now,
            updated_at=now,
            stage="uploaded",
        )
        self.store.create_video_job(job)
        return self.store.transition_video_job(
            job_id,
            expected={VideoJobStatus.UPLOADED},
            status=VideoJobStatus.QUEUED,
            stage="queued_for_extraction",
        )

    def find_cvat_job(self, cvat_job_id: int) -> VideoJob | None:
        if isinstance(cvat_job_id, bool) or cvat_job_id < 1:
            raise ValueError("CVAT job ID must be positive")
        return self.store.find_video_job_by_cvat_job(cvat_job_id)

    async def prepare_sequence_labels(self, cvat_job_id: int, classes: tuple[str, ...]) -> list[dict]:
        """Prepare pending-review attributes before spending any GPU quota."""
        labels = await self.seed_reader.client.get_job_labels(cvat_job_id)
        selected = self._selected_sequence_labels(labels, classes)
        attributes = _labels_with_autolabel_attributes([CvatLabelSpec(name="car")])[0].attributes
        assert attributes is not None
        await self.task_client.ensure_label_attributes(selected, attributes)
        return await self.seed_reader.client.get_job_labels(cvat_job_id)

    @staticmethod
    def _selected_sequence_labels(labels: list[dict], classes: tuple[str, ...]) -> list[dict]:
        available = {str(label["name"]) for label in labels if isinstance(label.get("name"), str)}
        missing = [name for name in classes if name not in available]
        if missing:
            names = ", ".join(sorted(available))
            raise ValueError(
                f"Requested CVAT classes not found: {', '.join(missing)}. "
                f"Available classes: {names or '(none)'}. Use the exact CVAT class names."
            )
        return [label for label in labels if label.get("name") in classes]

    async def get_sequence_class_names(self, cvat_job_id: int) -> list[str]:
        labels = await self.seed_reader.client.get_job_labels(cvat_job_id)
        return [label["name"] for label in labels if isinstance(label.get("name"), str)]

    async def import_cvat_job(self, cvat_job_id: int, sampling: CvatSamplingRequest | None = None) -> VideoJob:
        """Use the frames of an existing CVAT job without uploading the video twice."""
        if sampling is not None and sampling.auto_start_sequence and self.box_rescue_enabled:
            sampling = sampling.model_copy(update={"vlm_box_rescue": True})
        if isinstance(cvat_job_id, bool) or cvat_job_id < 1:
            raise ValueError("CVAT job ID must be positive")
        if sampling is not None and sampling.vlm_box_rescue and not self.box_rescue_enabled:
            raise ValueError("Claude box suggestions are disabled or missing an OpenRouter key")
        lock = self._task_locks.setdefault(f"cvat:{cvat_job_id}", asyncio.Lock())
        async with lock:
            existing = self.find_cvat_job(cvat_job_id)
            if existing is not None:
                if sampling is not None and sampling.auto_start_sequence:
                    previous = self.workspace.read_sampling_request(existing.job_id)
                    if (
                        previous.sam_execution_provider != sampling.sam_execution_provider
                        or previous.sam3_model_variant != sampling.sam3_model_variant
                    ):
                        raise ValueError(
                            "Existing job keeps its SAM provider/model; create a fresh CVAT job to compare"
                        )
                    if not previous.auto_start_sequence:
                        raise ValueError("CVAT job is linked to a legacy seed-first run; use a fresh job")
                return existing
            cvat_job, meta = await asyncio.gather(
                self.seed_reader.client.get_job(cvat_job_id),
                self.seed_reader.client.get_job_data_meta(cvat_job_id),
            )
            task_id = cvat_job.get("task_id")
            count = meta.get("size")
            frames = meta.get("frames")
            if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id < 1:
                raise ValueError("CVAT job has no valid task ID")
            if cvat_job.get("dimension") != "2d" or cvat_job.get("start_frame") != 0:
                raise ValueError("AutoLabel requires a 2D CVAT job starting at frame 0")
            if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 999999:
                raise ValueError("AutoLabel supports CVAT jobs with 1–999999 frames")
            if cvat_job.get("stop_frame") != count - 1 or meta.get("frame_filter") not in (None, ""):
                raise ValueError("AutoLabel requires a contiguous, unfiltered CVAT job")
            if meta.get("deleted_frames") or meta.get("included_frames") is not None:
                raise ValueError("AutoLabel does not support deleted or selected CVAT frames")
            if not isinstance(frames, list) or not frames or not isinstance(frames[0], dict):
                raise ValueError("CVAT frame metadata is incomplete")

            sampling = sampling or CvatSamplingRequest()
            if sampling.source_fps is None:
                raise ValueError(
                    "CVAT does not expose a reliable video FPS here; supply source_fps "
                    "for the contiguous frames in this CVAT job"
                )
            if sampling.auto_start_sequence:
                labels = await self.seed_reader.client.get_job_labels(cvat_job_id)
                self._selected_sequence_labels(labels, tuple(sampling.sequence_classes))
            job_id = str(uuid4())
            source_name = f"cvat-job-{cvat_job_id}.frames"
            source_metadata = json.dumps(
                {"job": cvat_job_id, "task": task_id, "frames": frames}, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            now = datetime.now(UTC)
            job = VideoJob(
                job_id=job_id,
                status=VideoJobStatus.QUEUED,
                source_name=source_name,
                source_sha256=hashlib.sha256(source_metadata).hexdigest(),
                source_size_bytes=len(source_metadata),
                created_at=now,
                updated_at=now,
                stage="queued_for_cvat_import",
                progress_current=0,
                progress_total=count,
                cvat_task_id=task_id,
                cvat_job_id=cvat_job_id,
            )
            self.workspace.write_sampling_request(job_id, sampling)
            self.store.create_video_job(job)
            return job

    def retry_cvat_import(self, job_id: str) -> VideoJob:
        return self.store.retry_video_cvat_import(job_id)

    async def capture_current_seed(self, job_id: str) -> SeedSet:
        job = self.get_job(job_id)
        if job.cvat_job_id is None:
            raise ValueError("video job is not bound to CVAT")
        annotations = await self.seed_reader.client.get_job_annotations(job.cvat_job_id)
        revision = annotations.get("version")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise ValueError("CVAT annotation revision is invalid")
        return await self.attach_seed(job_id, cvat_job_id=job.cvat_job_id, annotation_revision=revision)

    def queue_auto_seed(self, job_id: str) -> VideoJob:
        return self.store.queue_video_auto_seed(job_id)

    def get_job(self, job_id: str) -> VideoJob:
        return self.store.get_video_job(job_id)

    def get_manifest(self, job_id: str) -> FrameManifest:
        job = self.store.get_video_job(job_id)
        if job.manifest_sha256 is None:
            raise ValueError(f"video job {job_id} does not have a frame manifest yet")
        return self.workspace.read_manifest(job_id, expected_sha256=job.manifest_sha256)

    def get_seed(self, job_id: str) -> SeedSet:
        job = self.store.get_video_job(job_id)
        if job.seed_sha256 is None:
            raise ValueError(f"video job {job_id} does not have a seed set")
        encoded = self.store.read_video_seed(job_id).encode("utf-8")
        if hashlib.sha256(encoded).hexdigest() != job.seed_sha256:
            raise ValueError(f"stored seed hash does not match video job {job_id}")
        seed_set = SeedSet.model_validate_json(encoded)
        self.workspace.write_seed_payload(job_id, encoded)
        return seed_set

    def cancel_job(self, job_id: str) -> VideoJob:
        return self.store.cancel_video_job(job_id)

    def start_propagation(self, job_id: str) -> VideoJob:
        job = self.store.get_video_job(job_id)
        retryable_failure = job.status is VideoJobStatus.FAILED and bool(
            job.error_code and job.error_code.startswith(("propagation_", "video_propagation_"))
        )
        if job.status is not VideoJobStatus.SEED_READY and not retryable_failure:
            raise ValueError(f"video job {job_id} is not ready for propagation")
        if job.manifest_sha256 is None or job.seed_sha256 is None:
            raise ValueError(f"video job {job_id} propagation provenance is incomplete")
        manifest = self.workspace.read_manifest(job_id, expected_sha256=job.manifest_sha256)
        self.workspace.verify_frame_files(manifest)
        encoded = self.store.read_video_seed(job_id).encode("utf-8")
        if hashlib.sha256(encoded).hexdigest() != job.seed_sha256:
            raise ValueError(f"stored seed hash does not match video job {job_id}")
        SeedSet.model_validate_json(encoded)
        return self.store.queue_video_propagation(job_id, frame_count=manifest.frame_count)

    def start_sequence(self, job_id: str) -> VideoJob:
        """Resume a seed-free request if extraction completed before it was queued."""
        self.store.get_video_job(job_id)
        request = self.workspace.read_sampling_request(job_id)
        if not request.auto_start_sequence or request.strategy not in {"uniform", "all_frames"}:
            raise ValueError("job was not imported for seed-free sequence labeling")
        manifest = self.get_manifest(job_id)
        self.workspace.verify_frame_files(manifest)
        return self.store.queue_video_sequence(job_id, frame_count=manifest.frame_count)

    async def retry_sequence_with_classes(self, job_id: str, classes: list[str]) -> VideoJob:
        """Correct a failed class request using the existing imported frames."""
        lock = self._task_locks.setdefault(f"sequence:{job_id}", asyncio.Lock())
        async with lock:
            job = self.get_job(job_id)
            if (
                job.status is not VideoJobStatus.FAILED
                or not (job.error_code and job.error_code.startswith("sequence_"))
                or job.cvat_job_id is None
            ):
                raise ValueError("only a failed CVAT sequence can change requested classes")
            request = self.workspace.read_sampling_request(job_id)
            updated = CvatSamplingRequest.model_validate(
                {
                    **request.model_dump(),
                    "sequence_classes": classes,
                }
            )
            if not updated.auto_start_sequence or updated.strategy not in {"uniform", "all_frames"}:
                raise ValueError("job was not imported for seed-free sequence labeling")
            labels = await self.seed_reader.client.get_job_labels(job.cvat_job_id)
            self._selected_sequence_labels(labels, tuple(updated.sequence_classes))
            manifest = self.get_manifest(job_id)
            self.workspace.verify_frame_files(manifest)
            self.workspace.write_sampling_request(job_id, updated)
            return self.store.queue_video_sequence(job_id, frame_count=manifest.frame_count)

    def get_tracks(self, job_id: str) -> TrackSet:
        job = self.store.get_video_job(job_id)
        if job.track_set_sha256 is None:
            raise ValueError(f"video job {job_id} does not have a propagated track set")
        encoded = self.store.read_video_track_set(job_id).encode("utf-8")
        if hashlib.sha256(encoded).hexdigest() != job.track_set_sha256:
            raise ValueError(f"stored track-set hash does not match video job {job_id}")
        track_set = TrackSet.model_validate_json(encoded)
        self.workspace.write_track_set_payload(job_id, encoded)
        return track_set

    def retry_discovery(self, job_id: str) -> VideoJob:
        job = self.store.get_video_job(job_id)
        if job.status is not VideoJobStatus.FAILED or not (
            job.error_code and job.error_code.startswith(("discovery_", "video_discovery_"))
        ):
            raise ValueError(f"video job {job_id} is not ready for discovery retry")
        manifest = self.get_manifest(job_id)
        self.workspace.verify_frame_files(manifest)
        self.get_seed(job_id)
        self.get_tracks(job_id)
        return self.store.queue_video_discovery(job_id, frame_count=manifest.frame_count)

    def get_discovery(self, job_id: str) -> DiscoveryResult:
        job = self.store.get_video_job(job_id)
        if job.discovery_sha256 is None:
            raise ValueError(f"video job {job_id} does not have discovery evidence")
        encoded = self.store.read_video_discovery(job_id).encode("utf-8")
        if hashlib.sha256(encoded).hexdigest() != job.discovery_sha256:
            raise ValueError(f"stored discovery hash does not match video job {job_id}")
        result = DiscoveryResult.model_validate_json(encoded)
        self.workspace.write_discovery_payload(job_id, encoded)
        return result

    def get_sequence_audit_state(self, job_id: str) -> dict[str, object] | None:
        self.store.get_video_job(job_id)
        return self.store.get_sequence_audit_state(job_id)

    def get_box_rescue_state(self, job_id: str) -> dict:
        self.store.get_video_job(job_id)
        return self.store.get_video_box_rescue_state(job_id) or {"status": "disabled", "proposals": []}

    def retry_box_rescue(self, job_id: str) -> dict[str, object]:
        job = self.store.get_video_job(job_id)
        request = self.workspace.read_sampling_request(job_id)
        if not self.box_rescue_enabled or not request.vlm_box_rescue:
            raise ValueError("VLM box suggestions are not enabled for this job")
        if job.status is not VideoJobStatus.AWAITING_REVIEW:
            raise ValueError("SAM sequence must finish before VLM box rescue can be retried")
        self.store.retry_video_box_rescue(job_id)
        return self.get_box_rescue_state(job_id)

    async def accept_box_rescue(self, job_id: str, proposal_id: str, *, expected_revision: int) -> dict:
        # One API process is supported by the internal launcher. Serialize retries
        # inside it, and persist the reservation so Reject also sees it.
        lock = self._proposal_locks.setdefault(f"{job_id}:{proposal_id}", asyncio.Lock())
        async with lock:
            return await self._accept_box_rescue(job_id, proposal_id, expected_revision=expected_revision)

    async def _accept_box_rescue(self, job_id: str, proposal_id: str, *, expected_revision: int) -> dict:
        job = self.store.get_video_job(job_id)
        if job.cvat_job_id is None or job.manifest_sha256 is None:
            raise ValueError("video job has no CVAT annotation target")
        state = self.store.get_video_box_rescue_state(job_id)
        if state is None or state["status"] != "done":
            raise ValueError("VLM box proposals are not ready")
        proposal = next((row for row in state["proposals"] if row.get("proposal_id") == proposal_id), None)
        if proposal is None:
            raise KeyError(proposal_id)
        if proposal.get("status") == "accepted":
            return {"proposal_id": proposal_id, "status": "accepted", "revision_after": None}
        if proposal.get("status") not in {"pending", "accepting"}:
            raise ValueError("VLM proposal has already been reviewed")
        previous_status = proposal["status"]
        manifest = self.get_manifest(job_id)
        frame_index = proposal["frame_index"]
        if not 0 <= frame_index < manifest.frame_count:
            raise ValueError("proposal frame is outside the source manifest")
        frame = manifest.frames[frame_index]
        source_index = frame.source_frame_index if frame.source_frame_index is not None else frame_index
        if proposal.get("source_frame_index", source_index) != source_index:
            raise ValueError("proposal source frame mapping does not match manifest")
        x1, y1, x2, y2 = proposal["bbox"]
        from src.models.video import BoundingBox2D

        bbox = BoundingBox2D(xtl=x1 * frame.width, ytl=y1 * frame.height, xbr=x2 * frame.width, ybr=y2 * frame.height)
        labels = await self.seed_reader.client.get_job_labels(job.cvat_job_id)
        label = next((row for row in labels if row.get("name") == proposal["label"]), None)
        if label is None or isinstance(label.get("id"), bool) or not isinstance(label.get("id"), int):
            raise CvatWriteError("proposal class is no longer in this CVAT job", status_code=409)
        decision = self.store.set_video_box_rescue_proposal_status(job_id, proposal_id=proposal_id, status="accepting")
        if decision["status"] == "accepted":
            return {"proposal_id": proposal_id, "status": "accepted", "revision_after": None}
        try:
            revision_after = await self.track_client.accept_rescue_track(
                cvat_job_id=job.cvat_job_id,
                frame=source_index,
                label_id=label["id"],
                bbox=bbox,
                expected_revision=expected_revision,
                proposal_id=proposal_id,
                occluded=proposal.get("occlusion", "none") != "none",
                box_extent=proposal.get("box_extent", "visible"),
            )
        except CvatReadError:
            if previous_status == "pending":
                self.store.set_video_box_rescue_proposal_status(job_id, proposal_id=proposal_id, status="pending")
            raise
        except CvatWriteError as error:
            # An unknown network outcome stays reserved: Reject must not undo
            # the local decision while a CVAT track may already exist.
            if previous_status == "pending" and error.status_code in {400, 409}:
                self.store.set_video_box_rescue_proposal_status(job_id, proposal_id=proposal_id, status="pending")
            raise
        self.store.set_video_box_rescue_proposal_status(job_id, proposal_id=proposal_id, status="accepted")
        return {"proposal_id": proposal_id, "status": "accepted", "revision_after": revision_after}

    def reject_box_rescue(self, job_id: str, proposal_id: str) -> dict:
        self.store.get_video_job(job_id)
        proposal = self.store.set_video_box_rescue_proposal_status(
            job_id,
            proposal_id=proposal_id,
            status="rejected",
        )
        return {"proposal_id": proposal_id, "status": proposal["status"]}

    def get_review_queue(self, job_id: str) -> ReviewQueue:
        job = self.store.get_video_job(job_id)
        if job.review_queue_sha256 is None:
            raise ValueError(f"video job {job_id} does not have a review queue")
        encoded = self.store.read_video_review_queue(job_id).encode()
        if hashlib.sha256(encoded).hexdigest() != job.review_queue_sha256:
            raise ValueError(f"stored review-queue hash does not match video job {job_id}")
        queue = ReviewQueue.model_validate_json(encoded)
        self.workspace.write_review_queue_payload(job_id, encoded)
        return queue

    def get_publish_result(self, job_id: str) -> CvatPublishResult:
        job = self.store.get_video_job(job_id)
        if job.publish_sha256 is None:
            raise ValueError(f"video job {job_id} does not have a CVAT publish result")
        encoded = self.store.read_video_publish_result(job_id).encode()
        if hashlib.sha256(encoded).hexdigest() != job.publish_sha256:
            raise ValueError(f"stored publish-result hash does not match video job {job_id}")
        result = CvatPublishResult.model_validate_json(encoded)
        self.workspace.write_publish_result_payload(job_id, encoded)
        return result

    async def publish_tracks(
        self,
        job_id: str,
        *,
        expected_revision: int | None,
        worker_export: bool = False,
    ) -> CvatPublishResult:
        job = self.store.get_video_job(job_id)
        if job.status is not VideoJobStatus.AWAITING_REVIEW:
            raise ValueError(f"video job {job_id} is not ready for CVAT publishing")
        if job.stage == "sequence_draft_exporting" and not worker_export:
            raise ValueError("automatic CVAT draft export is in progress; wait before retrying")
        if job.cvat_job_id is None or job.manifest_sha256 is None or job.track_set_sha256 is None:
            raise ValueError(f"video job {job_id} CVAT publish provenance is incomplete")
        manifest = self.get_manifest(job_id)
        tracks = self.get_tracks(job_id)
        review_queue = self.get_review_queue(job_id)
        priority_order = {"critical": 0, "high": 1, "normal": 2, "low": 3}
        priority_by_track: dict[str, str] = {}
        score_by_track: dict[str, float] = {}
        for unit in review_queue.units:
            for track_id in unit.track_ids:
                current = priority_by_track.get(track_id)
                if current is None or priority_order[unit.priority] < priority_order[current]:
                    priority_by_track[track_id] = unit.priority
                if unit.review_score is not None:
                    score_by_track[track_id] = max(score_by_track.get(track_id, 0.0), unit.review_score)
        sparse_kwargs: dict[str, Any] = {}
        sparse_sidecar: dict = {}
        if isinstance(self.track_client, CvatTrackClient) and self.track_client.sparse_config.enabled:
            mandatory_frames = {
                int(manifest.frames[u.frame_start].source_frame_index or u.frame_start)
                for u in review_queue.units
                if set(u.reason_codes) - {"routine_review", "class_policy"}
            }
            sparse_kwargs = {"sparse_sidecar": sparse_sidecar, "mandatory_sparse_frames": mandatory_frames}
        from src.services.video.track_filter import classify_tracks

        options = self.store.get_export_options(job_id)
        filter_config = self.track_filter_config.model_copy(
            update={"ignore_regions": options.get("ignore_regions", self.track_filter_config.ignore_regions)}
        )
        suspects = classify_tracks(tracks.tracks, filter_config, width=manifest.width, height=manifest.height)
        if self.track_filter_config.enabled:
            self.workspace.job_dir(job_id).joinpath("suspect-tracks.json").write_text(
                json.dumps(
                    {
                        "schema": "autolabel-suspect-tracks/v1",
                        "track_set_sha256": job.track_set_sha256,
                        "config": filter_config.model_dump(),
                        "suspects": suspects,
                        "annotations_hidden": False,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            for track_id in suspects:
                if priority_by_track.get(track_id) != "critical":
                    priority_by_track[track_id] = "high"
        result, _request_payload = await self.track_client.publish_tracks(
            job_id=job_id,
            cvat_job_id=job.cvat_job_id,
            manifest=manifest,
            tracks=tracks,
            review_priority_by_track=priority_by_track,
            review_score_by_track=score_by_track,
            expected_revision=expected_revision,
            **sparse_kwargs,
        )
        if sparse_sidecar:
            sidecar = self.workspace.job_dir(job_id) / "cvat-keyframes.json"
            sidecar.write_text(
                json.dumps(
                    {
                        "schema": "autolabel-cvat-keyframes/v1",
                        "job_id": job_id,
                        "track_set_sha256": job.track_set_sha256,
                        "config": self.track_client.sparse_config.model_dump(),
                        "tracks": sparse_sidecar,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        encoded = self.workspace.encode_publish_result(result)
        self.store.record_video_publish(
            job_id,
            cvat_job_id=job.cvat_job_id,
            publish_sha256=hashlib.sha256(encoded).hexdigest(),
            publish_payload=encoded.decode(),
            cvat_annotation_revision=result.revision_after,
            published_track_count=result.published_track_count,
        )
        self.workspace.write_publish_result_payload(job_id, encoded)
        return result

    async def create_cvat_task(self, job_id: str, request: CreateCvatTaskRequest) -> CvatTaskBinding:
        lock = self._task_locks.setdefault(job_id, asyncio.Lock())
        async with lock:
            job = self.store.get_video_job(job_id)
            if job.status not in {VideoJobStatus.AWAITING_SEED, VideoJobStatus.SEED_READY}:
                raise ValueError(f"video job {job_id} is not ready for a CVAT task")
            if job.manifest_sha256 is None:
                raise ValueError(f"video job {job_id} does not have a frame manifest yet")
            if job.cvat_task_id is not None and job.cvat_job_id is not None:
                return _task_binding(job, frame_count=self.get_manifest(job_id).frame_count)
            manifest = self.get_manifest(job_id)
            name = request.task_name or f"AutoLabel {job.source_name} [{job.job_id[:8]}]"
            task_id, cvat_job_id = await self.task_client.create_image_task(
                name=name,
                labels=_labels_with_autolabel_attributes(request.labels),
                frame_paths=self.workspace.frame_paths(manifest),
            )
            try:
                bound = self.store.bind_cvat_task(
                    job_id,
                    task_id=task_id,
                    cvat_job_id=cvat_job_id,
                    manifest_sha256=job.manifest_sha256,
                )
            except ValueError as error:
                raise CvatWriteError(
                    "CVAT task was created but the video job binding changed concurrently",
                    task_id=task_id,
                ) from error
            return _task_binding(bound, frame_count=manifest.frame_count)

    async def attach_seed(self, job_id: str, *, cvat_job_id: int, annotation_revision: int) -> SeedSet:
        job = self.store.get_video_job(job_id)
        if job.status not in {VideoJobStatus.AWAITING_SEED, VideoJobStatus.SEED_READY} and not (
            job.status is VideoJobStatus.FAILED and job.error_code == "auto_seed_failed"
        ):
            raise ValueError(f"video job {job_id} is not ready for seed capture")
        if job.cvat_task_id is None or job.cvat_job_id is None:
            raise ValueError(f"video job {job_id} does not have a bound CVAT task")
        if cvat_job_id != job.cvat_job_id:
            raise ValueError(f"CVAT job {cvat_job_id} is not bound to video job {job_id}")
        if job.manifest_sha256 is None:
            raise ValueError(f"video job {job_id} does not have a frame manifest yet")
        manifest = self.workspace.read_manifest(job_id, expected_sha256=job.manifest_sha256)
        self.workspace.verify_frame_files(manifest)
        seed_set = await self.seed_reader.capture(
            job_id=job_id,
            cvat_job_id=cvat_job_id,
            expected_revision=annotation_revision,
            manifest=manifest,
            expected_task_id=job.cvat_task_id,
        )
        encoded = self.workspace.encode_seed_set(seed_set)
        seed_sha256 = hashlib.sha256(encoded).hexdigest()
        self.store.attach_video_seed(
            job_id,
            cvat_job_id=cvat_job_id,
            seed_revision=annotation_revision,
            seed_count=len(seed_set.seeds),
            seed_sha256=seed_sha256,
            seed_payload=encoded.decode("utf-8"),
            expected_updated_at=job.updated_at,
        )
        self.workspace.write_seed_payload(job_id, encoded)
        return seed_set


def _task_binding(job: VideoJob, *, frame_count: int) -> CvatTaskBinding:
    if job.cvat_task_id is None or job.cvat_job_id is None or job.manifest_sha256 is None:
        raise ValueError(f"video job {job.job_id} does not have a complete CVAT binding")
    return CvatTaskBinding(
        job_id=job.job_id,
        cvat_task_id=job.cvat_task_id,
        cvat_job_id=job.cvat_job_id,
        manifest_sha256=job.manifest_sha256,
        frame_count=frame_count,
    )


def _labels_with_autolabel_attributes(labels: list[CvatLabelSpec]) -> list[CvatLabelSpec]:
    required = [
        CvatAttributeSpec(
            name="confidence",
            mutable=False,
            input_type="number",
            default_value="0",
            values=["0", "1", "0.001"],
        ),
        CvatAttributeSpec(
            name="routing",
            mutable=False,
            input_type="select",
            default_value="review",
            values=["review", "auto-accept", "drop"],
        ),
        CvatAttributeSpec(
            name="source",
            mutable=False,
            input_type="select",
            default_value="human",
            values=["human", "auto"],
        ),
        CvatAttributeSpec(
            name="review_status",
            mutable=True,
            input_type="select",
            default_value="pending",
            values=["pending", "accepted", "rejected"],
        ),
        CvatAttributeSpec(
            name="autolabel_track_id",
            mutable=False,
            input_type="text",
            default_value="",
            values=[],
        ),
        CvatAttributeSpec(
            name="review_priority",
            mutable=False,
            input_type="select",
            default_value="normal",
            values=["critical", "high", "normal", "low"],
        ),
        CvatAttributeSpec(
            name="review_score",
            mutable=False,
            input_type="number",
            default_value="0",
            values=["0", "1", "0.001"],
        ),
        CvatAttributeSpec(
            name="box_extent",
            mutable=False,
            input_type="select",
            default_value="visible",
            values=["visible", "amodal_estimate"],
        ),
    ]
    enriched = []
    for label in labels:
        supplied = {item.name: item for item in label.attributes or []}
        for attribute in required:
            supplied.setdefault(attribute.name, attribute)
        enriched.append(label.model_copy(update={"attributes": list(supplied.values())}))
    return enriched
