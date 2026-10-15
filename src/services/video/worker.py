"""Restart-safe extraction worker over the persistent SQLite queue."""

from __future__ import annotations

import asyncio
import hashlib
import logging

from src.models.video import VideoJob
from src.services.cvat import CvatReadClient, CvatReadError
from src.services.storage.repository import Store
from src.services.video.cvat_import import import_cvat_frames
from src.services.video.extractor import FFmpegVideoExtractor, VideoExtractionCancelledError, VideoExtractionError
from src.services.video.profile import Profile2D
from src.services.video.workspace import VideoWorkspace


class VideoWorker:
    def __init__(
        self,
        *,
        store: Store,
        workspace: VideoWorkspace,
        extractor: FFmpegVideoExtractor,
        cvat_client: CvatReadClient | None = None,
        profile_enabled: bool = False,
        lease_seconds: float = 60,
    ) -> None:
        self.profile_enabled = profile_enabled
        self.store = store
        self.workspace = workspace
        self.extractor = extractor
        self.cvat_client = cvat_client
        self.lease_seconds = lease_seconds

    def process_next(self) -> VideoJob | None:
        self.store.requeue_expired_video_extractions()
        claim = self.store.claim_next_video_job(lease_seconds=self.lease_seconds)
        if claim is None:
            return None

        def report_progress(current: int, total: int) -> None:
            try:
                self.store.update_video_progress(
                    claim.job.job_id,
                    claim_token=claim.claim_token,
                    current=current,
                    total=total,
                    lease_seconds=self.lease_seconds,
                )
            except ValueError as error:
                raise VideoExtractionCancelledError("video extraction claim was cancelled or expired") from error

        try:
            if claim.job.cvat_job_id is not None:
                if self.cvat_client is None:
                    raise VideoExtractionError("CVAT read client is not configured on the video worker")
                manifest = asyncio.run(
                    import_cvat_frames(
                        claim.job,
                        client=self.cvat_client,
                        workspace=self.workspace,
                        progress=report_progress,
                        profile_enabled=self.profile_enabled,
                    )
                )
            else:
                profile = Profile2D(self.workspace.job_dir(claim.job.job_id), enabled=self.profile_enabled)
                with profile.activate(), profile.stage("decode_extract.total"):
                    manifest = self.extractor.extract(claim.job, progress=report_progress)
            encoded = self.workspace.write_manifest(manifest)
            completed = self.store.complete_video_extraction(
                claim.job.job_id,
                claim_token=claim.claim_token,
                manifest_sha256=hashlib.sha256(encoded).hexdigest(),
                frame_count=manifest.frame_count,
            )
            if self.workspace.read_sampling_request(claim.job.job_id).auto_start_sequence:
                return self.store.queue_video_sequence(completed.job_id, frame_count=manifest.frame_count)
            return completed
        except VideoExtractionCancelledError:
            return self.store.get_video_job(claim.job.job_id)
        except (VideoExtractionError, CvatReadError, ValueError) as error:
            return self._fail_claim(claim.job.job_id, claim.claim_token, "video_extraction_failed", str(error))
        except Exception:
            logging.exception("unexpected extraction failure for video job %s", claim.job.job_id)
            return self._fail_claim(
                claim.job.job_id,
                claim.claim_token,
                "video_extraction_internal_error",
                "video extraction failed unexpectedly; inspect worker logs",
            )

    def _fail_claim(self, job_id: str, claim_token: str, code: str, message: str) -> VideoJob:
        try:
            return self.store.fail_video_job(job_id, code=code, message=message, claim_token=claim_token)
        except ValueError:
            return self.store.get_video_job(job_id)
