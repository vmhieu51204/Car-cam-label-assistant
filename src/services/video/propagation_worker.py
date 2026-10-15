"""Persistent leased worker for seeded video propagation."""

from __future__ import annotations

import hashlib
import logging
import threading

from pydantic import ValidationError

from src.models.video import FrameManifest, SeedSet, VideoJob
from src.services.storage.repository import Store, VideoJobClaim
from src.services.video.propagation import VideoPropagationError, VideoPropagator
from src.services.video.workspace import VideoWorkspace


class PropagationWorker:
    def __init__(
        self,
        *,
        store: Store,
        workspace: VideoWorkspace,
        propagator: VideoPropagator,
        lease_seconds: float = 120,
    ) -> None:
        if lease_seconds < 3:
            raise ValueError("propagation lease must be at least three seconds")
        self.store = store
        self.workspace = workspace
        self.propagator = propagator
        self.lease_seconds = lease_seconds

    def process_next(self) -> VideoJob | None:
        self.store.requeue_expired_video_propagations()
        claim = self.store.claim_next_video_propagation(lease_seconds=self.lease_seconds)
        if claim is None:
            return None
        try:
            manifest, seed_set = self._inputs(claim)
            heartbeat = _PropagationHeartbeat(
                store=self.store,
                claim=claim,
                total=manifest.frame_count,
                lease_seconds=self.lease_seconds,
            )
            with heartbeat:
                track_set = self.propagator.propagate(
                    manifest=manifest,
                    manifest_sha256=claim.job.manifest_sha256 or "",
                    seed_set=seed_set,
                    seed_sha256=claim.job.seed_sha256 or "",
                )
            if heartbeat.lost:
                return self.store.get_video_job(claim.job.job_id)
            self.store.update_video_propagation_progress(
                claim.job.job_id,
                claim_token=claim.claim_token,
                current=manifest.frame_count,
                total=manifest.frame_count,
                lease_seconds=self.lease_seconds,
            )
            encoded = self.workspace.encode_track_set(track_set)
            completed = self.store.complete_video_propagation(
                claim.job.job_id,
                claim_token=claim.claim_token,
                track_set_sha256=hashlib.sha256(encoded).hexdigest(),
                track_payload=encoded.decode("utf-8"),
                component_versions=track_set.component_versions,
                frame_count=manifest.frame_count,
            )
            self.workspace.write_track_set_payload(claim.job.job_id, encoded)
            return completed
        except (FileNotFoundError, ValidationError, ValueError) as error:
            return self._fail(claim, "propagation_input_invalid", str(error))
        except VideoPropagationError as error:
            return self._fail(claim, "video_propagation_failed", str(error))
        except Exception:
            logging.exception("unexpected propagation failure for video job %s", claim.job.job_id)
            return self._fail(
                claim,
                "video_propagation_internal_error",
                "video propagation failed unexpectedly; inspect worker logs",
            )

    def _inputs(self, claim: VideoJobClaim) -> tuple[FrameManifest, SeedSet]:
        job = claim.job
        if job.manifest_sha256 is None or job.seed_sha256 is None:
            raise ValueError("video propagation provenance is incomplete")
        manifest = self.workspace.read_manifest(job.job_id, expected_sha256=job.manifest_sha256)
        self.workspace.verify_frame_files(manifest)
        encoded = self.store.read_video_seed(job.job_id).encode("utf-8")
        if hashlib.sha256(encoded).hexdigest() != job.seed_sha256:
            raise ValueError("stored seed hash does not match the video job")
        return manifest, SeedSet.model_validate_json(encoded)

    def _fail(self, claim: VideoJobClaim, code: str, message: str) -> VideoJob:
        try:
            return self.store.fail_video_job(
                claim.job.job_id,
                code=code,
                message=message,
                claim_token=claim.claim_token,
            )
        except ValueError:
            return self.store.get_video_job(claim.job.job_id)


class _PropagationHeartbeat:
    def __init__(self, *, store: Store, claim: VideoJobClaim, total: int, lease_seconds: float) -> None:
        self.store = store
        self.claim = claim
        self.total = total
        self.lease_seconds = lease_seconds
        self.stop = threading.Event()
        self.lost = False
        self.thread = threading.Thread(target=self._run, daemon=True, name=f"propagation-{claim.job.job_id[:8]}")

    def __enter__(self) -> _PropagationHeartbeat:
        self._renew()
        self.thread.start()
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.stop.set()
        self.thread.join(timeout=min(5, self.lease_seconds))

    def _run(self) -> None:
        interval = max(1.0, min(30.0, self.lease_seconds / 3))
        while not self.stop.wait(interval):
            try:
                self._renew()
            except ValueError:
                self.lost = True
                return

    def _renew(self) -> None:
        self.store.update_video_propagation_progress(
            self.claim.job.job_id,
            claim_token=self.claim.claim_token,
            current=0,
            total=self.total,
            lease_seconds=self.lease_seconds,
        )
