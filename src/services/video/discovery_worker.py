"""Persistent worker for independent detection and missing-object proposals."""

from __future__ import annotations

import hashlib
import logging
import threading

from pydantic import ValidationError

from src.models.video import DiscoveryResult, FrameManifest, SeedSet, TrackSet, VideoJob
from src.services.storage.repository import Store, VideoJobClaim
from src.services.video.discovery import VideoDetector, VideoDiscoveryError, discover_missing_objects
from src.services.video.review import build_review_queue
from src.services.video.workspace import VideoWorkspace


class DiscoveryWorker:
    def __init__(
        self,
        *,
        store: Store,
        workspace: VideoWorkspace,
        detector: VideoDetector,
        lease_seconds: float = 120,
    ) -> None:
        if lease_seconds < 3:
            raise ValueError("discovery lease must be at least three seconds")
        self.store = store
        self.workspace = workspace
        self.detector = detector
        self.lease_seconds = lease_seconds

    def process_next(self) -> VideoJob | None:
        self.store.requeue_expired_video_discovery()
        claim = self.store.claim_next_video_discovery(lease_seconds=self.lease_seconds)
        if claim is None:
            return None
        try:
            manifest, seeds, propagated = self._inputs(claim)
            heartbeat = _DiscoveryHeartbeat(
                store=self.store,
                claim=claim,
                total=manifest.frame_count,
                lease_seconds=self.lease_seconds,
            )
            with heartbeat:
                detections = self.detector.detect(
                    manifest=manifest,
                    manifest_sha256=claim.job.manifest_sha256 or "",
                    labels={seed.label_id: seed.label_name for seed in seeds.seeds},
                )
            if heartbeat.lost:
                return self.store.get_video_job(claim.job.job_id)
            discovery = discover_missing_objects(
                manifest=manifest,
                propagated=propagated,
                detections=detections,
                propagated_track_sha256=claim.job.track_set_sha256 or "",
            )
            combined = propagated.model_copy(
                update={
                    "component_versions": propagated.component_versions + discovery.component_versions,
                    "tracks": propagated.tracks + discovery.proposal_tracks,
                }
            )
            self.store.update_video_discovery_progress(
                claim.job.job_id,
                claim_token=claim.claim_token,
                current=manifest.frame_count,
                total=manifest.frame_count,
                lease_seconds=self.lease_seconds,
            )
            return self._complete(claim, manifest, combined, discovery)
        except (FileNotFoundError, ValidationError, ValueError) as error:
            return self._fail(claim, "discovery_input_invalid", str(error))
        except VideoDiscoveryError as error:
            return self._fail(claim, "video_discovery_failed", str(error))
        except Exception:
            logging.exception("unexpected discovery failure for video job %s", claim.job.job_id)
            return self._fail(
                claim,
                "video_discovery_internal_error",
                "video discovery failed unexpectedly; inspect worker logs",
            )

    def _inputs(self, claim: VideoJobClaim) -> tuple[FrameManifest, SeedSet, TrackSet]:
        job = claim.job
        if job.manifest_sha256 is None or job.seed_sha256 is None or job.track_set_sha256 is None:
            raise ValueError("video discovery provenance is incomplete")
        manifest = self.workspace.read_manifest(job.job_id, expected_sha256=job.manifest_sha256)
        self.workspace.verify_frame_files(manifest)
        seed_bytes = self.store.read_video_seed(job.job_id).encode("utf-8")
        track_bytes = self.store.read_video_track_set(job.job_id).encode("utf-8")
        if hashlib.sha256(seed_bytes).hexdigest() != job.seed_sha256:
            raise ValueError("stored seed hash does not match the video job")
        if hashlib.sha256(track_bytes).hexdigest() != job.track_set_sha256:
            raise ValueError("stored track-set hash does not match the video job")
        return manifest, SeedSet.model_validate_json(seed_bytes), TrackSet.model_validate_json(track_bytes)

    def _complete(
        self, claim: VideoJobClaim, manifest: FrameManifest, tracks: TrackSet, discovery: DiscoveryResult
    ) -> VideoJob:
        discovery_bytes = self.workspace.encode_discovery(discovery)
        track_bytes = self.workspace.encode_track_set(tracks)
        track_sha = hashlib.sha256(track_bytes).hexdigest()
        discovery_sha = hashlib.sha256(discovery_bytes).hexdigest()
        queue = build_review_queue(
            tracks=tracks,
            discovery=discovery,
            manifest=manifest,
            track_set_sha256=track_sha,
            discovery_sha256=discovery_sha,
        )
        queue_bytes = self.workspace.encode_review_queue(queue)
        completed = self.store.complete_video_discovery(
            claim.job.job_id,
            claim_token=claim.claim_token,
            track_set_sha256=track_sha,
            track_payload=track_bytes.decode("utf-8"),
            discovery_sha256=discovery_sha,
            discovery_payload=discovery_bytes.decode("utf-8"),
            component_versions=tracks.component_versions,
            frame_count=manifest.frame_count,
            review_queue_sha256=hashlib.sha256(queue_bytes).hexdigest(),
            review_queue_payload=queue_bytes.decode("utf-8"),
        )
        self.workspace.write_track_set_payload(claim.job.job_id, track_bytes)
        self.workspace.write_discovery_payload(claim.job.job_id, discovery_bytes)
        self.workspace.write_review_queue_payload(claim.job.job_id, queue_bytes)
        return completed

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


class _DiscoveryHeartbeat:
    def __init__(self, *, store: Store, claim: VideoJobClaim, total: int, lease_seconds: float) -> None:
        self.store = store
        self.claim = claim
        self.total = total
        self.lease_seconds = lease_seconds
        self.stop = threading.Event()
        self.lost = False
        self.thread = threading.Thread(target=self._run, daemon=True, name=f"discovery-{claim.job.job_id[:8]}")

    def __enter__(self) -> _DiscoveryHeartbeat:
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
        self.store.update_video_discovery_progress(
            self.claim.job.job_id,
            claim_token=self.claim.claim_token,
            current=0,
            total=self.total,
            lease_seconds=self.lease_seconds,
        )
