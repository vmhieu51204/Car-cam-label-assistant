"""Leased seed-free SAM3 Kaggle worker for a full-frame CVAT video job."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.models.video import DiscoveryResult, VideoJob
from src.services.cvat.read_client import CvatReadClient
from src.services.storage.repository import Store, VideoJobClaim
from src.services.video.kaggle_batch import KaggleBatchRunner
from src.services.video.profile import current_profile, profiled_worker
from src.services.video.sequence_checkpoints import SequenceCheckpoints
from src.services.video.sequence_kaggle import make_track_set, parse_sequence_result, sequence_payload
from src.services.video.sequence_review import build_sequence_review_queue
from src.services.video.service import VideoJobService
from src.services.video.workspace import VideoWorkspace


class SequenceWorker:
    def __init__(
        self,
        *,
        profile_enabled: bool = False,
        stitch_decision_log_enabled: bool = True,
        store: Store,
        workspace: VideoWorkspace,
        cvat: CvatReadClient,
        runner: KaggleBatchRunner | None,
        notebook: Path,
        lease_seconds: float = 120,
        draft_service: VideoJobService | None = None,
        local_runner: Any = None,
    ) -> None:
        self.profile_enabled = profile_enabled
        self.stitch_decision_log_enabled = stitch_decision_log_enabled
        self.store = store
        self.workspace = workspace
        self.cvat = cvat
        self.runner = runner
        self.notebook = notebook
        self.lease_seconds = lease_seconds
        self.draft_service = draft_service
        self.local_runner = local_runner

    def process_next(self) -> VideoJob | None:
        if self.draft_service is not None:
            draft = self.store.claim_next_video_sequence_draft(lease_seconds=self.lease_seconds)
            if draft is not None:
                return self._export_draft(draft)
        self.store.requeue_expired_video_sequences()
        claim = self.store.claim_next_video_sequence(lease_seconds=self.lease_seconds)
        if claim is None:
            return None
        try:
            completed = self._process(claim)
            if self.draft_service is not None and completed.stage == "sequence_draft_ready":
                draft = self.store.claim_next_video_sequence_draft(lease_seconds=self.lease_seconds)
                if draft is not None:
                    return self._export_draft(draft)
            return completed
        except Exception as error:
            logging.exception("SAM3 sequence job %s failed", claim.job.job_id)
            try:
                return self.store.fail_video_job(
                    claim.job.job_id,
                    code="sequence_failed",
                    message=f"{type(error).__name__}: {str(error)[:380]}",
                    claim_token=claim.claim_token,
                )
            except ValueError:
                return self.store.get_video_job(claim.job.job_id)

    @profiled_worker("cvat.write")
    def _export_draft(self, claim: VideoJobClaim) -> VideoJob:
        """Append pending tracks automatically; failures retain outputs for a CPU-only retry."""
        assert self.draft_service is not None
        try:
            asyncio.run(
                self.draft_service.publish_tracks(
                    claim.job.job_id,
                    expected_revision=None,
                    worker_export=True,
                )
            )
        except Exception as error:
            logging.exception("CVAT draft export for %s failed", claim.job.job_id)
            return self.store.fail_video_sequence_draft(
                claim.job.job_id,
                claim_token=claim.claim_token,
                message=f"{type(error).__name__}: {str(error)[:380]}",
            )
        return self.store.get_video_job(claim.job.job_id)

    @profiled_worker("sequence.total")
    def _process(self, claim: VideoJobClaim) -> VideoJob:
        with _SequenceHeartbeat(self.store, claim, claim.job.progress_total, self.lease_seconds):
            return self._process_with_lease(claim)

    def _process_with_lease(self, claim: VideoJobClaim) -> VideoJob:
        job = claim.job
        if job.cvat_job_id is None or job.manifest_sha256 is None:
            raise ValueError("seed-free sequence requires a bound CVAT job and manifest")
        manifest = self.workspace.read_manifest(job.job_id, expected_sha256=job.manifest_sha256)
        request = self.workspace.read_sampling_request(job.job_id)
        if not request.auto_start_sequence or request.strategy not in {"uniform", "all_frames"}:
            raise ValueError("job is not a seed-free uniform sequence request")
        classes = tuple(request.sequence_classes)
        labels = asyncio.run(
            self.draft_service.prepare_sequence_labels(job.cvat_job_id, classes)
            if self.draft_service is not None
            else self.cvat.get_job_labels(job.cvat_job_id)
        )
        if any(name not in {label["name"] for label in labels} for name in classes):
            raise ValueError("CVAT job does not contain every requested class")
        model_variant = request.sam3_model_variant
        runner: Any = self.local_runner if request.sam_execution_provider == "local" else self.runner
        if runner is None:
            raise ValueError(f"{request.sam_execution_provider} SAM worker is not configured")
        files, config = sequence_payload(manifest, self.workspace, classes, model_variant=model_variant)
        code_hash = hashlib.sha256(self.notebook.read_bytes()).hexdigest()
        if request.sam_execution_provider == "local":
            code_hash = runner.fingerprint(model_variant)
        config_bytes = files["benchmark-config.json"]
        frame_bytes = files["frames.json"]
        assert isinstance(config_bytes, bytes) and isinstance(frame_bytes, bytes)
        benchmark_config = json.loads(config_bytes)
        benchmark_config["code_sha256"] = code_hash
        files["benchmark-config.json"] = json.dumps(benchmark_config).encode()
        checkpoints = SequenceCheckpoints(
            self.store,
            self.workspace.job_dir(job.job_id),
            manifest,
            json.loads(frame_bytes)["frames"],
            config,
            code_hash,
        )
        saved = checkpoints.resume()
        files["resume.json"] = json.dumps({"windows": saved}).encode()
        checkpoints.claim_pending()
        try:
            if len(saved) == len(checkpoints.plans):
                model = saved[0]["model"]
                result: dict[str, Any] = {
                    "schema": "autolabel-sam3-benchmark/v1",
                    "model_runtime": {
                        "environment": {"versions": {"sam3_commit": model.get("sam3_commit")}},
                        "checkpoint_sha256": model.get("checkpoint_sha256"),
                        "model_variant": model_variant,
                    },
                    "configurations": [
                        {
                            "id": config["id"],
                            "config": config,
                            "status": "completed",
                            "selected_source_frames": list(range(manifest.frame_count)),
                            "evaluated_source_frames": list(range(manifest.frame_count)),
                            "window_predictions": [row for entry in saved for row in entry["predictions"]],
                        }
                    ],
                }
            else:
                with current_profile().stage(f"{request.sam_execution_provider}.batch_wait"):
                    result = runner.run(
                        stage="sam3-sequence",
                        job_id=job.job_id,
                        files=files,
                        code_file=self.notebook,
                        model_variant=model_variant,
                    )
            rows = result.get("configurations", [])
            if len(rows) == 1 and rows[0].get("config") == config:
                checkpoints.ingest(rows[0].get("window_checkpoints", []))
            if len(rows) == 1 and rows[0].get("status") != "completed":
                raise ValueError(
                    f"SAM3 incomplete: {len(checkpoints.resume())}/{len(checkpoints.plans)} "
                    "window checkpoints stored; retry resumes only verified windows"
                )
        finally:
            checkpoints.fail_remaining("window output unavailable; cannot claim successful checkpoint")
        with current_profile().stage("sequence.stitch"):
            if manifest.sampling_strategy == "all_frames" and len(checkpoints.resume()) != len(checkpoints.plans):
                raise ValueError("full-frame result lacks verified checkpoints for every window")
            stitched, component = parse_sequence_result(
                result,
                expected_config=config,
                frame_count=manifest.frame_count,
                decision_log_enabled=self.stitch_decision_log_enabled,
            )
        with current_profile().stage("sequence.prepare_tracks_review"):
            track_set = make_track_set(
                manifest=manifest,
                manifest_sha256=job.manifest_sha256,
                stitched=stitched,
                labels=labels,
                classes=classes,
                component=component,
            )
            track_bytes = self.workspace.encode_track_set(track_set)
            track_sha = hashlib.sha256(track_bytes).hexdigest()
            discovery = DiscoveryResult(
                job_id=job.job_id,
                manifest_sha256=job.manifest_sha256,
                propagated_track_sha256=track_sha,
                component_versions=[component],
                proposal_tracks=[],
                events=[],
            )
            discovery_bytes = self.workspace.encode_discovery(discovery)
            discovery_sha = hashlib.sha256(discovery_bytes).hexdigest()
            review = build_sequence_review_queue(
                tracks=track_set,
                manifest=manifest,
                track_set_sha256=track_sha,
                discovery_sha256=discovery_sha,
            )
            review_bytes = self.workspace.encode_review_queue(review)
            review_sha = hashlib.sha256(review_bytes).hexdigest()
        artifact = self.workspace.job_dir(job.job_id) / "sequence-stitch.json"
        with current_profile().stage("sequence.write_artifacts"):
            artifact.write_text(
                json.dumps(
                    {
                        "schema": "autolabel-sequence-stitch/v1",
                        "model_result_sha256": hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest(),
                        "decisions": [asdict(item) for item in stitched.decisions],
                        "boundary_logs": stitched.boundary_logs,
                        "decision_log_enabled": self.stitch_decision_log_enabled,
                        "tracks": [asdict(item) for item in stitched.tracks],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            completed = self.store.complete_video_sequence(
                job.job_id,
                claim_token=claim.claim_token,
                tracks=track_bytes.decode(),
                track_sha256=track_sha,
                discovery=discovery_bytes.decode(),
                discovery_sha256=discovery_sha,
                review=review_bytes.decode(),
                review_sha256=review_sha,
                component_versions=[component],
                frame_count=manifest.frame_count,
                queue_box_rescue=request.vlm_box_rescue,
            )
            self.workspace.write_track_set_payload(job.job_id, track_bytes)
            self.workspace.write_discovery_payload(job.job_id, discovery_bytes)
            self.workspace.write_review_queue_payload(job.job_id, review_bytes)
        return completed


class _SequenceHeartbeat:
    def __init__(self, store: Store, claim: VideoJobClaim, total: int, lease_seconds: float) -> None:
        self.store = store
        self.claim = claim
        self.total = total
        self.lease_seconds = lease_seconds
        self.stop = threading.Event()
        self.lost = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> _SequenceHeartbeat:
        self._renew()
        self.thread.start()
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.stop.set()
        self.thread.join(timeout=5)

    def _run(self) -> None:
        while not self.stop.wait(max(1, min(30, self.lease_seconds / 3))):
            try:
                self._renew()
            except ValueError:
                self.lost = True
                return

    def _renew(self) -> None:
        self.store.renew_video_sequence(
            self.claim.job.job_id,
            claim_token=self.claim.claim_token,
            lease_seconds=self.lease_seconds,
            progress=0,
            total=self.total,
        )
