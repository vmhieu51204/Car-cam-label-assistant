"""SAM3 frame-zero detection, CVAT suggestions, and restart-safe seed capture."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import threading
from pathlib import Path
from typing import Any

from src.models.video import BoundingBox2D, FrameManifest, SeedSet, VideoJob
from src.services.cvat import CvatReadClient, CvatReadError, CvatTrackClient, CvatWriteError
from src.services.storage.repository import Store, VideoJobClaim
from src.services.video.batch_runner import VideoBatchError, VideoBatchRunner
from src.services.video.seeds import CvatSeedReader, SeedContractError, _label_map
from src.services.video.workspace import VideoWorkspace


class AutoSeedError(ValueError):
    """A recoverable frame-zero labeling error shown on the job."""


class KaggleSam3Seeder:
    def __init__(self, *, runner: VideoBatchRunner, workspace: VideoWorkspace, code_file: Path) -> None:
        self.runner = runner
        self.workspace = workspace
        self.code_file = code_file

    def detect(self, manifest: FrameManifest, labels: dict[int, str]) -> list[dict[str, Any]]:
        if not 1 <= len(labels) <= 12:
            raise AutoSeedError("SAM3 frame-0 labeling supports 1–12 rectangle labels per job")
        frame = manifest.frames[0]
        path = self.workspace.job_dir(manifest.job_id) / frame.relative_path
        if hashlib.sha256(path.read_bytes()).hexdigest() != frame.sha256:
            raise AutoSeedError("frame 0 does not match the video manifest")
        request = {
            "job_id": manifest.job_id,
            "frame_sha256": frame.sha256,
            "width": frame.width,
            "height": frame.height,
            "labels": [{"id": key, "prompt": value} for key, value in sorted(labels.items())],
        }
        result = self.runner.run(
            stage="sam3-seed",
            job_id=manifest.job_id,
            files={"request.json": json.dumps(request, sort_keys=True).encode(), "frame.jpg": path},
            code_file=self.code_file,
        )
        raw = result.get("detections")
        if not isinstance(raw, list) or len(raw) > 80:
            raise AutoSeedError("SAM3 returned an invalid or oversized frame-0 result")
        detections: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict) or item.get("label_id") not in labels:
                raise AutoSeedError("SAM3 returned an unsupported label")
            score = item.get("score")
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
                or not 0 <= score <= 1
            ):
                raise AutoSeedError("SAM3 returned an invalid confidence")
            try:
                box = BoundingBox2D.model_validate(item.get("bbox"))
            except ValueError as error:
                raise AutoSeedError("SAM3 returned an invalid bounding box") from error
            if box.xbr > frame.width or box.ybr > frame.height:
                raise AutoSeedError("SAM3 box exceeds frame-0 bounds")
            detections.append({"label_id": item["label_id"], "score": float(score), "bbox": box})
        if not detections:
            raise AutoSeedError("SAM3 found no frame-0 objects. Draw boxes manually and capture them instead.")
        return detections


class AutoSeedWorker:
    def __init__(
        self,
        *,
        store: Store,
        workspace: VideoWorkspace,
        read_client: CvatReadClient,
        track_client: CvatTrackClient,
        seeder: KaggleSam3Seeder,
        lease_seconds: float = 120,
    ) -> None:
        self.store = store
        self.workspace = workspace
        self.read_client = read_client
        self.track_client = track_client
        self.seeder = seeder
        self.lease_seconds = lease_seconds

    def process_next(self) -> VideoJob | None:
        self.store.requeue_expired_video_auto_seeds()
        claim = self.store.claim_next_video_auto_seed(lease_seconds=self.lease_seconds)
        if claim is None:
            return None
        try:
            manifest = self.workspace.read_manifest(claim.job.job_id, expected_sha256=claim.job.manifest_sha256)
            if claim.job.cvat_job_id is None or claim.job.cvat_task_id is None:
                raise AutoSeedError("video job has no CVAT binding")
            with _SeedHeartbeat(self.store, claim, self.lease_seconds) as heartbeat:
                seed_set = asyncio.run(self._process(claim, manifest))
            if heartbeat.lost:
                return self.store.get_video_job(claim.job.job_id)
            encoded = self.workspace.encode_seed_set(seed_set)
            completed = self.store.complete_video_auto_seed(
                claim.job.job_id,
                claim_token=claim.claim_token,
                seed_revision=seed_set.annotation_revision,
                seed_count=len(seed_set.seeds),
                seed_sha256=hashlib.sha256(encoded).hexdigest(),
                seed_payload=encoded.decode(),
            )
            self.workspace.write_seed_payload(claim.job.job_id, encoded)
            return completed
        except (
            AutoSeedError,
            SeedContractError,
            CvatReadError,
            CvatWriteError,
            VideoBatchError,
            ValueError,
            FileNotFoundError,
        ) as error:
            return self._fail(claim, str(error))
        except Exception:
            logging.exception("unexpected SAM3 frame-0 labeling failure for %s", claim.job.job_id)
            return self._fail(claim, "SAM3 frame-0 labeling failed; inspect worker logs")

    async def _process(self, claim: VideoJobClaim, manifest: FrameManifest) -> SeedSet:
        cvat_job_id = claim.job.cvat_job_id
        assert cvat_job_id is not None
        annotations, labels = await asyncio.gather(
            self.read_client.get_job_annotations(cvat_job_id),
            self.read_client.get_job_labels(cvat_job_id),
        )
        frame_zero = [shape for shape in annotations.get("shapes", []) if shape.get("frame") == 0]
        if frame_zero:
            if any(shape.get("source") != "auto" for shape in frame_zero):
                raise AutoSeedError("frame 0 already has manual boxes; capture those or clear them before using SAM3")
            # A previous worker may have written suggestions before its lease expired.
        else:
            if any(
                any(shape.get("frame") == 0 for shape in track.get("shapes", []))
                for track in annotations.get("tracks", [])
            ):
                raise AutoSeedError("frame 0 already has CVAT tracks; capture those instead")
            label_map = _label_map(labels)
            detections = await asyncio.to_thread(self.seeder.detect, manifest, label_map)
            # Re-read before writing: an annotator may have edited frame 0 during GPU inference.
            latest = await self.read_client.get_job_annotations(cvat_job_id)
            if latest.get("version") != annotations.get("version"):
                raise AutoSeedError("CVAT annotations changed during SAM3 inference; retry after reviewing frame 0")
            await self.track_client.publish_seed_shapes(
                cvat_job_id=cvat_job_id,
                expected_revision=annotations["version"],
                detections=detections,
            )
        current = await self.read_client.get_job_annotations(cvat_job_id)
        return await CvatSeedReader(self.read_client).capture(
            job_id=claim.job.job_id,
            cvat_job_id=cvat_job_id,
            expected_revision=current["version"],
            manifest=manifest,
            expected_task_id=claim.job.cvat_task_id,
        )

    def _fail(self, claim: VideoJobClaim, message: str) -> VideoJob:
        try:
            return self.store.fail_video_job(
                claim.job.job_id,
                code="auto_seed_failed",
                message=message,
                claim_token=claim.claim_token,
            )
        except ValueError:
            return self.store.get_video_job(claim.job.job_id)


class _SeedHeartbeat:
    def __init__(self, store: Store, claim: VideoJobClaim, lease_seconds: float) -> None:
        self.store, self.claim, self.lease_seconds = store, claim, lease_seconds
        self.stop = threading.Event()
        self.lost = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> _SeedHeartbeat:
        self._renew()
        self.thread.start()
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.stop.set()
        self.thread.join(timeout=min(5, self.lease_seconds))

    def _renew(self) -> None:
        self.store.renew_video_auto_seed(
            self.claim.job.job_id,
            claim_token=self.claim.claim_token,
            lease_seconds=self.lease_seconds,
        )

    def _run(self) -> None:
        while not self.stop.wait(max(1, min(30, self.lease_seconds / 3))):
            try:
                self._renew()
            except ValueError:
                self.lost = True
                return
