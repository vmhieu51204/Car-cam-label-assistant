"""Single-GPU owner of privacy, CVAT binding and streaming annotation delivery."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import signal
import subprocess
import threading
import time
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx

from src.config import PROJECT_ROOT, Settings
from src.models.video import CvatLabelSpec
from src.services.cvat import CvatTaskClient
from src.services.cvat.read_client import cvat_route_host
from src.services.cvat.session_authorizer import CvatSessionAuthorizer
from src.services.cvat.session_read_client import CvatSessionReadClient
from src.services.video.manual_frame_rescue import propose_frame
from src.services.video.service import _labels_with_autolabel_attributes
from src.services.vm.media import verified_media
from src.services.vm.reader import VerifiedFrameReader
from src.services.vm.storage import VmStore, atomic_json, encode, worker_lock
from src.services.vm.streaming import StreamingEngine
from src.services.vm.vlm import Secrets, VlmProfiles


class UserTaskClient(CvatTaskClient):
    def __init__(self, settings: Settings, credentials: dict, intent: Path | None = None):
        super().__init__(
            base_url=settings.cvat_url,
            token="",
            ready_timeout_s=settings.cvat_task_ready_timeout_s,
            timeout_s=settings.cvat_timeout_s,
        )
        self.credentials, self.origin = credentials, settings.cvat_public_url
        self.host_header = cvat_route_host(settings)
        self.intent = intent

    async def _create_task(
        self, client: httpx.AsyncClient, name: str, labels: list[CvatLabelSpec], frame_count: int
    ) -> int:
        task = await super()._create_task(client, name, labels, frame_count)
        if self.intent:
            atomic_json(self.intent, {"task_id": task, "name": name, "phase": "created_waiting_media"})
        return task

    def _headers(self) -> dict[str, str]:
        return {
            **({"Host": self.host_header} if self.host_header else {}),
            "Cookie": f"sessionid={self.credentials['sessionid']}; csrftoken={self.credentials['csrftoken']}",
            "X-CSRFToken": self.credentials["csrftoken"],
            "Origin": self.origin,
            "Accept": "application/vnd.cvat+json",
        }


class IncrementalPublisher:
    def __init__(self, settings: Settings, store: VmStore, job: dict, labels: list[dict], credentials: dict):
        self.settings, self.store, self.job, self.labels, self.credentials = settings, store, job, labels, credentials
        self.read = CvatSessionReadClient(settings, credentials["sessionid"])
        self.authorizer = CvatSessionAuthorizer(settings)
        self.pending: list[dict] = []
        self.pending_events: list[str] = []
        self.last_write = 0.0
        self.published: set[int] = set()
        self.event_keys: set[tuple[int, str]] = set()
        with store.connect() as db:
            rows = db.execute(
                "SELECT payload FROM vm_events WHERE job=? AND event LIKE 'published:%'", (job["job_id"],)
            ).fetchall()
        for row in rows:
            record = json.loads(row[0])
            self.event_keys.add((record["frame"], record["class"]))
            self.published.add(record["frame"])
        self.lock = threading.RLock()
        self.stopped = threading.Event()
        self.failure: Exception | None = None

        def flush_periodically() -> None:
            while not self.stopped.wait(1):
                try:
                    self.flush()
                except Exception as error:
                    self.failure = error
                    return

        self.thread = threading.Thread(target=flush_periodically, daemon=True)
        self.thread.start()

    def __call__(self, event: dict, mapping: dict[str, str]) -> None:
        with self.lock:
            if self.failure:
                raise self.failure
            self.enqueue(event, mapping)

    def enqueue(self, event: dict, mapping: dict[str, str]) -> None:
        key = (event["frame"], event["class"])
        if key in self.event_keys:
            return
        self.event_keys.add(key)
        label = next(row for row in self.labels if row["name"] == event["class"])
        specs = {row["name"]: row["id"] for row in label["attributes"]}
        mutable = {row["name"] for row in label["attributes"] if row["mutable"]}
        for observation in event["observations"]:
            identity = f"vm:{self.job['job_id']}:{mapping[observation['track_id']]}"
            values = {
                "autolabel_track_id": identity,
                "review_status": "pending",
                "source": "auto",
                "routing": "review",
                "confidence": str(observation["score"]),
                "review_priority": "normal",
            }
            self.pending.append(
                {
                    "identity": identity,
                    "label_id": label["id"],
                    "frame": event["frame"],
                    "points": observation["bbox"],
                    "attributes": [
                        {"spec_id": specs[k], "value": v} for k, v in values.items() if k in specs and k not in mutable
                    ],
                    "shape_attributes": [
                        {"spec_id": specs[k], "value": v} for k, v in values.items() if k in specs and k in mutable
                    ],
                    "total_frames": self.job["total_frames"],
                }
            )
        self.pending_events.append(f"{event['frame']}:{event['class']}")
        self.published.add(event["frame"])
        if time.monotonic() - self.last_write >= 1:
            self.flush()

    def flush(self) -> None:
        with self.lock:
            if self.failure:
                raise self.failure
            self.flush_locked()

    def close(self) -> None:
        self.stopped.set()
        self.thread.join(timeout=self.settings.cvat_timeout_s * 4 + 5)

    def flush_locked(self) -> None:
        if not self.pending_events:
            return
        batch = {
            "operation_id": self.job["job_id"]
            + ":"
            + hashlib.sha256(encode([self.pending_events, self.pending])).hexdigest(),
            "operations": self.pending,
            "frame_keys": self.pending_events,
            "expected_revision": 0,
        }
        # Persist the exact outbound intent; lost responses always replay this batch.
        intent = self.store.directory(self.job["job_id"]) / "private" / "delivery.json"
        atomic_json(intent, batch)
        asyncio.run(self.send(batch))
        self.confirm(batch)
        atomic_json(intent.with_name("delivery-complete.json"), batch)
        intent.unlink(missing_ok=True)
        self.store.update(
            self.job["job_id"],
            published_frames=len(self.published),
            publication_version=self.store.get(self.job["job_id"]).get("publication_version", 0) + 1,
        )
        self.pending, self.pending_events = [], []
        self.last_write = time.monotonic()

    def confirm(self, batch: dict) -> None:
        for key in batch.get("frame_keys", []):
            frame, label = key.split(":", 1)
            self.store.event(self.job["job_id"], "published:" + key, {"frame": int(frame), "class": label})
            self.event_keys.add((int(frame), label))
            self.published.add(int(frame))

    async def send(self, batch: dict) -> None:
        if self.store.get(self.job["job_id"])["status"] == "cancelled":
            raise ValueError("VM job cancelled")
        job_id = self.job["cvat_job_id"]
        owner = await self.authorizer.authorize(self.credentials["sessionid"], job_id, write=True)
        if owner != self.job["owner"]:
            raise ValueError("CVAT session identity changed")
        for _attempt in range(3):
            current = await self.read.get_guarded_job_annotations(job_id)
            batch["expected_revision"] = current["autolabel_revision"]
            async with httpx.AsyncClient(
                base_url=self.settings.cvat_url,
                headers=UserTaskClient(self.settings, self.credentials)._headers(),
                timeout=self.settings.cvat_timeout_s,
            ) as client:
                response = await client.post(
                    f"/api/jobs/{job_id}/autolabel-stream", json={"mode": "incremental", **batch}
                )
            if response.status_code == 409:
                continue
            if not response.is_success:
                raise ValueError(f"CVAT incremental delivery HTTP {response.status_code}; annotations remain pending")
            return
        raise ValueError("CVAT changed during incremental delivery; retry after saving")


class VmWorker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = VmStore(settings.db_path, settings.vm_work_root)

    def privacy(self, job: dict) -> dict:
        directory = self.store.directory(job["job_id"])
        output = directory / "public" / "redacted.mp4"
        receipt_path = output.with_suffix(".privacy.json")
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if "model" in receipt or job.get("cvat_job_id"):
                return verified_media(self.store, job)[1]
            # Recover the old two-write finalization crash without reusing unverified media.
            receipt_path.replace(directory / "private" / ("incomplete-receipt-" + str(time.time_ns()) + ".json"))
        if output.exists():
            # Keep incomplete output private; never expose or overwrite it on retry.
            output.replace(directory / "private" / ("incomplete-redacted-" + str(time.time_ns()) + ".mp4"))
        profile = job["privacy_profile"]
        request = {
            "source": str((directory / "private" / ("source" + job["source_extension"])).resolve()),
            "source_sha256": job["source_sha256"],
            "output": str(output.resolve()),
            "regions": job["manual_regions"],
            "face": profile["privacy_face_checkpoint"],
            "plate": profile["privacy_plate_checkpoint"],
            "face_sha256": profile["privacy_face_sha256"],
            "plate_sha256": profile["privacy_plate_sha256"],
            "source_commit": profile["privacy_source_commit"],
            "ffmpeg": self.settings.ffmpeg_path,
            "ffprobe": self.settings.ffprobe_path,
        }
        path = directory / "private" / "privacy-request.json"
        atomic_json(path, request)
        with (directory / "private" / "privacy.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [
                    str(self.settings.privacy_python.absolute()),
                    "-m",
                    "scripts.run_privacy",
                    "--request",
                    str(path.resolve()),
                ],
                cwd=PROJECT_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=os.name != "nt",
            )
            deadline = time.monotonic() + self.settings.privacy_timeout_s
            try:
                while process.poll() is None:
                    if self.store.get(job["job_id"])["status"] == "cancelled" or time.monotonic() > deadline:
                        raise ValueError("Privacy cancelled or timed out")
                    progress = path.with_name("privacy-progress.json")
                    if progress.exists():
                        counts = json.loads(progress.read_text())
                        self.store.update(job["job_id"], privacy_frames=counts["current"], total_frames=counts["total"])
                    time.sleep(1)
                if process.returncode or not receipt_path.exists():
                    raise ValueError("Privacy failed; no raw-media fallback. Inspect private privacy log")
            finally:
                if process.poll() is None:
                    if os.name != "nt":
                        # Privacy owns an FFmpeg decoder child; stop the entire
                        # isolated session when cancelling or timing out.
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    else:
                        process.kill()
                    process.wait()
        return verified_media(self.store, job)[1]

    async def bind(self, job: dict, receipt: dict, credentials: dict) -> tuple[int, int]:
        if job.get("in_place"):
            read = CvatSessionReadClient(self.settings, credentials["sessionid"])
            scope = await read.get_job(job["cvat_job_id"])
            if scope["task_id"] != job["cvat_task_id"]:
                raise ValueError("CVAT target task identity changed")
            owner = await CvatSessionAuthorizer(self.settings).authorize(
                credentials["sessionid"], job["cvat_job_id"], write=True
            )
            if owner != job["owner"]:
                raise ValueError("CVAT session identity changed")
            labels = await read.get_job_labels(job["cvat_job_id"])
            selected = [label for label in labels if label["name"] in job["classes"]]
            if {label["name"] for label in selected} != set(job["classes"]):
                raise ValueError("CVAT target labels changed; start a new AutoLabel run")
            required = _labels_with_autolabel_attributes([CvatLabelSpec(name="review")])[0].attributes
            await UserTaskClient(self.settings, credentials).ensure_label_attributes(selected, required or [])
            return job["cvat_task_id"], job["cvat_job_id"]
        if job.get("cvat_job_id"):
            return job["cvat_task_id"], job["cvat_job_id"]
        marker = self.store.directory(job["job_id"]) / "private" / "cvat-create-intent.json"
        # Task creation has an unknown outcome if interrupted: do not create a replacement.
        if marker.exists():
            reserved = json.loads(marker.read_text())
            if not reserved.get("task_id"):
                raise ValueError("CVAT creation outcome unknown; inspect the reserved task. No automatic replacement")
            client = UserTaskClient(self.settings, credentials)
            async with httpx.AsyncClient(
                base_url=self.settings.cvat_url, headers=client._headers(), timeout=self.settings.cvat_timeout_s
            ) as session:
                response = await session.get(f"/api/tasks/{reserved['task_id']}")
                if not response.is_success or response.json().get("name") != "AutoLabel redacted " + job["job_id"]:
                    raise ValueError("Reserved redacted CVAT task identity changed")
                cvat_job = await client._wait_for_single_job(session, reserved["task_id"])
            self.store.update(job["job_id"], cvat_task_id=reserved["task_id"], cvat_job_id=cvat_job)
            return reserved["task_id"], cvat_job
        atomic_json(marker, {"job": job["job_id"]})
        client = UserTaskClient(self.settings, credentials, marker)
        task, cvat_job = await client.create_video_task(
            name="AutoLabel redacted " + job["job_id"],
            labels=_labels_with_autolabel_attributes([CvatLabelSpec(name=c) for c in job["classes"]]),
            video_path=self.store.directory(job["job_id"]) / "public" / "redacted.mp4",
        )
        self.store.update(job["job_id"], cvat_task_id=task, cvat_job_id=cvat_job)
        return task, cvat_job

    def process_next(self) -> dict | None:
        if not self.settings.vm_workflow_enabled:
            return None
        with worker_lock(self.store.root / "gpu-worker.lock"):
            job = self.store.next()
            if job is None:
                return None
            identity = job["job_id"]
            publisher = None
            try:
                credentials = Secrets(self.settings).open(self.store.credentials(identity))
                if asyncio.run(CvatSessionAuthorizer(self.settings).identity(credentials["sessionid"])) != job["owner"]:
                    raise ValueError("Queued CVAT session identity changed")
                if job.get("workflow") == "lidar3d":
                    from src.services.vm.lidar3d import Lidar3DWorkflow

                    return Lidar3DWorkflow(self.settings, self.store).run(job, credentials)
                self.store.update(identity, status="running", stage="privacy")
                receipt = self.privacy(job)
                if self.store.get(identity)["status"] == "cancelled":
                    raise ValueError("VM job cancelled")
                _task, cvat_job = asyncio.run(self.bind(job, receipt, credentials))
                read: Any = CvatSessionReadClient(self.settings, credentials["sessionid"])
                meta = asyncio.run(read.get_job_data_meta(cvat_job))
                if meta["size"] != receipt["total_frames"]:
                    raise ValueError("CVAT video coverage differs from privacy output")
                scope = asyncio.run(read.get_job(cvat_job))
                if (
                    scope.get("start_frame", 0) != 0
                    or scope.get("stop_frame") != receipt["total_frames"] - 1
                    or meta.get("deleted_frames")
                ):
                    raise ValueError("CVAT redacted frame mapping was modified")
                job = self.store.update(
                    identity,
                    total_frames=receipt["total_frames"],
                    privacy_frames=receipt["total_frames"],
                    redacted_sha256=receipt["output_sha256"],
                    source_fps=float(Fraction(receipt["fps"])),
                    stage="sam_starting",
                )
                job["owner"] = self.store.get(identity)["owner"]
                read = VerifiedFrameReader(self.store, job, read)
                labels = asyncio.run(read.get_job_labels(cvat_job))
                publisher = IncrementalPublisher(self.settings, self.store, job, labels, credentials)
                asyncio.run(
                    publisher.send({"operation_id": "init:" + identity, "operations": [], "expected_revision": 0})
                )
                self.store.update(identity, incremental_ready=True)
                pending = self.store.directory(identity) / "private" / "delivery.json"
                if pending.exists():
                    batch = json.loads(pending.read_text())
                    asyncio.run(publisher.send(batch))
                    publisher.confirm(batch)
                    pending.unlink()
                StreamingEngine(self.store, self.settings, read, publisher).run(job)
                publisher.flush()
                self.store.update(identity, status="completed", stage="awaiting_human_review")
                if job.get("vlm_enabled"):
                    self.vlm(job, read)
            except Exception as error:
                current = self.store.get(identity)
                atomic_json(
                    self.store.directory(identity) / "private" / "failure.json",
                    {"error_class": type(error).__name__, "stage": current["stage"], "time": time.time()},
                )
                if current["status"] != "cancelled":
                    # Only allowlisted messages are public; paths/credentials/provider payloads stay private.
                    message = (
                        str(error)
                        if isinstance(error, (ValueError, TimeoutError))
                        else "VM processing failed; inspect private worker log"
                    )
                    self.store.update(
                        identity,
                        status="failed",
                        stage="failed",
                        error=message[:300],
                        failed_frames=current.get("active_window_frames", [])
                        if current["stage"] == "sam_streaming"
                        else [],
                    )
            finally:
                if publisher is not None:
                    publisher.close()
            return self.store.get(identity)

    def vlm(self, job: dict, read: Any) -> None:
        """The upload Run intent authorizes one bounded audit; never opening the panel."""
        identity = job["job_id"]
        directory = self.store.directory(identity)
        journal = directory / "private" / "vlm-audit.json"
        count = min(20, job["total_frames"])
        frames = sorted({round(i * (job["total_frames"] - 1) / max(1, count - 1)) for i in range(count)})
        state: dict = json.loads(journal.read_text()) if journal.exists() else {"frames": frames, "results": {}}

        async def run() -> None:
            auditor = VlmProfiles(self.store, self.settings).auditor(job["vlm_snapshot"])
            service = SimpleNamespace(
                box_rescue_enabled=True,
                store=SimpleNamespace(get_video_job=lambda _id: SimpleNamespace(cvat_job_id=job["cvat_job_id"])),
                workspace=SimpleNamespace(job_dir=lambda _id: directory),
                seed_reader=SimpleNamespace(client=read),
            )
            for frame in state["frames"]:
                if str(frame) in state["results"]:
                    continue
                if self.store.get(identity)["status"] == "cancelled":
                    raise ValueError("VM job cancelled")
                verified_media(self.store, job)
                result = await propose_frame(service, self.settings, identity, frame, auditor=auditor)
                state["results"][str(frame)] = result["result_id"]
                atomic_json(journal, state)
                self.store.update(identity, vlm_status="running", vlm_frames=len(state["results"]))

        try:
            self.store.update(identity, vlm_status="running")
            asyncio.run(run())
            self.store.update(identity, vlm_status="completed")
        except Exception:
            if self.store.get(identity)["status"] != "cancelled":
                self.store.update(
                    identity, vlm_status="failed", vlm_error="VLM audit failed; SAM results remain editable"
                )
