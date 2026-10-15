"""Window engine, durable frame events and conservative boundary IDs."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import queue
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.config import PROJECT_ROOT, Settings
from src.services.video.stitching import ChunkObservation, ChunkTrack, MaskRLE, decode_mask, stitch_tracks
from src.services.vm.contracts import VmProfile, file_sha256
from src.services.vm.frames import FrameProvider, plan_window
from src.services.vm.storage import VmStore, atomic_json, encode


def code_hash() -> str:
    names = [
        "scripts/run_vm_sam.py",
        "scripts/run_vm_hybrid.py",
        "src/services/vm/hybrid.py",
        "src/services/vm/hybrid_contracts.py",
        "scripts/run_local_sam.py",
        "kaggle/sam3_benchmark/main.py",
        "src/services/vm/streaming.py",
        "src/services/vm/frames.py",
        "src/services/video/stitching.py",
    ]
    return hashlib.sha256(b"".join((PROJECT_ROOT / name).read_bytes() for name in names)).hexdigest()


def validate_event(event: dict, records: dict[int, dict], classes: list[str], profile: VmProfile | None = None) -> None:
    if event.get("type") != "frame" or event.get("class") not in classes or type(event.get("frame")) is not int:
        raise ValueError("Invalid GPU frame event")
    record = records[event["frame"]]
    if event["image_sha256"] != record["sha256"]:
        raise ValueError("GPU event input hash mismatch")
    if profile is not None and profile.pipeline != "sam" and event.get("pipeline") != profile.pipeline:
        raise ValueError("GPU hybrid mode differs from the frozen job")
    identities = set()
    for observation in event["observations"]:
        if (
            not isinstance(observation["track_id"], str)
            or not math.isfinite(observation["score"])
            or not 0 <= observation["score"] <= 1
        ):
            raise ValueError("Invalid GPU track/score")
        if observation["track_id"] in identities or observation["label"] != event["class"]:
            raise ValueError("GPU event track/class mismatch")
        identities.add(observation["track_id"])
        if profile is not None and profile.pipeline != "sam":
            from src.services.vm.hybrid import valid_box

            valid_box(observation["prompt_bbox"], record["width"], record["height"])
            if observation["detector_bbox"] is not None:
                valid_box(observation["detector_bbox"], record["width"], record["height"])
            if observation["detector_source_frame"] > event["frame"]:
                raise ValueError("Hybrid detector provenance refers to a future observation")
        mask = MaskRLE(**{**observation["mask"], "counts": tuple(observation["mask"]["counts"])})
        if (mask.width, mask.height) != (record["width"], record["height"]):
            raise ValueError("GPU mask dimensions mismatch")
        decoded = decode_mask(mask)
        from kaggle.sam3_benchmark.main import _mask_bbox

        box = _mask_bbox(decoded)
        if box is None or list(box) != list(observation["bbox"]):
            raise ValueError("GPU box does not match mask extent")


class GpuProcess:
    def __init__(self, settings: Settings, directory: Path, profile: VmProfile):
        self.profile = profile
        self.log = (directory / "private" / "sam.log").open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            [
                str(settings.local_sam_python.absolute()),
                "-m",
                "scripts.run_vm_sam" if profile.pipeline == "sam" else "scripts.run_vm_hybrid",
                "--job-root",
                str(directory.resolve()),
                "--profile",
                str((directory / "profile.json").resolve()),
            ],
            cwd=PROJECT_ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            bufsize=1,
        )
        self.events: queue.Queue[Any] = queue.Queue(maxsize=8)
        self.stopping = threading.Event()
        self.deadline = time.monotonic() + settings.local_sam_timeout_s

        def reader() -> None:
            assert self.process.stdout is not None
            for line in self.process.stdout:
                try:
                    value = json.loads(line)
                except ValueError:
                    value = {"type": "error", "error": "Invalid GPU protocol"}
                while not self.stopping.is_set():
                    try:
                        self.events.put(value, timeout=0.2)
                        break
                    except queue.Full:
                        continue
                if self.stopping.is_set():
                    return
            if not self.stopping.is_set():
                try:
                    self.events.put({"type": "error", "error": "GPU process exited"}, timeout=1)
                except queue.Full:
                    pass

        self.thread = threading.Thread(target=reader, daemon=True)
        self.thread.start()
        self.cancel_path = directory
        try:
            loaded = self.receive()
        except Exception:
            self.close()
            raise
        if (
            loaded.get("type") != "loaded"
            or loaded.get("checkpoint_sha256") != profile.checkpoint_sha256
            or loaded.get("source_commit") != profile.source_commit
            or (
                profile.hybrid is not None
                and (
                    loaded.get("pipeline") != profile.pipeline
                    or loaded.get("detector_sha256") != profile.hybrid.checkpoint_sha256
                    or loaded.get("tracker_sha256") != profile.hybrid.tracker_sha256
                )
            )
        ):
            self.close()
            raise ValueError("GPU model preflight failed; see private SAM log")

    def receive(self) -> dict:
        while time.monotonic() < self.deadline:
            if (self.cancel_path / "private" / "cancelled.json").exists():
                raise ValueError("VM job cancelled")
            try:
                return self.events.get(timeout=1)
            except queue.Empty:
                if self.process.poll() is not None:
                    break
        raise TimeoutError("SAM process exited or job deadline exceeded")

    def run(self, command: dict) -> Iterator[dict]:
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(command) + "\n")
        self.process.stdin.flush()
        while True:
            event = self.receive()
            if event.get("type") == "error":
                if event.get("error") == "hybrid_object_cap":
                    raise ValueError("Hybrid object cap exceeded; results retained, no objects silently dropped")
                raise ValueError("SAM failed; verified completed windows remain reusable")
            yield event
            if event.get("type") == "complete":
                break

    def close(self) -> None:
        self.stopping.set()
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.process.stdin:
            self.process.stdin.close()
        self.thread.join(timeout=2)
        if self.process.stdout:
            self.process.stdout.close()
        self.log.close()


def chunks(events: list[dict], window: int, label: str) -> list[ChunkTrack]:
    tracks: dict[str, list[ChunkObservation]] = {}
    for event in events:
        if event["class"] != label:
            continue
        for row in event["observations"]:
            mask = MaskRLE(**{**row["mask"], "counts": tuple(row["mask"]["counts"])})
            tracks.setdefault(row["track_id"], []).append(
                ChunkObservation(event["frame"], tuple(row["bbox"]), mask, row["score"])
            )
    return [
        ChunkTrack(window, identity, label, sorted(rows, key=lambda r: r.frame)) for identity, rows in tracks.items()
    ]


def boundary_mapping(
    previous: list[dict], current: list[dict], label: str, window: int, old_mapping: dict[str, str], reset: bool
) -> tuple[dict, list, list]:
    previous_tracks = [] if reset else chunks(previous, 0, label)
    current_tracks = chunks(current, 1, label)
    result = stitch_tracks(previous_tracks + current_tracks, planned_window_ids=[0, 1])
    mapping = {}
    logs = result.boundary_logs[-1]["tracks"] if result.boundary_logs else []
    for row in logs:
        old = row.get("selected_previous_local_id")
        if row["result"] == "matched" and old in old_mapping:
            mapping[row["local_track_id"]] = old_mapping[old]
    for track in current_tracks:
        mapping.setdefault(track.local_track_id, hashlib.sha256(track.local_track_id.encode()).hexdigest()[:24])
    return mapping, [asdict(item) for item in result.decisions], logs


class StreamingEngine:
    def __init__(
        self,
        store: VmStore,
        settings: Settings,
        source: Any,
        publish: Callable[[dict, dict[str, str]], None],
        gpu_factory: Any = GpuProcess,
    ):
        self.store, self.settings, self.source, self.publish, self.gpu_factory = (
            store,
            settings,
            source,
            publish,
            gpu_factory,
        )

    def run(self, job: dict) -> None:
        job_id = job["job_id"]
        directory = self.store.directory(job_id)
        profile = VmProfile.model_validate(job["profile"])
        atomic_json(directory / "profile.json", profile.model_dump(mode="json", by_alias=True))
        fingerprint = code_hash()
        if job.get("code_sha256") not in (None, fingerprint):
            raise ValueError("Streaming code identity changed; resume with the original deployment")
        self.store.update(job_id, code_sha256=fingerprint)
        provider = FrameProvider(
            self.store, job_id, self.source, job["cvat_job_id"], self.settings.local_frame_cache_max_bytes
        )
        total, classes = job["total_frames"], job["classes"]
        start, window, gpu = 0, 0, None
        previous: list[dict] = []
        mapping: dict[str, str] = {}
        processed: set[int] = set()
        complete: set[int] = set()
        class_frames: dict[str, set[int]] = {c: set() for c in classes}
        mask_absences: set[tuple[int, str, str]] = set()
        attempt = uuid4().hex
        executor = ThreadPoolExecutor(max_workers=1)
        future = None
        last_progress = 0.0
        try:
            while start < total:
                indexes = list(range(start, min(total, start + profile.window + 1)))
                self.store.update(job_id, stage="sam_streaming", current_window=window, active_window_frames=indexes)
                records = future.result() if future else asyncio.run(provider.acquire(indexes))
                selected, next_start = plan_window(records, total, profile.window, profile.overlap)
                self.store.update(job_id, active_window_frames=[r["frame_index"] for r in selected])
                identity = hashlib.sha256(
                    encode(
                        {
                            "frames": selected,
                            "profile": job["profile"],
                            "source": job["redacted_sha256"],
                            "code": fingerprint,
                        }
                    )
                ).hexdigest()
                by_frame = {r["frame_index"]: r for r in selected}
                checkpoint = directory / "checkpoints" / f"{window:06d}.json"
                partial_path = directory / "checkpoints" / f"{window:06d}-partial.json"
                partial_events = []
                partial_mapping = {}
                if not checkpoint.exists() and partial_path.exists():
                    partial = json.loads(partial_path.read_text())
                    if partial["identity"] != identity:
                        raise ValueError("Partial window identity changed")
                    partial_mapping = partial["mapping"]
                    for entry in partial["events"]:
                        path = directory / entry["path"]
                        if file_sha256(path) != entry["sha256"]:
                            raise ValueError("Corrupt partial frame event")
                        value = json.loads(path.read_text())
                        validate_event(value, by_frame, classes, profile)
                        previous_end = max((e["frame"] for e in previous), default=-1)
                        if value["frame"] > previous_end or not previous or selected[0]["scene_cut_before"]:
                            partial_events.append(value)
                frozen = {(e["frame"], e["class"]): e for e in partial_events}
                saved = None
                if checkpoint.exists():
                    saved = json.loads(checkpoint.read_text())
                    if saved["identity"] != identity:
                        raise ValueError("Streaming checkpoint identity differs")
                    events = []
                    for entry in saved["events"]:
                        path = directory / entry["path"]
                        if file_sha256(path) != entry["sha256"]:
                            raise ValueError("Corrupt frame checkpoint")
                        event = json.loads(path.read_text())
                        validate_event(event, by_frame, classes, profile)
                        events.append(event)
                    incoming = iter(events + [saved["completion"]])
                else:
                    if gpu is None:
                        gpu = self.gpu_factory(self.settings, directory, profile)
                    incoming = gpu.run(
                        {
                            "type": "window",
                            "window": window,
                            "frames": selected,
                            "classes": classes,
                            "identity": identity,
                            "redacted_sha256": job["redacted_sha256"],
                        }
                    )
                future = None
                if next_start is not None:
                    future_indexes = list(range(next_start, min(total, next_start + profile.window + 1)))
                    future = executor.submit(asyncio.run, provider.acquire(future_indexes))
                events = []
                live_events = []
                entries: list[dict] = []
                current_mapping: dict[str, str] = {}
                seen: set[tuple[int, str]] = set()
                linked_classes = set()
                reset = selected[0]["scene_cut_before"]
                overlap_end = max((e["frame"] for e in previous), default=-1)
                for event in incoming:
                    if self.store.get(job_id)["status"] == "cancelled":
                        raise ValueError("Job cancelled")
                    if event["type"] == "complete":
                        expected = {(f, c) for f in by_frame for c in classes}
                        if seen != expected or event["identity"] != identity or event["frames"] != list(by_frame):
                            raise ValueError("Incomplete window coverage")
                        if saved is None:
                            atomic_json(
                                checkpoint,
                                {
                                    "identity": identity,
                                    "events": entries,
                                    "completion": event,
                                    "mapping": current_mapping,
                                },
                            )
                        break
                    validate_event(event, by_frame, classes, profile)
                    if event.get("window") != window:
                        raise ValueError("GPU event window mapping differs")
                    key = (event["frame"], event["class"])
                    if key in seen:
                        raise ValueError("Duplicate class/frame result")
                    seen.add(key)
                    live_events.append(event)
                    label = event["class"]
                    if key in frozen:
                        event = frozen[key]
                        current_mapping.update(partial_mapping)
                    elif saved is None and (event["frame"] > overlap_end or not previous or reset):
                        if profile.pipeline != "sam":
                            # ByteTrack owns identity across windows. SAM instance
                            # slots are local and never fed into overlap stitching.
                            for row in event["observations"]:
                                current_mapping[row["track_id"]] = hashlib.sha256(row["track_id"].encode()).hexdigest()[
                                    :24
                                ]
                        elif label not in linked_classes:
                            anchors = [e for e in partial_events if e["class"] == label]
                            links, decisions, logs = boundary_mapping(
                                anchors or previous,
                                live_events,
                                label,
                                window,
                                partial_mapping if anchors else mapping,
                                False if anchors else reset,
                            )
                            current_mapping.update(links)
                            # An unproven retry identity cannot collide with a previously edited/deleted track.
                            if anchors:
                                matched = {r["local_track_id"] for r in logs if r["result"] == "matched"}
                                for raw in links:
                                    if raw not in matched:
                                        current_mapping[raw] = hashlib.sha256((attempt + raw).encode()).hexdigest()[:24]
                            atomic_json(
                                directory / "checkpoints" / f"{window:06d}-{classes.index(label)}-boundary.json",
                                {"decisions": decisions, "logs": logs, "partial_retry": bool(anchors)},
                            )
                            linked_classes.add(label)
                        normalized = []
                        for row in event["observations"]:
                            raw = row["track_id"]
                            current_mapping.setdefault(
                                raw,
                                hashlib.sha256(((attempt if partial_events else "") + raw).encode()).hexdigest()[:24],
                            )
                            stable = current_mapping[raw]
                            normalized.append({**row, "local_track_id": raw, "track_id": "global:" + stable})
                            current_mapping["global:" + stable] = stable
                        event = {**event, "observations": normalized}
                    elif saved:
                        current_mapping.update(saved["mapping"])
                    relative = (
                        f"events/{attempt}-{window:06d}-{event['frame']:06d}-{classes.index(event['class'])}.json"
                    )
                    digest = atomic_json(directory / relative, event)
                    entries.append({"path": relative, "sha256": digest})
                    events.append(event)
                    for absence in event.get("mask_absences", []):
                        mask_absences.add((event["frame"], event["class"], absence["track_id"]))
                    if saved is None:
                        atomic_json(
                            partial_path,
                            {
                                "identity": identity,
                                "events": entries,
                                "mapping": current_mapping,
                                "native_state_resume": False,
                                "attempt": attempt,
                            },
                        )
                    self.store.event(
                        job_id,
                        f"result:{event['frame']}:{event['class']}",
                        {"frame": event["frame"], "class": event["class"]},
                    )
                    processed.add(event["frame"])
                    class_frames[event["class"]].add(event["frame"])
                    if all(event["frame"] in class_frames[c] for c in classes):
                        complete.add(event["frame"])
                    if event["frame"] > overlap_end or not previous or reset:
                        for row in event["observations"]:
                            current_mapping.setdefault(
                                row["track_id"], hashlib.sha256(row["track_id"].encode()).hexdigest()[:24]
                            )
                        self.publish(event, current_mapping)
                    if time.monotonic() - last_progress >= 1:
                        self.store.update(
                            job_id,
                            status="running",
                            stage="sam_streaming",
                            current_window=window,
                            delivered_frames=min(total, start + len(records)),
                            processed_frames=len(processed),
                            complete_frames=len(complete),
                            class_coverage={c: len(v) for c, v in class_frames.items()},
                            cache_bytes=provider.size,
                            mask_absence_count=len(mask_absences),
                            cap_warning=any(len(e["observations"]) >= profile.max_objects for e in events)
                            if profile.variant == "sam3.1"
                            else False,
                        )
                        last_progress = time.monotonic()
                else:
                    raise ValueError("Window returned no completion checkpoint")
                if saved:
                    current_mapping = saved["mapping"]
                previous, mapping = events, current_mapping
                provider.release(indexes)
                window += 1
                self.store.update(job_id, completed_windows=window)
                if next_start is None:
                    break
                start = next_start
            if len(complete) != total:
                raise ValueError("Job coverage incomplete")
            self.store.update(
                job_id,
                processed_frames=len(processed),
                complete_frames=len(complete),
                class_coverage={c: len(v) for c, v in class_frames.items()},
                cache_bytes=provider.size,
                mask_absence_count=len(mask_absences),
            )
            atomic_json(
                directory / "public" / "input-manifest.json",
                {
                    "schema": "autolabel-redacted-input/v1",
                    "redacted_sha256": job["redacted_sha256"],
                    "frames": [provider.record(i) for i in range(total)],
                },
            )
            atomic_json(
                directory / "public" / "coverage.json",
                {
                    "schema": "autolabel-stream-coverage/v1",
                    "total": total,
                    "class_coverage": {c: len(v) for c, v in class_frames.items()},
                    "code_sha256": fingerprint,
                    "profile": job["profile"],
                    "completed_windows": window,
                    "complete": True,
                },
            )
        finally:
            provider.close()
            executor.shutdown(wait=True, cancel_futures=True)
            if gpu is not None:
                gpu.close()
