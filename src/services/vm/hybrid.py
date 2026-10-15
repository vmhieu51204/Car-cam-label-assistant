"""Detector journals and stable ByteTrack IDs, independent of SAM's object slots.

Only this coordinator is CPU-testable. Real detector/SAM adapters live in the
separate CUDA process, and never substitute detector boxes for missing masks.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from src.services.vm.hybrid_contracts import HybridProfile
from src.services.vm.storage import atomic_json, encode


def valid_box(box: list, width: int, height: int) -> list[float]:
    if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
        raise ValueError("Non-finite detector box")
    x1, y1, x2, y2 = [float(v) for v in box]
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("Detector/prompt coordinates outside source image")
    return [x1, y1, x2, y2]


class TrackerPool:
    """One explicit ByteTrack per class; class and scene namespace every ID."""

    def __init__(self, profile: HybridProfile, factory: Callable[[], Any], boxes: Callable):
        self.profile, self.factory, self.boxes = profile, factory, boxes
        self.scene = -1
        self.trackers: dict[str, Any] = {}

    def update(self, frame: dict, detections: list[dict]) -> list[dict]:
        if self.scene < 0 or frame["scene_cut_before"]:
            self.scene += 1
            self.trackers = {c: self.factory() for c in self.profile.class_mapping}
        result = []
        for label, detector_class in self.profile.class_mapping.items():
            selected = [d for d in detections if d["class_id"] == detector_class]
            array = np.asarray([d["box"] + [d["score"], detector_class] for d in selected], dtype=np.float32)
            # Empty frames advance ByteTrack too; they are not skipped observations.
            output = self.trackers[label].update(self.boxes(array.reshape(-1, 6), (frame["height"], frame["width"])))
            seen = set()
            for row in output:
                if len(row) != 8 or not np.isfinite(row).all():
                    raise ValueError("Pinned ByteTrack output signature differs")
                track, index = int(row[4]), int(row[7])
                if row[4] != track or row[7] != index or index not in range(len(selected)) or track in seen:
                    raise ValueError("Invalid ByteTrack detection/ID mapping")
                seen.add(track)
                if int(row[6]) != detector_class:
                    raise ValueError("ByteTrack changed detector class")
                # Kalman boxes can slightly leave the image; explicit clipping is
                # for prompt geometry only. The original detector box is retained.
                prompt = np.clip(row[:4], [0, 0, 0, 0], [frame["width"], frame["height"]] * 2).tolist()
                result.append(
                    {
                        "track_id": f"byte:{self.scene}:{detector_class}:{track}",
                        "label": label,
                        "prompt_bbox": valid_box(prompt, frame["width"], frame["height"]),
                        "detector_bbox": selected[index]["box"],
                        "score": selected[index]["score"],
                        "detector_source_frame": frame["frame_index"],
                    }
                )
        return result


class IndexedBoxes:
    """Carry original detection indices through ByteTrack's confidence slicing."""

    def __init__(self, boxes: Any, indices: np.ndarray | None = None):
        self.boxes = boxes
        self.indices = np.arange(len(boxes)) if indices is None else indices

    def __len__(self) -> int:
        return len(self.boxes)

    def __getitem__(self, index: Any) -> IndexedBoxes:
        return IndexedBoxes(self.boxes[index], self.indices[index])

    @property
    def conf(self) -> Any:
        return self.boxes.conf

    @property
    def cls(self) -> Any:
        return self.boxes.cls

    @property
    def xywh(self) -> Any:
        return self.boxes.xywh


def indexed_tracker(base: type) -> type:
    """Narrow adapter around the real pinned BYTETracker, preserving its logic."""

    class IndexedByteTracker(base):
        def init_track(self, results: IndexedBoxes, img: Any = None) -> list:
            tracks = super().init_track(results, img)
            for track, index in zip(tracks, results.indices, strict=True):
                track.idx = int(index)
            return tracks

    return IndexedByteTracker


class DetectionJournal:
    """Write-before-SAM journal; retry replays CPU tracking, not GPU detection.

    Journals are chained and immutable. Overlap reuses existing results without
    advancing the tracker twice. A missing/corrupt prefix fails closed.
    """

    def __init__(self, directory: Path, profile: HybridProfile, pool: TrackerPool, identity: str = ""):
        self.directory, self.profile, self.pool = directory, profile, pool
        self.directory.mkdir(parents=True, exist_ok=True)
        self.fingerprint = hashlib.sha256(
            encode({"profile": profile.model_dump(mode="json"), "identity": identity})
        ).hexdigest()
        self.next_frame = 0
        self.chain = "0" * 64
        self.replay_ms = 0.0
        names = sorted(directory.glob("*.json"))
        if [p.stem for p in names] != [f"{i:06d}" for i in range(len(names))]:
            raise ValueError("Hybrid detector journal prefix is incomplete")

    def read(self, frame: int) -> dict:
        value = json.loads((self.directory / f"{frame:06d}.json").read_text())
        if value["sha256"] != hashlib.sha256(encode(value["payload"])).hexdigest():
            raise ValueError("Hybrid detector journal checksum differs")
        payload = value["payload"]
        if payload["profile"] != self.fingerprint or payload["frame"]["frame_index"] != frame:
            raise ValueError("Hybrid journal mode/profile/frame identity differs")
        return payload

    def replay_to(self, frame: int) -> None:
        while self.next_frame < frame:
            payload = self.read(self.next_frame)
            self._replay(payload)

    def _replay(self, payload: dict) -> None:
        started = time.perf_counter()
        if payload["previous_sha256"] != self.chain:
            raise ValueError("Hybrid detector journal chain differs")
        tracked = self.pool.update(payload["frame"], payload["detections"])
        if encode(tracked) != encode(payload["tracks"]):
            raise ValueError("ByteTrack replay differs; original IDs must be preserved")
        self.chain = hashlib.sha256(encode(payload)).hexdigest()
        self.next_frame += 1
        self.replay_ms += (time.perf_counter() - started) * 1000

    def detect(self, frame: dict, detector: Callable[[dict], list[dict]], cap: int) -> dict:
        number = frame["frame_index"]
        replay_before = self.replay_ms
        self.replay_to(number)
        path = self.directory / f"{number:06d}.json"
        if path.exists():
            payload = self.read(number)
            if payload["frame"] != frame:
                raise ValueError("Hybrid detector input hash/mapping changed")
            if number == self.next_frame:
                self._replay(payload)
            return {**payload, "reused": True, "replay_ms": self.replay_ms - replay_before}
        if number != self.next_frame:
            raise ValueError("Hybrid detector must receive every frame in order")
        started = time.perf_counter()
        detections = detector(frame)
        detection_ms = (time.perf_counter() - started) * 1000
        if len(detections) > cap:
            raise ValueError("Hybrid object cap exceeded; no detections truncated")
        for d in detections:
            valid_box(d["box"], frame["width"], frame["height"])
            if d["class_id"] not in self.profile.class_mapping.values() or not 0 <= d["score"] <= 1:
                raise ValueError("YOLO class or score differs from frozen profile")
        started = time.perf_counter()
        tracks = self.pool.update(frame, detections)
        payload = {
            "profile": self.fingerprint,
            "frame": frame,
            "detections": detections,
            "tracks": tracks,
            "previous_sha256": self.chain,
            "detector_ms": detection_ms,
            "tracker_ms": (time.perf_counter() - started) * 1000,
        }
        self.chain = hashlib.sha256(encode(payload)).hexdigest()
        atomic_json(path, {"payload": payload, "sha256": self.chain})
        self.next_frame += 1
        return {**payload, "reused": False, "replay_ms": self.replay_ms - replay_before}


def prompt_plan(records: list[dict], detections: list[dict], cap: int, stride: int, lost_frames: int) -> dict:
    """All new object slots are seeded before SAM propagation starts."""
    identities: dict[str, dict] = {}
    for local, payload in enumerate(detections):
        for row in payload["tracks"]:
            track = identities.setdefault(
                row["track_id"],
                {
                    "row": row,
                    "first": local,
                    "last": local,
                    "anchors": [],
                    "last_anchor": -stride,
                },
            )
            track["last"] = local
            if local - track["last_anchor"] >= stride:
                track["anchors"].append((local, row["prompt_bbox"]))
                track["last_anchor"] = local
    if len(identities) > cap:
        raise ValueError("Hybrid window object cap exceeded; split/reconfigure pilot, no silent drop")
    for slot, track in enumerate(identities.values(), 1):
        track["slot"] = slot
        track["end"] = min(len(records) - 1, track["last"] + lost_frames)
    return identities
