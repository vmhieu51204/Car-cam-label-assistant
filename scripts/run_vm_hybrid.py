"""Local-only YOLO26s/explicit ByteTrack/SAM instance CUDA worker.

Uses the existing redacted frame and durable per-window event protocol. No
weights download, runtime dependency install, text-prompt ID injection or CPU
inference fallback. SAM3.1 remains available through the baseline worker.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import inspect
import json
import os
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

from kaggle.runtime_probe import sam3_commit
from kaggle.sam3_benchmark.main import _mask_bbox
from scripts.run_vm_sam import emit
from src.services.video.stitching import encode_mask
from src.services.vm.contracts import VmProfile, file_sha256
from src.services.vm.hybrid import DetectionJournal, IndexedBoxes, TrackerPool, indexed_tracker, prompt_plan, valid_box
from src.services.vm.storage import encode


def array(value: Any) -> np.ndarray:
    return np.asarray(value.detach().cpu().numpy() if hasattr(value, "detach") else value)


def observation(track: dict, mask: Any, frame: dict, quality: float | None = None) -> dict | None:
    pixels = array(mask).reshape(frame["height"], frame["width"]) > 0
    box = _mask_bbox(pixels)
    if box is None:
        return None  # Explicit absence evidence is recorded by the caller.
    return {
        **track,
        "bbox": list(box),
        "mask": asdict(encode_mask(pixels)),
        "score_source": "detector",
        "sam_quality": quality,
    }


def relative_box(box: list, frame: dict) -> np.ndarray:
    # The official tracker multiplies relative coordinates by image_size. Its
    # normalize_coords argument is unused; never pass raw source pixels there.
    return np.asarray(valid_box(box, frame["width"], frame["height"]), dtype=np.float32) / np.array(
        [frame["width"], frame["height"], frame["width"], frame["height"]], dtype=np.float32
    )


def validate_detector_metadata(metadata: dict, task: str) -> None:
    filename = str(metadata.get("yaml_file", "")).replace("\\", "/").rsplit("/", 1)[-1]
    if task != "detect" or metadata.get("scale") != "s" or filename not in {"yolo26.yaml", "yolo26s.yaml"}:
        raise ValueError("Checkpoint architecture is not the pinned YOLO26s detector")


class HybridBackend:
    def __init__(self, root: Path, profile: VmProfile):
        import torch

        hybrid = profile.hybrid
        if hybrid is None or profile.pipeline == "sam" or profile.variant != "sam3":
            raise ValueError("Hybrid requires an explicit SAM3 instance profile")
        os.environ["YOLO_AUTOINSTALL"] = "false"
        (root / "private/yolo-settings/Ultralytics").mkdir(parents=True, exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = str(root / "private/yolo-settings")
        if importlib.metadata.version("ultralytics") != hybrid.ultralytics_version:
            raise ValueError("Ultralytics dependency pin differs")
        if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).major < 8:
            raise ValueError("Hybrid pilot requires Ampere or newer CUDA; no CPU fallback")
        if sam3_commit() != profile.source_commit:
            raise ValueError("Installed SAM source commit differs")
        for checkpoint, digest in (
            (profile.checkpoint, profile.checkpoint_sha256),
            (hybrid.checkpoint, hybrid.checkpoint_sha256),
        ):
            if not Path(checkpoint).is_file() or file_sha256(Path(checkpoint)) != digest:
                raise ValueError("Hybrid checkpoint SHA-256 differs")
        from sam3.model_builder import build_sam3_image_model, build_sam3_video_model
        from ultralytics import YOLO, settings
        from ultralytics.engine.results import Boxes
        from ultralytics.trackers.byte_tracker import BYTETracker

        self.root, self.profile, self.hybrid, self.torch = root, profile, hybrid, torch
        receipt = json.loads((root / "public/redacted.privacy.json").read_text())
        self.redacted_sha256 = receipt.get("output_sha256")
        if receipt.get("status") != "complete" or file_sha256(root / "public/redacted.mp4") != self.redacted_sha256:
            raise ValueError("Hybrid requires a completed verified redacted video")
        settings.update({"sync": False})
        torch.cuda.set_per_process_memory_fraction(profile.vram_fraction)
        torch.backends.cuda.matmul.allow_tf32 = profile.tf32
        started = time.perf_counter()
        self.detector = YOLO(hybrid.checkpoint, task="detect")
        validate_detector_metadata(self.detector.model.yaml, self.detector.task)
        if {int(k): v for k, v in self.detector.names.items()} != hybrid.detector_names:
            raise ValueError("Checkpoint detector classes differ from the frozen manifest")
        tracker_type = indexed_tracker(BYTETracker)
        self.pool = TrackerPool(
            hybrid,
            lambda: tracker_type(SimpleNamespace(**hybrid.tracker_config)),
            lambda data, shape: IndexedBoxes(Boxes(data, shape)),
        )
        self.journal: DetectionJournal | None = None
        self.absences: dict[int, list[dict]] = {}
        if profile.pipeline == "hybrid_frame":
            from sam3.model.sam3_image_processor import Sam3Processor

            if "enable_inst_interactivity" not in inspect.signature(build_sam3_image_model).parameters:
                raise ValueError("SAM source lacks the pinned instance image API")
            self.model = build_sam3_image_model(
                checkpoint_path=profile.checkpoint,
                load_from_HF=False,
                enable_inst_interactivity=True,
                compile=False,
                device="cuda",
            )
            if not hasattr(self.model, "predict_inst_batch"):
                raise ValueError("SAM instance batch API unavailable")
            self.processor = Sam3Processor(self.model)
        else:
            self.model = build_sam3_video_model(
                checkpoint_path=profile.checkpoint, load_from_HF=False, compile=False, device="cuda"
            )
            self.tracker = self.model.tracker
            self.tracker.backbone = self.model.detector.backbone
            if not {"box", "rel_coordinates", "normalize_coords"} <= set(
                inspect.signature(self.tracker.add_new_points_or_box).parameters
            ):
                raise ValueError("SAM source cannot honor pixel-coordinate instance prompts")
            if "propagate_preflight" not in inspect.signature(self.tracker.propagate_in_video).parameters:
                raise ValueError("SAM instance video propagation API differs")
        torch.cuda.synchronize()
        self.load_ms = (time.perf_counter() - started) * 1000

    def image_path(self, frame: dict) -> Path:
        path = (self.root / frame["relative_path"]).resolve()
        cache = (self.root / "cache").resolve()
        if path.parent != cache or file_sha256(path) != frame["sha256"]:
            raise ValueError("Hybrid accepts only the verified redacted JPEG cache")
        with Image.open(path) as image:
            if image.size != (frame["width"], frame["height"]):
                raise ValueError("Hybrid frame geometry differs")
        return path

    def detect(self, frame: dict) -> list[dict]:
        result = self.detector.predict(
            source=str(self.image_path(frame)),
            classes=list(self.hybrid.class_mapping.values()),
            conf=self.hybrid.detector_conf,
            iou=self.hybrid.detector_iou,
            imgsz=self.hybrid.image_size,
            device=0,
            half=True,
            max_det=self.profile.max_objects + 1,
            save=False,
            verbose=False,
        )[0]
        return [
            {"box": row[:4].tolist(), "score": float(row[4]), "class_id": int(row[5])}
            for row in result.boxes.cpu().numpy().data
        ]

    def masks_by_frame(self, frames: list[dict], detections: list[dict]) -> Iterator[tuple[dict, list[dict], dict]]:
        for frame, payload in zip(frames, detections, strict=True):
            tracks = payload["tracks"]
            rows: list[dict] = []
            started = time.perf_counter()
            embedding_ms = 0.0
            if tracks:
                with Image.open(self.image_path(frame)) as source:
                    state = self.processor.set_image_batch([source.convert("RGB")])
                self.torch.cuda.synchronize()
                embedding_ms = (time.perf_counter() - started) * 1000
                for start in range(0, len(tracks), self.hybrid.object_batch):
                    batch = tracks[start : start + self.hybrid.object_batch]
                    boxes = np.asarray([row["prompt_bbox"] for row in batch], dtype=np.float32)
                    masks, quality, _ = self.model.predict_inst_batch(
                        state,
                        None,
                        None,
                        box_batch=[boxes],
                        multimask_output=False,
                    )
                    if len(masks) != 1 or len(masks[0]) != len(batch) or len(quality[0]) != len(batch):
                        raise ValueError("SAM instance batch mapping differs")
                    for track, mask, score in zip(batch, masks[0], quality[0], strict=True):
                        row = observation(track, mask, frame, float(np.asarray(score).reshape(-1)[0]))
                        if row is not None:
                            rows.append(row)
                        else:
                            self.absences.setdefault(frame["frame_index"], []).append(
                                {
                                    "track_id": track["track_id"],
                                    "label": track["label"],
                                    "reason": "empty_sam_mask",
                                }
                            )
                del state
            self.torch.cuda.synchronize()
            yield frame, rows, {"sam_ms": (time.perf_counter() - started) * 1000, "embedding_ms": embedding_ms}

    def masks_propagated(self, frames: list[dict], detections: list[dict]) -> Iterator[tuple[dict, list[dict], dict]]:
        plan = prompt_plan(
            frames, detections, self.profile.max_objects, self.hybrid.prompt_stride, self.hybrid.lost_mask_frames
        )
        if not plan:
            for frame in frames:
                yield frame, [], {"sam_ms": 0.0}
            return
        with tempfile.TemporaryDirectory(prefix="hybrid-window-", dir=self.root / "private") as temporary:
            stage = Path(temporary)
            for i, frame in enumerate(frames):
                os.link(self.image_path(frame), stage / f"{i:06d}.jpg")
            state = self.tracker.init_state(
                video_path=str(stage), offload_video_to_cpu=True, offload_state_to_cpu=True, async_loading_frames=False
            )
            started = time.perf_counter()
            # Late-arriving IDs must all be registered before propagation, as
            # required by the official instance API. Slot IDs never become track IDs.
            for track in plan.values():
                for local, box in track["anchors"]:
                    self.tracker.add_new_points_or_box(
                        state,
                        frame_idx=local,
                        obj_id=track["slot"],
                        box=relative_box(box, frames[local]),
                        rel_coordinates=True,
                    )
            slots = {track["slot"]: track for track in plan.values()}
            iterator = self.tracker.propagate_in_video(
                state,
                start_frame_idx=0,
                max_frame_num_to_track=len(frames) - 1,
                reverse=False,
                propagate_preflight=True,
            )
            seen: set[int] = set()
            try:
                for value in iterator:
                    if len(value) != 5:
                        raise ValueError("SAM instance propagation return signature differs")
                    local, ids, _low, masks, _presence = value
                    if local != len(seen) or local not in range(len(frames)) or len(ids) != len(slots):
                        raise ValueError("SAM instance frame/object coverage differs")
                    seen.add(local)
                    mask_arrays = array(masks)
                    rows = []
                    for slot, mask in zip(ids, mask_arrays, strict=True):
                        track = slots[int(slot)]
                        if track["first"] <= local <= track["end"]:
                            # No output before first ByteTrack observation; mask
                            # extension after disappearance is explicitly bounded.
                            nearest = next(
                                (r for r in detections[local]["tracks"] if r["track_id"] == track["row"]["track_id"]),
                                None,
                            )
                            last = next(
                                r
                                for past in reversed(detections[: local + 1])
                                for r in past["tracks"]
                                if r["track_id"] == track["row"]["track_id"]
                            )
                            if (
                                frames[local]["frame_index"] - last["detector_source_frame"]
                                > self.hybrid.lost_mask_frames
                            ):
                                self.absences.setdefault(frames[local]["frame_index"], []).append(
                                    {
                                        "track_id": last["track_id"],
                                        "label": last["label"],
                                        "reason": "detector_gap_limit",
                                    }
                                )
                                continue
                            row = observation(nearest or last, mask, frames[local])
                            if row is None:
                                self.absences.setdefault(frames[local]["frame_index"], []).append(
                                    {
                                        "track_id": last["track_id"],
                                        "label": last["label"],
                                        "reason": "empty_sam_mask",
                                    }
                                )
                                continue
                            if nearest is None:
                                row["detector_bbox"] = None
                                row["score_source"] = "last_detector_observation"
                            rows.append(row)
                    self.torch.cuda.synchronize()
                    yield frames[local], rows, {"sam_ms": (time.perf_counter() - started) * 1000}
                    started = time.perf_counter()
            finally:
                if hasattr(iterator, "close"):
                    iterator.close()
                del state
            if len(seen) != len(frames):
                raise ValueError("SAM propagation omitted source frames")

    def run(self, command: dict, output: Callable[[dict], None]) -> None:
        window_started = time.perf_counter()
        self.absences = {}
        frames = command["frames"]
        if set(command["classes"]) != set(self.hybrid.class_mapping):
            raise ValueError("Hybrid job classes changed")
        if command["redacted_sha256"] != self.redacted_sha256 or any(f["scene_cut_before"] for f in frames[1:]):
            raise ValueError("Hybrid source identity or scene mapping changed")
        for frame in frames:
            self.image_path(frame)
        # Isolation includes mode/SAM profile AND the redacted source hash.
        identity = encode([self.profile.model_dump(mode="json"), command["redacted_sha256"]]).hex()
        if self.journal is None:
            self.journal = DetectionJournal(self.root / "private/hybrid-detections", self.hybrid, self.pool, identity)
        self.torch.cuda.reset_peak_memory_stats()
        detections = [self.journal.detect(frame, self.detect, self.profile.max_objects) for frame in frames]
        rows = self.masks_by_frame if self.profile.pipeline == "hybrid_frame" else self.masks_propagated
        for local, (frame, observations, timing) in enumerate(rows(frames, detections)):
            payload = detections[local]
            for label in command["classes"]:
                output(
                    {
                        "type": "frame",
                        "window": command["window"],
                        "frame": frame["frame_index"],
                        "class": label,
                        "image_sha256": frame["sha256"],
                        "observations": [r for r in observations if r["label"] == label],
                        "detector_boxes": [
                            d for d in payload["detections"] if d["class_id"] == self.hybrid.class_mapping[label]
                        ],
                        "tracker_ids": [t["track_id"] for t in payload["tracks"] if t["label"] == label],
                        "mask_absences": [
                            r for r in self.absences.get(frame["frame_index"], []) if r["label"] == label
                        ],
                        "pipeline": self.profile.pipeline,
                        "runtime": timing
                        | {
                            "detector_ms": 0.0 if payload["reused"] else payload["detector_ms"],
                            "tracker_ms": 0.0 if payload["reused"] else payload["tracker_ms"],
                            "tracker_replay_ms": payload["replay_ms"],
                            "detector_reused": payload["reused"],
                            "shared_across_classes": True,
                        },
                    }
                )
        output(
            {
                "type": "complete",
                "window": command["window"],
                "identity": command["identity"],
                "frames": [f["frame_index"] for f in frames],
                "profile": self.profile.model_dump(mode="json", by_alias=True),
                "runtime": {
                    "model_load_ms": self.load_ms,
                    "peak_cuda_allocated_bytes": self.torch.cuda.max_memory_allocated(),
                    "detector": self.hybrid.model_dump(mode="json"),
                    "native_state_resume": False,
                    "window_ms": (time.perf_counter() - window_started) * 1000,
                    "gpu": self.torch.cuda.get_device_name(0),
                    "rss_end_bytes": __import__("psutil").Process().memory_info().rss,
                },
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    args = parser.parse_args()
    profile = VmProfile.model_validate_json(args.profile.read_text())
    if profile.hybrid is None:
        raise ValueError("Hybrid profile missing")
    protocol_stdout = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        backend = HybridBackend(args.job_root, profile)
    emit(
        {
            "type": "loaded",
            "checkpoint_sha256": profile.checkpoint_sha256,
            "source_commit": profile.source_commit,
            "pipeline": profile.pipeline,
            "detector_sha256": profile.hybrid.checkpoint_sha256,
            "tracker_sha256": profile.hybrid.tracker_sha256,
        }
    )
    for line in sys.stdin:
        command = json.loads(line)
        if command["type"] == "close":
            break
        try:

            def send(event: dict) -> None:
                with contextlib.redirect_stdout(protocol_stdout):
                    emit(event)

            with (
                contextlib.redirect_stdout(sys.stderr),
                backend.torch.inference_mode(),
                backend.torch.autocast("cuda", dtype=backend.torch.bfloat16),
            ):
                backend.run(command, send)
        except Exception as error:
            emit({"type": "error", "error": "hybrid_object_cap" if "cap exceeded" in str(error) else "hybrid_failed"})
            raise


if __name__ == "__main__":
    main()
