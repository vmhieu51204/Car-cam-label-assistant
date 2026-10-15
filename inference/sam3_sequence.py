"""SAM3 video-text backend for a whole sequence, loaded once per process.

This module imports torch and sam3 only in ``load``. It can therefore be tested
without a GPU. The image-text/box-seed fallback is deliberately not silent:
an incompatible SAM3 checkout raises a clear error rather than returning
incomplete whole-video labels.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import math
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np

from src.services.video.stitching import ChunkObservation, ChunkTrack, encode_mask


def require_video_text_api(model: object) -> None:
    required = ("init_state", "add_prompt", "propagate_in_video")
    if not all(callable(getattr(model, name, None)) for name in required):
        raise NotImplementedError(
            "installed SAM3 has no video text-prompt API; image-text/box-seed "
            "fallback is not yet validated for full-video labeling"
        )


def _array(mask: object) -> np.ndarray:
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    result = np.asarray(mask).squeeze() > 0
    if result.ndim != 2:
        raise ValueError("SAM3 returned a mask that is not 2D")
    return result


def collect_text_window(model: object, *, directory: Path, window_id: int,
                        frames: list[int], labels: tuple[str, ...],
                        on_class_start: Callable[[str], None] | None = None,
                        on_class_end: Callable[[str, float], None] | None = None) -> list[ChunkTrack]:
    """Use a fresh SAM3 state for each class; preserve every visible overlap mask."""
    require_video_text_api(model)
    by_track: dict[tuple[str, int], list[ChunkObservation]] = defaultdict(list)
    for label in labels:
        if on_class_start:
            on_class_start(label)
        started = time.perf_counter()
        state = model.init_state(resource_path=str(directory))
        model.add_prompt(inference_state=state, frame_idx=0, text_str=label)
        seen: set[int] = set()
        for local, output in model.propagate_in_video(
                state, start_frame_idx=0, max_frame_num_to_track=len(frames), reverse=False):
            if local in seen or not 0 <= local < len(frames):
                raise ValueError("SAM3 returned a duplicate or out-of-range frame")
            seen.add(local)
            for object_id, mask, probability in zip(
                    output["out_obj_ids"], output["out_binary_masks"],
                    output["out_probs"], strict=True):
                array = _array(mask)
                pixels = np.argwhere(array)
                if not pixels.size:
                    continue
                ymin, xmin = pixels.min(axis=0)
                ymax, xmax = pixels.max(axis=0)
                score = float(probability)
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("SAM3 returned a non-probability track score")
                by_track[(label, int(object_id))].append(ChunkObservation(
                    frame=frames[local],
                    bbox=(float(xmin), float(ymin), float(xmax + 1), float(ymax + 1)),
                    mask=encode_mask(array), score=score,
                ))
        if seen != set(range(len(frames))):
            raise ValueError("SAM3 did not return every frame in this window")
        del state
        if on_class_end:
            on_class_end(label, time.perf_counter() - started)
    return [ChunkTrack(window_id, f"{label}:{object_id}", label, observations)
            for (label, object_id), observations in sorted(by_track.items())]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _installed_commit() -> str | None:
    try:
        distribution = importlib.metadata.distribution("sam3")
        direct_url = distribution.read_text("direct_url.json")
        if direct_url:
            commit = json.loads(direct_url).get("vcs_info", {}).get("commit_id")
            if commit:
                return commit
    except (importlib.metadata.PackageNotFoundError, ValueError, TypeError):
        pass
    try:
        import sam3

        result = subprocess.run(["git", "-C", str(Path(sam3.__file__).resolve().parent),
                                 "rev-parse", "HEAD"], capture_output=True, text=True,
                                timeout=5, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    except (ImportError, OSError, subprocess.TimeoutExpired, TypeError):
        return None


class Sam3SequenceBackend:
    """Actual GPU implementation behind ChunkProcessor; no network calls."""

    def __init__(self, *, frame_paths: dict[int, Path], checkpoint: Path,
                 checkpoint_sha256: str, sam3_commit: str) -> None:
        if len(checkpoint_sha256) != 64 or len(sam3_commit) != 40:
            raise ValueError("pin checkpoint SHA256 and SAM3 git commit")
        self.frame_paths = frame_paths
        self.checkpoint = checkpoint
        self.checkpoint_sha256 = checkpoint_sha256
        self.expected_commit = sam3_commit
        self.version = f"sam3@{sam3_commit}:{checkpoint_sha256}"
        self.model = None
        self.torch = None
        self.dtype = None
        self.runtime: dict = {"model_load_seconds": None, "class_seconds": [],
                              "window_seconds": [], "peak_cuda_allocated_bytes": []}

    def load(self) -> None:
        import torch
        from sam3.model_builder import build_sam3_video_model

        if not torch.cuda.is_available():
            raise RuntimeError("SAM3 video-text inference requires a CUDA GPU")
        actual_commit = _installed_commit()
        if actual_commit != self.expected_commit:
            raise RuntimeError(f"SAM3 commit mismatch: expected {self.expected_commit}, got {actual_commit}")
        if _sha256(self.checkpoint) != self.checkpoint_sha256:
            raise ValueError("SAM3 checkpoint SHA256 mismatch")
        self.torch = torch
        device = torch.cuda.get_device_properties(0)
        self.dtype = torch.bfloat16 if device.major >= 8 else torch.float16
        self.runtime["dtype"] = str(self.dtype)
        self.runtime["gpu"] = {"name": device.name, "total_vram_bytes": device.total_memory,
                               "compute_capability": [device.major, device.minor]}
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.autocast("cuda", dtype=self.dtype), torch.inference_mode():
            self.model = build_sam3_video_model(checkpoint_path=str(self.checkpoint), load_from_HF=False)
        self.model.tracker.backbone = self.model.detector.backbone
        require_video_text_api(self.model)
        from kaggle.tracking_parameters import tracking_parameters

        self.runtime["tracking_parameters"] = tracking_parameters(self.model)
        torch.cuda.synchronize()
        self.runtime["model_load_seconds"] = time.perf_counter() - started
        self.runtime["model_load_peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        self.runtime["sam3_commit"] = actual_commit
        self.runtime["checkpoint_sha256"] = self.checkpoint_sha256

    def run_chunk(self, *, index: int, frames: list[int],
                  labels: tuple[str, ...]) -> list[ChunkTrack]:
        if self.model is None or self.torch is None:
            raise RuntimeError("load SAM3 before processing chunks")
        torch = self.torch
        with tempfile.TemporaryDirectory(prefix="autolabel-sam3-") as temp:
            directory = Path(temp)
            for local, frame in enumerate(frames):
                source = self.frame_paths.get(frame)
                if source is None or not source.is_file():
                    raise FileNotFoundError(f"missing input frame {frame}")
                shutil.copyfile(source, directory / f"{local:06d}.jpg")
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            class_peaks: list[int] = []

            def class_start(_label: str) -> None:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

            def class_end(label: str, seconds: float) -> None:
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated()
                class_peaks.append(peak)
                self.runtime["class_seconds"].append({"window_id": index, "label": label,
                                                       "seconds": seconds,
                                                       "peak_cuda_allocated_bytes": peak})

            with torch.autocast("cuda", dtype=self.dtype), torch.inference_mode():
                tracks = collect_text_window(self.model, directory=directory,
                                             window_id=index, frames=frames, labels=labels,
                                             on_class_start=class_start, on_class_end=class_end)
            torch.cuda.synchronize()
            self.runtime["window_seconds"].append({"window_id": index,
                                                    "seconds": time.perf_counter() - started})
            self.runtime["peak_cuda_allocated_bytes"].append(
                {"window_id": index, "bytes": max(class_peaks, default=0)})
            torch.cuda.empty_cache()
            return tracks

    def close(self) -> None:
        self.model = None
        gc.collect()
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
