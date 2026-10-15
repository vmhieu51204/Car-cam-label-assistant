"""Kaggle GPU script: propagate CVAT frame-0 boxes with Meta SAM 3."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working")
CHUNK_FRAMES = 50


def _setup() -> None:
    if sys.version_info < (3, 12):
        raise RuntimeError("SAM3 requires Python 3.12+ in the Kaggle GPU runtime")
    if not torch.cuda.is_available():
        raise RuntimeError("SAM3 Kaggle kernel requires a CUDA GPU")
    # Kaggle imports NumPy before this cell runs. SAM3's NumPy<2 dependency would
    # replace that live module and break PyTorch's NumPy integration in-process.
    # Keep the Kaggle image's NumPy and install only SAM3's other runtime deps.
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "-q", "--no-deps",
            "git+https://github.com/facebookresearch/sam3.git",
            "timm>=1.0.17", "ftfy==6.1.1", "iopath>=0.1.10", "portalocker",
            "huggingface_hub",
        ],
        check=True,
    )


def _checkpoint() -> Path:
    candidates = list(INPUT_ROOT.glob("**/sam3.pt"))
    if len(candidates) != 1 or candidates[0].stat().st_size < 1_000_000:
        raise RuntimeError("Expected one sam3.pt in the attached private checkpoint dataset")
    return candidates[0]


def _inputs() -> tuple[dict, Path]:
    archives = list(INPUT_ROOT.glob("**/autolabel-sam3-*/payload.zip"))
    extracted = list(INPUT_ROOT.glob("**/autolabel-sam3-*/request.json"))
    if len(archives) + len(extracted) != 1:
        raise RuntimeError("Expected exactly one private AutoLabel SAM3 dataset")
    if extracted:
        target = extracted[0].parent
    else:
        target = WORK_ROOT / "autolabel-input"
        target.mkdir(exist_ok=True)
        with zipfile.ZipFile(archives[0]) as archive:
            for member in archive.infolist():
                path = Path(member.filename)
                if path.is_absolute() or ".." in path.parts:
                    raise RuntimeError("Unsafe input archive")
            archive.extractall(target)
    request = json.loads((target / "request.json").read_text(encoding="utf-8"))
    for record in request["frames"]:
        path = target / record["relative_path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise RuntimeError(f"Frame {record['frame_index']} does not match the request")
    return request, target


def _box_from_mask(mask: np.ndarray, previous: dict) -> tuple[dict, bool, int]:
    pixels = np.argwhere(mask)
    if len(pixels) == 0:
        return previous, True, 0
    y_min, x_min = pixels.min(axis=0)
    y_max, x_max = pixels.max(axis=0)
    return {
        "xtl": float(x_min), "ytl": float(y_min),
        "xbr": float(x_max + 1), "ybr": float(y_max + 1),
    }, False, int(len(pixels))


def _propagate(request: dict, root: Path, checkpoint: Path, *, work_root: Path = WORK_ROOT,
               runtime: dict | None = None) -> dict:
    model_import_started = time.perf_counter()
    from sam3.model_builder import build_sam3_video_model

    from kaggle.runtime_probe import begin_cuda, duration_stats, end_cuda

    if runtime is not None:
        runtime["timing_s"]["model_import"] = time.perf_counter() - model_import_started
    begin_cuda(torch)
    load_started = time.perf_counter()
    model = build_sam3_video_model(checkpoint_path=str(checkpoint), load_from_HF=False)
    if runtime is not None:
        runtime["peak_cuda_allocated_bytes"]["model_load"] = end_cuda(torch)
        runtime["timing_s"]["model_load"] = time.perf_counter() - load_started
    predictor = model.tracker
    predictor.backbone = model.detector.backbone
    frames = request["frames"][: request["end_frame_index"] + 1]
    seeds = request["seed_set"]["seeds"]
    if not frames or not seeds or len(frames) > 300:
        raise RuntimeError("SAM3 requires 1–300 frames and at least one frame-0 seed")
    tracks = [{
        "track_id": seed["object_id"], "label_id": seed["label_id"],
        "label_name": seed["label_name"], "source": "human_seed", "observations": [],
    } for seed in seeds]
    previous = [seed["bbox"] for seed in seeds]
    active = [True] * len(seeds)
    window_times: list[float] = []
    window_peaks: list[int | None] = []
    for start in range(0, len(frames), CHUNK_FRAMES):
        begin_cuda(torch)
        window_started = time.perf_counter()
        batch = frames[start:start + CHUNK_FRAMES]
        if not any(active):
            for local_index in range(len(batch)):
                for object_id, track in enumerate(tracks):
                    track["observations"].append({
                        "frame_index": start + local_index,
                        "bbox": previous[object_id], "score": 0.0,
                        "mask_area_px": 0, "outside": True,
                    })
            window_times.append(0.0)
            continue
        chunk_dir = work_root / f"sam3-chunk-{start:04d}"
        chunk_dir.mkdir(exist_ok=True)
        for local_index, frame in enumerate(batch):
            shutil.copyfile(root / frame["relative_path"], chunk_dir / f"{local_index:06d}.jpg")
        state = predictor.init_state(video_path=str(chunk_dir))
        width, height = batch[0]["width"], batch[0]["height"]
        for object_id, box in enumerate(previous, start=1):
            if not active[object_id - 1]:
                continue
            relative = np.array([[
                box["xtl"] / width, box["ytl"] / height,
                box["xbr"] / width, box["ybr"] / height,
            ]], dtype=np.float32)
            predictor.add_new_points_or_box(inference_state=state, frame_idx=0, obj_id=object_id, box=relative)
        seen = set()
        with torch.inference_mode():
            iterator = predictor.propagate_in_video(
                state, start_frame_idx=0, max_frame_num_to_track=len(batch),
                reverse=False, propagate_preflight=True,
            )
            for frame_index, object_ids, _low, masks, _scores in iterator:
                if frame_index in seen or frame_index < 0 or frame_index >= len(batch):
                    raise RuntimeError("SAM3 returned an invalid frame index")
                seen.add(frame_index)
                returned = {int(object_id): masks[index] for index, object_id in enumerate(object_ids)}
                for object_id, track in enumerate(tracks, start=1):
                    raw = returned.get(object_id) if active[object_id - 1] else None
                    mask = np.asarray(raw.detach().cpu().numpy()).squeeze() > 0 if raw is not None else np.zeros(
                        (batch[frame_index]["height"], batch[frame_index]["width"]), dtype=bool,
                    )
                    if mask.ndim != 2:
                        raise RuntimeError("SAM3 returned a mask with invalid dimensions")
                    bbox, outside, area = _box_from_mask(mask, previous[object_id - 1])
                    previous[object_id - 1] = bbox
                    if outside:
                        active[object_id - 1] = False
                    track["observations"].append({
                        "frame_index": start + frame_index, "bbox": bbox,
                        "score": 0.0 if outside else 1.0,
                        "mask_area_px": area, "outside": outside,
                    })
        if seen != set(range(len(batch))):
            raise RuntimeError("SAM3 did not return every frame in the chunk")
        del state
        shutil.rmtree(chunk_dir)
        if torch.cuda.is_available():
            window_peaks.append(end_cuda(torch))
            torch.cuda.empty_cache()
        window_times.append(time.perf_counter() - window_started)
    if runtime is not None:
        runtime["timing_s"]["inference_window"] = duration_stats(window_times)
        runtime["peak_cuda_allocated_bytes"]["inference_windows"] = window_peaks
    return {
        "schema": "autolabel-track-set/v1",
        "job_id": request["job_id"],
        "manifest_sha256": request["manifest_sha256"],
        "seed_sha256": request["seed_sha256"],
        "component_versions": [{"component": "sam3-video", "version": "facebook/sam3"}],
        "tracks": tracks,
    }


def main() -> None:
    from kaggle.runtime_probe import environment, sha256_file

    input_started = time.perf_counter()
    request, root = _inputs()
    input_seconds = time.perf_counter() - input_started
    setup_started = time.perf_counter()
    _setup()
    runtime = environment(torch, include_sam3=True)
    runtime["timing_s"]["input_validation"] = input_seconds
    runtime["timing_s"]["dependency_setup"] = time.perf_counter() - setup_started
    device = torch.cuda.get_device_properties(0)
    dtype = torch.bfloat16 if device.major >= 8 else torch.float16
    checkpoint = _checkpoint()
    runtime["checkpoint_sha256"] = sha256_file(checkpoint)
    runtime["dtype"] = str(dtype)
    runtime["autocast"] = {"enabled": True, "device": "cuda", "dtype": str(dtype)}
    runtime["timing_s"]["checkpoint_download"] = None  # Kaggle mounts the dataset before script start.
    with torch.autocast("cuda", dtype=dtype):
        track_set = _propagate(request, root, checkpoint, runtime=runtime)
    result = {"job_id": request["job_id"], "stage": "sam3", "track_set": track_set,
              "runtime": runtime}
    (WORK_ROOT / "result.json").write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()
