"""Kaggle GPU script: detect editable CVAT frame-0 boxes using SAM3 text prompts."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import torch
from PIL import Image

INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working")
MAX_BOXES = 80
MIN_SCORE = 0.55


def _setup() -> None:
    if sys.version_info < (3, 12) or not torch.cuda.is_available():
        raise RuntimeError("SAM3 frame-0 labeling requires Kaggle Python 3.12+ and a CUDA GPU")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps",
         "git+https://github.com/facebookresearch/sam3.git",
         "timm>=1.0.17", "ftfy==6.1.1", "iopath>=0.1.10", "portalocker", "huggingface_hub"],
        check=True,
    )


def _checkpoint() -> Path:
    candidates = list(INPUT_ROOT.glob("**/sam3.pt"))
    if len(candidates) != 1 or candidates[0].stat().st_size < 1_000_000:
        raise RuntimeError("Expected one sam3.pt in the private checkpoint dataset")
    return candidates[0]


def _inputs() -> tuple[dict, Path]:
    archives = list(INPUT_ROOT.glob("**/autolabel-sam3-seed-*/payload.zip"))
    extracted = list(INPUT_ROOT.glob("**/autolabel-sam3-seed-*/request.json"))
    if len(archives) + len(extracted) != 1:
        raise RuntimeError("Expected exactly one private AutoLabel SAM3 seed input")
    if extracted:
        target = extracted[0].parent
    else:
        target = WORK_ROOT / "autolabel-seed-input"
        target.mkdir(exist_ok=True)
        with zipfile.ZipFile(archives[0]) as archive:
            if set(archive.namelist()) != {"request.json", "frame.jpg"}:
                raise RuntimeError("Unexpected SAM3 seed input files")
            archive.extractall(target)
    request = json.loads((target / "request.json").read_text(encoding="utf-8"))
    if hashlib.sha256((target / "frame.jpg").read_bytes()).hexdigest() != request["frame_sha256"]:
        raise RuntimeError("SAM3 seed frame does not match request")
    return request, target / "frame.jpg"


def _iou(left: dict, right: dict) -> float:
    a, b = left["bbox"], right["bbox"]
    width = max(0.0, min(a["xbr"], b["xbr"]) - max(a["xtl"], b["xtl"]))
    height = max(0.0, min(a["ybr"], b["ybr"]) - max(a["ytl"], b["ytl"]))
    intersection = width * height
    area_a = (a["xbr"] - a["xtl"]) * (a["ybr"] - a["ytl"])
    area_b = (b["xbr"] - b["xtl"]) * (b["ybr"] - b["ytl"])
    return intersection / max(area_a + area_b - intersection, 1e-9)


def _detect(request: dict, image_path: Path, checkpoint: Path,
            runtime: dict | None = None) -> list[dict]:
    model_import_started = time.perf_counter()
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    from kaggle.runtime_probe import begin_cuda, duration_stats, end_cuda

    if runtime is not None:
        runtime["timing_s"]["model_import"] = time.perf_counter() - model_import_started
    image = Image.open(image_path).convert("RGB")
    if image.size != (request["width"], request["height"]):
        raise RuntimeError("SAM3 seed image size does not match request")
    begin_cuda(torch)
    load_started = time.perf_counter()
    model = build_sam3_image_model(checkpoint_path=str(checkpoint), load_from_HF=False)
    if runtime is not None:
        runtime["peak_cuda_allocated_bytes"]["model_load"] = end_cuda(torch)
        runtime["timing_s"]["model_load"] = time.perf_counter() - load_started
    processor = Sam3Processor(model, confidence_threshold=MIN_SCORE)
    begin_cuda(torch)
    image_started = time.perf_counter()
    state = processor.set_image(image)
    if runtime is not None:
        runtime["peak_cuda_allocated_bytes"]["image_encode"] = end_cuda(torch)
        runtime["timing_s"]["image_encode"] = time.perf_counter() - image_started
    candidates = []
    prompt_times: list[float] = []
    prompt_peaks: list[int | None] = []
    for label in request["labels"]:
        prompt = label["prompt"].replace("_", " ").strip()
        if not prompt:
            continue
        processor.reset_all_prompts(state)
        begin_cuda(torch)
        prompt_started = time.perf_counter()
        with torch.inference_mode():
            output = processor.set_text_prompt(state=state, prompt=prompt)
        prompt_peaks.append(end_cuda(torch))
        prompt_times.append(time.perf_counter() - prompt_started)
        for raw_box, raw_score in zip(output["boxes"].tolist(), output["scores"].tolist(), strict=True):
            coordinates = [float(value) for value in raw_box]
            score = float(raw_score)
            if not all(math.isfinite(value) for value in (*coordinates, score)) or score < MIN_SCORE:
                continue
            x1, y1, x2, y2 = coordinates
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(image.width), x2), min(float(image.height), y2)
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            candidates.append({
                "label_id": label["id"], "score": score,
                "bbox": {"xtl": x1, "ytl": y1, "xbr": x2, "ybr": y2},
            })
    selected: list[dict] = []
    for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
        if all(_iou(candidate, previous) < 0.8 for previous in selected):
            selected.append(candidate)
        if len(selected) == MAX_BOXES:
            break
    if runtime is not None:
        runtime["timing_s"]["inference_prompt"] = duration_stats(prompt_times)
        runtime["peak_cuda_allocated_bytes"]["inference_prompts"] = prompt_peaks
    return selected


def main() -> None:
    from kaggle.runtime_probe import environment, sha256_file

    input_started = time.perf_counter()
    request, image_path = _inputs()
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
        detections = _detect(request, image_path, checkpoint, runtime=runtime)
    result = {"stage": "sam3-seed", "job_id": request["job_id"],
              "detections": detections, "runtime": runtime}
    (WORK_ROOT / "result.json").write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()
