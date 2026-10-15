"""Compare SAM3 frame spacing and window policies in one Kaggle GPU session."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working")
BENCHMARK_SESSION_STARTED_AT: float | None = None


def validate_config(config: dict) -> None:
    for key in ("max_total_seconds", "max_config_seconds"):
        if type(config.get(key)) not in {int, float} or not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if type(config.get("source_frame_limit")) is not int or not 1 <= config["source_frame_limit"] <= 10000:
        raise ValueError("source_frame_limit must be 1..10000")
    seen: set[str] = set()
    for row in config.get("configurations", []):
        if row.get("tracking_policy", "independent_windows") not in {"independent_windows", "native_scene"}:
            raise ValueError("unsupported tracking policy")
        if row.get("tracking_policy") == "native_scene" and (
                row.get("mode") != "text_prompt" or type(row.get("max_scene_frames")) is not int
                or not 1 <= row["max_scene_frames"] <= 250):
            raise ValueError("native_scene requires text prompt and bounded max_scene_frames")
        if row.get("model_variant", "sam3") not in {"sam3", "sam3.1"}:
            raise ValueError("unsupported SAM model variant")
        if type(row.get("id")) is not str or row["id"] in seen:
            raise ValueError("configuration IDs must be unique strings")
        seen.add(row["id"])
        forced = row.get("force_include_frames", [])
        if (not isinstance(forced, list) or any(type(index) is not int or
                not 0 <= index < config["source_frame_limit"] for index in forced)
                or forced != sorted(set(forced))):
            raise ValueError("force_include_frames must be unique ordered source indices")
        if row.get("selection") not in {"optical", "uniform", "dense"}:
            raise ValueError("unsupported frame selection")
        if row["selection"] == "uniform":
            if "target_fps" in row:
                if any(type(row.get(key)) not in {int, float} or not math.isfinite(row[key]) or row[key] <= 0
                       for key in ("source_fps", "target_fps")) or row["target_fps"] > row["source_fps"]:
                    raise ValueError("invalid uniform FPS")
            elif row.get("gap") not in {2, 3, 5, 10, 20}:
                raise ValueError("uniform gap must be 2, 3, 5, 10 or 20")
        if type(row.get("window_size")) is not int or not 2 <= row["window_size"] <= 250:
            raise ValueError("window_size must be 2..250")
        if type(row.get("overlap")) is not int or not 0 <= row["overlap"] < row["window_size"]:
            raise ValueError("overlap must be smaller than window_size")
        if type(row.get("preserve_overlap", False)) is not bool:
            raise ValueError("preserve_overlap must be boolean")
        if row.get("mode") == "box_seed":
            if row.get("reprompt") not in {"box_last", "mask_last"} or row.get("seed_type") not in {"box", "mask"}:
                raise ValueError("invalid box-seed prompt policy")
        elif row.get("mode") in {"text_prompt", "tile_probe"}:
            if row.get("reprompt") != "none" or row.get("seed_type") != "none":
                raise ValueError("text mode uses its own semantic prompt")
            prompts = row.get("text_prompts")
            if not isinstance(prompts, list) or not prompts or any(
                    not isinstance(prompt, str) or not prompt.strip() for prompt in prompts
            ) or len(prompts) != len(set(prompts)):
                raise ValueError("text mode needs unique nonempty text_prompts")
            mapping = row.get("prompt_to_label")
            if mapping is not None and (not isinstance(mapping, dict) or set(mapping) != set(prompts)
                                        or any(not isinstance(value, str) or not value.strip()
                                               for value in mapping.values())):
                raise ValueError("prompt_to_label must map every text prompt to a nonempty class")
            for name in ("score_threshold_detection", "new_det_thresh"):
                value = row.get(name)
                if value is not None and (type(value) not in (int, float) or not math.isfinite(value)
                                          or not 0 <= value <= 1):
                    raise ValueError(f"{name} must be in [0, 1]")
            if (row.get("score_threshold_detection") is None) != (row.get("new_det_thresh") is None):
                raise ValueError("both SAM3 detection thresholds must be set together")
            if (row.get("score_threshold_detection") is not None
                    and row["new_det_thresh"] < row["score_threshold_detection"]):
                raise ValueError("new_det_thresh must not be below score_threshold_detection")
            if row["mode"] == "tile_probe":
                tile_prompts = row.get("tile_prompts")
                if (not isinstance(tile_prompts, list) or not tile_prompts
                        or any(prompt not in prompts for prompt in tile_prompts)
                        or type(row.get("tile_probe_stride")) is not int
                        or not 1 <= row["tile_probe_stride"] <= 50
                        or type(row.get("tile_max_candidates_per_window")) is not int
                        or not 1 <= row["tile_max_candidates_per_window"] <= 5):
                    raise ValueError("invalid tile_probe settings")
        else:
            raise ValueError("unsupported SAM3 mode")


def select_frames(frames: list[dict], optical_indices: list[int], row: dict) -> list[dict]:
    available = {item["source_frame_index"]: item for item in frames}
    if row["selection"] == "optical":
        indices = [index for index in optical_indices if index in available]
    elif row["selection"] == "uniform":
        if "target_fps" in row:
            count = max(available) + 1
            step = row["source_fps"] / row["target_fps"]
            indices = sorted({min(count - 1, round(i * step))
                              for i in range(math.ceil((count - 1) / step) + 1)} | {0, count - 1})
            if any(index not in available for index in indices):
                raise ValueError("FPS selection requires every requested source frame")
        else:
            indices = [index for index in sorted(available) if index % row["gap"] == 0]
    else:
        indices = sorted(available)
    indices = sorted(set(indices) | set(row.get("force_include_frames", [])))
    if any(index not in available for index in indices):
        raise ValueError("forced frame is absent from the payload")
    if not indices or indices[0] != 0:
        raise ValueError("selection must include source frame zero")
    cuts = set(row.get("scene_cut_frames", []))
    selected = []
    previous = None
    for index in indices:
        frame = dict(available[index])
        current = frame["source_frame_index"]
        frame["scene_cut_before"] = frame.get("scene_cut_before", False) or any(
            (previous is None and cut == current) or (previous is not None and previous < cut <= current)
            for cut in cuts
        )
        selected.append(frame)
        previous = current
    return selected


def windows(frames: list[dict], size: int, overlap: int) -> list[tuple[int, list[dict]]]:
    """Window each sampled scene independently, preserving overlap only inside it."""
    result = []
    boundaries = [0] + [i for i, frame in enumerate(frames) if i and frame.get("scene_cut_before")] + [len(frames)]
    for left, right in zip(boundaries, boundaries[1:]):
        start = left
        while start < right:
            result.append((start, frames[start:min(right, start + size)]))
            if start + size >= right:
                break
            start += size - overlap
    return result


def checkpoint_identity(frames: list[dict], row: dict, code_sha256: str = "") -> str:
    from kaggle.checkpoint_contract import checkpoint_identity as identity
    return identity(frames, row, code_sha256)


def model_identity(probe: dict) -> dict:
    from kaggle.checkpoint_contract import model_identity as identity
    return identity(probe)


def validate_checkpoint(entry: dict, *, identity: str, window_id: int,
                        frame_ids: list[int], model: dict | None = None) -> None:
    from kaggle.checkpoint_contract import validate_checkpoint as validate
    validate(entry, identity=identity, window_id=window_id, frame_ids=frame_ids, model=model)


def _save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    temporary.replace(path)


def _rle(mask: Any) -> dict:
    """Persist the uncompressed mask with the same row-major contract as the backend."""
    import numpy as np

    flat = np.asarray(mask, dtype=bool).reshape(-1)
    height, width = mask.shape
    transitions = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    counts = np.diff(np.concatenate(([0], transitions, [flat.size]))).tolist()
    if flat[0]:
        counts.insert(0, 0)
    canonical = json.dumps([height, width, counts], separators=(",", ":")).encode("ascii")
    return {"height": height, "width": width, "counts": counts,
            "sha256": hashlib.sha256(canonical).hexdigest()}


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p95": None}
    ordered = sorted(values)
    def quantile(fraction: float) -> float:
        return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]
    return {"mean": sum(values) / len(values), "p50": quantile(0.5), "p95": quantile(0.95)}


def load_payload(root: Path) -> tuple[dict, list[dict], list[int], list[dict]]:
    config = json.loads((root / "benchmark-config.json").read_text(encoding="utf-8"))
    validate_config(config)
    index = json.loads((root / "frames.json").read_text(encoding="utf-8"))
    frames = index["frames"]
    if not frames or len(frames) > config["source_frame_limit"]:
        raise ValueError("payload exceeds configured source frame limit")
    ids = [item["source_frame_index"] for item in frames]
    if ids != sorted(set(ids)) or ids[0] != 0:
        raise ValueError("payload frames must be ordered from source frame zero")
    for item in frames:
        path = Path(item["relative_path"])
        if path.is_absolute() or ".." in path.parts or not (root / path).is_file():
            raise ValueError("unsafe or missing frame")
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("frame checksum mismatch")
    optical = index["optical_source_indices"]
    if 0 not in optical or optical != sorted(set(optical)):
        raise ValueError("optical keyframes must be ordered and include frame zero")
    seeds = json.loads((root / "seed.json").read_text(encoding="utf-8"))["seeds"]
    if not seeds and any(row["mode"] == "box_seed" for row in config["configurations"]):
        raise ValueError("box-seed mode needs at least one frame-zero seed")
    for seed in seeds:
        if seed.get("mask_file"):
            path = Path(seed["mask_file"])
            if path.is_absolute() or ".." in path.parts or not (root / path).is_file():
                raise ValueError("unsafe or missing seed mask")
    return config, frames, optical, seeds


def run_benchmark(root: Path, output: Path, backend: Any, *, clock: Any = time.monotonic) -> dict:
    config, frames, optical, seeds = load_payload(root)
    started = BENCHMARK_SESSION_STARTED_AT if BENCHMARK_SESSION_STARTED_AT is not None else clock()
    result: dict = {"schema": "autolabel-sam3-benchmark/v1", "configurations": [], "completed": False}
    try:
        load_started = clock()
        backend.load(root)
        result["model_load_seconds"] = clock() - load_started
        result["model_runtime"] = backend.probe()
        result["model_runtime"]["timing_s"] = {
            "model_load": result["model_load_seconds"]}
        load_error = None
    except Exception as exc:
        load_error = f"{type(exc).__name__}: {exc}"
        result["model_load_error"] = load_error
    _save(output, result)
    print(f"SAM3 model load: {'failed: ' + load_error if load_error else 'completed'}", flush=True)
    for row in config["configurations"]:
        item: dict = {"id": row["id"], "config": row, "status": "pending", "predictions": [],
                      "evaluated_source_frames": [], "runtime": {}, "window_checkpoints": []}
        if row.get("preserve_overlap"):
            item["window_predictions"] = []
        result["configurations"].append(item)
        if clock() >= started + config["max_total_seconds"]:
            item.update(status="skipped_budget", error="total time budget reached")
            _save(output, result)
            print(f"SAM3 config {row['id']}: {item['status']}", flush=True)
            continue
        if load_error:
            item.update(status="failed", error=load_error)
            _save(output, result)
            print(f"SAM3 config {row['id']}: {item['status']}", flush=True)
            continue
        try:
            selected = select_frames(frames, optical, row)
            item["selected_source_frames"] = [frame["source_frame_index"] for frame in selected]
            if row["seed_type"] == "mask" and any(not seed.get("mask_file") for seed in seeds):
                raise NotImplementedError("ground truth does not supply every frame-0 seed mask")
            backend.begin_config()
            config_started = clock()
            if hasattr(backend, "configure"):
                backend.configure(row)
            if row.get("tracking_policy") == "native_scene":
                from kaggle.native_memory import run_native

                if (root / "resume.json").is_file():
                    raise ValueError("native output checkpoints cannot restore memory; replay scene without resume.json")
                run_native(backend, root, selected, row, item, save=lambda: _save(output, result),
                           deadline=min(started + config["max_total_seconds"],
                                        config_started + config["max_config_seconds"]),
                           identity=checkpoint_identity(frames, row, config.get("code_sha256", "")),
                           model=model_identity(result["model_runtime"]), clock=clock)
                backend.end_config()
                _save(output, result)
                continue
            accepted: dict[int, list[dict]] = {}
            carry_masks: dict[int, dict[str, Any]] = {}
            durations: list[float] = []
            window_timings: list[dict] = []
            batches = windows(selected, row["window_size"], row["overlap"])
            identity = checkpoint_identity(frames, row, config.get("code_sha256", ""))
            probe_identity = model_identity(result["model_runtime"])
            resume_path = root / "resume.json"
            resume = json.loads(resume_path.read_text()) if resume_path.is_file() else {"windows": []}
            saved = {entry["window_id"]: entry for entry in resume.get("windows", [])
                     if entry.get("config_id") == row["id"]}
            item["unique_frame_count"] = len(selected)
            item["window_frame_count"] = sum(len(batch) for _, batch in batches)
            item["resumed_windows"] = []
            for window_index, (start, batch) in enumerate(batches):
                if (clock() >= started + config["max_total_seconds"]
                        or clock() >= config_started + config["max_config_seconds"]):
                    item.update(status="timed_out", error="time budget reached between SAM3 windows")
                    break
                source_start = batch[0]["source_frame_index"]
                previous = accepted.get(source_start)
                if previous is None and accepted:
                    previous = accepted[max(index for index in accepted if index < source_start)]
                tick = clock()
                frame_ids = [frame["source_frame_index"] for frame in batch]
                window_result: dict[int, list[dict]]
                if window_index in saved:
                    if row["mode"] != "text_prompt" or not row.get("preserve_overlap"):
                        raise ValueError("resume only supports independent text-prompt windows with masks")
                    entry = saved[window_index]
                    validate_checkpoint(entry, identity=identity, window_id=window_index,
                                        frame_ids=frame_ids, model=probe_identity)
                    window_result = {frame: [] for frame in frame_ids}
                    for prediction in entry["predictions"]:
                        window_result[prediction["frame_index"]].append({
                            key: value for key, value in prediction.items() if key not in {"window_id", "frame_index"}})
                    item["resumed_windows"].append(window_index)
                else:
                    window_result = backend.run_window(root, batch, seeds, row, window_index,
                                                       previous or [], carry_masks.get(source_start, {}))
                durations.append(clock() - tick)
                window_timings.append({"window_id": window_index,
                                       "seconds": durations[-1]})
                if set(window_result) != {frame["source_frame_index"] for frame in batch}:
                    raise ValueError("SAM3 window did not return every selected frame")
                next_source_start = (batches[window_index + 1][1][0]["source_frame_index"]
                                     if window_index + 1 < len(batches) else None)
                carry_masks = ({next_source_start: {
                    obs["track_id"]: obs["_mask"] for obs in window_result[next_source_start]
                    if obs.get("_mask") is not None}}
                    if next_source_start is not None and next_source_start in window_result else {})
                if row.get("preserve_overlap"):
                    predictions = [{
                        "window_id": window_index, "frame_index": source_index,
                        **{key: value for key, value in obs.items() if key != "_mask"},
                        "mask_rle": _rle(obs["_mask"]) if obs.get("_mask") is not None else obs.get("mask_rle"),
                    } for source_index, observations in window_result.items()
                      for obs in observations if not obs.get("outside", False)]
                    item["window_predictions"].extend(predictions)
                    entry = {"schema": "autolabel-sam3-window/v1", "config_id": row["id"],
                             "input_sha256": identity, "window_id": window_index,
                             "frame_ids": frame_ids, "model": probe_identity, "predictions": predictions}
                    entry["output_sha256"] = hashlib.sha256(json.dumps(entry, sort_keys=True,
                                                       separators=(",", ":")).encode()).hexdigest()
                    item["window_checkpoints"].append(entry)
                for source_index, observations in window_result.items():
                    if source_index not in accepted:
                        accepted[source_index] = [{key: value for key, value in obs.items() if key != "_mask"}
                                                  for obs in observations]
                if row.get("preserve_overlap"):
                    item["evaluated_source_frames"] = sorted(accepted)
                    _save(output, result)
            item["evaluated_source_frames"] = sorted(accepted)
            item["predictions"] = [{"frame_index": frame, **obs}
                                   for frame in sorted(accepted) for obs in accepted[frame]
                                   if not obs.get("outside", False)]
            item["runtime"] = {**backend.probe(), "window_seconds": _stats(durations),
                               "total_seconds": clock() - config_started,
                               "windows_completed": len(durations)}
            item["runtime"]["timing_s"] = {
                "total_inference": item["runtime"]["total_seconds"],
                "inference_window": _stats(durations),
                "windows": window_timings,
                "classes": getattr(backend, "class_timings", []),
            }
            if item["status"] == "pending":
                item["status"] = "completed"
        except NotImplementedError as exc:
            item.update(status="unsupported", error=str(exc))
        except Exception as exc:
            item.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            try:
                backend.recover()
            except Exception as recovery_error:
                item["recovery_error"] = f"{type(recovery_error).__name__}: {recovery_error}"
        if hasattr(backend, "end_config"):
            backend.end_config()
        # Preserve measurements even when inference fails halfway through a config.
        if not item["runtime"] and load_error is None:
            item["runtime"] = {**backend.probe(), "timing_s": {
                "windows": locals().get("window_timings", []),
                "classes": getattr(backend, "class_timings", [])}}
        _save(output, result)
        print(f"SAM3 config {row['id']}: {item['status']}", flush=True)
    backend.release()
    result["completed"] = True
    _save(output, result)
    return result


class Sam3Backend:
    """Actual SAM3 API adapter; imported only inside the manually run GPU notebook."""

    def __init__(self, work_root: Path = WORK_ROOT) -> None:
        self.work_root = work_root
        self.model: Any = None
        self.torch: Any = None
        self.numpy: Any = None
        self.checkpoint: Path | None = None
        self.checkpoint_sha256 = None
        self.fallbacks = 0
        self.class_timings: list[dict] = []
        self.image_model: Any = None
        self.image_processor: Any = None
        self.model_variant = "sam3"
        self.session_ids: set[Any] = set()

    def load(self, root: Path) -> None:
        import numpy as np
        import torch

        config = json.loads((root / "benchmark-config.json").read_text(encoding="utf-8"))
        variants = {row.get("model_variant", "sam3") for row in config.get("configurations", [])}
        if len(variants) != 1:
            raise ValueError("one Kaggle run must use exactly one SAM model variant")
        self.model_variant = variants.pop()
        if self.model_variant != "sam3":
            raise NotImplementedError("canonical Kaggle backend supports SAM3 only; use local multiplex adapter for SAM3.1")
        checkpoint_name = "sam3.pt"
        checkpoint = list(INPUT_ROOT.glob(f"**/{checkpoint_name}"))
        if len(checkpoint) != 1:
            raise RuntimeError(f"attach exactly one private {checkpoint_name} checkpoint dataset")
        self.checkpoint = checkpoint[0]
        from kaggle.runtime_probe import sha256_file

        self.checkpoint_sha256 = sha256_file(self.checkpoint)
        self.torch, self.numpy = torch, np
        from sam3.model_builder import build_sam3_video_model

        self.model = build_sam3_video_model(checkpoint_path=str(self.checkpoint), load_from_HF=False)
        self.model.tracker.backbone = self.model.detector.backbone

    def begin_config(self) -> None:
        self.last_memory_selection = {"policy": "group"}
        self.fallbacks = 0
        self.class_timings = []
        for device in range(self.torch.cuda.device_count()):
            self.torch.cuda.reset_peak_memory_stats(device)

    def configure(self, row: dict) -> None:
        """Change only the two pinned SAM3 detector gates for an explicit benchmark row."""
        policy = row.get("memory_policy", "group")
        if policy not in {"group", "dms_singleton"}:
            raise ValueError("unsupported memory policy")
        if policy == "dms_singleton":
            from kaggle.dms_memory import DecoupledMemory

            self.dms = DecoupledMemory(self.model)
            self.dms.install()
        if row["mode"] not in {"text_prompt", "tile_probe"}:
            return
        for name, default in (("score_threshold_detection", 0.5), ("new_det_thresh", 0.7)):
            if not hasattr(self.model, name):
                raise NotImplementedError(f"installed SAM3 lacks {name}")
            setattr(self.model, name, row.get(name, default))
        if row["mode"] == "tile_probe":
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model

            self.image_model = build_sam3_image_model(
                checkpoint_path=str(self.checkpoint), load_from_HF=False)
            self.image_processor = Sam3Processor(self.image_model, confidence_threshold=0.5)

    def end_config(self) -> None:
        if getattr(self, "dms", None) is not None:
            self.last_memory_selection = self.dms.evidence()
            self.dms.restore()
            self.dms = None
        self.image_processor = None
        self.image_model = None
        for session_id in list(self.session_ids):
            try:
                self.model.handle_request({"type": "close_session", "session_id": session_id})
            except Exception:
                pass
            self.session_ids.discard(session_id)
        self.recover()

    def recover(self) -> None:
        gc.collect()
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def release(self) -> None:
        self.end_config()
        self.model = None
        self.recover()

    def probe(self) -> dict:
        from kaggle.runtime_probe import environment
        from kaggle.tracking_parameters import tracking_parameters

        peaks = {str(device): self.torch.cuda.max_memory_allocated(device)
                 for device in range(self.torch.cuda.device_count())}
        if self.class_timings:
            peaks["0"] = max(peaks.get("0", 0), *(row["peak_cuda_allocated_bytes"]
                                                   for row in self.class_timings
                                                   if row["peak_cuda_allocated_bytes"] is not None))
        threshold_owner = self.model
        return {"environment": environment(self.torch, include_sam3=True),
                "tracking_parameters": tracking_parameters(self.model),
                "memory_selection": self.dms.evidence() if getattr(self, "dms", None) else getattr(self, "last_memory_selection", {"policy": "group"}),
                "checkpoint_sha256": self.checkpoint_sha256,
                "model_variant": self.model_variant,
                "sam3_detection_thresholds": {
                    name: getattr(threshold_owner, name, None)
                    for name in ("score_threshold_detection", "new_det_thresh")},
                "dtype": "bf16" if self.torch.cuda.get_device_properties(0).major >= 8 else "fp16",
                "autocast": "cuda", "peak_cuda_allocated_bytes_by_device": peaks,
                "peak_cuda_allocated_bytes": max(peaks.values(), default=0),
                "mask_to_box_fallbacks": self.fallbacks}

    def run_window(self, root: Path, batch: list[dict], seeds: list[dict], row: dict,
                   window_index: int, previous: list[dict], previous_masks: dict[str, Any]) -> dict[int, list[dict]]:
        torch = self.torch
        directory = self.work_root / f"sam3-benchmark-{row['id']}-{window_index:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        try:
            for local, frame in enumerate(batch):
                shutil.copyfile(root / frame["relative_path"], directory / f"{local:06d}.jpg")
            dtype = torch.bfloat16 if torch.cuda.get_device_properties(0).major >= 8 else torch.float16
            with torch.autocast("cuda", dtype=dtype), torch.inference_mode():
                if row["mode"] == "text_prompt":
                    return self._text_window(directory, batch, row, window_index)
                if row["mode"] == "tile_probe":
                    return self._tile_window(directory, batch, row, window_index)
                return self._box_window(directory, batch, seeds, row, previous, previous_masks, root)
        finally:
            shutil.rmtree(directory)
            self.recover()

    def _box_window(self, directory: Path, batch: list[dict], seeds: list[dict], row: dict,
                    previous: list[dict], previous_masks: dict[str, Any], root: Path) -> dict[int, list[dict]]:
        predictor = self.model.tracker
        if row["reprompt"] == "mask_last" and not hasattr(predictor, "add_new_mask"):
            raise NotImplementedError("installed SAM3 tracker lacks add_new_mask")
        state = predictor.init_state(video_path=str(directory))
        width, height = batch[0]["width"], batch[0]["height"]
        prior = {item["track_id"]: item for item in previous}
        for index, seed in enumerate(seeds, start=1):
            track_id = seed["track_id"]
            mask = previous_masks.get(track_id)
            if not previous and row["seed_type"] == "mask":
                from PIL import Image

                with Image.open(root / seed["mask_file"]) as opened:
                    mask = self.numpy.asarray(opened.convert("L")) > 0
            if mask is not None and (row["reprompt"] == "mask_last" or row["seed_type"] == "mask"):
                predictor.add_new_mask(inference_state=state, frame_idx=0, obj_id=index,
                                       mask=self.torch.as_tensor(mask))
            else:
                if previous and row["reprompt"] == "mask_last":
                    self.fallbacks += 1
                box = prior.get(track_id, seed)["bbox"]
                normalized = self.numpy.array([[box[0] / width, box[1] / height,
                                                box[2] / width, box[3] / height]], dtype=self.numpy.float32)
                predictor.add_new_points_or_box(inference_state=state, frame_idx=0,
                                                obj_id=index, box=normalized)
        results: dict[int, list[dict]] = {}
        iterator = predictor.propagate_in_video(
            state, start_frame_idx=0, max_frame_num_to_track=len(batch),
            reverse=False, propagate_preflight=True,
        )
        for local, object_ids, _low, masks, logits in iterator:
            returned = {int(object_id): (masks[position], logits[position])
                        for position, object_id in enumerate(object_ids)}
            observations = []
            for index, seed in enumerate(seeds, start=1):
                raw, logit = returned.get(index, (None, None))
                mask = (self.numpy.asarray(raw.detach().cpu().numpy()).squeeze() > 0
                        if raw is not None else self.numpy.zeros((height, width), dtype=bool))
                bbox = _mask_bbox(mask)
                observations.append({"track_id": seed["track_id"], "label": seed["label"],
                                     "bbox": bbox or seed["bbox"],
                                     "score": (float(logit.sigmoid().item())
                                               if bbox is not None and logit is not None else 0.0),
                                     "outside": bbox is None, "_mask": mask})
            results[batch[local]["source_frame_index"]] = observations
        del state
        return results

    def _text_window(self, directory: Path, batch: list[dict], row: dict,
                     window_index: int) -> dict[int, list[dict]]:
        if not all(hasattr(self.model, name) for name in ("init_state", "add_prompt", "propagate_in_video")):
            raise NotImplementedError("installed SAM3 lacks dense video text-prompt API")
        results: dict[int, list[dict]] = {item["source_frame_index"]: [] for item in batch}
        mapping = row.get("prompt_to_label") or {}
        for prompt in row["text_prompts"]:
            class_started = time.perf_counter()
            if self.torch is not None:
                self.torch.cuda.reset_peak_memory_stats()
            state = self.model.init_state(resource_path=str(directory))
            self.model.add_prompt(inference_state=state, frame_idx=0, text_str=prompt)
            for local, output in self.model.propagate_in_video(
                    state, start_frame_idx=0, max_frame_num_to_track=len(batch), reverse=False):
                for object_id, mask, probability in zip(
                        output["out_obj_ids"], output["out_binary_masks"],
                        output["out_probs"], strict=True):
                    array = self.numpy.asarray(mask.detach().cpu().numpy()
                                               if hasattr(mask, "detach") else mask).squeeze() > 0
                    bbox = _mask_bbox(array)
                    if bbox is not None:
                        results[batch[local]["source_frame_index"]].append({
                            "track_id": f"text:{prompt}:window{window_index}:{int(object_id)}",
                            "label": mapping.get(prompt, prompt), "bbox": bbox, "score": float(probability),
                            "outside": False,
                            "_mask": array,
                        })
            del state
            if self.torch is not None:
                self.torch.cuda.synchronize()
            self.class_timings.append({
                "window_id": window_index, "label": prompt,
                "seconds": time.perf_counter() - class_started,
                "peak_cuda_allocated_bytes": (self.torch.cuda.max_memory_allocated()
                                              if self.torch is not None else None),
            })
        if mapping:
            for frame in results:
                results[frame] = _deduplicate_prompt_boxes(results[frame])
        return results


    def _tile_window(self, directory: Path, batch: list[dict], row: dict,
                     window_index: int) -> dict[int, list[dict]]:
        """Experimental small-object rescue: image tiles seed short video tracks."""
        from PIL import Image

        if self.image_processor is None or not hasattr(self.model.tracker, "add_new_points_or_box"):
            raise NotImplementedError("installed SAM3 lacks image tile or box tracking API")
        results = self._text_window(directory, batch, row, window_index)
        width, height = batch[0]["width"], batch[0]["height"]
        candidates: list[tuple[int, dict]] = []
        for local in range(0, len(batch), row["tile_probe_stride"]):
            if len(candidates) >= row["tile_max_candidates_per_window"]:
                break
            frame_index = batch[local]["source_frame_index"]
            with Image.open(directory / f"{local:06d}.jpg") as opened:
                image = opened.convert("RGB")
            for left, top, right, bottom in _tile_bounds(width, height):
                state = self.image_processor.set_image(image.crop((left, top, right, bottom)))
                for prompt in row["tile_prompts"]:
                    self.image_processor.reset_all_prompts(state)
                    detected = self.image_processor.set_text_prompt(state=state, prompt=prompt)
                    for raw_box, probability in zip(detected["boxes"].tolist(),
                                                    detected["scores"].tolist(), strict=True):
                        box = _restore_tile_box(raw_box, (left, top, right, bottom), width, height)
                        if (box is None or box[3] - box[1] > 40
                                or min(box[2] - box[0], box[3] - box[1]) < 4):
                            continue
                        label = (row.get("prompt_to_label") or {}).get(prompt, prompt)
                        candidate = {"label": label, "bbox": box, "score": float(probability),
                                     "tile_source_frame": frame_index}
                        if any(other["label"] == label and _box_iou(box, other["bbox"]) >= 0.5
                               for other in results[frame_index]):
                            continue
                        if any(other["label"] == label and _box_iou(box, other["bbox"]) >= 0.8
                               for candidate_local, other in candidates if candidate_local == local):
                            continue
                        candidates.append((local, candidate))
                        if len(candidates) >= row["tile_max_candidates_per_window"]:
                            break
                    if len(candidates) >= row["tile_max_candidates_per_window"]:
                        break
                if len(candidates) >= row["tile_max_candidates_per_window"]:
                    break
        for ordinal, (local, candidate) in enumerate(candidates):
            tracker = self.model.tracker
            state = tracker.init_state(video_path=str(directory))
            box = candidate["bbox"]
            normalized = self.numpy.array([[box[0] / width, box[1] / height,
                                            box[2] / width, box[3] / height]], dtype=self.numpy.float32)
            tracker.add_new_points_or_box(inference_state=state, frame_idx=local,
                                          obj_id=1, box=normalized)
            for current, object_ids, _low, masks, logits in tracker.propagate_in_video(
                    state, start_frame_idx=local,
                    max_frame_num_to_track=min(50, len(batch) - local),
                    reverse=False, propagate_preflight=True):
                if 1 not in object_ids:
                    continue
                position = list(object_ids).index(1)
                mask = self.numpy.asarray(masks[position].detach().cpu().numpy()).squeeze() > 0
                detected_box = _mask_bbox(mask)
                if detected_box is None:
                    continue
                frame_index = batch[current]["source_frame_index"]
                if any(other["label"] == candidate["label"]
                       and _box_iou(detected_box, other["bbox"]) >= 0.5
                       for other in results[frame_index]):
                    continue
                results[frame_index].append({
                    "track_id": f"tile:{window_index}:{local}:{ordinal}",
                    "label": candidate["label"], "bbox": detected_box,
                    "score": float(logits[position].sigmoid().item()), "outside": False,
                    "_mask": mask, "tile_source_frame": candidate["tile_source_frame"],
                })
            del state
        return results


def _tile_bounds(width: int, height: int) -> list[tuple[int, int, int, int]]:
    """Two full-height crops with 20% horizontal overlap, covering every pixel."""
    if width < 2 or height < 1:
        raise ValueError("tile source image is too small")
    middle = width // 2
    margin = max(1, round(width * 0.1))
    return [(0, 0, min(width, middle + margin), height),
            (max(0, middle - margin), 0, width, height)]


def _restore_tile_box(box: list[float], tile: tuple[int, int, int, int],
                      width: int, height: int) -> list[float] | None:
    """Map a crop-local XYXY detection to original pixels and clip to the image."""
    if len(box) != 4 or any(not math.isfinite(value) for value in box):
        return None
    left, top, _, _ = tile
    restored = [max(0.0, min(float(width), box[0] + left)),
                max(0.0, min(float(height), box[1] + top)),
                max(0.0, min(float(width), box[2] + left)),
                max(0.0, min(float(height), box[3] + top))]
    return restored if restored[2] > restored[0] and restored[3] > restored[1] else None


def _box_iou(left: list[float], right: list[float]) -> float:
    overlap = max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0.0, min(left[3], right[3]) - max(left[1], right[1]))
    area_left = (left[2] - left[0]) * (left[3] - left[1])
    area_right = (right[2] - right[0]) * (right[3] - right[1])
    union = area_left + area_right - overlap
    return overlap / union if union > 0 else 0.0


def _deduplicate_prompt_boxes(observations: list[dict], minimum_iou: float = 0.8) -> list[dict]:
    """Suppress same-class aliases on one frame without relying on uncalibrated SAM3 scores."""
    kept: list[dict] = []
    for item in observations:
        left = item["bbox"]
        duplicate = False
        for previous in kept:
            if previous["label"] != item["label"]:
                continue
            if _box_iou(left, previous["bbox"]) >= minimum_iou:
                duplicate = True
                break
        if not duplicate:
            kept.append(item)
    return kept






def _mask_bbox(mask: Any) -> list[float] | None:
    import numpy as np

    pixels = np.argwhere(mask)
    if not len(pixels):
        return None
    ymin, xmin = pixels.min(axis=0)
    ymax, xmax = pixels.max(axis=0)
    return [float(xmin), float(ymin), float(xmax + 1), float(ymax + 1)]


def _extract(archive: Path, root: Path) -> None:
    with zipfile.ZipFile(archive) as source:
        names = [item.filename for item in source.infolist()]
        if len(names) != len(set(names)) or any(Path(name).is_absolute() or ".." in Path(name).parts
                                                for name in names):
            raise ValueError("unsafe SAM3 benchmark archive")
        source.extractall(root)


def locate_payload(input_root: Path, work_root: Path) -> Path:
    """Find one Kaggle payload, whether Kaggle preserved or expanded its ZIP."""
    archives = list(input_root.glob("**/payload.zip"))
    expanded = [path.parent for path in input_root.glob("**/benchmark-config.json")
                if (path.parent / "frames.json").is_file()
                and (path.parent / "seed.json").is_file()]
    if len(archives) + len(expanded) != 1:
        raise RuntimeError("attach exactly one private SAM3 benchmark payload dataset")
    if expanded:
        return expanded[0]
    root = work_root / "sam3-benchmark-input"
    root.mkdir(parents=True, exist_ok=True)
    _extract(archives[0], root)
    return root


def main() -> None:
    global BENCHMARK_SESSION_STARTED_AT

    BENCHMARK_SESSION_STARTED_AT = time.monotonic()
    root = locate_payload(INPUT_ROOT, WORK_ROOT)
    if sys.version_info < (3, 12):
        raise RuntimeError("SAM3 requires Python 3.12+")
    configured = json.loads((root / "benchmark-config.json").read_text(encoding="utf-8"))
    validate_config(configured)
    # Kaggle Linux: preempt long native calls as well as checking between windows.
    # Partial result/checkpoint files are written after every completed window.
    import signal
    if hasattr(signal, "setitimer"):
        def deadline_expired(_signal, _frame):
            raise TimeoutError("SAM3 kernel total time budget reached")
        signal.signal(signal.SIGALRM, deadline_expired)
        signal.setitimer(signal.ITIMER_REAL, max(1, configured["max_total_seconds"] -
                        (time.monotonic() - BENCHMARK_SESSION_STARTED_AT)))
    variants = {row.get("model_variant", "sam3") for row in configured.get("configurations", [])}
    if len(variants) != 1:
        raise ValueError("one Kaggle kernel must use one SAM model variant")
    commit = "2345a4ad109ac29c569da749c91d84f10dc08c40"
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                    f"git+https://github.com/facebookresearch/sam3.git@{commit}", "timm>=1.0.17",
                    "ftfy==6.1.1", "iopath>=0.1.10", "portalocker", "huggingface_hub"], check=True)
    try:
        run_benchmark(root, WORK_ROOT / "result.json", Sam3Backend())
    finally:
        if hasattr(signal, "setitimer"):
            signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    main()
