"""Experimental scene-continuous SAM3; checkpoints contain outputs, never memory.

Use one propagation generator per class/scene. Reopening the generator at each
checkpoint would reset SAM3's hot-start/output-confirmation buffers. No internal
state fields are copied or deleted. A failed scene must be replayed in full.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


def scenes(frames: list[dict]) -> list[list[dict]]:
    result: list[list[dict]] = []
    for frame in frames:
        if not result or frame.get("scene_cut_before"):
            result.append([])
        result[-1].append(frame)
    return result


def run_native(backend: Any, root: Path, frames: list[dict], row: dict,
               item: dict, *, save: Callable[[], None], deadline: float,
               identity: str, model: dict, clock: Callable[[], float] = time.monotonic) -> None:
    """Keep bounded scenes on CPU; fail explicitly on missing/duplicate outputs."""
    from kaggle.sam3_benchmark.main import _mask_bbox, _rle

    if not all(callable(getattr(backend.model, name, None))
               for name in ("init_state", "add_prompt", "propagate_in_video")):
        raise NotImplementedError("SAM3 lacks native scene video API")
    item.update(resume_policy="replay_entire_native_config_no_native_state_checkpoint",
                native_state_checkpoint=False, predictions=[], native_checkpoints=[],
                evaluated_source_frames=[], window_predictions=[])
    all_frames: set[int] = set()
    timings: list[dict] = []
    started = clock()
    for scene_id, batch in enumerate(scenes(frames)):
        if len(batch) > row.get("max_scene_frames", 250):
            raise ValueError("scene exceeds experimental max_scene_frames; no implicit reset")
        directory = backend.work_root / f"native-{row['id']}-scene{scene_id}"
        directory.mkdir(parents=True, exist_ok=True)
        try:
            for local, frame in enumerate(batch):
                shutil.copyfile(root / frame["relative_path"], directory / f"{local:06d}.jpg")
            scene_seen: list[set[int]] = []
            for prompt in row["text_prompts"]:
                seen: set[int] = set()
                tick = clock()
                state = None
                generator = None
                pending: list[dict] = []
                pending_frames: list[int] = []
                backend.torch.cuda.reset_peak_memory_stats()

                def checkpoint() -> None:
                    entry = dict(schema="autolabel-native-output/v1", scene_id=scene_id,
                                 prompt=prompt, input_sha256=identity, model=model,
                                 frame_ids=list(pending_frames), predictions=list(pending),
                                 native_state_checkpoint=False,
                                 resume_policy="replay_entire_native_config")
                    entry["output_sha256"] = hashlib.sha256(json.dumps(
                        entry, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                    item["native_checkpoints"].append(entry)
                    pending.clear()
                    pending_frames.clear()
                    save()

                try:
                    dtype = (backend.torch.bfloat16 if backend.torch.cuda.get_device_properties(0).major >= 8
                             else backend.torch.float16)
                    with backend.torch.autocast("cuda", dtype=dtype), backend.torch.inference_mode():
                        state = backend.model.init_state(resource_path=str(directory),
                            offload_video_to_cpu=True, offload_state_to_cpu=True)
                        backend.model.add_prompt(inference_state=state, frame_idx=0, text_str=prompt)
                        generator = backend.model.propagate_in_video(state, start_frame_idx=0,
                            max_frame_num_to_track=len(batch) - 1, reverse=False)
                        for local, output in generator:
                            if clock() >= deadline:
                                raise TimeoutError("native scene deadline reached; replay scene on retry")
                            if local != len(seen) or local >= len(batch):
                                raise ValueError("native propagation has duplicate/out-of-order frame")
                            seen.add(local)
                            source = batch[local]["source_frame_index"]
                            pending_frames.append(source)
                            for obj, mask, score in zip(output["out_obj_ids"], output["out_binary_masks"],
                                                        output["out_probs"], strict=True):
                                array = backend.numpy.asarray(mask.detach().cpu().numpy()
                                    if hasattr(mask, "detach") else mask).squeeze() > 0
                                box = _mask_bbox(array)
                                if box is not None:
                                    observation = dict(frame_index=source,
                                        track_id=f"native:{prompt}:scene{scene_id}:{int(obj)}",
                                        label=(row.get("prompt_to_label") or {}).get(prompt, prompt),
                                        bbox=box, score=float(score), outside=False, mask_rle=_rle(array))
                                    pending.append(observation)
                                    item["predictions"].append(observation)
                            if len(pending_frames) == row["window_size"]:
                                checkpoint()
                        if seen != set(range(len(batch))):
                            raise ValueError("native propagation did not observe every scene frame")
                finally:
                    try:
                        if pending_frames:
                            checkpoint()
                    finally:
                        if generator is not None:
                            generator.close()
                        del state
                        backend.torch.cuda.synchronize()
                        timings.append(dict(scene_id=scene_id, label=prompt, seconds=clock() - tick,
                            observed_frames=len(seen), peak_cuda_allocated_bytes=backend.torch.cuda.max_memory_allocated()))
                        backend.class_timings = timings
                        item["runtime"] = dict(backend.probe(), timing_s=dict(
                            classes=timings, total_inference=clock() - started))
                        backend.recover()
                    save()
                scene_seen.append(seen)
            if all(x == set(range(len(batch))) for x in scene_seen):
                all_frames.update(f["source_frame_index"] for f in batch)
                item["evaluated_source_frames"] = sorted(all_frames)
                save()
        finally:
            shutil.rmtree(directory)
    item.update(status="completed", unique_frame_count=len(frames), window_frame_count=len(frames),
                resumed_windows=[])
