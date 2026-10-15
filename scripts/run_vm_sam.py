"""Persistent single-job local SAM process with per-class/frame output events."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from kaggle.sam3_benchmark.main import _mask_bbox
from scripts.run_local_sam import LocalBackend
from src.services.video.stitching import encode_mask
from src.services.vm.contracts import VmProfile
from src.services.vm.storage import atomic_json


def emit(value: dict) -> None:
    sys.stdout.write(json.dumps(value, allow_nan=False) + "\n")
    sys.stdout.flush()


class StreamingBackend(LocalBackend):
    def run_window(
        self,
        root: Path,
        batch: list[dict],
        seeds: list[dict],
        row: dict,
        window_index: int,
        previous: list[dict],
        previous_masks: dict[str, Any],
    ) -> dict[int, list[dict]]:
        # Hard links reuse the pinned JPEG bytes; no second full window cache.
        with tempfile.TemporaryDirectory(prefix="vm-sam-window-", dir=self.work_root) as temporary:
            directory = Path(temporary)
            atomic_json(
                directory / "stage.json", {"owner": str(self.work_root.resolve()), "schema": "autolabel-stage/v1"}
            )
            for local, frame in enumerate(batch):
                os.link(root / frame["relative_path"], directory / f"{local:06d}.jpg")
            with self.torch.autocast("cuda", dtype=self.torch.bfloat16), self.torch.inference_mode():
                return self._text_window(directory, batch, row, window_index)

    def _text_window(self, directory: Path, batch: list[dict], row: dict, window_index: int) -> dict[int, list[dict]]:
        results: dict[int, list[dict]] = {item["source_frame_index"]: [] for item in batch}
        for prompt in row["text_prompts"]:
            started = time.perf_counter()
            self.torch.cuda.reset_peak_memory_stats()
            with contextlib.redirect_stdout(sys.stderr):
                state = self.model.init_state(resource_path=str(directory))
                self.model.add_prompt(inference_state=state, frame_idx=0, text_str=prompt)
                iterator = self.model.propagate_in_video(
                    state, start_frame_idx=0, max_frame_num_to_track=len(batch), reverse=False
                )
                seen: set[int] = set()
                try:
                    for local, output in iterator:
                        if type(local) is not int or local != len(seen) or local not in range(len(batch)):
                            raise ValueError("SAM frame mapping invalid")
                        seen.add(local)
                        observations = []
                        for object_id, mask, probability in zip(
                            output["out_obj_ids"], output["out_binary_masks"], output["out_probs"], strict=True
                        ):
                            array = (
                                self.numpy.asarray(
                                    mask.detach().cpu().numpy() if hasattr(mask, "detach") else mask
                                ).squeeze()
                                > 0
                            )
                            box = _mask_bbox(array)
                            if box is not None:
                                observations.append(
                                    {
                                        "track_id": f"text:{prompt}:window{window_index}:{int(object_id)}",
                                        "label": prompt,
                                        "bbox": box,
                                        "score": float(probability),
                                        "mask": asdict(encode_mask(array)),
                                    }
                                )
                        frame = batch[local]["source_frame_index"]
                        results[frame].extend(observations)
                        # emit outside redirect so only protocol JSON goes to stdout
                        with contextlib.redirect_stdout(self.protocol_stdout):
                            emit(
                                {
                                    "type": "frame",
                                    "window": window_index,
                                    "frame": frame,
                                    "class": prompt,
                                    "image_sha256": batch[local]["sha256"],
                                    "observations": observations,
                                }
                            )
                finally:
                    if hasattr(iterator, "close"):
                        iterator.close()
                if len(seen) != len(batch):
                    raise ValueError("SAM propagation omitted frames")
            del state
            self.torch.cuda.synchronize()
            self.class_timings.append(
                {
                    "window_id": window_index,
                    "label": prompt,
                    "seconds": time.perf_counter() - started,
                    "peak_cuda_allocated_bytes": self.torch.cuda.max_memory_allocated(),
                }
            )
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    args = parser.parse_args()
    # Remove only our abandoned hard-link aliases, not source media or model outputs.
    for directory in args.job_root.glob("vm-sam-window-*"):
        if directory.is_symlink() or directory.resolve().parent != args.job_root.resolve():
            raise ValueError("Unsafe staging directory")
        files = list(directory.iterdir())
        if not files:
            directory.rmdir()
            continue
        marker = directory / "stage.json"
        if not marker.exists() or json.loads(marker.read_text()) != {
            "owner": str(args.job_root.resolve()),
            "schema": "autolabel-stage/v1",
        }:
            raise ValueError("Unverified abandoned staging directory")
        if any(
            f.is_symlink()
            or not f.is_file()
            or (f.name != "stage.json" and not (f.suffix == ".jpg" and f.stem.isdigit()))
            for f in files
        ):
            raise ValueError("Unowned file in staging directory; retained")
        for file in files:
            file.unlink()
        directory.rmdir()
    profile = VmProfile.model_validate_json(args.profile.read_text())
    backend = StreamingBackend(
        Path(profile.checkpoint), profile.checkpoint_sha256, args.job_root, profile.source_commit
    )
    backend.protocol_stdout = sys.stdout
    backend.vm_profile = profile
    config = args.job_root / "benchmark-config.json"
    atomic_json(config, {"configurations": [{"model_variant": profile.variant}]})
    with contextlib.redirect_stdout(sys.stderr):
        backend.load(args.job_root)
        torch = backend.torch
        if torch.cuda.get_device_properties(0).major < 8:
            raise ValueError("VM pilot bf16 requires Ampere or newer CUDA GPU")
        torch.cuda.set_per_process_memory_fraction(profile.vram_fraction)
        torch.backends.cuda.matmul.allow_tf32 = profile.tf32
        backend.model.score_threshold_detection = profile.detection_threshold
        backend.model.new_det_thresh = profile.new_detection_threshold
    emit({"type": "loaded", "checkpoint_sha256": profile.checkpoint_sha256, "source_commit": profile.source_commit})
    for line in sys.stdin:
        command = json.loads(line)
        if command["type"] == "close":
            break
        try:
            frames = command["frames"]
            result = backend.run_window(
                args.job_root,
                frames,
                [],
                {"id": "vm", "mode": "text_prompt", "text_prompts": command["classes"]},
                command["window"],
                [],
                {},
            )
            emit(
                {
                    "type": "complete",
                    "window": command["window"],
                    "identity": command["identity"],
                    "frames": list(result),
                    "runtime": backend.probe(),
                    "profile": profile.model_dump(mode="json", by_alias=True),
                }
            )
        except Exception:
            emit(
                {
                    "type": "error",
                    "window": command.get("window"),
                    "error": "SAM window failed; inspect private GPU log",
                }
            )
            raise


if __name__ == "__main__":
    main()
