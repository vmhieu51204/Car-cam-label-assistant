"""Run the canonical SAM window engine locally; no pip install or remote calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from kaggle.runtime_probe import sam3_commit, sha256_file
from kaggle.sam3_benchmark.main import Sam3Backend, run_benchmark


class MultiplexBridge:
    """Map official predictor sessions to the original dense-window interface."""
    def __init__(self, predictor: Any):
        self.predictor = predictor

    @property
    def score_threshold_detection(self) -> float:
        return float(self.predictor.model.score_threshold_detection)

    @score_threshold_detection.setter
    def score_threshold_detection(self, value: float) -> None:
        self.predictor.model.score_threshold_detection = value

    @property
    def new_det_thresh(self) -> float:
        return float(self.predictor.model.new_det_thresh)

    @new_det_thresh.setter
    def new_det_thresh(self, value: float) -> None:
        self.predictor.model.new_det_thresh = value

    def init_state(self, *, resource_path: str) -> str:
        return self.predictor.handle_request({"type": "start_session", "resource_path": resource_path})["session_id"]

    def add_prompt(self, *, inference_state: str, frame_idx: int, text_str: str) -> Any:
        try:
            return self.predictor.handle_request({"type": "add_prompt", "session_id": inference_state,
                                                  "frame_index": frame_idx, "text": text_str})
        except Exception:
            self.predictor.handle_request({"type": "close_session", "session_id": inference_state})
            raise

    def propagate_in_video(self, state: str, **kwargs: Any) -> Any:
        try:
            for item in self.predictor.handle_stream_request({"type": "propagate_in_video", "session_id": state}):
                yield item["frame_index"], item["outputs"]
        finally:
            self.predictor.handle_request({"type": "close_session", "session_id": state})

    def handle_request(self, request: dict) -> Any:
        return self.predictor.handle_request(request)


class LocalBackend(Sam3Backend):
    def __init__(self, checkpoint: Path, digest: str, output: Path, commit: str):
        super().__init__(output)
        self.checkpoint, self.checkpoint_sha256 = checkpoint, digest.lower()
        self.expected_commit = commit.lower()

    def load(self, root: Path) -> None:
        import numpy as np
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Local SAM requires CUDA; no CPU fallback")
        if sam3_commit() != self.expected_commit:
            raise ValueError("installed SAM source commit differs from LOCAL_SAM_COMMIT")
        assert self.checkpoint is not None
        config = json.loads((root / "benchmark-config.json").read_text())
        variants = {r.get("model_variant", "sam3") for r in config["configurations"]}
        if len(variants) != 1:
            raise ValueError("cannot mix SAM variants in one local run")
        self.model_variant = variants.pop()
        expected_name = "sam3.1_multiplex.pt" if self.model_variant == "sam3.1" else "sam3.pt"
        if self.checkpoint.name != expected_name or sha256_file(self.checkpoint) != self.checkpoint_sha256:
            raise ValueError("checkpoint filename or SHA-256 mismatch")
        self.torch, self.numpy = torch, np
        profile = getattr(self, "vm_profile", None)
        if profile is not None:
            if torch.cuda.get_device_properties(0).major < 8:
                raise ValueError("VM bf16 profile requires Ampere or newer")
            torch.cuda.set_per_process_memory_fraction(profile.vram_fraction)
            torch.backends.cuda.matmul.allow_tf32 = profile.tf32
        if self.model_variant == "sam3.1":
            from sam3.model_builder import build_sam3_multiplex_video_predictor
            kwargs = {}
            if profile is not None:
                import inspect
                required = {"max_num_objects"}
                if not required <= set(inspect.signature(build_sam3_multiplex_video_predictor).parameters):
                    raise ValueError("SAM builder cannot honor the pinned VM profile")
                kwargs = {"max_num_objects": profile.max_objects}
            self.model = MultiplexBridge(build_sam3_multiplex_video_predictor(
                checkpoint_path=str(self.checkpoint), use_fa3=False, compile=False, warm_up=False, **kwargs))
            if profile is not None:
                if not hasattr(self.model.predictor.model, "batched_grounding_batch_size"):
                    raise ValueError("SAM model cannot honor batched grounding profile")
                self.model.predictor.model.batched_grounding_batch_size = profile.grounding_batch_size
        else:
            from sam3.model_builder import build_sam3_video_model

            original = build_sam3_video_model(checkpoint_path=str(self.checkpoint), load_from_HF=False)
            original.tracker.backbone = original.detector.backbone
            self.model = original


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    run_benchmark(args.input, args.output, LocalBackend(args.checkpoint, args.sha256, args.output, args.commit))


if __name__ == "__main__":
    main()
