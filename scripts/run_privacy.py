"""EgoBlur Gen1 on the VM GPU; bounded decode/encode, no download or CPU fallback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.services.vm.contracts import file_sha256
from src.services.vm.privacy import redact_video
from src.services.vm.storage import atomic_json


class EgoBlur:
    def __init__(self, paths: list[Path], hashes: list[str]):
        import torch

        # Register torchvision custom ops before deserializing EgoBlur TorchScript.
        import torchvision  # noqa: F401

        if not torch.cuda.is_available():
            raise ValueError("Privacy requires CUDA on the VM")
        if len(paths) != 2 or any(file_sha256(p) != h for p, h in zip(paths, hashes, strict=True)):
            raise ValueError("Both privacy checkpoint hashes must match")
        self.torch = torch
        self.models = [torch.jit.load(str(path), map_location="cpu").eval().to("cuda") for path in paths]

    def detect(self, bgr: Any) -> list[list[float]]:
        from torchvision.ops import nms

        tensor = self.torch.from_numpy(bgr.transpose(2, 0, 1).copy()).to("cuda")
        result = []
        with self.torch.inference_mode():
            for model in self.models:
                boxes, _labels, scores, _dims = model(tensor)
                indexes = nms(boxes, scores, 0.3)
                result.extend(boxes[indexes][scores[indexes] > 0.1].cpu().tolist())
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    detector = EgoBlur(
        [Path(request["face"]), Path(request["plate"])], [request["face_sha256"], request["plate_sha256"]]
    )

    def progress(current: int, total: int) -> None:
        atomic_json(args.request.with_name("privacy-progress.json"), {"current": current, "total": total})

    redact_video(
        Path(request["source"]),
        Path(request["output"]),
        detector,
        expected_sha256=request["source_sha256"],
        manual_regions=request.get("regions", []),
        progress=progress,
        ffmpeg=request.get("ffmpeg", "ffmpeg"),
        ffprobe=request.get("ffprobe", "ffprobe"),
        model_metadata={key: request[key] for key in ("face_sha256", "plate_sha256", "source_commit")},
    )


if __name__ == "__main__":
    main()
