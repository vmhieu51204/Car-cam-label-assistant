"""Package one unlabeled image sequence for SAM3 text prompts on Kaggle.

The output is compatible with the existing SAM3 benchmark notebook but has no
ground-truth seed. Every configuration is text-prompt-only. The private archive
contains sampled frames, exact source-index mapping, checksums, and one config.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import tempfile
import zipfile
from pathlib import Path

from PIL import Image

from kaggle.sam3_benchmark.main import validate_config


def _images(directory: Path) -> list[Path]:
    files = sorted(path for path in directory.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not files or [path.stem for path in files] != [f"{index:06d}" for index in range(len(files))]:
        raise ValueError("sequence must contain contiguous 000000.jpg/png frames")
    return files


def package_sequence(
    *,
    images: Path,
    output: Path,
    source_fps: float,
    target_fps: float = 5.0,
    classes: tuple[str, ...] = ("car", "person"),
    window_size: int = 50,
    overlap: int = 5,
    max_bytes: int = 2_000_000_000,
    fps_source: str = "operator",
) -> dict:
    if (
        not math.isfinite(source_fps)
        or not math.isfinite(target_fps)
        or not 0 < target_fps <= source_fps
        or max_bytes < 1
        or not fps_source.strip()
    ):
        raise ValueError("source/target FPS or payload budget is invalid")
    files = _images(images)
    source_indices = sorted(
        {
            min(len(files) - 1, round(index * source_fps / target_fps))
            for index in range(math.ceil(len(files) * target_fps / source_fps))
        }
        | {0, len(files) - 1}
    )
    if len(source_indices) < 2:
        raise ValueError("SAM3 sequence needs at least two selected frames")
    config = {
        "schema": "autolabel-sam3-benchmark-config/v1",
        "max_total_seconds": 21_600,
        "max_config_seconds": 20_000,
        "source_frame_limit": len(source_indices),
        "configurations": [
            {
                "id": "sequence-text",
                "selection": "dense",
                "gap": None,
                "window_size": window_size,
                "overlap": overlap,
                "mode": "text_prompt",
                "seed_type": "none",
                "reprompt": "none",
                "text_prompts": list(classes),
                "preserve_overlap": True,
            }
        ],
    }
    validate_config(config)
    records = []
    total = 0
    expected_size = None
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".zip", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for selected, source in enumerate(source_indices):
                with Image.open(files[source]) as opened:
                    rgb = opened.convert("RGB")
                    if expected_size is None:
                        expected_size = rgb.size
                    if rgb.size != expected_size:
                        raise ValueError("input images have inconsistent dimensions")
                    buffer = io.BytesIO()
                    rgb.save(buffer, format="JPEG", quality=95)
                data = buffer.getvalue()
                total += len(data)
                if total > max_bytes:
                    raise ValueError("sampled sequence exceeds max_bytes; lower target_fps")
                record = {
                    "source_frame_index": selected,
                    "original_source_index": source,
                    "timestamp_s": source / source_fps,
                    "relative_path": f"frames/frame_{selected:06d}.jpg",
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "width": expected_size[0],
                    "height": expected_size[1],
                }
                records.append(record)
                archive.writestr(record["relative_path"], data)
            archive.writestr("benchmark-config.json", json.dumps(config))
            archive.writestr(
                "frames.json",
                json.dumps(
                    {
                        "frames": records,
                        "optical_source_indices": list(range(len(records))),
                        "source_fps": source_fps,
                        "target_fps": target_fps,
                        "fps_source": fps_source,
                    }
                ),
            )
            archive.writestr("seed.json", json.dumps({"seeds": []}))
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "archive": str(output),
        "source_frames": len(files),
        "selected_frames": len(records),
        "uncompressed_frame_bytes": total,
        "zip_bytes": output.stat().st_size,
        "source_indices": source_indices,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-fps", type=float, required=True)
    parser.add_argument("--target-fps", type=float, default=5)
    parser.add_argument(
        "--fps-source", default="operator", help="Provenance of --source-fps, e.g. ffprobe or KITTI fallback"
    )
    parser.add_argument("--classes", default="car,person")
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--overlap", type=int, default=5)
    parser.add_argument("--max-bytes", type=int, default=2_000_000_000)
    args = parser.parse_args()
    report = package_sequence(
        images=args.images,
        output=args.output,
        source_fps=args.source_fps,
        target_fps=args.target_fps,
        classes=tuple(item.strip() for item in args.classes.split(",")),
        window_size=args.window_size,
        overlap=args.overlap,
        max_bytes=args.max_bytes,
        fps_source=args.fps_source,
    )
    print(json.dumps({key: value for key, value in report.items() if key != "source_indices"}))


if __name__ == "__main__":
    main()
