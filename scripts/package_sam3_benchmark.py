"""Build a private SAM3 benchmark payload from original frames, keyframes and GT seed."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw

from kaggle.sam3_benchmark.main import validate_config
from scripts.eval_autolabel import read_boxes


def _seed_masks(gt: Path, first_image_size: tuple[int, int]) -> dict[str, bytes]:
    if gt.suffix.lower() != ".json":
        return {}
    data = json.loads(gt.read_text(encoding="utf-8"))
    if not {"images", "annotations"} <= data.keys():
        return {}
    first_ids = {image["id"] for image in data["images"] if image.get("frame_index", image["id"]) == 0}
    masks = {}
    for annotation in data["annotations"]:
        polygons = annotation.get("segmentation")
        track_id = annotation.get("track_id")
        if annotation.get("image_id") not in first_ids or track_id is None or not isinstance(polygons, list):
            continue
        mask = Image.new("L", first_image_size, 0)
        draw = ImageDraw.Draw(mask)
        for polygon in polygons:
            if isinstance(polygon, list) and len(polygon) >= 6 and len(polygon) % 2 == 0:
                draw.polygon(list(zip(polygon[::2], polygon[1::2])), fill=255)
        buffer = io.BytesIO()
        mask.save(buffer, format="PNG")
        masks[str(track_id)] = buffer.getvalue()
    return masks


def package(
    *,
    frames_dir: Path,
    manifest_path: Path,
    ground_truth: Path,
    config_path: Path,
    output: Path,
    max_bytes: int = 2_000_000_000,
    include_dense: bool = False,
) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if include_dense:
        config["configurations"].append(config.pop("optional_dense_configuration"))
    validate_config(config)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    limit = config["source_frame_limit"]
    optical = sorted(
        {
            frame.get("source_frame_index", frame["frame_index"])
            for frame in manifest["frames"]
            if frame.get("source_frame_index", frame["frame_index"]) < limit
        }
    )
    if not optical or optical[0] != 0:
        raise ValueError("optical keyframes must include source frame zero")
    selected = set(optical)
    for row in config["configurations"]:
        if row["selection"] == "uniform":
            selected.update(range(0, limit, row["gap"]))
        elif row["selection"] == "dense":
            selected.update(range(limit))
    source_paths = {index: frames_dir / f"frame_{index:06d}.jpg" for index in selected}
    if any(not path.is_file() or path.is_symlink() for path in source_paths.values()):
        missing = next(index for index, path in sorted(source_paths.items()) if not path.is_file() or path.is_symlink())
        raise FileNotFoundError(f"missing original source frame {missing}")
    frame_zero = source_paths[0]
    with Image.open(frame_zero) as opened:
        size = opened.size
    seeds = [box for box in read_boxes(ground_truth) if box.frame == 0]
    if not seeds or any(seed.track_id is None for seed in seeds):
        raise ValueError("ground truth needs frame-0 boxes with stable track IDs")
    if len({seed.track_id for seed in seeds}) != len(seeds):
        raise ValueError("frame-0 seed track IDs must be unique")
    mask_bytes = _seed_masks(ground_truth, size)
    seed_payload = {
        "seeds": [
            {
                "track_id": box.track_id,
                "label": box.label,
                "bbox": list(box.xyxy),
                **({"mask_file": f"seed_masks/seed_{index:04d}.png"} if box.track_id in mask_bytes else {}),
            }
            for index, box in enumerate(seeds, start=1)
        ]
    }
    frame_records = []
    total_bytes = 0
    for index, path in sorted(source_paths.items()):
        total_bytes += path.stat().st_size
        with Image.open(path) as opened:
            width, height = opened.size
        frame_records.append(
            {
                "source_frame_index": index,
                "relative_path": f"frames/frame_{index:06d}.jpg",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "width": width,
                "height": height,
            }
        )
    if total_bytes > max_bytes:
        raise ValueError(f"frame payload {total_bytes} bytes exceeds {max_bytes}; reduce source_frame_limit")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("benchmark-config.json", json.dumps(config))
        archive.writestr("frames.json", json.dumps({"frames": frame_records, "optical_source_indices": optical}))
        archive.writestr("seed.json", json.dumps(seed_payload))
        for record in frame_records:
            archive.write(source_paths[record["source_frame_index"]], record["relative_path"])
        for index, box in enumerate(seeds, start=1):
            if box.track_id in mask_bytes:
                archive.writestr(f"seed_masks/seed_{index:04d}.png", mask_bytes[box.track_id])
    return {
        "source_frames_packaged": len(frame_records),
        "optical_keyframes": len(optical),
        "frame_zero_seeds": len(seeds),
        "ground_truth_masks": len(mask_bytes),
        "uncompressed_frame_bytes": total_bytes,
        "zip_bytes": output.stat().st_size,
        "archive": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("kaggle/sam3_benchmark/benchmark-config.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-bytes", type=int, default=2_000_000_000)
    parser.add_argument("--include-dense", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            package(
                frames_dir=args.frames_dir,
                manifest_path=args.manifest,
                ground_truth=args.ground_truth,
                config_path=args.config,
                output=args.output,
                max_bytes=args.max_bytes,
                include_dense=args.include_dense,
            )
        )
    )


if __name__ == "__main__":
    main()
