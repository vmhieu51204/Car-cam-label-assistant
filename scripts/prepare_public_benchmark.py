"""Prepare SAM3 and future VLM payloads for two validated, licensed local sequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

from scripts.import_public_gt import _digest
from scripts.package_sam3_benchmark import package as package_sam3
from scripts.validate_public_gt import validate

ROOT = Path(__file__).resolve().parents[1]


def prepare(segment: Path, *, max_bytes: int = 2_000_000_000, include_vlm: bool = False) -> dict:
    report = validate(segment)
    if report["result"] != "passed":
        raise ValueError("public ground truth did not validate")
    manifest = json.loads((segment / "manifest.json").read_text(encoding="utf-8"))
    names = sorted(manifest["classes"])
    config = json.loads((ROOT / "kaggle/sam3_benchmark/benchmark-config.json").read_text(encoding="utf-8"))
    config["source_frame_limit"] = manifest["frame_count"]
    if manifest["dataset"] == "kitti":
        # KITTI images were already reduced from assumed 10 fps to 5 fps by the
        # converter. Compare useful 5 / 2.5 / 1.67 / 1 fps densities here.
        for row, gap in zip((item for item in config["configurations"] if item["selection"] == "uniform"), (2, 3, 5)):
            row["gap"] = gap
            row["id"] = f"uniform-{gap}-window-{row['window_size']}"
    for row in config["configurations"]:
        if row["mode"] == "text_prompt":
            row["text_prompts"] = names
    config["note"] = (
        "Input images are already at target_fps; 'optical' selection means all converted "
        "frames, and uniform gaps are multiples of that target_fps. KITTI source fps "
        "is assumed 10 unless overridden, as annotation timestamps are absent."
    )
    config_path = segment / "benchmark-config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    # The SAM3 benchmark indexes the converted 5-fps timeline. The original source mapping
    # remains in the public manifest and is not silently used as a local JPEG file index.
    benchmark_manifest = segment / "benchmark-manifest.json"
    benchmark_manifest.write_text(
        json.dumps(
            {
                "frames": [
                    {"frame_index": row["frame_index"], "source_frame_index": row["frame_index"]}
                    for row in manifest["frames"]
                ]
            }
        ),
        encoding="utf-8",
    )
    archive = segment / "sam3-payload.zip"
    sam3 = package_sam3(
        frames_dir=segment / "frames",
        manifest_path=benchmark_manifest,
        ground_truth=segment / "ground_truth.coco.json",
        config_path=config_path,
        output=archive,
        max_bytes=max_bytes,
    )
    attribution = {
        "dataset": manifest["dataset"],
        "source": manifest["format_url"],
        "license": manifest["license_url"],
        "license_note": manifest["license_note"],
        "changes": "selected frames downsampled; KITTI PNG frames converted to JPEG when applicable",
        "use": "private non-commercial evaluation only",
    }
    with ZipFile(archive, "a") as packed:
        packed.writestr("dataset-attribution.json", json.dumps(attribution, indent=2))
    sam3["zip_bytes"] = archive.stat().st_size
    result = {
        "schema": "public-benchmark-payloads/v1",
        "segment": manifest["sequence"],
        "sam3": {**sam3, "sha256": _digest(archive)},
        "text_prompts": names,
        "license_note": manifest["license_note"],
    }
    if include_vlm:
        raise ValueError("Kaggle VLM benchmark is retired; use Claude benchmark")
    (segment / "payloads.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=Path("eval/public/bdd100k/selection.json"))
    parser.add_argument("--max-bytes", type=int, default=2_000_000_000)
    parser.add_argument("--sam3-only", action="store_true", help="Skip VLM fault images and its larger payload")
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    for role in ("hard", "control"):
        result = prepare(Path(selection[role]["path"]), max_bytes=args.max_bytes, include_vlm=False)
        print(json.dumps({"role": role, **result}, indent=2))


if __name__ == "__main__":
    main()
