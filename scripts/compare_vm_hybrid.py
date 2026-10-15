"""Offline three-mode comparison on identical redacted images/SAM profile.

Reads immutable model checkpoints, not human edits or unaccepted VLM proposals.
No model/provider calls. Mask GT is optional and never synthesized from boxes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from scripts.eval_autolabel import Box
from src.services.evaluation.manual import load_manual_reference
from src.services.evaluation.phase0 import canonical_predictions
from src.services.evaluation.tracking import tracking_diagnostics
from src.services.video.stitching import MaskRLE, decode_mask
from src.services.vm.contracts import VmProfile, file_sha256
from src.services.vm.storage import atomic_json, encode
from src.services.vm.streaming import validate_event


def read_run(root: Path) -> dict:
    profile = VmProfile.model_validate_json((root / "profile.json").read_text())
    manifest = json.loads((root / "public/input-manifest.json").read_text())
    coverage = json.loads((root / "public/coverage.json").read_text())
    if VmProfile.model_validate(coverage["profile"]) != profile:
        raise ValueError("Coverage profile differs from model snapshot")
    records = {row["frame_index"]: row for row in manifest["frames"]}
    classes = list(coverage["class_coverage"])
    if not coverage["complete"] or set(records) != set(range(coverage["total"])):
        raise ValueError("Compare completed, fully covered jobs only")
    events: dict[tuple[int, str], dict] = {}
    runtime, hashes = [], []
    for path in sorted((root / "checkpoints").glob("*.json")):
        if not path.stem.isdecimal():
            continue
        checkpoint = json.loads(path.read_text())
        selected = [records[i] for i in checkpoint["completion"]["frames"]]
        identity = hashlib.sha256(
            encode(
                {
                    "frames": selected,
                    "profile": coverage["profile"],
                    "source": manifest["redacted_sha256"],
                    "code": coverage["code_sha256"],
                }
            )
        ).hexdigest()
        if checkpoint["identity"] != identity or checkpoint["completion"]["identity"] != identity:
            raise ValueError("Model checkpoint source/profile identity differs")
        hashes.append(file_sha256(path))
        runtime.append(checkpoint["completion"].get("runtime", {}))
        for entry in checkpoint["events"]:
            event_path = (root / entry["path"]).resolve()
            if event_path.parent != (root / "events").resolve() or file_sha256(event_path) != entry["sha256"]:
                raise ValueError("Model event path or checksum differs")
            event = json.loads(event_path.read_text())
            validate_event(event, records, classes, profile)
            event["observations"] = [
                {**row, "track_id": checkpoint["mapping"].get(row["track_id"], row["track_id"])}
                for row in event["observations"]
            ]
            # Earlier overlap output is what the incremental writer published.
            events.setdefault((event["frame"], event["class"]), event)
    if set(events) != {(i, c) for i in records for c in classes}:
        raise ValueError("Model class/frame event coverage incomplete")
    image_identity = [{k: row[k] for k in ("frame_index", "sha256", "width", "height")} for row in manifest["frames"]]
    return {
        "profile": profile.model_dump(mode="json"),
        "manifest": manifest,
        "records": records,
        "images_sha256": hashlib.sha256(encode(image_identity)).hexdigest(),
        "events": events,
        "runtime": runtime,
        "checkpoint_sha256": hashes,
        "classes": classes,
    }


def mask_metrics(run: dict, truth: dict) -> dict:
    if truth.get("schema") != "autolabel-manual-masks/v1" or truth.get("images_sha256") != run["images_sha256"]:
        raise ValueError("Mask GT must identify the same redacted source images")
    matches, total_gt, total_pred, ious = 0, 0, 0, []
    seen = set()
    for row in truth["frames"]:
        key = (row["frame"], row["class"])
        if key in seen or key not in run["events"]:
            raise ValueError("Mask GT frame/class is duplicated or outside run coverage")
        seen.add(key)

        def decode(mask: dict) -> np.ndarray:
            result = decode_mask(MaskRLE(**{**mask, "counts": tuple(mask["counts"])}))
            record = run["records"][row["frame"]]
            if result.shape != (record["height"], record["width"]):
                raise ValueError("Mask GT dimensions differ")
            return result

        gt = [decode(mask) for mask in row["masks"]]
        predictions = [decode(obs["mask"]) for obs in run["events"][key]["observations"]]
        total_gt += len(gt)
        total_pred += len(predictions)
        if not gt or not predictions:
            continue
        matrix = np.array(
            [[np.count_nonzero(a & b) / max(1, np.count_nonzero(a | b)) for b in predictions] for a in gt]
        )
        # Maximize eligible pairs first, then IoU. A larger subthreshold overlap
        # must not steal a prediction from an otherwise valid 0.5 match.
        left, right = linear_sum_assignment(-(min(matrix.shape) + 1) * (matrix >= 0.5) - matrix)
        selected = [float(matrix[i, j]) for i, j in zip(left, right, strict=True) if matrix[i, j] >= 0.5]
        matches += len(selected)
        ious.extend(selected)
    return {
        "matched_mask_mean_iou_at_0_5": float(np.mean(ious)) if ious else None,
        "tp": matches,
        "fp": total_pred - matches,
        "fn": total_gt - matches,
        "declared_frame_classes": len(seen),
    }


def compare(roots: list[Path], ground_truth: Path, masks: Path | None = None) -> dict:
    runs = [read_run(root) for root in roots]
    if len(runs) != 3 or {r["profile"]["pipeline"] for r in runs} != {"sam", "hybrid_frame", "hybrid_propagate"}:
        raise ValueError("Provide one completed run for each of the three modes")

    def common(r: dict) -> dict:
        return {k: v for k, v in r["profile"].items() if k not in ("pipeline", "hybrid")}

    if any(
        r["images_sha256"] != runs[0]["images_sha256"]
        or r["classes"] != runs[0]["classes"]
        or common(r) != common(runs[0])
        or r["manifest"]["redacted_sha256"] != runs[0]["manifest"]["redacted_sha256"]
        for r in runs
    ):
        raise ValueError(
            "Use identical redacted images, classes and SAM profile; original-image baselines are not equivalent"
        )
    hybrids = [r["profile"]["hybrid"] for r in runs if r["profile"]["pipeline"] != "sam"]
    if hybrids[0] != hybrids[1]:
        raise ValueError("Hybrid detector/tracker profiles differ")
    gpu_names = {
        runtime.get("gpu")
        if isinstance(runtime.get("gpu"), str)
        else (runtime.get("environment", {}).get("gpu") or {}).get("name")
        for run in runs
        for runtime in run["runtime"]
    }
    known_gpu = gpu_names - {None}
    if len(known_gpu) > 1:
        raise ValueError("GPU devices differ across comparison runs")
    reference, occlusions = load_manual_reference(ground_truth)
    if any(box.class_name not in runs[0]["classes"] for box in reference.boxes):
        raise ValueError("Ground truth classes differ from the frozen job classes")
    if len({f.key.sequence.sequence_uid for f in reference.frames}) != 1:
        raise ValueError("One video per hybrid comparison")
    keys = {f.source_ordinal: f.key for f in reference.frames}
    for f in reference.frames:
        record = runs[0]["records"].get(f.source_ordinal)
        if record is None or (f.width, f.height) != (record["width"], record["height"]):
            raise ValueError("GT frame or source dimensions differ")
    result = {}
    for run in runs:
        mode = run["profile"]["pipeline"]
        boxes = [
            Box(event["frame"], row["label"], tuple(row["bbox"]), row["track_id"])
            for event in run["events"].values()
            if event["frame"] in keys
            for row in event["observations"]
        ]
        result[mode] = {
            "boxes_and_tracking": tracking_diagnostics(
                reference.boxes,
                canonical_predictions(boxes, keys, mode),
                frames=tuple(keys.values()),
                occlusions=occlusions,
                tracking_ids_available=all(b.native_track_id is not None for b in reference.boxes),
            ),
            "mask_metrics": mask_metrics(run, json.loads(masks.read_text()))
            if masks
            else "unavailable_without_mask_GT",
            "runtime_by_window": run["runtime"],
            "checkpoint_sha256": run["checkpoint_sha256"],
        }
    return {
        "schema": "autolabel-hybrid-comparison/v1",
        "images_sha256": runs[0]["images_sha256"],
        "ground_truth_sha256": file_sha256(ground_truth),
        "modes": result,
        "human_correction_effort": "not_measured",
        "same_gpu_verified_from_runtime": bool(known_gpu) and None not in gpu_names,
        "note": "Model output on identical redacted frames. Box diagnostics are not official KITTI/HOTA. Runtime requires VM pilot evidence.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-root", type=Path, nargs=3, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--masks", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    if any(
        root.resolve() == destination or root.resolve() in destination.parents for root in args.job_root
    ) or destination in (args.ground_truth.resolve(), args.masks.resolve() if args.masks else None):
        raise ValueError("Write comparison to a separate output location")
    atomic_json(destination, compare(args.job_root, args.ground_truth, args.masks))


if __name__ == "__main__":
    main()
