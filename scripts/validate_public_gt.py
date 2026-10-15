"""Validate converted images/CVAT/COCO and the offline tracking evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image

from scripts.eval_autolabel import Box, read_boxes, tracking_metrics


def _key(box: Box) -> tuple:
    return box.frame, box.label, box.track_id


def validate(segment: Path, *, seed: int = 42) -> dict:
    manifest = json.loads((segment / "manifest.json").read_text(encoding="utf-8"))
    coco_path = segment / "ground_truth.coco.json"
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    frames = manifest["frames"]
    if [frame["frame_index"] for frame in frames] != list(range(len(frames))):
        raise ValueError("manifest frame indices are not contiguous")
    if any(right["timestamp_s"] <= left["timestamp_s"] for left, right in zip(frames, frames[1:])):
        raise ValueError("manifest timestamps are not increasing")
    if any(right["source_frame_index"] <= left["source_frame_index"] for left, right in zip(frames, frames[1:])):
        raise ValueError("source frame mapping is not increasing")
    image_map = {image["id"]: image for image in coco["images"]}
    if len(image_map) != len(frames):
        raise ValueError("COCO image count does not match manifest")
    for frame in frames:
        image = image_map.get(frame["frame_index"])
        if image is None or image["frame_index"] != frame["frame_index"]:
            raise ValueError("COCO image indices do not match manifest")
        path = segment / "frames" / frame["file_name"]
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"missing or linked converted frame: {path}")
        with Image.open(path) as opened:
            if opened.size != (frame["width"], frame["height"]):
                raise ValueError(f"image size mismatch at frame {frame['frame_index']}")
        if (image["width"], image["height"]) != (frame["width"], frame["height"]):
            raise ValueError("COCO dimensions do not match manifest")
        converted_digest = manifest.get("converted_files_sha256", {}).get(frame["file_name"])
        if converted_digest is None:
            converted_digest = manifest["source_files_sha256"]["images"][frame["source_file_name"]]
        if hashlib.sha256(path.read_bytes()).hexdigest() != converted_digest:
            raise ValueError(f"image checksum mismatch at frame {frame['frame_index']}")
    xml_boxes = read_boxes(segment / "ground_truth.xml")
    coco_boxes = read_boxes(coco_path)
    xml_index = {_key(box): box for box in xml_boxes}
    coco_index = {_key(box): box for box in coco_boxes}
    if (
        len(xml_index) != len(xml_boxes)
        or len(coco_index) != len(coco_boxes)
        or xml_index.keys() != coco_index.keys()
        or any(
            any(abs(left - right) > 1e-4 for left, right in zip(xml_index[key].xyxy, coco_index[key].xyxy))
            for key in xml_index
        )
    ):
        raise ValueError("CVAT XML and COCO boxes/IDs/classes disagree")
    seen = set()
    by_track: dict[str, list[Box]] = defaultdict(list)
    for box in coco_boxes:
        identifier = (box.frame, box.track_id)
        if identifier in seen:
            raise ValueError(f"track ID {box.track_id} repeats in frame {box.frame}")
        seen.add(identifier)
        width, height = frames[box.frame]["width"], frames[box.frame]["height"]
        x1, y1, x2, y2 = box.xyxy
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"box outside frame {box.frame}")
        by_track[box.track_id or ""].append(box)
    warnings = []
    for track_id, boxes in by_track.items():
        if len({box.label for box in boxes}) != 1:
            raise ValueError(f"track {track_id} changes class")
        boxes.sort(key=lambda box: box.frame)
        for left, right in zip(boxes, boxes[1:]):
            if right.frame == left.frame + 1:
                dx = (right.xyxy[0] + right.xyxy[2] - left.xyxy[0] - left.xyxy[2]) / 2
                dy = (right.xyxy[1] + right.xyxy[3] - left.xyxy[1] - left.xyxy[3]) / 2
                diagonal = (frames[left.frame]["width"] ** 2 + frames[left.frame]["height"] ** 2) ** 0.5
                if (dx * dx + dy * dy) ** 0.5 > 0.25 * diagonal:
                    warnings.append(
                        {
                            "track_id": track_id,
                            "frame": right.frame,
                            "reason": "large adjacent-frame box-center jump; inspect manually",
                        }
                    )
    baseline = tracking_metrics(coco_boxes, coco_boxes)
    perfect = baseline["thresholds"]["0.5"]["overall"]
    if perfect["precision"] != 1 or perfect["recall"] != 1 or baseline["id_switches"] != 0:
        raise ValueError("GT-as-prediction tracking metrics are not perfect")
    rng = random.Random(seed)
    deleted = set(rng.sample(range(len(coco_boxes)), max(1, round(len(coco_boxes) * 0.1))))
    longest = max(by_track.values(), key=len)
    switch_track = longest[0].track_id if len(longest) >= 2 else None
    switch_frame = longest[len(longest) // 2].frame if switch_track else None
    predictions = []
    shifted = False
    for index, box in enumerate(coco_boxes):
        if index in deleted:
            continue
        xyxy = box.xyxy
        if not shifted:
            width = xyxy[2] - xyxy[0]
            xyxy = (xyxy[0] + width * 0.05, xyxy[1], xyxy[2] + width * 0.05, xyxy[3])
            shifted = True
        identity = (
            f"{box.track_id}:switched" if box.track_id == switch_track and box.frame >= switch_frame else box.track_id
        )
        predictions.append(Box(box.frame, box.label, xyxy, identity))
    damaged = tracking_metrics(coco_boxes, predictions)
    changed = damaged["thresholds"]["0.5"]["overall"]
    if changed["recall"] is None or changed["recall"] >= 1:
        raise ValueError("deleted GT boxes did not reduce recall")
    if (
        switch_track
        and sum(box.track_id == switch_track for box in predictions) >= 1
        and sum(box.track_id == f"{switch_track}:switched" for box in predictions) >= 1
        and damaged["id_switches"] < 1
    ):
        raise ValueError("controlled track-ID change did not produce an ID switch")
    report = {
        "schema": "public-gt-validation/v1",
        "dataset": manifest["dataset"],
        "sequence": manifest["sequence"],
        "frames": len(frames),
        "boxes": len(coco_boxes),
        "tracks": len(by_track),
        "cross_format_equal": True,
        "gt_as_prediction": {
            "precision_iou_05": perfect["precision"],
            "recall_iou_05": perfect["recall"],
            "precision_iou_07": baseline["thresholds"]["0.7"]["overall"]["precision"],
            "recall_iou_07": baseline["thresholds"]["0.7"]["overall"]["recall"],
            "id_switches": baseline["id_switches"],
        },
        "controlled_fault": {
            "deleted_boxes": len(deleted),
            "shifted_boxes": int(shifted),
            "switched_track": switch_track,
            "recall_iou_05": changed["recall"],
            "id_switches": damaged["id_switches"],
        },
        "warnings": warnings,
        "result": "passed",
    }
    (segment / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(validate(args.segment, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
