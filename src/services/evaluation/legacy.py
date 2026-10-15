"""Legacy sampled-frame readers and diagnostics; original outputs are preserved."""

from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .matching import MatchInput, bbox_iou, greedy_matches

IOU_THRESHOLDS = (0.5, 0.7)


@dataclass(frozen=True)
class Box:
    frame: int
    label: str
    xyxy: tuple[float, float, float, float]
    track_id: str | None = None


def _box(frame: Any, label: Any, coordinates: Any, track_id: Any = None) -> Box:
    values = tuple(float(value) for value in coordinates)
    if (
        type(frame) is not int
        or frame < 0
        or not isinstance(label, str)
        or not label
        or len(values) != 4
        or not all(math.isfinite(value) for value in values)
        or values[2] <= values[0]
        or values[3] <= values[1]
    ):
        raise ValueError("invalid annotation box")
    return Box(frame, label, values, str(track_id) if track_id is not None else None)


def read_cvat_xml(path: Path) -> list[Box]:
    root = ET.parse(path).getroot()
    if root.tag != "annotations" or root.findtext("version") != "1.1":
        raise ValueError("expected CVAT XML annotations version 1.1")
    boxes = []
    for track in root.findall("track"):
        keyframes = sorted(track.findall("box"), key=lambda item: int(item.attrib["frame"]))
        for index, item in enumerate(keyframes):
            if item.get("outside") == "1":
                continue
            start = int(item.attrib["frame"])
            left = tuple(float(item.attrib[key]) for key in ("xtl", "ytl", "xbr", "ybr"))
            next_item = keyframes[index + 1] if index + 1 < len(keyframes) else None
            stop = int(next_item.attrib["frame"]) if next_item is not None else start + 1
            right = (
                tuple(float(next_item.attrib[key]) for key in ("xtl", "ytl", "xbr", "ybr"))
                if next_item is not None
                else left
            )
            for frame in range(start, stop):
                fraction = (frame - start) / (stop - start)
                coordinates = tuple(a + (b - a) * fraction for a, b in zip(left, right))
                boxes.append(_box(frame, track.attrib["label"], coordinates, track.attrib["id"]))
    for image in root.findall("image"):
        for item in image.findall("box"):
            boxes.append(
                _box(
                    int(image.attrib["id"]),
                    item.attrib["label"],
                    (item.attrib[key] for key in ("xtl", "ytl", "xbr", "ybr")),
                )
            )
    return boxes


def read_coco(path: Path) -> list[Box]:
    data = json.loads(path.read_text(encoding="utf-8"))
    categories = {item["id"]: item["name"] for item in data["categories"]}
    frames = {}
    for image in data["images"]:
        match = re.search(r"(\d+)(?:\.[^.]+)?$", image.get("file_name", ""))
        frame = image.get("frame_index", int(match.group(1)) if match else image["id"])
        frames[image["id"]] = frame
    result = []
    for item in data["annotations"]:
        x, y, width, height = item["bbox"]
        result.append(
            _box(
                frames[item["image_id"]],
                categories[item["category_id"]],
                (x, y, x + width, y + height),
                item.get("track_id"),
            )
        )
    return result


def read_state_tracks(job_dir: Path) -> tuple[list[Box], dict[int, int]]:
    manifest = json.loads((job_dir / "manifest.json").read_text(encoding="utf-8"))
    tracks = json.loads((job_dir / "tracks.json").read_text(encoding="utf-8"))
    mapping = {item["frame_index"]: item.get("source_frame_index", item["frame_index"]) for item in manifest["frames"]}
    boxes = []
    for track in tracks["tracks"]:
        for observation in track["observations"]:
            if observation.get("outside", False):
                continue
            b = observation["bbox"]
            boxes.append(
                _box(
                    mapping[observation["frame_index"]],
                    track["label_name"],
                    (b["xtl"], b["ytl"], b["xbr"], b["ybr"]),
                    track["track_id"],
                )
            )
    return boxes, mapping


def read_boxes(path: Path) -> list[Box]:
    if path.suffix.lower() == ".xml":
        return read_cvat_xml(path)
    return read_coco(path)


def iou(left: Box, right: Box) -> float:
    return bbox_iou(left.xyxy, right.xyxy)


def _matches(ground_truth: list[Box], predictions: list[Box], threshold: float) -> list[tuple[int, int, float]]:
    gt = tuple(MatchInput(box.frame, box.label, box.xyxy, i) for i, box in enumerate(ground_truth))
    pred = tuple(MatchInput(box.frame, box.label, box.xyxy, i) for i, box in enumerate(predictions))
    return greedy_matches(gt, pred, threshold=threshold)


def _rates(gt_count: int, pred_count: int, matches: list[tuple[int, int, float]]) -> dict[str, float | int | None]:
    true_positive = len(matches)
    return {
        "gt": gt_count,
        "predicted": pred_count,
        "tp": true_positive,
        "precision": true_positive / pred_count if pred_count else None,
        "recall": true_positive / gt_count if gt_count else None,
        "mean_iou": sum(row[2] for row in matches) / true_positive if true_positive else None,
    }


def tracking_metrics(ground_truth: list[Box], predictions: list[Box], frames: set[int] | None = None) -> dict[str, Any]:
    """Evaluate only selected frames when a prediction manifest provides them."""
    if frames is not None:
        ground_truth = [box for box in ground_truth if box.frame in frames]
        predictions = [box for box in predictions if box.frame in frames]
    result: dict[str, Any] = {
        "evaluated_frames": sorted(frames if frames is not None else {box.frame for box in ground_truth + predictions}),
        "thresholds": {},
        "id_switches": None,
        "fragmentations": None,
    }
    labels = sorted({box.label for box in ground_truth + predictions})
    for threshold in IOU_THRESHOLDS:
        matches = _matches(ground_truth, predictions, threshold)
        result["thresholds"][str(threshold)] = {
            "overall": _rates(len(ground_truth), len(predictions), matches),
            "by_class": {
                label: _rates(
                    sum(box.label == label for box in ground_truth),
                    sum(box.label == label for box in predictions),
                    [row for row in matches if ground_truth[row[0]].label == label],
                )
                for label in labels
            },
        }
    matches = _matches(ground_truth, predictions, 0.5)
    if ground_truth and all(box.track_id is not None for box in ground_truth):
        associated: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for gi, pi, _ in matches:
            if predictions[pi].track_id is not None:
                associated[ground_truth[gi].track_id or ""].append(
                    (ground_truth[gi].frame, predictions[pi].track_id or "")
                )
        result["id_switches"] = 0
        result["fragmentations"] = 0
        for observations in associated.values():
            ordered = sorted(observations)
            result["id_switches"] += sum(left[1] != right[1] for left, right in zip(ordered, ordered[1:]))
        # Count matched -> unmatched -> matched on annotated source observations.
        # Gaps in GT itself are not detector failures.
        matched_gt = {gi for gi, _, _ in matches}
        per_track: dict[str, list[tuple[int, bool]]] = defaultdict(list)
        for gi, box in enumerate(ground_truth):
            if box.track_id is not None:
                per_track[box.track_id].append((box.frame, gi in matched_gt))
        for presence in per_track.values():
            seen_match, missed_after_match = False, False
            for _, matched in sorted(presence):
                if matched:
                    result["fragmentations"] += int(missed_after_match)
                    seen_match, missed_after_match = True, False
                elif seen_match:
                    missed_after_match = True
    if ground_truth:
        all_frames = sorted(frames if frames is not None else {box.frame for box in ground_truth})
        thirds = {frame: min(2, index * 3 // len(all_frames)) for index, frame in enumerate(all_frames)}
        matched_gt = {gi for gi, _, _ in matches}
        result["recall_by_segment"] = {
            name: _rates(
                sum(thirds[box.frame] == segment for box in ground_truth),
                0,
                [(gi, -1, 0.0) for gi in matched_gt if thirds[ground_truth[gi].frame] == segment],
            )["recall"]
            for segment, name in enumerate(("early", "middle", "late"))
        }
        first_frame = min(all_frames)
        first_seen: dict[str, int] = {}
        for box in ground_truth:
            if box.track_id is not None:
                first_seen[box.track_id] = min(first_seen.get(box.track_id, box.frame), box.frame)
        entering = {
            gi
            for gi, box in enumerate(ground_truth)
            if box.track_id is not None and first_seen[box.track_id] > first_frame
        }
        result["new_object_recall"] = len(entering & matched_gt) / len(entering) if entering else None
        result["new_object_gt_boxes"] = len(entering)
    else:
        result["recall_by_segment"] = {name: None for name in ("early", "middle", "late")}
        result["new_object_recall"] = None
        result["new_object_gt_boxes"] = 0
    return result
