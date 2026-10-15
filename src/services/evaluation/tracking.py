"""Versioned offline tracking diagnostics; never used to alter annotations.

Box P/R retains the project's deterministic greedy matcher. Identity uses global
trajectory assignment (TrackEval Identity), not the greedy detection pairs.
Temporal matching prioritizes continuity then IoU (CLEAR). Fragmentation counts
matched -> missed while GT is present -> matched; absence of GT is not a miss.
Only explicitly annotated/evaluation frames participate; IDs are sequence-local.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.models.evaluation import FrameKey, PredictionBox, ReferenceBox
from src.services.evaluation.matching import bbox_iou, match_boxes

REVISION = "autolabel-tracking-diagnostics/v1"


def _track(box: PredictionBox) -> str:
    return box.stitched_track_id or box.raw_track_id


def _identity(gt: list[ReferenceBox], pred: list[PredictionBox], threshold: float) -> dict[str, Any]:
    left = list(dict.fromkeys((x.frame.sequence, x.native_track_id) for x in gt))
    right = list(dict.fromkeys((x.frame.sequence, _track(x)) for x in pred))
    gi, pi = {key: i for i, key in enumerate(left)}, {key: i for i, key in enumerate(right)}
    counts = np.zeros((len(left), len(right)), dtype=np.int64)
    gt_count = np.zeros(len(left), dtype=np.int64)
    pred_count = np.zeros(len(right), dtype=np.int64)
    by_frame: dict[FrameKey, list[PredictionBox]] = defaultdict(list)
    for box in pred:
        by_frame[box.frame].append(box)
        pred_count[pi[(box.frame.sequence, _track(box))]] += 1
    for box in gt:
        i = gi[(box.frame.sequence, box.native_track_id)]
        gt_count[i] += 1
        for other in by_frame[box.frame]:
            if box.class_name == other.class_name and bbox_iou(box.bbox_xyxy, other.bbox_xyxy) >= threshold:
                counts[i, pi[(other.frame.sequence, _track(other))]] += 1
    # Dummy rows/columns permit unmatched trajectories. Each real pair pays
    # GT detections + prediction detections - 2 * potential co-occurrences.
    n, m = len(left), len(right)
    cost = np.full((n + m, n + m), 1e12, dtype=float)
    cost[:n, :m] = gt_count[:, None] + pred_count[None, :] - 2 * counts
    cost[n:, m:] = 0
    for i in range(n):
        cost[i, m + i] = gt_count[i]
    for j in range(m):
        cost[n + j, j] = pred_count[j]
    rows, columns = linear_sum_assignment(cost)
    tp = int(sum(counts[i, j] for i, j in zip(rows, columns) if i < n and j < m))
    fn, fp = len(gt) - tp, len(pred) - tp
    denominator = 2 * tp + fp + fn
    return {"idtp": tp, "idfp": fp, "idfn": fn, "idf1": 2 * tp / denominator if denominator else None}


def _temporal(
    gt: list[ReferenceBox],
    pred: list[PredictionBox],
    frames: tuple[FrameKey, ...],
    threshold: float,
    *,
    include_events: bool = False,
) -> dict[str, Any]:
    gt_frames: dict[FrameKey, list[ReferenceBox]] = defaultdict(list)
    pred_frames: dict[FrameKey, list[PredictionBox]] = defaultdict(list)
    for box in gt:
        gt_frames[box.frame].append(box)
    for box in pred:
        pred_frames[box.frame].append(box)
    last: dict[tuple[Any, str | None], str] = {}
    previous: dict[tuple[Any, str | None], str] = {}
    matched_count: dict[tuple[Any, str | None], int] = defaultdict(int)
    support: dict[tuple[Any, str | None], int] = defaultdict(int)
    missed: set[tuple[Any, str | None]] = set()
    switches = fragments = 0
    events = []
    last_frame: dict[tuple[Any, str | None], int] = {}
    missed_annotations: dict[tuple[Any, str | None], list[str]] = defaultdict(list)
    last_annotation: dict[tuple[Any, str | None], str] = {}
    for frame in frames:
        a, b = gt_frames[frame], pred_frames[frame]
        pairs: dict[int, int] = {}
        if a and b:
            similarities = np.array(
                [[bbox_iou(x.bbox_xyxy, y.bbox_xyxy) if x.class_name == y.class_name else -1 for y in b] for x in a]
            )
            score = similarities.copy()
            for i, box in enumerate(a):
                identity = (frame.sequence, box.native_track_id)
                for j, other in enumerate(b):
                    if previous.get(identity) == _track(other):
                        score[i, j] += 1000
            score[similarities < threshold] = 0
            rows, columns = linear_sum_assignment(-score)
            pairs = {int(i): int(j) for i, j in zip(rows, columns) if similarities[i, j] >= threshold}
        current = {key: value for key, value in previous.items() if key[0] != frame.sequence}
        for i, box in enumerate(a):
            key = (frame.sequence, box.native_track_id)
            support[key] += 1
            if i not in pairs:
                if key in last:
                    missed.add(key)
                    if include_events:
                        missed_annotations[key].append(box.annotation_id)
                continue
            track = _track(b[pairs[i]])
            if include_events and key in last and last[key] != track:
                events.append(
                    {
                        "kind": "id_switch",
                        "source_frame": int(frame.frame_uid),
                        "gt_id": box.native_track_id,
                        "annotation_id": box.annotation_id,
                        "previous_pred_id": last[key],
                        "pred_id": track,
                        "previous_matched_source_frame": last_frame[key],
                    }
                )
            if include_events and key in missed:
                events.append(
                    {
                        "kind": "fragmentation",
                        "source_frame": int(frame.frame_uid),
                        "gt_id": box.native_track_id,
                        "annotation_id": box.annotation_id,
                        "previous_pred_id": last.get(key),
                        "pred_id": track,
                        "previous_matched_source_frame": last_frame[key],
                        "previous_matched_annotation_id": last_annotation[key],
                        "missed_annotation_ids": list(missed_annotations[key]),
                    }
                )
            switches += int(key in last and last[key] != track)
            fragments += int(key in missed)
            missed.discard(key)
            last[key] = current[key] = track
            if include_events:
                last_frame[key] = int(frame.frame_uid)
                last_annotation[key] = box.annotation_id
                missed_annotations[key].clear()
            matched_count[key] += 1
        previous = current
    tracks = [
        {
            "video_id": key[0].sequence_uid,
            "track_id": key[1],
            "gt_frames": total,
            "matched_frames": matched_count[key],
            "coverage": matched_count[key] / total,
        }
        for key, total in support.items()
    ]
    return {
        "id_switches": switches,
        "fragmentations": fragments,
        "track_recall": sum(x["matched_frames"] > 0 for x in tracks) / len(tracks) if tracks else None,
        "tracks": tracks,
        **({"events": events} if include_events else {}),
    }


def tracking_diagnostics(
    gt: tuple[ReferenceBox, ...],
    pred: tuple[PredictionBox, ...],
    *,
    frames: tuple[FrameKey, ...],
    occlusions: Mapping[str, str] | None = None,
    thresholds: tuple[float, ...] = (0.5, 0.7),
    size_limits: tuple[float, float] = (32**2, 96**2),
    tracking_ids_available: bool = True,
) -> dict[str, Any]:
    """Rates carry counts. Unknown metadata/zero support are never invented.

    Size limits refer to area in original-image pixels and are reporting bins,
    not production filters. Per-size/occlusion FP attribution is not meaningful.
    """
    if not 0 < size_limits[0] < size_limits[1]:
        raise ValueError("size limits must be positive ascending areas")
    if len(set(frames)) != len(frames):
        raise ValueError("duplicate evaluation frame")
    # Numeric video indices are chronological even when JSON rows arrive shuffled.
    # Non-numeric sensor frame UIDs retain the caller's explicit timeline order.
    if all(frame.frame_uid.isdecimal() for frame in frames):
        frames = tuple(
            sorted(
                frames,
                key=lambda x: (x.sequence.dataset_id, x.sequence.split, x.sequence.sequence_uid, int(x.frame_uid)),
            )
        )
    known = set(frames)
    if any(x.frame not in known for x in (*gt, *pred)):
        raise ValueError("box outside annotated evaluation frames")
    for boxes, field in ((gt, "native_track_id"), (pred, "raw_track_id")):
        seen = set()
        for box in boxes:
            identity = _track(box) if isinstance(box, PredictionBox) else getattr(box, field)
            if identity is None:
                tracking_ids_available = False
                continue
            key = (box.frame, identity)
            if key in seen:
                raise ValueError("track has duplicate observations in one frame")
            seen.add(key)
    metadata = occlusions or {}
    if any(value not in {"none", "partial", "heavy", "unknown"} for value in metadata.values()):
        raise ValueError("invalid occlusion")
    result: dict[str, Any] = {"revision": REVISION, "annotated_frames": len(frames), "thresholds": {}}
    gt_by_frame: dict[FrameKey, list[ReferenceBox]] = defaultdict(list)
    for reference in gt:
        gt_by_frame[reference.frame].append(reference)
    for threshold in thresholds:
        matches = match_boxes(gt, pred, iou_threshold=threshold)
        matched = {pair.annotation_id for pair in matches.pairs}
        match_iou = {pair.annotation_id: pair.iou for pair in matches.pairs}
        tp, fp, fn = len(matches.pairs), len(matches.unmatched_prediction_ids), len(matches.unmatched_annotation_ids)
        per_frame = []
        for frame in frames:
            frame_boxes = gt_by_frame[frame]
            found = sum(x.annotation_id in matched for x in frame_boxes)
            per_frame.append(
                {
                    "video_id": frame.sequence.sequence_uid,
                    "frame": frame.frame_uid,
                    "tp": found,
                    "gt": len(frame_boxes),
                    "recall": found / len(frame_boxes) if frame_boxes else None,
                }
            )
        groups: dict[str, Any] = {}
        for dimension in ("class", "size", "occlusion"):
            counts: dict[str, list[float]] = defaultdict(lambda: [0, 0, 0])
            for box in gt:
                area = (box.bbox_xyxy[2] - box.bbox_xyxy[0]) * (box.bbox_xyxy[3] - box.bbox_xyxy[1])
                category = {
                    "class": box.class_name,
                    "occlusion": metadata.get(box.annotation_id, "unknown"),
                    "size": "small" if area < size_limits[0] else "medium" if area < size_limits[1] else "large",
                }[dimension]
                counts[category][0] += int(box.annotation_id in matched)
                counts[category][1] += 1
                counts[category][2] += match_iou.get(box.annotation_id, 0)
            if dimension == "class":
                for box in pred:
                    counts[box.class_name]
            groups[dimension] = {
                key: {
                    "tp": int(a),
                    "gt": int(b),
                    "recall": a / b if b else None,
                    "mean_iou": iou_sum / a if a else None,
                }
                for key, (a, b, iou_sum) in sorted(counts.items())
            }
            if dimension == "class":
                unmatched = set(matches.unmatched_prediction_ids)
                for label, group in groups[dimension].items():
                    group["fp"] = sum(x.prediction_id in unmatched and x.class_name == label for x in pred)
                    denominator = group["tp"] + group["fp"]
                    group["precision"] = group["tp"] / denominator if denominator else None
                    group["fp_per_frame"] = group["fp"] / len(frames) if frames else None
        per_class = {}
        for label in sorted({x.class_name for x in (*gt, *pred)}):
            a, b = [x for x in gt if x.class_name == label], [x for x in pred if x.class_name == label]
            per_class[label] = (
                {**_identity(a, b, threshold), **_temporal(a, b, frames, threshold)}
                if tracking_ids_available
                else {"availability": "missing_track_ids"}
            )
        tracking = (
            {**_identity(list(gt), list(pred), threshold), **_temporal(list(gt), list(pred), frames, threshold)}
            if tracking_ids_available
            else {
                "idf1": None,
                "id_switches": None,
                "fragmentations": None,
                "track_recall": None,
                "availability": "missing_track_ids",
            }
        )
        result["thresholds"][str(threshold)] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "mean_iou": sum(x.iou for x in matches.pairs) / tp if tp else None,
            "fp_per_frame": fp / len(frames) if frames else None,
            "per_frame": per_frame,
            "groups": groups,
            "tracking": tracking,
            "tracking_by_class": per_class,
        }
    return result
