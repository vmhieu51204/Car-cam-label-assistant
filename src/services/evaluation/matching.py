"""One deterministic continuous-box diagnostic matcher shared by all callers."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Hashable
from dataclasses import dataclass

from src.models.evaluation import BBox, MatchedPair, MatchResult, PredictionBox, ReferenceBox


@dataclass(frozen=True)
class MatchInput:
    frame: Hashable
    label: str
    bbox: BBox
    ordinal: int


def bbox_iou(a: BBox, b: BBox) -> float:
    """Continuous xyxy area (no pixel +1); invalid extents are rejected."""
    for box in (a, b):
        if not all(math.isfinite(x) for x in box) or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("invalid box extent")
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return intersection / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection)


def greedy_matches(
    gt: tuple[MatchInput, ...], pred: tuple[MatchInput, ...], *, threshold: float, class_aware: bool = True
) -> list[tuple[int, int, float]]:
    """Descending IoU then GT/pred ordinals, with an inclusive threshold."""
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("IoU threshold must be finite and in [0,1]")
    for rows in (gt, pred):
        if len({(x.frame, x.ordinal) for x in rows}) != len(rows):
            raise ValueError("duplicate within-frame tie ordinal")
        for row in rows:
            bbox_iou(row.bbox, row.bbox)
    by_frame: dict[tuple[Hashable, str | None], list[tuple[int, MatchInput]]] = defaultdict(list)
    for pi, row in enumerate(pred):
        by_frame[(row.frame, row.label if class_aware else None)].append((pi, row))
    candidates = [
        (bbox_iou(row.bbox, other.bbox), row.ordinal, other.ordinal, gi, pi)
        for gi, row in enumerate(gt)
        for pi, other in by_frame[(row.frame, row.label if class_aware else None)]
    ]
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    result = []
    for overlap, _, _, gi, pi in sorted(candidates, key=lambda row: (-row[0], row[1], row[2], row[3], row[4])):
        if overlap < threshold:
            break
        if gi not in matched_gt and pi not in matched_pred:
            matched_gt.add(gi)
            matched_pred.add(pi)
            result.append((gi, pi, overlap))
    return result


def match_boxes(
    gt: tuple[ReferenceBox, ...], pred: tuple[PredictionBox, ...], *, iou_threshold: float, class_aware: bool = True
) -> MatchResult:
    if len({x.annotation_id for x in gt}) != len(gt) or len({x.prediction_id for x in pred}) != len(pred):
        raise ValueError("duplicate box IDs")
    rows = greedy_matches(
        tuple(MatchInput(x.frame, x.class_name, x.bbox_xyxy, x.source_ordinal) for x in gt),
        tuple(MatchInput(x.frame, x.class_name, x.bbox_xyxy, x.scoring_ordinal) for x in pred),
        threshold=iou_threshold,
        class_aware=class_aware,
    )
    matched_gt = {gi for gi, _, _ in rows}
    matched_pred = {pi for _, pi, _ in rows}
    return MatchResult(
        iou_threshold=iou_threshold,
        class_aware=class_aware,
        pairs=tuple(
            MatchedPair(
                annotation_id=gt[gi].annotation_id,
                prediction_id=pred[pi].prediction_id,
                frame=gt[gi].frame,
                iou=overlap,
            )
            for gi, pi, overlap in rows
        ),
        unmatched_annotation_ids=tuple(x.annotation_id for i, x in enumerate(gt) if i not in matched_gt),
        unmatched_prediction_ids=tuple(x.prediction_id for i, x in enumerate(pred) if i not in matched_pred),
    )
