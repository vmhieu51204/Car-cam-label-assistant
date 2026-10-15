"""Offline, selected-frame evaluation of 2D tracks, VLM audits and review queues."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from src.services.evaluation.legacy import (  # compatibility re-exports
    IOU_THRESHOLDS as IOU_THRESHOLDS,
)
from src.services.evaluation.legacy import (
    Box as Box,
)
from src.services.evaluation.legacy import (
    _box as _box,
)
from src.services.evaluation.legacy import (
    _matches as _matches,
)
from src.services.evaluation.legacy import (
    _rates as _rates,
)
from src.services.evaluation.legacy import (
    iou as iou,
)
from src.services.evaluation.legacy import (
    read_boxes as read_boxes,
)
from src.services.evaluation.legacy import (
    read_coco as read_coco,
)
from src.services.evaluation.legacy import (
    read_cvat_xml as read_cvat_xml,
)
from src.services.evaluation.legacy import (
    read_state_tracks as read_state_tracks,
)
from src.services.evaluation.legacy import (
    tracking_metrics as tracking_metrics,
)

VERDICTS = ("agree", "possible_missing", "wrong_class", "bad_extent", "uncertain")
PRIORITIES = ("critical", "high", "normal", "low")


def _pearson(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    mean_x = sum(x for x, _ in pairs) / len(pairs)
    mean_y = sum(y for _, y in pairs) / len(pairs)
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    variance_x = sum((x - mean_x) ** 2 for x, _ in pairs)
    variance_y = sum((y - mean_y) ** 2 for _, y in pairs)
    return covariance / math.sqrt(variance_x * variance_y) if variance_x and variance_y else None


def audit_metrics(
    faults: list[dict], audit: dict, queue: dict | None = None, mapping: dict[int, int] | None = None
) -> dict[str, Any]:
    mapping = mapping or {}
    truth = {item["frame_index"]: item for item in faults}
    verdicts = {mapping.get(item["frame_index"], item["frame_index"]): item["verdict"] for item in audit["evidence"]}
    confusion = {expected: {observed: 0 for observed in VERDICTS} for expected in VERDICTS}
    for frame, item in truth.items():
        if frame in verdicts:
            confusion[item["verdict"]][verdicts[frame]] += 1
    per_class = {}
    for label in VERDICTS:
        tp = confusion[label][label]
        predicted = sum(confusion[actual][label] for actual in VERDICTS)
        actual = sum(confusion[label].values())
        per_class[label] = {
            "precision": tp / predicted if predicted else None,
            "recall": tp / actual if actual else None,
            "support": actual,
        }
    result: dict[str, Any] = {
        "coverage": len(set(truth) & set(verdicts)),
        "fault_frames": len(truth),
        "confusion_matrix": confusion,
        "by_verdict": per_class,
    }
    if queue is not None:
        result.update(review_queue_metrics(faults, queue, mapping))
    return result


def review_queue_metrics(faults: list[dict], queue: dict, mapping: dict[int, int] | None = None) -> dict[str, Any]:
    mapping = mapping or {}
    truth = {item["frame_index"]: item for item in faults}
    units = queue["units"]
    counts = Counter(unit["priority"] for unit in units)
    ordered = sorted(
        units,
        key=lambda unit: (
            (-unit["review_score"], unit["frame_start"])
            if unit.get("review_score") is not None
            else (PRIORITIES.index(unit["priority"]), unit["frame_start"])
        ),
    )
    errors = {frame for frame, item in truth.items() if item["verdict"] != "agree"}
    result: dict[str, Any] = {
        "review_distribution": {priority: counts[priority] for priority in PRIORITIES},
        "review_recall_at_k": {},
    }
    for percent in (5, 10, 20, 50, 100):
        limit = math.ceil(len(ordered) * percent / 100)
        reviewed = {mapping.get(unit["frame_start"], unit["frame_start"]) for unit in ordered[:limit]}
        result["review_recall_at_k"][str(percent)] = len(errors & reviewed) / len(errors) if errors else None
    priority_value = {"critical": 1.0, "high": 2 / 3, "normal": 1 / 3, "low": 0.0}
    best: dict[int, float] = {}
    for unit in units:
        frame = mapping.get(unit["frame_start"], unit["frame_start"])
        value = unit.get("review_score")
        best[frame] = max(best.get(frame, 0.0), float(value) if value is not None else priority_value[unit["priority"]])
    result["priority_severity_pearson"] = _pearson(
        [(best[frame], float(item["severity"])) for frame, item in truth.items() if frame in best]
    )
    return result


def evaluate(
    gt_path: Path,
    prediction_path: Path,
    *,
    faults_path: Path | None = None,
    audit_path: Path | None = None,
    queue_path: Path | None = None,
    proposal_review_path: Path | None = None,
) -> dict[str, Any]:
    gt = read_boxes(gt_path)
    if prediction_path.is_dir():
        predicted, mapping = read_state_tracks(prediction_path)
        frames = set(mapping.values())
        audit_path = audit_path or prediction_path / "audit.json"
        queue_path = queue_path or prediction_path / "review-queue.json"
    else:
        predicted, mapping, frames = read_boxes(prediction_path), {}, None
    report = {"tracking": tracking_metrics(gt, predicted, frames)}
    if faults_path is not None:
        faults = json.loads(faults_path.read_text(encoding="utf-8"))["faults"]
        queue = json.loads(queue_path.read_text(encoding="utf-8")) if queue_path and queue_path.is_file() else None
        if audit_path is not None and audit_path.is_file():
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            report["audit"] = audit_metrics(faults, audit, queue, mapping)
        elif queue is not None:
            report["review"] = review_queue_metrics(faults, queue, mapping)
        else:
            raise ValueError("fault evaluation requires audit.json or review-queue.json")
    if proposal_review_path is not None:
        report["proposal_review"] = proposal_review_metrics(proposal_review_path)
    return report


def proposal_review_metrics(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len({row["proposal_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate proposal IDs in review CSV")
    marked = []
    for row in rows:
        value = row["is_correct"].strip().lower()
        if value in {"", "skip"}:
            continue
        if value not in {"true", "false", "1", "0", "yes", "no"}:
            raise ValueError(f"invalid is_correct for {row['proposal_id']}")
        marked.append((row["class"], value in {"true", "1", "yes"}))
    classes = sorted({row["class"] for row in rows})
    return {
        "sampled": len(rows),
        "reviewed": len(marked),
        "precision": sum(correct for _, correct in marked) / len(marked) if marked else None,
        "by_class": {
            label: {
                "reviewed": sum(name == label for name, _ in marked),
                "precision": (
                    sum(correct for name, correct in marked if name == label) / sum(name == label for name, _ in marked)
                    if any(name == label for name, _ in marked)
                    else None
                ),
            }
            for label in classes
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--faults", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--review-queue", type=Path)
    parser.add_argument("--proposal-review", type=Path)
    args = parser.parse_args()
    if args.ground_truth is None and args.predictions is None and args.proposal_review is not None:
        report = {"proposal_review": proposal_review_metrics(args.proposal_review)}
    elif args.ground_truth is not None and args.predictions is not None:
        report = evaluate(
            args.ground_truth,
            args.predictions,
            faults_path=args.faults,
            audit_path=args.audit,
            queue_path=args.review_queue,
            proposal_review_path=args.proposal_review,
        )
    else:
        parser.error("provide ground truth and predictions, or --proposal-review alone")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
