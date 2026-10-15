"""Add evidence to CLEAR fragmentation events without changing temporal matching."""

from collections import Counter
from typing import Any

import numpy as np

from src.models.evaluation import FrameKey, PredictionBox, ReferenceBox
from src.services.evaluation.matching import bbox_iou
from src.services.evaluation.tracking import _track

CATEGORIES = (
    "continuing_track_low_iou",
    "id_break",
    "no_prediction",
    "other_class",
    "continuing_track_below_threshold",
    "assignment_conflict",
    "unresolved",
    "mixed",
)
RULES = {
    "revision": "autolabel-fragmentation-evidence/v1",
    "frame_priority": [
        "continuing_track_low_iou",
        "id_break",
        "other_class",
        "no_prediction",
        "continuing_track_below_threshold",
        "assignment_conflict",
        "unresolved",
    ],
    "continuing_track_low_iou": "Same class, either matched anchor ID, 0.1 <= IoU < 0.5",
    "id_break": "Same class, neither matched anchor ID, IoU >= 0.5; overlap is not proof of identity",
    "other_class": "Different class in evaluated raw population, IoU >= 0.5",
    "no_prediction": "No prediction of any evaluated class has IoU >= 0.1",
    "continuing_track_below_threshold": "Same class/anchor ID, 0.5 <= IoU < evaluation threshold (0.7 only)",
    "assignment_conflict": "Same class/anchor ID IoU >= evaluation threshold but GT unmatched one-to-one",
    "unresolved": "Remaining overlap evidence (including another ID with IoU below 0.5)",
    "event_rule": "All actual GT-present missed frames same category => that category; otherwise mixed",
    "incidence_rule": "Events containing at least one frame of category; incidences may overlap",
    "occlusion_rule": "Use missed-frame GT occlusion; event incidence denominator is events touching that occlusion",
    "population": "Raw Car->car/Pedestrian->person; other_class cannot detect classes outside this population",
    "limits": "Evidence only, not confirmed causes. GT absence is excluded. No relinking or box changes.",
}


def fragmentation_evidence(
    event: dict,
    by_gt: dict[str, ReferenceBox],
    by_frame: dict[FrameKey, list[PredictionBox]],
    occlusions: dict[str, str],
) -> dict:
    anchors = {event["previous_pred_id"], event["pred_id"]}
    threshold = event["iou_threshold"]
    rows = []
    for annotation_id in event["missed_annotation_ids"]:
        gt = by_gt[annotation_id]
        candidates: list[dict[str, Any]] = []
        for pred in by_frame.get(gt.frame, []):
            iou = bbox_iou(gt.bbox_xyxy, pred.bbox_xyxy)
            if iou >= 0.1:
                candidates.append(
                    {
                        "prediction_id": pred.prediction_id,
                        "track_id": _track(pred),
                        "class": pred.class_name,
                        "iou": iou,
                        "same_class": pred.class_name == gt.class_name,
                        "anchor_id": _track(pred) in anchors,
                    }
                )
        candidates.sort(key=lambda p: (-p["iou"], p["prediction_id"]))
        if any(p["same_class"] and p["anchor_id"] and p["iou"] < 0.5 for p in candidates):
            category = "continuing_track_low_iou"
        elif any(p["same_class"] and not p["anchor_id"] and p["iou"] >= 0.5 for p in candidates):
            category = "id_break"
        elif any(not p["same_class"] and p["iou"] >= 0.5 for p in candidates):
            category = "other_class"
        elif not candidates:
            category = "no_prediction"
        elif any(p["same_class"] and p["anchor_id"] and 0.5 <= p["iou"] < threshold for p in candidates):
            category = "continuing_track_below_threshold"
        elif any(p["same_class"] and p["anchor_id"] and p["iou"] >= threshold for p in candidates):
            category = "assignment_conflict"
        else:
            category = "unresolved"
        rows.append(
            {
                "source_frame": int(gt.frame.frame_uid),
                "annotation_id": annotation_id,
                "gt_bbox_xyxy": gt.bbox_xyxy,
                "gt_class": gt.class_name,
                "occlusion": occlusions.get(annotation_id, "unknown"),
                "category": category,
                "candidates": candidates,
            }
        )
    if not rows:
        raise ValueError("fragmentation must contain actual GT-present missed observations")
    categories = sorted({row["category"] for row in rows})
    return {
        "fragmentation_type": categories[0] if len(categories) == 1 else "mixed",
        "fragmentation_types_present": categories,
        "missed_frames": rows,
    }


def _rate(events: list[dict], selected: list[dict], repeats: int) -> dict[str, Any]:
    counts = Counter((e["sequence"], str(e["gt_id"])) for e in events)
    outcomes = Counter((e["sequence"], str(e["gt_id"])) for e in selected)
    result: dict[str, Any] = {
        "count": len(selected),
        "denominator_events": len(events),
        "rate": len(selected) / len(events) if events else None,
        "supporting_tracks": len(outcomes),
        "denominator_tracks": len(counts),
        "ci95": None,
        "availability": "unavailable",
    }
    if len(outcomes) < 2:
        result["reason"] = "fewer_than_two_supporting_tracks"
        return result
    tracks = sorted(counts)
    totals = np.array([counts[t] for t in tracks])
    values = np.array([outcomes[t] for t in tracks])
    rng = np.random.default_rng(42)
    numerator = np.zeros(repeats)
    denominator = np.zeros(repeats)
    for sequence in sorted({t[0] for t in tracks}):
        indices = np.array([i for i, t in enumerate(tracks) if t[0] == sequence])
        sampled = indices[rng.integers(0, len(indices), size=(repeats, len(indices)))]
        numerator += values[sampled].sum(axis=1)
        denominator += totals[sampled].sum(axis=1)
    rates = numerator / denominator
    result.update(
        availability="available",
        ci95=np.percentile(rates, [2.5, 97.5]).tolist(),
        method="sequence-local whole GT-track percentile bootstrap",
        seed=42,
        repeats=repeats,
        valid_repeats=repeats,
    )
    return result


def fragmentation_summary(events: list[dict], *, repeats: int = 10000) -> dict:
    events = [e for e in events if e["kind"] == "fragmentation"]
    exclusive = {c: _rate(events, [e for e in events if e["fragmentation_type"] == c], repeats) for c in CATEGORIES}
    incidence = {
        c: _rate(events, [e for e in events if c in e["fragmentation_types_present"]], repeats)
        for c in CATEGORIES
        if c != "mixed"
    }
    occlusion = {}
    for label in ("none", "partial", "heavy", "unknown"):
        subset = [e for e in events if any(r["occlusion"] == label for r in e["missed_frames"])]
        rows = [r for e in subset for r in e["missed_frames"] if r["occlusion"] == label]
        occlusion[label] = {
            "missed_frame_count": len(rows),
            "frame_counts": dict(Counter(r["category"] for r in rows)),
            "event_incidence": {
                c: _rate(
                    subset,
                    [
                        e
                        for e in subset
                        if any(r["occlusion"] == label and r["category"] == c for r in e["missed_frames"])
                    ],
                    repeats,
                )
                for c in CATEGORIES
                if c != "mixed"
            },
        }
    majority = bool(events) and exclusive["continuing_track_low_iou"]["count"] > len(events) / 2
    return {
        "event_count": len(events),
        "exclusive_event_types": exclusive,
        "event_incidence": incidence,
        "missed_frame_count": sum(len(e["missed_frames"]) for e in events),
        "by_occlusion": occlusion,
        "continuing_low_iou_majority": majority,
        "conclusion": "relink tracklet không giải quyết phần lớn sự kiện"
        if majority
        else "No homogeneous continuing-low-IoU majority; report only, no algorithm recommendation",
    }
