"""Measure how many erroneous predicted boxes a track-level review budget reaches.

An erroneous box is an unmatched prediction at the configured 2D IoU threshold.
This is a correction-work proxy, not a human-review-time measurement. Missing GT
objects are excluded because no predicted track can place them in this queue.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts.eval_autolabel import Box, _matches, read_boxes
from src.services.video.sequence_review import score_tracks
from src.services.video.stitching import ChunkObservation, UnifiedTrack


def review_recall(
    tracks: list[UnifiedTrack],
    ground_truth: list[Box],
    *,
    iou_threshold: float = 0.5,
    budgets: tuple[float, ...] = (0.1, 0.2, 0.4),
) -> dict:
    """Report box-error recall for top-K percent of tracks, sorted by cheap score.

    The threshold and budgets are evaluation parameters, not calibrated defaults.
    """
    if not 0 < iou_threshold <= 1 or any(not 0 < value <= 1 for value in budgets):
        raise ValueError("invalid review evaluation threshold or budget")
    predictions = [
        Box(obs.frame, track.label, obs.bbox, track.track_id) for track in tracks for obs in track.observations
    ]
    matched = {pred_index for _, pred_index, _ in _matches(ground_truth, predictions, iou_threshold)}
    errors_by_track = {track.track_id: 0 for track in tracks}
    for index, prediction in enumerate(predictions):
        if index not in matched:
            errors_by_track[prediction.track_id] += 1
    ranked = score_tracks(tracks)
    total_errors = sum(errors_by_track.values())
    rows = []
    for budget in budgets:
        chosen = ranked[: math.ceil(budget * len(ranked))]
        found = sum(errors_by_track[item.track_id] for item in chosen)
        rows.append(
            {
                "top_fraction": budget,
                "reviewed_tracks": len(chosen),
                "found_erroneous_boxes": found,
                "error_recall": found / total_errors if total_errors else None,
            }
        )
    return {
        "iou_threshold": iou_threshold,
        "total_tracks": len(ranked),
        "total_predictions": len(predictions),
        "erroneous_boxes": total_errors,
        "buckets": {
            name: sum(item.priority == name for item in ranked) for name in ("critical", "high", "normal", "low")
        },
        "recall_at_k": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stitch-result", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()
    artifact = json.loads(args.stitch_result.read_text(encoding="utf-8"))
    tracks = [
        UnifiedTrack(
            row["track_id"],
            row["label"],
            [ChunkObservation(obs["frame"], tuple(obs["bbox"])) for obs in row["observations"]],
            row["needs_review"],
            row["review_reasons"],
        )
        for row in artifact["tracks"]
    ]
    result = review_recall(tracks, read_boxes(args.ground_truth), iou_threshold=args.iou)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
