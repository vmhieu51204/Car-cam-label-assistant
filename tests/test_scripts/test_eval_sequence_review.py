from __future__ import annotations

from scripts.eval_autolabel import Box
from scripts.eval_sequence_review import review_recall
from src.services.video.stitching import ChunkObservation, UnifiedTrack


def test_review_recall_counts_only_unmatched_predictions() -> None:
    good = UnifiedTrack("good", "car", [ChunkObservation(0, (0, 0, 10, 10))])
    bad = UnifiedTrack("bad", "car", [ChunkObservation(0, (50, 50, 60, 60))], needs_review=True)
    result = review_recall([good, bad], [Box(0, "car", (0, 0, 10, 10), "gt")], budgets=(0.5, 1.0))
    assert result["erroneous_boxes"] == 1
    assert result["recall_at_k"][0]["error_recall"] == 1.0
    assert result["recall_at_k"][0]["reviewed_tracks"] == 1


def test_review_recall_is_undefined_without_errors() -> None:
    track = UnifiedTrack("t", "car", [ChunkObservation(0, (0, 0, 10, 10))])
    result = review_recall([track], [Box(0, "car", (0, 0, 10, 10), "gt")])
    assert result["recall_at_k"][0]["error_recall"] is None
