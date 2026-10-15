"""Conservative review ordering without VLM or uncalibrated SAM3 score filtering."""

from src.services.video.sequence_review import score_tracks
from src.services.video.stitching import ChunkObservation, UnifiedTrack


def test_ambiguous_person_track_ranks_above_stable_car():
    car = UnifiedTrack(
        "car",
        "car",
        [
            ChunkObservation(0, (0, 0, 20, 20)),
            ChunkObservation(1, (1, 0, 21, 20)),
            ChunkObservation(2, (2, 0, 22, 20)),
        ],
    )
    person = UnifiedTrack(
        "person",
        "person",
        [
            ChunkObservation(0, (5, 5, 8, 10)),
        ],
        needs_review=True,
        review_reasons=["ambiguous_overlap"],
    )
    queue = score_tracks([car, person])
    assert [item.track_id for item in queue] == ["person", "car"]
    assert 0 <= queue[0].score <= 1
    assert queue[0].priority == "critical"
    assert queue[0].needs_review
    assert queue[1].priority == "low"
