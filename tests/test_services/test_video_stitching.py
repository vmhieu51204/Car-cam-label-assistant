"""Cross-window identity uses the same overlap frames, never adjacent-frame IoU."""

from dataclasses import asdict

import numpy as np
import pytest

from src.services.video.stitching import (
    ChunkObservation,
    ChunkTrack,
    StitchConfig,
    decode_mask,
    encode_mask,
    stitch_tracks,
)


def _track(window, local, frames, boxes, *, label="car", masks=None):
    return ChunkTrack(
        window_id=window,
        local_track_id=local,
        label=label,
        observations=[
            ChunkObservation(frame=frame, bbox=box, mask=masks[i] if masks else None, score=0.6)
            for i, (frame, box) in enumerate(zip(frames, boxes, strict=True))
        ],
    )


def test_mask_rle_round_trip_and_checksum():
    mask = np.zeros((8, 10), dtype=bool)
    mask[2:6, 3:8] = True
    encoded = encode_mask(mask)
    assert encoded.sha256 and len(encoded.sha256) == 64
    assert np.array_equal(decode_mask(encoded), mask)
    assert np.array_equal(decode_mask(encode_mask(np.ones((2, 3), dtype=bool))), np.ones((2, 3), dtype=bool))
    assert np.array_equal(decode_mask(encode_mask(np.zeros((2, 3), dtype=bool))), np.zeros((2, 3), dtype=bool))


def test_fast_moving_object_stitches_from_same_overlap_frames():
    left = _track(0, "a", [0, 1, 2, 3], [(0, 0, 20, 20), (120, 0, 140, 20), (240, 0, 260, 20), (360, 0, 380, 20)])
    right = _track(1, "x", [2, 3, 4], [(240, 0, 260, 20), (360, 0, 380, 20), (480, 0, 500, 20)])
    result = stitch_tracks([left, right], StitchConfig(min_overlap_iou=0.5))
    assert len(result.tracks) == 1
    assert [obs.frame for obs in result.tracks[0].observations] == [0, 1, 2, 3, 4]
    assert result.decisions[0].action == "matched"
    assert result.decisions[0].mean_iou == 1.0


def test_ambiguous_overlap_keeps_new_id_and_flags_review():
    left_a = _track(0, "a", [3, 4], [(0, 0, 20, 20)] * 2)
    left_b = _track(0, "b", [3, 4], [(1, 0, 21, 20)] * 2)
    right = _track(1, "x", [3, 4, 5], [(0, 0, 20, 20)] * 3)
    result = stitch_tracks([left_a, left_b, right], StitchConfig(min_overlap_iou=0.5, ambiguity_margin=0.2))
    assert len(result.tracks) == 3
    assert result.decisions[0].action == "new_id"
    assert result.decisions[0].needs_review
    assert result.decisions[0].reason == "ambiguous_overlap"


def test_different_classes_never_stitch():
    left = _track(0, "a", [3, 4], [(0, 0, 20, 20)] * 2)
    right = _track(1, "x", [3, 4], [(0, 0, 20, 20)] * 2, label="person")
    result = stitch_tracks([left, right])
    assert len(result.tracks) == 2
    assert result.decisions[0].action == "new_id"


def test_empty_middle_chunk_breaks_identity_without_crashing():
    left = _track(0, "a", [0, 1], [(0, 0, 20, 20)] * 2)
    right = _track(2, "x", [8, 9], [(0, 0, 20, 20)] * 2)
    result = stitch_tracks([left, right])
    assert len(result.tracks) == 2
    assert result.tracks[1].needs_review
    assert result.tracks[1].review_reasons == ["empty_chunk_gap"]
    assert result.decisions[0].reason == "empty_chunk_gap"


@pytest.mark.parametrize("mode", ["matched", "ambiguous", "different_class", "gap", "cut", "row_competition"])
def test_boundary_logging_does_not_change_tracks_or_decisions(mode):
    left = _track(0, "a", [3, 4], [(0, 0, 20, 20)] * 2)
    right = _track(
        2 if mode == "gap" else 1,
        "x",
        [3, 4, 5],
        [(0, 0, 20, 20)] * 3,
        label="person" if mode == "different_class" else "car",
    )
    chunks = [left, right]
    if mode == "ambiguous":
        chunks.append(_track(0, "b", [3, 4], [(1, 0, 21, 20)] * 2))
    if mode == "row_competition":
        chunks.append(_track(1, "y", [3, 4], [(0, 0, 20, 20)] * 2))
    cuts = {1} if mode == "cut" else set()
    on = stitch_tracks(chunks, reset_windows=cuts)
    off = stitch_tracks(chunks, StitchConfig(decision_log_enabled=False), reset_windows=cuts)
    assert [asdict(t) for t in on.tracks] == [asdict(t) for t in off.tracks]
    assert on.decisions == off.decisions and not off.boundary_logs
    assert len(on.boundary_logs) == (2 if mode == "gap" else 1)
    row = on.boundary_logs[-1]["tracks"][0]
    if mode == "ambiguous":
        assert row["candidate_count"] == 2 and row["eligible_candidate_count"] == 2
        assert row["best_iou"] == 1 and row["second_iou"] == pytest.approx(19 / 21)
        assert row["margin"] == pytest.approx(2 / 21) and row["result"] == "ambiguous"
    if mode == "row_competition":
        assert row["selected_row_margin"] == 0 and row["result"] == "ambiguous"
    if mode in {"gap", "cut", "different_class"}:
        assert row["candidate_count"] == 0 and row["best_iou"] is None


def test_logging_covers_initial_trailing_and_all_empty_planned_windows():
    chunks = [_track(1, "a", [3, 4], [(0, 0, 20, 20)] * 2)]
    on = stitch_tracks(chunks, planned_window_ids=[0, 1, 2, 3])
    off = stitch_tracks(chunks, StitchConfig(decision_log_enabled=False), planned_window_ids=[0, 1, 2, 3])
    assert on.tracks == off.tracks and on.decisions == off.decisions
    assert [(r["previous_window_id"], r["window_id"]) for r in on.boundary_logs] == [(0, 1), (1, 2), (2, 3)]
    empty = stitch_tracks([], planned_window_ids=[0, 1, 2])
    assert len(empty.boundary_logs) == 2 and not empty.tracks and not empty.decisions
    assert all(r["candidate_pair_count"] == 0 for r in empty.boundary_logs)
