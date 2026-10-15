"""Behavior tests at the approved DiscrepancyFilter.evaluate interface."""

import math

import pytest

from src.services.discrepancy import DiscrepancyFilter


def observed_frame(index=0):
    return {
        "frame_index": index,
        "timestamp_s": index * 0.5,
        "label": "car",
        "box": {"center": [index, 0, 1], "size_lwh": [4, 2, 2], "yaw": 0},
        "detector": {"raw_score": 0.99, "probability": 0.99, "calibration_id": "fixture"},
        "points": {"robust_z": 0, "calibration_id": "fixture"},
        "projection": {"iou": 1, "calibration_id": "fixture", "calibration_valid": True},
    }


def evaluate(frames, label="car"):
    return DiscrepancyFilter().evaluate(
        {"track_id": "scene:1", "label": label, "data_source": "mock", "frames": frames}
    )


@pytest.mark.parametrize(
    "change,expected",
    [
        ({"projection": None, "points": None, "detector": {"raw_score": 0.99}}, "insufficient_evidence"),
        ({"projection": {"iou": 1, "calibration_id": "broken", "calibration_valid": False}}, "insufficient_evidence"),
        ({"timestamps_aligned": False}, "insufficient_evidence"),
        ({"vlm_status": "failed"}, "ambiguous"),
    ],
)
def test_absent_or_invalid_evidence_cannot_become_a_consistent_candidate(change, expected):
    result = evaluate([observed_frame() | change])
    assert result.status == expected
    assert result.suggested_action == "review"
    assert result.p_correct is None
    if change.get("points", "present") is None:
        assert result.discrepancy_score is None


def test_duplicate_frame_is_invalid():
    result = evaluate([observed_frame(), observed_frame()])
    assert result.status == "invalid"
    assert result.suggested_action == "exclude"
    assert "DUPLICATE_FRAME" in result.hard_flags


def test_vlm_dimensions_are_separate_proposals_and_occlusion_reduces_evidence():
    frame = observed_frame() | {
        "vlm_status": "completed",
        "dimensions": {"size_lwh": [8, 2, 2], "source_id": "fixture-model", "identifiable": True},
    }
    result = evaluate([frame])
    assert result.status == "discrepancy"
    assert frame["box"]["size_lwh"] == [4, 2, 2]
    assert result.frame_results[0].signals["dimensions"].provenance == "vlm_estimate:fixture-model"
    occluded = evaluate([frame | {"occlusion": "heavy"}])
    assert occluded.frame_results[0].signals["dimensions"].available is False
    assert occluded.evidence_coverage < result.evidence_coverage


def test_short_failure_in_long_track_survives_percentile_aggregation():
    frames = [observed_frame(i) for i in range(30)]
    frames[12]["projection"]["iou"] = 0
    result = evaluate(frames)
    assert result.status == "discrepancy"
    assert result.worst_frame == 12
    assert any(reason.frame_index == 12 for reason in result.reasons)


def test_global_motion_and_wrapped_yaw_do_not_create_false_conflict():
    frames = [observed_frame(i) for i in range(3)]
    for i, frame in enumerate(frames):
        frame["box"]["yaw"] = math.pi - 0.01 if i == 0 else -math.pi + 0.01
    result = evaluate(frames)
    assert result.status == "consistent_candidate"
    assert result.frame_results[-1].signals["temporal"].available is True
    assert result.frame_results[-1].signals["temporal"].value < 0.1


def test_near_square_boxes_ignore_yaw_but_still_detect_dimension_jump():
    first, second = observed_frame(), observed_frame(1)
    first["box"]["size_lwh"] = [2, 2, 2]
    second["box"].update(size_lwh=[2, 2, 2], yaw=math.pi / 2)
    assert evaluate([first, second]).status == "consistent_candidate"
    second["box"]["size_lwh"] = [5, 2, 2]
    assert evaluate([first, second]).status == "discrepancy"


def test_gaps_and_class_flips_are_not_averaged_away():
    assert evaluate([observed_frame(), observed_frame(10)]).status == "discrepancy"
    result = evaluate([observed_frame(), observed_frame(1) | {"label": "truck"}])
    assert result.status == "discrepancy"
    assert any(reason.code == "CLASS_FLIP" for reason in result.reasons)


def test_pedestrians_do_not_expect_factory_dimension_evidence():
    frame = observed_frame() | {"label": "pedestrian", "vlm_status": "completed"}
    pedestrian = evaluate([frame], label="pedestrian")
    car = evaluate([observed_frame() | {"vlm_status": "completed"}])
    assert "dimensions" not in pedestrian.frame_results[0].signals
    assert pedestrian.evidence_coverage > car.evidence_coverage


def test_uncalibrated_pedestrian_detector_cannot_become_consistent():
    frame = observed_frame() | {"label": "pedestrian", "detector": {"raw_score": 0.99}}
    result = evaluate([frame], label="pedestrian")
    assert result.status == "ambiguous"
    assert result.needs_vlm is False


def test_vlm_heavy_occlusion_is_the_canonical_occlusion_signal():
    frame = observed_frame() | {
        "vlm_status": "completed",
        "dimensions": {"size_lwh": [8, 2, 2], "source_id": "fixture-model", "identifiable": True},
        "semantic": {
            "model_id": "fixture-model",
            "verdict": {"is_real_object": True, "class": "car", "occlusion": "heavy"},
        },
    }
    result = evaluate([frame])
    signals = result.frame_results[0].signals
    assert signals["dimensions"].available is False
    assert signals["points"].reliability == pytest.approx(0.25)


def test_any_heavy_occlusion_source_conservatively_suppresses_dimensions():
    frame = observed_frame() | {
        "occlusion": "heavy",
        "vlm_status": "completed",
        "dimensions": {"size_lwh": [8, 2, 2], "source_id": "fixture-model", "identifiable": True},
        "semantic": {
            "model_id": "fixture-model",
            "verdict": {"is_real_object": True, "class": "car", "occlusion": "none"},
        },
    }
    signals = evaluate([frame]).frame_results[0].signals
    assert signals["dimensions"].available is False
    assert signals["points"].reliability == pytest.approx(0.25)


def test_two_frame_constant_velocity_does_not_assume_zero_motion():
    first, second = observed_frame(), observed_frame(1)
    second["box"]["center"] = [5, 0, 1]
    result = evaluate([first, second])
    assert result.frame_results[-1].signals["temporal"].value == pytest.approx(0)
    assert result.status == "consistent_candidate"


def test_evidence_free_result_has_no_arbitrary_worst_frame():
    frame = observed_frame() | {"projection": None, "points": None, "detector": {"raw_score": 0.99}}
    result = evaluate([frame])
    assert result.worst_frame is None


def test_unsupported_observation_class_routes_to_insufficient_evidence():
    result = evaluate([observed_frame() | {"label": "bus"}])
    assert result.status == "insufficient_evidence"
    assert "UNSUPPORTED_CLASS" in result.hard_flags


def test_severe_projection_conflict_is_visible_despite_other_agreement():
    evidence = {
        "track_id": "scene:1",
        "label": "car",
        "data_source": "mock",
        "frames": [
            {
                "frame_index": 0,
                "timestamp_s": 0,
                "label": "car",
                "box": {"center": [0, 0, 1], "size_lwh": [4, 2, 2], "yaw": 0},
                "detector": {"raw_score": 0.99, "probability": 0.99, "calibration_id": "fixture"},
                "points": {"robust_z": 0, "calibration_id": "fixture"},
                "projection": {"iou": 0, "calibration_id": "fixture", "calibration_valid": True},
            }
        ],
    }
    result = DiscrepancyFilter().evaluate(evidence)
    assert result.status == "discrepancy"
    assert result.discrepancy_score > 0.6
    assert result.p_correct is None
    assert result.review_status == "pending"
    assert result.worst_frame == 0
    assert any(reason.code == "PROJECTION_CONFLICT" for reason in result.reasons)
