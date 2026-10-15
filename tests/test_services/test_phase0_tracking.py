from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.eval_autolabel import Box
from src.services.evaluation.manual import load_manual_reference
from src.services.evaluation.phase0 import canonical_predictions
from src.services.evaluation.tracking import tracking_diagnostics

FIXTURE = Path(__file__).parents[1] / "fixtures/evaluation/phase0/manual-gt.json"


def inputs() -> tuple:
    ref, occlusion = load_manual_reference(FIXTURE)
    keys = {x.source_ordinal: x.key for x in ref.frames}
    boxes = [Box(int(x.frame.frame_uid), x.class_name, x.bbox_xyxy, "p") for x in ref.boxes]
    return ref, occlusion, keys, canonical_predictions(boxes, keys, "test")


def test_perfect_identity_and_true_negative_support() -> None:
    ref, occlusion, keys, predictions = inputs()
    report = tracking_diagnostics(ref.boxes, predictions, frames=tuple(keys.values()), occlusions=occlusion)
    for metrics in report["thresholds"].values():
        assert metrics["recall"] == metrics["precision"] == metrics["tracking"]["idf1"] == 1
        assert metrics["tracking"]["fragmentations"] == metrics["tracking"]["id_switches"] == 0
        assert len(metrics["per_frame"]) == 6
        assert metrics["per_frame"][-1]["recall"] is None
        assert metrics["groups"]["occlusion"]["unknown"]["gt"] == 1


def test_switch_gap_duplicate_false_positive_and_identity() -> None:
    ref, occlusion, keys, predictions = inputs()
    predictions = tuple(x for i, x in enumerate(predictions) if i != 3)
    predictions = predictions[:-1] + (predictions[-1].model_copy(update={"raw_track_id": "new"}),)
    predictions += canonical_predictions([Box(5, "car", (100, 10, 140, 50), "noise")], keys, "noise")
    metrics = tracking_diagnostics(ref.boxes, predictions, frames=tuple(keys.values()), occlusions=occlusion)[
        "thresholds"
    ]["0.5"]
    assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (4, 1, 1)
    assert metrics["fp_per_frame"] == pytest.approx(1 / 6)
    assert metrics["tracking"]["id_switches"] == metrics["tracking"]["fragmentations"] == 1
    assert metrics["tracking"]["idf1"] == pytest.approx(0.6)
    assert metrics["tracking"]["tracks"][0]["coverage"] == 0.8
    assert metrics["tracking"]["track_recall"] == 1


def test_duplicate_boxes_wrong_class_and_missing_ids() -> None:
    ref, _, keys, predictions = inputs()
    duplicate = predictions[0].model_copy(
        update={"prediction_id": "duplicate", "raw_track_id": "duplicate", "scoring_ordinal": 90}
    )
    predictions += (duplicate,)
    wrong = predictions[1].model_copy(update={"class_name": "truck"})
    predictions = predictions[:1] + (wrong,) + predictions[2:]
    metrics = tracking_diagnostics(ref.boxes, predictions, frames=tuple(keys.values()), tracking_ids_available=False)[
        "thresholds"
    ]["0.5"]
    assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (4, 2, 1)
    assert metrics["tracking"]["idf1"] is None


def test_undeclared_frames_duplicate_track_and_invalid_gt(tmp_path: Path) -> None:
    ref, _, keys, predictions = inputs()
    with pytest.raises(ValueError, match="outside annotated"):
        tracking_diagnostics(ref.boxes, predictions, frames=tuple(keys.values())[:-2])
    with pytest.raises(ValueError, match="duplicate observations"):
        tracking_diagnostics(ref.boxes, predictions + (predictions[0],), frames=tuple(keys.values()))
    document = json.loads(FIXTURE.read_text())
    document["annotations"][0]["frame"] = 99
    path = tmp_path / "gt.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="undeclared frame"):
        load_manual_reference(path)


def test_sizes_at_exact_boundaries_and_unknown_occlusion() -> None:
    ref, _, keys, predictions = inputs()
    sizes = ((0, 0, 31, 32), (0, 0, 32, 32), (0, 0, 96, 96))
    gt = tuple(x.model_copy(update={"bbox_xyxy": sizes[i]}) for i, x in enumerate(ref.boxes[:3]))
    pred = tuple(x.model_copy(update={"bbox_xyxy": sizes[i]}) for i, x in enumerate(predictions[:3]))
    groups = tracking_diagnostics(gt, pred, frames=tuple(keys.values()))["thresholds"]["0.5"]["groups"]
    assert set(groups["size"]) == {"small", "medium", "large"}
    assert groups["occlusion"]["unknown"]["gt"] == 3


def test_id_namespace_is_sequence_local() -> None:
    ref, _, keys, predictions = inputs()
    first = ref.boxes[0]
    sequence = first.frame.sequence.model_copy(update={"sequence_uid": "other"})
    frame = first.frame.model_copy(update={"sequence": sequence})
    second = first.model_copy(update={"annotation_id": "other", "frame": frame})
    pred2 = predictions[0].model_copy(update={"prediction_id": "other", "frame": frame})
    report = tracking_diagnostics((first, second), (predictions[0], pred2), frames=(first.frame, frame))
    assert report["thresholds"]["0.5"]["tracking"]["idf1"] == 1
    assert len(report["thresholds"]["0.5"]["tracking"]["tracks"]) == 2


def test_global_identity_uses_all_potential_pairs_not_greedy_matches() -> None:
    ref, _, keys, _ = inputs()
    boxes = []
    for frame in range(3):
        for track, extent in (("a", (10, 10, 20, 20)), ("b", (14, 10, 24, 20))):
            if frame == 2 and track == "b":
                continue
            boxes.append(
                ref.boxes[0].model_copy(
                    update={
                        "annotation_id": f"{frame}/{track}",
                        "frame": keys[frame],
                        "native_track_id": track,
                        "bbox_xyxy": extent,
                        "source_ordinal": len(boxes),
                    }
                )
            )
    predictions = canonical_predictions(
        [
            Box(frame, "car", extent, track)
            for frame in range(3)
            for track, extent in (("p", (12, 10, 22, 20)), ("q", (8, 10, 18, 20)))
            if frame < 2 or track == "p"
        ],
        keys,
        "adversarial",
    )
    report = tracking_diagnostics(tuple(boxes), predictions, frames=tuple(keys.values()))
    # Potential counts [[3,2],[2,0]]: greedy chooses 3, global assignment 2+2.
    assert report["thresholds"]["0.5"]["tracking"]["idtp"] == 4
    assert report["thresholds"]["0.5"]["tracking"]["idf1"] == 0.8


def test_continuity_outweighs_higher_iou_and_order_is_chronological() -> None:
    ref, _, keys, _ = inputs()
    gt, boxes = [], []
    for frame in (0, 1):
        for track, left in (("a", 10), ("b", 12)):
            gt.append(
                ref.boxes[0].model_copy(
                    update={
                        "annotation_id": f"{frame}/{track}",
                        "frame": keys[frame],
                        "native_track_id": track,
                        "bbox_xyxy": (left, 10, left + 10, 20),
                        "source_ordinal": len(gt),
                    }
                )
            )
            position = left if frame == 0 else 22 - left
            boxes.append(Box(frame, "car", (position, 10, position + 10, 20), track))
    pred = canonical_predictions(boxes, keys, "crossing")
    report = tracking_diagnostics(tuple(gt), pred, frames=(keys[1], keys[0]))
    assert report["thresholds"]["0.5"]["tracking"]["id_switches"] == 0


def test_manual_loader_sorts_frames_and_preserves_unknown(tmp_path: Path) -> None:
    data = json.loads(FIXTURE.read_text())
    data["frames"] = list(reversed(data["frames"]))
    path = tmp_path / "gt.json"
    path.write_text(json.dumps(data))
    ref, occlusions = load_manual_reference(path)
    assert [x.source_ordinal for x in ref.frames] == list(range(6))
    assert occlusions["manual/3"] == "unknown"
