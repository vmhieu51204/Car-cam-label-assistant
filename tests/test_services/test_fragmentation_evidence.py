from collections import defaultdict

import pytest

from scripts.eval_autolabel import Box
from src.models.evaluation import FrameKey, ReferenceBox, SequenceKey
from src.services.evaluation.fragmentation import fragmentation_evidence, fragmentation_summary
from src.services.evaluation.phase0 import canonical_predictions
from src.services.evaluation.tracking import _temporal
from src.services.evaluation.tracking_diagnosis import event_breakdown, event_ci


def fixture(boxes, *, threshold=0.5, gt_frames=(0, 1, 2, 3)):
    keys = {
        i: FrameKey(sequence=SequenceKey(dataset_id="kitti", split="training", sequence_uid="0013"), frame_uid=str(i))
        for i in range(4)
    }
    gt = [
        ReferenceBox(
            annotation_id=f"gt-{i}",
            native_annotation_id=f"gt-{i}",
            frame=keys[i],
            class_name="car",
            bbox_xyxy=(0, 0, 10, 10),
            native_track_id="g",
            source_ordinal=i,
            reference_kind="native_2d",
            native_category="Car",
        )
        for i in gt_frames
    ]
    pred = canonical_predictions(
        [Box(0, "car", (0, 0, 10, 10), "p"), *boxes, Box(3, "car", (0, 0, 10, 10), "p")], keys, "test"
    )
    frames = tuple(keys.values())
    plain = _temporal(gt, list(pred), frames, threshold)
    ledger = _temporal(gt, list(pred), frames, threshold, include_events=True)
    assert {k: v for k, v in ledger.items() if k != "events"} == plain
    by_frame = defaultdict(list)
    for p in pred:
        by_frame[p.frame].append(p)
    events = []
    for e in ledger["events"]:
        if e["kind"] == "fragmentation":
            e.update(sequence="0013", iou_threshold=threshold)
            e.update(
                fragmentation_evidence(
                    e, {g.annotation_id: g for g in gt}, by_frame, {"gt-1": "partial", "gt-2": "heavy"}
                )
            )
            events.append(e)
    return events


@pytest.mark.parametrize(
    "box,category",
    [
        (Box(1, "car", (0, 0, 2, 10), "p"), "continuing_track_low_iou"),
        (Box(1, "person", (0, 0, 10, 10), "q"), "other_class"),
        (Box(1, "car", (30, 0, 40, 10), "p"), "no_prediction"),
        (Box(1, "car", (0, 0, 2, 10), "q"), "unresolved"),
    ],
)
def test_missed_frame_evidence_and_mixed_events(box, category):
    event = fixture([box])[0]
    assert event["previous_matched_source_frame"] == 0 and event["source_frame"] == 3
    assert event["missed_annotation_ids"] == ["gt-1", "gt-2"]
    assert event["missed_frames"][0]["category"] == category
    assert event["missed_frames"][0]["occlusion"] == "partial"
    assert event["fragmentation_type"] == (category if category == "no_prediction" else "mixed")


def test_iou_bounds_and_point_seven_localization_dip():
    events = fixture([Box(i, "car", (0, 0, 1, 10), "p") for i in (1, 2)])
    assert events[0]["fragmentation_type"] == "continuing_track_low_iou"  # exactly .1
    events = fixture([Box(i, "car", (0, 0, 5, 10), "p") for i in (1, 2)], threshold=0.7)
    assert events[0]["fragmentation_type"] == "continuing_track_below_threshold"  # exactly .5
    assert not fixture([Box(i, "car", (0, 0, 7, 10), "p") for i in (1, 2)], threshold=0.7)
    events = fixture([Box(i, "car", (0, 0, 5, 10), "new") for i in (1, 2)], threshold=0.7)
    assert events[0]["fragmentation_type"] == "id_break"


def test_absent_gt_is_not_a_missed_frame_and_low_iou_priority():
    event = fixture([Box(1, "car", (0, 0, 2, 10), "p"), Box(1, "person", (0, 0, 10, 10), "q")], gt_frames=(0, 1, 3))[0]
    assert [r["source_frame"] for r in event["missed_frames"]] == [1]
    assert event["fragmentation_type"] == "continuing_track_low_iou"


def test_track_bootstrap_counts_occlusion_and_minimum_support():
    event = fixture([Box(i, "car", (0, 0, 2, 10), "p") for i in (1, 2)])[0]
    one = fragmentation_summary([event], repeats=100)
    assert one["exclusive_event_types"]["continuing_track_low_iou"]["availability"] == "unavailable"
    two = fragmentation_summary([event, {**event, "gt_id": "g2"}], repeats=100)
    assert two == fragmentation_summary([event, {**event, "gt_id": "g2"}], repeats=100)
    assert two["exclusive_event_types"]["continuing_track_low_iou"]["ci95"] == [1, 1]
    assert two["continuing_low_iou_majority"]
    assert two["by_occlusion"]["heavy"]["frame_counts"] == {"continuing_track_low_iou": 2}
    assert sum(v["count"] for v in two["exclusive_event_types"].values()) == two["event_count"]
    assert fragmentation_summary([])["event_count"] == 0


def test_event_ci_uses_supporting_tracks_not_all_denominator_tracks():
    events = fixture([Box(i, "car", (0, 0, 2, 10), "p") for i in (1, 2)])
    frame = FrameKey(sequence=SequenceKey(dataset_id="kitti", split="training", sequence_uid="0013"), frame_uid="1")
    gt = tuple(
        ReferenceBox(
            annotation_id=f"g-{identity}",
            native_annotation_id=f"g-{identity}",
            frame=frame,
            class_name="car",
            bbox_xyxy=(0, 0, 10, 10),
            native_track_id=identity,
            source_ordinal=i,
            reference_kind="native_2d",
            native_category="Car",
        )
        for i, identity in enumerate(("g", "g2", "g3"))
    )
    events[0]["signal"] = "within_window_gap"
    one = event_ci(gt, events, repeats=100)
    assert one["denominator_tracks"] == 3 and one["supporting_tracks"] == 1
    assert one["availability"] == "unavailable" and one["ci95"] is None
    two = events + [{**events[0], "gt_id": "g2"}]
    assert event_ci(gt, two, repeats=100) == event_ci(gt, two, repeats=100)
    assert event_ci(gt, two, repeats=100)["availability"] == "available"
    breakdown = event_breakdown(gt, two)
    assert sum(c["event_count"] for c in breakdown["by_signal"].values()) == 2
    assert sum(c["event_count"] for c in breakdown["by_kind"].values()) == 2
