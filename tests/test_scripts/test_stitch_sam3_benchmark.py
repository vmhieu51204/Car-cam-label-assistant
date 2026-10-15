"""Stitching an already-run benchmark must reduce boundary ID changes."""

from scripts.eval_autolabel import Box
from scripts.stitch_sam3_benchmark import compare_stitch


def test_overlap_stitch_reuses_identity_on_next_window():
    gt = [Box(frame, "car", (frame * 2, 0, frame * 2 + 10, 10), "gt") for frame in range(4)]
    first = [
        {
            "window_id": 0,
            "frame_index": frame,
            "track_id": "a",
            "label": "car",
            "bbox": [frame * 2, 0, frame * 2 + 10, 10],
            "mask_rle": None,
        }
        for frame in (0, 1, 2)
    ]
    second = [
        {
            "window_id": 1,
            "frame_index": frame,
            "track_id": "b",
            "label": "car",
            "bbox": [frame * 2, 0, frame * 2 + 10, 10],
            "mask_rle": None,
        }
        for frame in (1, 2, 3)
    ]
    item = {
        "evaluated_source_frames": [0, 1, 2, 3],
        "window_predictions": first + second,
        "predictions": [
            {key: value for key, value in row.items() if key not in {"window_id", "mask_rle"}}
            for row in first + second
            if row["frame_index"] != 3 or row["window_id"] == 1
            if row["window_id"] == 0 or row["frame_index"] == 3
        ],
    }
    report, stitched = compare_stitch(item, gt)
    assert report["before"]["id_switches"] == 1
    assert report["after"]["id_switches"] == 0
    assert len(stitched.tracks) == 1
    assert report["decisions"][0]["action"] == "matched"
