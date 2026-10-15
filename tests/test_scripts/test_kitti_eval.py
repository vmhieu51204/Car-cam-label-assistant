"""Focused checks for KITTI ignore rules on a selected-frame benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.compare_sam3_configs import compare
from scripts.eval_autolabel import Box
from scripts.kitti_eval import KittiItem, KittiReference, evaluate_kitti_style, load_reference


def _line(
    frame: int, track: int, category: str, box: tuple[int, int, int, int], truncation: str = "0.00", occlusion: int = 0
) -> str:
    x1, y1, x2, y2 = box
    return f"{frame} {track} {category} {truncation} {occlusion} -10 {x1} {y1} {x2} {y2} 0 0 0 0 0 0 0"


def _reference(tmp_path: Path):
    labels = tmp_path / "0000.txt"
    labels.write_text(
        "\n".join(
            [
                _line(2, 1, "Car", (0, 0, 100, 100)),
                _line(2, 2, "Van", (120, 0, 220, 100)),
                _line(2, 3, "Car", (240, 0, 340, 100), "1.00"),
                _line(2, -1, "DontCare", (360, 0, 460, 100), "-1", -1),
                _line(1, 4, "Car", (0, 0, 100, 100)),
            ]
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "frames": [
                    {"frame_index": 0, "source_frame_index": 0},
                    {"frame_index": 1, "source_frame_index": 2},
                ]
            }
        ),
        encoding="utf-8",
    )
    return load_reference(labels, manifest)


def _predictions() -> list[Box]:
    return [
        Box(1, "car", box, str(index))
        for index, box in enumerate(
            [
                (0, 0, 100, 100),
                (120, 0, 220, 100),
                (240, 0, 340, 100),
                (360, 0, 460, 100),
                (500, 0, 530, 20),
                (600, 0, 700, 100),
            ]
        )
    ]


def test_kitti_style_ignores_distractor_truncation_dontcare_and_small(tmp_path: Path) -> None:
    reference = _reference(tmp_path)
    assert len(reference.boxes) == 3  # source frame 1 was not selected
    assert {item.box.frame for item in reference.boxes} == {1}
    result = evaluate_kitti_style(reference, _predictions(), {0, 1})
    rates = result["metrics"]["thresholds"]["0.5"]["overall"]
    assert (rates["gt"], rates["predicted"], rates["tp"]) == (1, 2, 1)
    assert (rates["precision"], rates["recall"]) == (0.5, 1.0)
    assert result["ignored_predictions"] == {"distractor": 2, "small": 1, "dontcare": 1}


def test_kitti_style_does_not_ignore_small_prediction_matched_to_valid_gt() -> None:
    reference = KittiReference((KittiItem(Box(1, "car", (0, 0, 100, 20), "gt"), True),), {})
    prediction = Box(1, "car", (0, 0, 100, 20), "track")
    result = evaluate_kitti_style(reference, [prediction], {1})
    assert result["ignored_predictions"]["small"] == 0
    assert result["retained_predictions"] == 1
    assert result["metrics"]["thresholds"]["0.5"]["overall"]["tp"] == 1


def test_compare_exposes_adjusted_rates_without_changing_raw_rates(tmp_path: Path) -> None:
    reference = _reference(tmp_path)
    predictions = _predictions()
    result = {
        "configurations": [
            {
                "id": "optical-text",
                "status": "completed",
                "config": {
                    "selection": "optical",
                    "window_size": 50,
                    "overlap": 0,
                    "reprompt": "box",
                    "mode": "text_prompt",
                },
                "evaluated_source_frames": [1],
                "predictions": [
                    {"frame_index": box.frame, "label": box.label, "bbox": box.xyxy, "track_id": box.track_id}
                    for box in predictions
                ],
            }
        ]
    }
    rows = compare(result, [item.box for item in reference.boxes if item.valid], kitti_reference=reference)
    assert rows[0]["precision_iou_05"] == pytest.approx(1 / 6)
    assert rows[0]["kitti_style_precision_iou_05"] == 0.5
    assert rows[0]["kitti_style_recall_iou_05"] == 1.0
    assert rows[0]["kitti_style_precision_iou_07"] == 0.5
    assert rows[0]["kitti_style_recall_iou_07"] == 1.0
    assert rows[0]["kitti_style_id_switches"] == 0


def test_hungarian_ignore_assignment_retained_before_greedy_diagnostics() -> None:
    reference = KittiReference(
        (
            KittiItem(Box(0, "car", (0, 0, 100, 100), "target"), True),
            KittiItem(Box(0, "car", (30, 0, 130, 100), "van"), False),
        ),
        {},
    )
    predictions = [Box(0, "car", (10, 0, 110, 100), "p"), Box(0, "car", (-10, 0, 90, 100), "q")]
    result = evaluate_kitti_style(reference, predictions, {0})
    assert result["ignored_predictions"]["distractor"] == 1
    assert result["retained_predictions"] == 1
    assert result["metrics"]["thresholds"]["0.5"]["overall"]["tp"] == 1


def test_ignore_counts_do_not_collapse_repeated_prediction_track_ids() -> None:
    predictions = [Box(0, "car", (0, 0, 20, 20), "same"), Box(0, "car", (50, 0, 70, 20), "same")]
    result = evaluate_kitti_style(KittiReference((), {}), predictions, {0})
    assert result["ignored_predictions"]["small"] == 2
