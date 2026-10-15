from __future__ import annotations

import json
from pathlib import Path

from scripts.eval_autolabel import evaluate, read_cvat_xml
from tools.inject_faults import inject

SAMPLE = Path(__file__).resolve().parents[2] / "eval/sample"


def test_coco_tracking_metrics_are_class_and_time_aware() -> None:
    report = evaluate(SAMPLE / "ground_truth.coco.json", SAMPLE / "predictions.coco.json")["tracking"]
    assert report["thresholds"]["0.5"]["overall"]["recall"] == 5 / 6
    assert report["thresholds"]["0.7"]["overall"]["precision"] == 1
    assert report["thresholds"]["0.5"]["by_class"]["person"]["recall"] == 0.5
    assert report["id_switches"] == 1
    assert report["new_object_recall"] == 0.5
    assert report["recall_by_segment"]["late"] == 0.5


def test_cvat_video_xml_skips_outside_and_preserves_track_id(tmp_path: Path) -> None:
    source = tmp_path / "annotations.xml"
    source.write_text(
        """<annotations><version>1.1</version>
      <track id="7" label="car">
        <box frame="0" outside="0" xtl="1" ytl="2" xbr="11" ybr="12" />
        <box frame="1" outside="1" xtl="1" ytl="2" xbr="11" ybr="12" />
      </track></annotations>""",
        encoding="utf-8",
    )
    boxes = read_cvat_xml(source)
    assert len(boxes) == 1
    assert (boxes[0].frame, boxes[0].track_id, boxes[0].xyxy) == (0, "7", (1, 2, 11, 12))


def test_cvat_video_xml_interpolates_visible_rectangles(tmp_path: Path) -> None:
    source = tmp_path / "annotations.xml"
    source.write_text(
        """<annotations><version>1.1</version>
      <track id="7" label="car">
        <box frame="0" outside="0" xtl="0" ytl="0" xbr="10" ybr="10" />
        <box frame="2" outside="0" xtl="2" ytl="2" xbr="12" ybr="12" />
        <box frame="3" outside="1" xtl="2" ytl="2" xbr="12" ybr="12" />
      </track></annotations>""",
        encoding="utf-8",
    )
    boxes = read_cvat_xml(source)
    assert [box.frame for box in boxes] == [0, 1, 2]
    assert boxes[1].xyxy == (1, 1, 11, 11)


def test_injected_faults_export_images_overlays_and_known_labels(tmp_path: Path) -> None:
    result = inject(SAMPLE / "ground_truth.coco.json", tmp_path)
    assert result["fault_counts"] == {"possible_missing": 1, "wrong_class": 1, "bad_extent": 1, "agree": 1}
    assert len(list((tmp_path / "frames").glob("*.png"))) == 4
    assert len(list((tmp_path / "overlays").glob("*.png"))) == 4
    report = evaluate(SAMPLE / "ground_truth.coco.json", tmp_path / "predictions.coco.json")
    assert report["tracking"]["thresholds"]["0.5"]["overall"]["recall"] < 1
    labels = json.loads((tmp_path / "faults.json").read_text())["faults"]
    assert all(item["raw_is_synthetic"] for item in labels)


def test_audit_confusion_priority_correlation_and_recall_at_k() -> None:
    report = evaluate(
        SAMPLE / "ground_truth.coco.json",
        SAMPLE / "predictions.coco.json",
        faults_path=SAMPLE / "faults.json",
        audit_path=SAMPLE / "audit.json",
        queue_path=SAMPLE / "review-queue.json",
    )["audit"]
    assert report["coverage"] == 4
    assert report["confusion_matrix"]["wrong_class"]["wrong_class"] == 1
    assert report["by_verdict"]["possible_missing"]["precision"] == 1
    assert report["review_recall_at_k"]["50"] == 2 / 3
    assert report["review_distribution"] == {"critical": 1, "high": 1, "normal": 1, "low": 1}
    assert report["priority_severity_pearson"] > 0


def test_review_recall_at_k_without_audit(tmp_path: Path) -> None:
    queue = {
        "units": [
            {"frame_start": index, "priority": priority, "review_score": score}
            for index, priority, score in ((0, "high", 0.3), (1, "critical", 0.9), (2, "normal", 0.2), (3, "low", 0.1))
        ]
    }
    path = tmp_path / "review-queue.json"
    path.write_text(json.dumps(queue))
    report = evaluate(
        SAMPLE / "ground_truth.coco.json",
        SAMPLE / "predictions.coco.json",
        faults_path=SAMPLE / "faults.json",
        queue_path=path,
    )
    assert "audit" not in report
    assert report["review"]["review_distribution"] == {"critical": 1, "high": 1, "normal": 1, "low": 1}
    assert report["review"]["review_recall_at_k"]["50"] == 2 / 3


def test_selected_frame_mapping_does_not_penalize_unselected_ground_truth(tmp_path: Path) -> None:
    manifest = {"frames": [{"frame_index": 0, "source_frame_index": 0}, {"frame_index": 1, "source_frame_index": 2}]}
    tracks = {
        "tracks": [
            {
                "track_id": "a",
                "label_name": "car",
                "observations": [
                    {"frame_index": 0, "bbox": {"xtl": 10, "ytl": 10, "xbr": 50, "ybr": 40}},
                    {"frame_index": 1, "bbox": {"xtl": 14, "ytl": 10, "xbr": 54, "ybr": 40}},
                ],
            }
        ]
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "tracks.json").write_text(json.dumps(tracks))
    report = evaluate(SAMPLE / "ground_truth.coco.json", tmp_path)["tracking"]
    assert report["evaluated_frames"] == [0, 2]
    assert report["thresholds"]["0.5"]["overall"]["recall"] == 2 / 3


def test_legacy_exports_and_ties_remain_compatible() -> None:
    from scripts.eval_autolabel import Box, _matches, iou, tracking_metrics
    from src.services.evaluation import legacy

    assert Box is legacy.Box and _matches is legacy._matches
    gt = [Box(0, "car", (0.0, 0.0, 10.0, 10.0), "g")]
    pred = [Box(0, "car", (0.0, 0.0, 10.0, 10.0), "first"), Box(0, "car", (0.0, 0.0, 10.0, 10.0), "second")]
    assert _matches(gt, pred, 0.5) == [(0, 0, 1.0)]
    assert iou(gt[0], pred[0]) == 1.0
    overall = tracking_metrics(gt, pred)["thresholds"]["0.5"]["overall"]
    assert overall == {"gt": 1, "predicted": 2, "tp": 1, "precision": 0.5, "recall": 1.0, "mean_iou": 1.0}
    assert tracking_metrics(gt, pred, {1})["evaluated_frames"] == [1]
    assert tracking_metrics(gt, pred, {1})["new_object_recall"] is None
