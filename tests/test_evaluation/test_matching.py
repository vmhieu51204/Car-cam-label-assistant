"""Independent tiny known-answer matching cases."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models.evaluation import FrameKey, ObservationSource, PredictionBox, ReferenceBox, Score, SequenceKey
from src.services.evaluation.matching import bbox_iou, match_boxes


def frame(index: int = 0, sequence: str = "0000", split: str = "training") -> FrameKey:
    return FrameKey(sequence=SequenceKey(dataset_id="kitti", split=split, sequence_uid=sequence), frame_uid=str(index))


def reference(
    ident: str = "g",
    ordinal: int = 0,
    label: str = "car",
    key: FrameKey | None = None,
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 10.0, 10.0),
) -> ReferenceBox:
    return ReferenceBox(
        annotation_id=ident,
        native_annotation_id=ident,
        frame=key or frame(),
        class_name=label,
        bbox_xyxy=bbox,
        source_ordinal=ordinal,
        reference_kind="native_2d",
        native_category=label,
        native_track_id="track",
    )


def prediction(
    ident: str = "p",
    ordinal: int = 0,
    label: str = "car",
    key: FrameKey | None = None,
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 10.0, 10.0),
) -> PredictionBox:
    return PredictionBox(
        prediction_id=ident,
        frame=key or frame(),
        class_name=label,
        bbox_xyxy=bbox,
        raw_track_id=ident,
        source=ObservationSource(
            mapping_id="map", source_ordinal=10 - ordinal, config_id="config", source_observation_id=ident
        ),
        scoring_ordinal=ordinal,
        score=Score(value=None, semantics="missing", source="fixture"),
        origin="model",
    )


def test_missing_extra_duplicate_and_wrong_class() -> None:
    gt = (reference(), reference("missing", 1, bbox=(20.0, 0.0, 30.0, 10.0)))
    pred = (prediction(), prediction("duplicate", 1), prediction("wrong", 2, "person", bbox=(20.0, 0.0, 30.0, 10.0)))
    for threshold in (0.5, 0.7):
        result = match_boxes(gt, pred, iou_threshold=threshold)
        assert [(p.annotation_id, p.prediction_id, p.iou) for p in result.pairs] == [("g", "p", 1.0)]
        assert result.unmatched_annotation_ids == ("missing",)
        assert result.unmatched_prediction_ids == ("duplicate", "wrong")
    assert len(match_boxes(gt, pred, iou_threshold=0.5, class_aware=False).pairs) == 2


def test_exact_threshold_inclusive_and_deterministic_ordinals() -> None:
    gt = (reference("later", 1), reference("first", 0))
    pred = (prediction("later", 1, bbox=(0.0, 0.0, 5.0, 10.0)), prediction("first", 0, bbox=(0.0, 0.0, 5.0, 10.0)))
    result = match_boxes(gt, pred, iou_threshold=0.5)
    assert [(p.annotation_id, p.prediction_id) for p in result.pairs] == [("first", "first"), ("later", "later")]
    assert not match_boxes(gt, pred, iou_threshold=0.7).pairs


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_threshold(threshold: float) -> None:
    with pytest.raises(ValueError):
        match_boxes((), (), iou_threshold=threshold)


def test_zero_area_and_nonfinite_rejected() -> None:
    with pytest.raises(ValidationError):
        reference(bbox=(0.0, 0.0, 0.0, 10.0))
    with pytest.raises(ValueError):
        bbox_iou((0.0, 0.0, 10.0, 0.0), (0.0, 0.0, 10.0, 10.0))


def test_empty_and_namespace_isolation() -> None:
    assert not match_boxes((), (), iou_threshold=0.5).pairs
    for key in (frame(sequence="0001"), frame(split="testing"), frame(1)):
        assert not match_boxes((reference(),), (prediction(key=key),), iou_threshold=0.5).pairs


def test_duplicate_ordinals_rejected() -> None:
    with pytest.raises(ValueError):
        match_boxes((reference(), reference("second")), (), iou_threshold=0.5)


def test_point_seven_threshold_is_inclusive() -> None:
    result = match_boxes((reference(),), (prediction(bbox=(0.0, 0.0, 7.0, 10.0)),), iou_threshold=0.7)
    assert result.pairs[0].iou == 0.7


def test_camera_namespaces_are_disjoint() -> None:
    left = FrameKey(
        sequence=SequenceKey(dataset_id="nuscenes", split="mini", sequence_uid="a" * 32, camera_channel="CAM_FRONT"),
        frame_uid="b" * 32,
    )
    right = FrameKey(
        sequence=SequenceKey(dataset_id="nuscenes", split="mini", sequence_uid="a" * 32, camera_channel="CAM_BACK"),
        frame_uid="b" * 32,
    )
    assert not match_boxes((reference(key=left),), (prediction(key=right),), iou_threshold=0.5).pairs
