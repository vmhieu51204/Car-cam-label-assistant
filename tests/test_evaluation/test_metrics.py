"""Count, grouping and metadata supports computed by hand."""

from __future__ import annotations

import pytest

from src.models.evaluation import (
    BoxMetadata,
    DiagnosticGroupingOptions,
    EvaluationScope,
    FrameReference,
    InterpolationPolicy,
    PreparedEvaluationView,
    ReferenceProfile,
    RunKey,
    SourceFile,
)
from src.services.evaluation.matching import match_boxes
from src.services.evaluation.metrics import box_metrics, group_box_metrics
from tests.test_evaluation.test_matching import frame, prediction, reference


def prepared(gt: tuple = (), pred: tuple = (), keys: tuple | None = None, frames: tuple = ()) -> PreparedEvaluationView:
    keys = keys or tuple(dict.fromkeys(x.frame for x in (*gt, *pred)))
    profile = ReferenceProfile(
        profile_id="raw",
        revision="raw/v1",
        designation="raw",
        classes=("car", "person"),
        class_map=(),
        eligibility_policy="all/v1",
        geometry_policy="raw/v1",
    )
    scope = EvaluationScope(
        scope_id="scope",
        kind="fixed_set",
        frames=keys,
        profile_id="raw",
        view="raw",
        cohort="text_discovery",
        target_population="all_objects",
        interpolation=InterpolationPolicy(revision="none/v1", algorithm="none"),
        run_ids=(RunKey(config_id="config", view="raw"),),
    )
    return PreparedEvaluationView(
        scope=scope, profile=profile, reference_boxes=gt, prediction_boxes=pred, reference_frames=frames
    )


def test_counts_micro_classes_sequences_and_absent_class() -> None:
    gt = (reference(), reference("miss", 1), reference("other", key=frame(sequence="0001")))
    pred = (
        prediction(),
        prediction("extra", 1, bbox=(20.0, 0.0, 30.0, 10.0)),
        prediction("otherp", key=frame(sequence="0001")),
    )
    view = prepared(gt, pred)
    matches = match_boxes(gt, pred, iou_threshold=0.5)
    metric = box_metrics(matches, view=view, declared_classes=("car", "person"))
    assert (metric.tp, metric.fp, metric.fn, metric.iou_sum) == (2, 1, 1, 2.0)
    assert metric.precision.value == metric.recall.value == metric.f1.value == 2 / 3
    assert metric.matched_mean_iou.value == 1
    groups = group_box_metrics(
        matches, view=view, declared_classes=("car", "person"), options=DiagnosticGroupingOptions()
    )
    absent = next(g for g in groups if g.class_name == "person").diagnostic
    assert absent is not None and absent.precision.value is None and absent.recall.denominator == 0
    sequences = [g.diagnostic for g in groups if g.aggregation == "per_sequence"]
    assert sum(g.tp for g in sequences if g) == 2
    assert sorted(g.recall.value for g in sequences if g) == [0.5, 1.0]


@pytest.mark.parametrize(
    "has_gt,has_pred,expected",
    [(False, False, (None, None, None)), (True, False, (None, 0.0, 0.0)), (False, True, (0.0, None, 0.0))],
)
def test_empty_denominators(has_gt: bool, has_pred: bool, expected: tuple) -> None:
    gt = (reference(),) if has_gt else ()
    pred = (prediction(),) if has_pred else ()
    view = prepared(gt, pred)
    metric = box_metrics(match_boxes(gt, pred, iou_threshold=0.5), view=view, declared_classes=("car", "person"))
    assert (metric.precision.value, metric.recall.value, metric.f1.value) == expected
    assert metric.matched_mean_iou.value is None
    assert metric.matched_mean_iou.availability == "undefined_denominator"


def test_wrong_class_and_reject_spatial_primary() -> None:
    gt = (reference(),)
    pred = (prediction(label="person"),)
    view = prepared(gt, pred)
    result = box_metrics(match_boxes(gt, pred, iou_threshold=0.5), view=view, declared_classes=("car", "person"))
    assert (result.tp, result.fp, result.fn) == (0, 1, 1)
    with pytest.raises(ValueError):
        box_metrics(
            match_boxes(gt, pred, iou_threshold=0.5, class_aware=False), view=view, declared_classes=("car", "person")
        )
    with pytest.raises(ValueError):
        box_metrics(match_boxes((), (), iou_threshold=0.5), view=view, declared_classes=("car", "person"))


def test_full_timeline_slices_and_unavailable_fp_difficulty() -> None:
    frames = tuple(
        FrameReference(
            key=frame(i),
            source_ordinal=i,
            width=100,
            height=100,
            media=SourceFile(relative_path=f"{i}.png", sha256="a" * 64, role="image"),
            timestamp_us=i * 100000,
            timestamp_provenance="declared_fps",
        )
        for i in range(10)
    )
    row = reference(key=frame(8))
    row = type(row).model_validate({**row.model_dump(), "metadata": BoxMetadata(ego_distance_m=10.0)})
    gt = (row,)
    pred = (prediction(key=frame(8)), prediction("extra", 1, key=frame(8), bbox=(20.0, 0.0, 30.0, 25.0)))
    view = prepared(gt, pred, frames=frames)
    options = DiagnosticGroupingOptions(
        dimensions=("slice",),
        slices=("temporal_third", "box_height", "occlusion", "ego_distance"),
        slice_revision="diagnostic-slices/v1",
    )
    groups = group_box_metrics(
        match_boxes(gt, pred, iou_threshold=0.5), view=view, declared_classes=("car", "person"), options=options
    )
    assert any(
        g.slice_name == "temporal_third:declared_fps:late" and g.diagnostic and g.diagnostic.tp == 1 for g in groups
    )
    assert any(g.slice_name == "box_height:25_50" and g.diagnostic and g.diagnostic.fp == 1 for g in groups)
    assert any(g.slice_name == "ego_distance:10_20" and g.diagnostic and g.diagnostic.tp == 1 for g in groups)
    assert any(g.slice_name == "occlusion:unavailable_gt_metadata" for g in groups)


def test_slice_policy_and_unknown_revision() -> None:
    pred = (prediction(),)
    view = prepared((), pred)
    matches = match_boxes((), pred, iou_threshold=0.5)
    options = DiagnosticGroupingOptions(
        dimensions=("slice",),
        slices=("box_height",),
        slice_revision="diagnostic-slices/v1",
        unmatched_prediction_attribution="separate_unattributed",
    )
    groups = group_box_metrics(matches, view=view, declared_classes=("car", "person"), options=options)
    assert groups[0].slice_name == "box_height:unattributed_prediction"
    assert groups[0].diagnostic and groups[0].diagnostic.fp == 1
    unknown = DiagnosticGroupingOptions(dimensions=("slice",), slices=("box_height",), slice_revision="unknown/v1")
    with pytest.raises(ValueError, match="unsupported"):
        group_box_metrics(matches, view=view, declared_classes=("car", "person"), options=unknown)


def test_native_difficulty_is_exact_and_never_imputed_to_fp() -> None:
    from src.models.evaluation import NativeKittiRow

    row = NativeKittiRow(
        frame=frame(),
        source_id="native",
        source_file=SourceFile(relative_path="labels.txt", sha256="b" * 64, role="labels"),
        line_number=1,
        source_frame_index=0,
        track_id=1,
        native_class="Car",
        truncated=0.5,
        occluded=2,
        alpha=0.0,
        bbox_xyxy=(0.0, 0.0, 10.0, 10.0),
        dimensions_hwl=(1.0, 1.0, 1.0),
        location_xyz=(0.0, 0.0, 5.0),
        rotation_y=0.0,
    )
    gtrow = reference()
    gtrow = type(gtrow).model_validate({**gtrow.model_dump(), "metadata": BoxMetadata(kitti_source_id="native")})
    view = prepared((gtrow,), (prediction(), prediction("extra", 1, bbox=(20.0, 0.0, 30.0, 10.0))))
    view = PreparedEvaluationView.model_validate({**view.model_dump(), "native_kitti_rows": (row,)})
    options = DiagnosticGroupingOptions(
        dimensions=("slice",), slices=("occlusion", "truncation"), slice_revision="diagnostic-slices/v1"
    )
    groups = group_box_metrics(
        match_boxes(view.reference_boxes, view.prediction_boxes, iou_threshold=0.5),
        view=view,
        declared_classes=("car", "person"),
        options=options,
    )
    assert any(g.slice_name == "occlusion:2" and g.diagnostic and g.diagnostic.tp == 1 for g in groups)
    assert any(g.slice_name == "truncation:0.5" and g.diagnostic and g.diagnostic.tp == 1 for g in groups)
    assert any(
        g.slice_name == "occlusion:unavailable_gt_metadata" and g.diagnostic and g.diagnostic.fp == 1 for g in groups
    )
