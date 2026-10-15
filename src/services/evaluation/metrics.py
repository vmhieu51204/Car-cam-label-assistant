"""Count-derived local diagnostics and frozen, explicit metadata grouping."""

from __future__ import annotations

from collections import defaultdict

from src.models.evaluation import (
    DiagnosticGroupingOptions,
    DiagnosticMetrics,
    FrameKey,
    MatchResult,
    MetricGroup,
    MetricValue,
    NamedMetric,
    PredictionBox,
    PreparedEvaluationView,
    ReferenceBox,
    SupportCount,
    canonical_json,
)

SLICE_REVISION = "diagnostic-slices/v1"


def _diagnostic(tp: int, fp: int, fn: int, iou_sum: float) -> DiagnosticMetrics:
    support = tuple(
        SupportCount(name=name, count=count)
        for name, count in (("tp", tp), ("fp", fp), ("fn", fn), ("matched_count", tp))
    )

    def rate(name: str, numerator: float, denominator: int) -> MetricValue:
        return MetricValue(
            value=numerator / denominator if denominator else None,
            unit="ratio",
            scale="0_to_1",
            numerator=numerator,
            denominator=float(denominator),
            support=support,
            availability="available" if denominator else "undefined_denominator",
            reason=None if denominator else "zero denominator",
            definition_id=f"local-box-{name}/v1",
        )

    return DiagnosticMetrics(
        tp=tp,
        fp=fp,
        fn=fn,
        iou_sum=iou_sum,
        matched_count=tp,
        precision=rate("precision", float(tp), tp + fp),
        recall=rate("recall", float(tp), tp + fn),
        f1=rate("f1", float(2 * tp), 2 * tp + fp + fn),
        matched_mean_iou=rate("matched-mean-iou", iou_sum, tp),
    )


def _validate(matches: MatchResult, view: PreparedEvaluationView, declared_classes: tuple[str, ...]) -> None:
    if not matches.class_aware:
        raise ValueError("class-agnostic matches are spatial diagnostics, not primary box metrics")
    if not declared_classes or len(set(declared_classes)) != len(declared_classes):
        raise ValueError("declare a unique nonempty taxonomy")
    gt = {x.annotation_id: x for x in view.reference_boxes}
    pred = {x.prediction_id: x for x in view.prediction_boxes}
    if any(x.class_name not in declared_classes for x in (*gt.values(), *pred.values())):
        raise ValueError("box outside declared taxonomy")
    if set(gt) != {p.annotation_id for p in matches.pairs} | set(matches.unmatched_annotation_ids):
        raise ValueError("match reference IDs do not cover prepared view")
    if set(pred) != {p.prediction_id for p in matches.pairs} | set(matches.unmatched_prediction_ids):
        raise ValueError("match prediction IDs do not cover prepared view")
    for pair in matches.pairs:
        left, right = gt[pair.annotation_id], pred[pair.prediction_id]
        if left.frame != right.frame or pair.frame != left.frame or left.class_name != right.class_name:
            raise ValueError("pair crosses canonical frame/class")


def box_metrics(
    matches: MatchResult, *, view: PreparedEvaluationView, declared_classes: tuple[str, ...]
) -> DiagnosticMetrics:
    _validate(matches, view, declared_classes)
    return _diagnostic(
        len(matches.pairs),
        len(matches.unmatched_prediction_ids),
        len(matches.unmatched_annotation_ids),
        sum(x.iou for x in matches.pairs),
    )


def group_box_metrics(
    matches: MatchResult,
    *,
    view: PreparedEvaluationView,
    declared_classes: tuple[str, ...],
    options: DiagnosticGroupingOptions,
) -> tuple[MetricGroup, ...]:
    """Group existing matches; never rematch or average rates to form micro metrics.

    diagnostic-slices/v1: height [0,25),[25,50),[50,100),[100,inf);
    distance [0,10),[10,20),[20,40),[40,inf). Native difficulty uses exact
    values. Temporal thirds divide full per-sequence timestamp extent, falling
    back to source ordinal. Matched predictions inherit GT slice; unmatched
    predictions use their frame/height or explicit unattributed policy. Native
    difficulty and distance are unavailable for unmatched predictions.
    """
    _validate(matches, view, declared_classes)
    if options.slices and options.slice_revision != SLICE_REVISION:
        raise ValueError("unsupported diagnostic slice revision")
    gt = {x.annotation_id: x for x in view.reference_boxes}
    pred = {x.prediction_id: x for x in view.prediction_boxes}
    native = {x.source_id: x for x in view.native_kitti_rows}
    frames = {x.key: x for x in view.reference_frames}
    sequence_frames: dict[object, list[FrameKey]] = defaultdict(list)
    for frame_key in frames:
        sequence_frames[frame_key.sequence].append(frame_key)

    def temporal(frame: FrameKey) -> str:
        keys = sequence_frames[frame.sequence]
        if not keys or frame not in frames:
            return "unavailable"
        recorded = all(frames[k].timestamp_us is not None for k in keys)
        values = {k: float(frames[k].timestamp_us or 0) if recorded else float(frames[k].source_ordinal) for k in keys}
        low, high = min(values.values()), max(values.values())
        third = min(2, int(3 * (values[frame] - low) / (high - low))) if high > low else 0
        provenance = frames[frame].timestamp_provenance if recorded else "source_ordinal"
        return f"{provenance}:{('early', 'middle', 'late')[third]}"

    def slice_key(row: ReferenceBox | PredictionBox, name: str) -> str:
        if name == "temporal_third":
            return temporal(row.frame)
        if name == "box_height":
            value = row.bbox_xyxy[3] - row.bbox_xyxy[1]
            return next(
                (label for limit, label in ((25, "0_25"), (50, "25_50"), (100, "50_100")) if value < limit), "100_plus"
            )
        if not isinstance(row, ReferenceBox):
            return "unavailable_gt_metadata"
        if name == "ego_distance":
            distance = row.metadata.ego_distance_m
            if distance is None:
                return "unavailable_gt_metadata"
            return next(
                (label for limit, label in ((10, "0_10"), (20, "10_20"), (40, "20_40")) if distance < limit), "40_plus"
            )
        source = native.get(row.metadata.kitti_source_id or "")
        if source is None:
            return "unavailable_gt_metadata"
        return str(source.occluded if name == "occlusion" else source.truncated)

    result = []
    for dimension in options.dimensions:
        slices = options.slices if dimension == "slice" else (None,)
        for slice_name in slices:

            def key(row: ReferenceBox | PredictionBox, *, unmatched: bool = False) -> str:
                if dimension == "micro":
                    return "all"
                if dimension == "per_class":
                    return row.class_name
                if dimension == "per_sequence":
                    return canonical_json(row.frame.sequence).decode()
                if dimension == "per_camera":
                    return row.frame.sequence.camera_channel or "unavailable_camera"
                if unmatched and options.unmatched_prediction_attribution == "separate_unattributed":
                    return "unattributed_prediction"
                assert slice_name is not None
                return slice_key(row, slice_name)

            counts: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
            sequences = {k.sequence for k in view.scope.frames}
            if dimension == "micro":
                counts["all"]
            if dimension == "per_class":
                for label in declared_classes:
                    counts[label]
            if dimension == "per_sequence":
                for sequence in sequences:
                    counts[canonical_json(sequence).decode()]
            if dimension == "per_camera":
                for sequence in sequences:
                    counts[sequence.camera_channel or "unavailable_camera"]
            for pair in matches.pairs:
                entry = counts[key(gt[pair.annotation_id])]
                entry[0] += 1
                entry[3] += pair.iou
            for ident in matches.unmatched_annotation_ids:
                counts[key(gt[ident])][2] += 1
            for ident in matches.unmatched_prediction_ids:
                counts[key(pred[ident], unmatched=True)][1] += 1
            for label, entry in sorted(counts.items()):
                diagnostic = _diagnostic(int(entry[0]), int(entry[1]), int(entry[2]), entry[3])
                seq = (
                    next((s for s in sequences if canonical_json(s).decode() == label), None)
                    if dimension == "per_sequence"
                    else None
                )
                metrics = tuple(
                    NamedMetric(name=n, metric=getattr(diagnostic, n))
                    for n in ("precision", "recall", "f1", "matched_mean_iou")
                )
                result.append(
                    MetricGroup(
                        group_id=f"{dimension}:{slice_name or 'none'}:{label}",
                        aggregation=dimension,
                        class_name=label if dimension == "per_class" else None,
                        sequence=seq,
                        slice_name=f"{slice_name}:{label}" if slice_name else None,
                        metrics=metrics,
                        diagnostic=diagnostic,
                    )
                )
    return tuple(result)
