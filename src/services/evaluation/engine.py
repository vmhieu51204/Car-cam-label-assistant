"""Offline benchmark evaluation engine coordinating matching, metrics, coverage, and arrivals."""

from __future__ import annotations

from pathlib import Path

from src.models.evaluation import (
    BenchmarkManifest,
    DatasetReference,
    MatchResult,
    PredictionRun,
    RunEvaluation,
    ScopeEvaluation,
    ScopeSet,
)
from src.services.evaluation.arrivals import evaluate_arrivals
from src.services.evaluation.coverage import build_scopes, evaluate_coverage
from src.services.evaluation.matching import match_boxes
from src.services.evaluation.metrics import group_box_metrics
from src.services.evaluation.references.kitti import prepare_local_profile


def evaluate_run(
    reference: DatasetReference,
    run: PredictionRun,
    *,
    manifest: BenchmarkManifest,
    dataset_root: Path,
    scopes: ScopeSet | None = None,
) -> RunEvaluation:
    """Evaluate one prediction run across fixed, observed, and pairwise scopes."""
    if scopes is None:
        scopes = build_scopes(manifest, (run,))

    # Scopes matching this run's config and view
    active_scopes = [
        s for s in scopes.scopes if any(r.config_id == run.config_id and r.view == run.view for r in s.run_ids)
    ]

    scope_evals: list[ScopeEvaluation] = []
    thresholds = manifest.matching.iou_thresholds

    for scope in active_scopes:
        prepared = prepare_local_profile(reference, run, manifest.profile, scope=scope)
        matches: list[MatchResult] = []
        for thresh in thresholds:
            m = match_boxes(
                prepared.reference_boxes,
                prepared.prediction_boxes,
                iou_threshold=thresh,
                class_aware=True,
            )
            matches.append(m)

        # Primary grouping metrics on first threshold (0.5)
        primary_match = (
            matches[0]
            if matches
            else match_boxes(prepared.reference_boxes, prepared.prediction_boxes, iou_threshold=0.5, class_aware=True)
        )
        groups = group_box_metrics(
            primary_match,
            view=prepared,
            declared_classes=manifest.profile.classes,
            options=manifest.grouping,
        )

        cov = evaluate_coverage(scope, reference, run)
        arr = evaluate_arrivals(reference, run, manifest=manifest, scope=scope)

        scope_evals.append(
            ScopeEvaluation(
                scope=scope,
                coverage=cov,
                matches=tuple(matches),
                groups=groups,
                arrivals=arr,
                prepared_view=prepared,
            )
        )

    seq_set = set(manifest.sequences)
    ref_frames = tuple(f for f in reference.frames if f.key.sequence in seq_set)
    native_rows = tuple(r for r in reference.native_kitti_rows if r.frame.sequence in seq_set)

    return RunEvaluation(
        config_id=run.config_id,
        view=run.view,
        cohort=manifest.cohort,
        status=run.status,
        raw_status=run.raw_status,
        provenance=run.provenance,
        runtime=run.runtime,
        scopes=tuple(scope_evals),
        processing=run.processing,
        windows=run.windows,
        normalization=run.normalization,
        normalization_sha256=run.normalization_sha256,
        reference_frames=ref_frames,
        native_kitti_rows=native_rows,
        failure_reason=run.failure_reason,
    )
