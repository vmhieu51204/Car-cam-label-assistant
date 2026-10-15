"""Scoping, frame coverage, and observation support computation."""

from __future__ import annotations

from src.models.evaluation import (
    BenchmarkManifest,
    CoverageResult,
    DatasetReference,
    EvaluationScope,
    Evidence,
    MetricValue,
    PredictionRun,
    RunKey,
    ScopeSet,
)


def build_scopes(manifest: BenchmarkManifest, runs: tuple[PredictionRun, ...]) -> ScopeSet:
    """Build fixed, observed, and pairwise common evaluation scopes."""
    scopes: list[EvaluationScope] = []
    excluded_seeds = set(manifest.excluded_seed_frames)
    eligible_annotations = tuple(b.annotation_id for b in manifest.seed_set.seeds) if manifest.seed_set.seeds else ()

    for run in runs:
        run_key = RunKey(config_id=run.config_id, view=run.view)

        # 1. Fixed-set scope
        fixed_frames = tuple(f for f in manifest.planned_frames if f not in excluded_seeds)
        fixed_scope = EvaluationScope(
            scope_id=f"{run.config_id}/{run.view}/fixed",
            kind="fixed_set",
            frames=fixed_frames,
            profile_id=manifest.profile.profile_id,
            view=run.view,
            cohort=manifest.cohort,
            target_population="all_objects",
            excluded_seed_frames=manifest.excluded_seed_frames,
            interpolation=run.normalization.interpolation,
            run_ids=(run_key,),
        )
        scopes.append(fixed_scope)

        # 2. Observed-frame scope
        observed_frames = tuple(
            fp.frame for fp in run.processing if fp.state == "processed" and fp.frame not in excluded_seeds
        )
        observed_scope = EvaluationScope(
            scope_id=f"{run.config_id}/{run.view}/observed",
            kind="observed",
            frames=observed_frames,
            profile_id=manifest.profile.profile_id,
            view=run.view,
            cohort=manifest.cohort,
            target_population="all_objects",
            excluded_seed_frames=manifest.excluded_seed_frames,
            interpolation=run.normalization.interpolation,
            run_ids=(run_key,),
        )
        scopes.append(observed_scope)

        # 3. Seeded propagation scope (if seeded cohort)
        if manifest.cohort != "text_discovery" and eligible_annotations:
            seeded_scope = EvaluationScope(
                scope_id=f"{run.config_id}/{run.view}/seeded",
                kind="fixed_set",
                frames=fixed_frames,
                profile_id=manifest.profile.profile_id,
                view=run.view,
                cohort=manifest.cohort,
                target_population="seeded_objects",
                excluded_seed_frames=manifest.excluded_seed_frames,
                interpolation=run.normalization.interpolation,
                run_ids=(run_key,),
                eligible_annotation_ids=eligible_annotations,
            )
            scopes.append(seeded_scope)

    # 4. Pairwise common scopes between runs sharing the same view
    by_view: dict[str, list[PredictionRun]] = {}
    for run in runs:
        by_view.setdefault(run.view, []).append(run)

    for view, run_list in by_view.items():
        if len(run_list) >= 2:
            for i in range(len(run_list)):
                for j in range(i + 1, len(run_list)):
                    r1, r2 = run_list[i], run_list[j]
                    p1 = {fp.frame for fp in r1.processing if fp.state == "processed"}
                    p2 = {fp.frame for fp in r2.processing if fp.state == "processed"}
                    common_frames = tuple(
                        f for f in manifest.planned_frames if f in p1 and f in p2 and f not in excluded_seeds
                    )
                    key1 = RunKey(config_id=r1.config_id, view=r1.view)
                    key2 = RunKey(config_id=r2.config_id, view=r2.view)
                    pairwise_scope = EvaluationScope(
                        scope_id=f"pairwise/{r1.config_id}_vs_{r2.config_id}/{view}",
                        kind="pairwise_common",
                        frames=common_frames,
                        profile_id=manifest.profile.profile_id,
                        view=r1.view,
                        cohort=manifest.cohort,
                        target_population="all_objects",
                        excluded_seed_frames=manifest.excluded_seed_frames,
                        interpolation=r1.normalization.interpolation,
                        run_ids=(key1, key2),
                    )
                    scopes.append(pairwise_scope)

    return ScopeSet(scopes=tuple(scopes))


def evaluate_coverage(
    scope: EvaluationScope,
    reference: DatasetReference,
    run: PredictionRun,
) -> CoverageResult:
    """Calculate frame and observation coverage against planned reference denominators."""
    scope_frame_set = set(scope.frames)
    planned_frames = len(scope.frames)

    processed_map = {fp.frame: fp for fp in run.processing}
    processed_count = sum(1 for f in scope.frames if f in processed_map and processed_map[f].state == "processed")

    evidences: set[Evidence] = set()
    for f in scope.frames:
        if f in processed_map:
            evidences.add(processed_map[f].evidence)

    # Frame coverage
    frame_cov_val = (processed_count / planned_frames) if planned_frames > 0 else None
    frame_cov = MetricValue(
        value=frame_cov_val,
        unit="ratio",
        scale="0-1",
        numerator=float(processed_count),
        denominator=float(planned_frames),
        availability="available" if frame_cov_val is not None else "undefined_denominator",
        definition_id="coverage/frame_coverage",
    )

    # GT observation coverage
    planned_gt_count = sum(1 for b in reference.boxes if b.frame in scope_frame_set)
    observed_gt_count = sum(
        1
        for b in reference.boxes
        if b.frame in scope_frame_set and b.frame in processed_map and processed_map[b.frame].state == "processed"
    )
    obs_cov_val = (observed_gt_count / planned_gt_count) if planned_gt_count > 0 else None
    obs_cov = MetricValue(
        value=obs_cov_val,
        unit="ratio",
        scale="0-1",
        numerator=float(observed_gt_count),
        denominator=float(planned_gt_count),
        availability="available" if obs_cov_val is not None else "undefined_denominator",
        definition_id="coverage/object_observation_coverage",
    )

    evidence_tuple = tuple(sorted(evidences)) if evidences else ("unavailable",)

    return CoverageResult(
        frame_coverage=frame_cov,
        object_observation_coverage=obs_cov,
        evidence=evidence_tuple,
        planned_frames=planned_frames,
        processed_frames=processed_count,
        planned_gt_observations=planned_gt_count,
        observed_gt_observations=observed_gt_count,
    )
