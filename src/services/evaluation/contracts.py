"""Typed public interface contract. Implementations remain with their owners.

EvaluationInterfaces is a structural protocol, not a service to instantiate.
Each method names its owning module's public function. Runtime roots/outputs
are call arguments only; persisted identities contain portable relative paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from src.models.evaluation import (
    ArrivalMetrics,
    BackendResult,
    BenchmarkManifest,
    BindingOptions,
    CocoReferenceOptions,
    DatasetReference,
    DiagnosticGroupingOptions,
    DiagnosticMetrics,
    EvaluationReport,
    EvaluationScope,
    ExampleOptions,
    FrameBinding,
    KittiReferenceOptions,
    MatchResult,
    MetricGroup,
    NuScenesReferenceOptions,
    PredictionBox,
    PredictionNormalizationOptions,
    PredictionRun,
    PreparedEvaluationView,
    ReferenceBox,
    ReferenceProfile,
    ReportArtifacts,
    ReviewTruth,
    RunEvaluation,
    ScopeSet,
    TrackEvalOptions,
)


class EvaluationInterfaces(Protocol):
    def load_nuscenes_reference(self, root: Path, *, options: NuScenesReferenceOptions) -> DatasetReference: ...

    def load_kitti_reference(self, root: Path, *, options: KittiReferenceOptions) -> DatasetReference: ...

    def load_coco_reference(
        self,
        path: Path,
        *,
        bindings: tuple[FrameBinding, ...],
        options: CocoReferenceOptions,
    ) -> DatasetReference: ...

    def build_frame_bindings(
        self,
        reference: DatasetReference,
        payload: Path,
        *,
        options: BindingOptions,
        source_manifests: tuple[Path, ...],
        result: Path | None = None,
    ) -> tuple[FrameBinding, ...]: ...

    def load_sam3_predictions(
        self,
        result: Path,
        *,
        manifest: BenchmarkManifest,
        payload: Path | None,
        options: PredictionNormalizationOptions,
    ) -> tuple[PredictionRun, ...]: ...

    def prepare_local_profile(
        self,
        reference: DatasetReference,
        run: PredictionRun,
        profile: ReferenceProfile,
        *,
        scope: EvaluationScope,
    ) -> PreparedEvaluationView: ...

    def match_boxes(
        self,
        gt: tuple[ReferenceBox, ...],
        pred: tuple[PredictionBox, ...],
        *,
        iou_threshold: float,
        class_aware: bool = True,
    ) -> MatchResult: ...

    def box_metrics(
        self,
        matches: MatchResult,
        *,
        view: PreparedEvaluationView,
        declared_classes: tuple[str, ...],
    ) -> DiagnosticMetrics: ...

    def group_box_metrics(
        self,
        matches: MatchResult,
        *,
        view: PreparedEvaluationView,
        declared_classes: tuple[str, ...],
        options: DiagnosticGroupingOptions,
    ) -> tuple[MetricGroup, ...]: ...

    def build_scopes(self, manifest: BenchmarkManifest, runs: tuple[PredictionRun, ...]) -> ScopeSet: ...

    def evaluate_arrivals(
        self,
        reference: DatasetReference,
        run: PredictionRun,
        *,
        manifest: BenchmarkManifest,
        scope: EvaluationScope,
    ) -> ArrivalMetrics: ...

    def evaluate_run(
        self,
        reference: DatasetReference,
        run: PredictionRun,
        *,
        manifest: BenchmarkManifest,
        dataset_root: Path,
        scopes: ScopeSet | None = None,
    ) -> RunEvaluation: ...

    def evaluate_trackeval(
        self,
        reference: DatasetReference,
        run: PredictionRun,
        *,
        manifest: BenchmarkManifest,
        options: TrackEvalOptions,
        scope: EvaluationScope,
        dataset_root: Path,
        work_dir: Path,
    ) -> BackendResult: ...

    def build_report(
        self,
        evaluations: tuple[RunEvaluation, ...],
        *,
        manifest: BenchmarkManifest,
        review_truth: ReviewTruth | None = None,
    ) -> EvaluationReport: ...

    def write_report(
        self,
        report: EvaluationReport,
        *,
        output_dir: Path,
        dataset_root: Path,
        examples: ExampleOptions = ExampleOptions(),
    ) -> ReportArtifacts: ...
