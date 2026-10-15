"""Tests for engine orchestration and end-to-end evaluation."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.models import evaluation as e
from src.services.evaluation.engine import evaluate_run

HASH = "a" * 64


@pytest.fixture
def engine_fixture() -> tuple[e.DatasetReference, e.PredictionRun, e.BenchmarkManifest]:
    sequence = e.SequenceKey(dataset_id="kitti", split="training", sequence_uid="0013")
    keys = tuple(e.FrameKey(sequence=sequence, frame_uid=str(i)) for i in range(2))
    image = e.SourceFile(relative_path="training/image_02/0013/000000.png", sha256=HASH, role="image")
    frames = tuple(
        e.FrameReference(
            key=key,
            source_ordinal=i,
            width=1242,
            height=375,
            media=image,
            timestamp_us=i * 100000,
            timestamp_provenance="declared_fps",
        )
        for i, key in enumerate(keys)
    )
    profile = e.ReferenceProfile(
        profile_id="kitti-raw",
        revision="kitti-raw/v1",
        designation="raw",
        classes=("car", "person"),
        class_map=(
            e.ClassMapping(native_class="Car", class_name="car"),
            e.ClassMapping(native_class="Pedestrian", class_name="person"),
        ),
        eligibility_policy="native-targets/v1",
        geometry_policy="native-unclipped/v1",
    )
    labels = e.SourceFile(relative_path="training/label_02/0013.txt", sha256=HASH, role="labels")
    box = e.ReferenceBox(
        annotation_id="gt-0",
        native_annotation_id="native-0",
        frame=keys[0],
        class_name="car",
        bbox_xyxy=(10.0, 20.0, 25.0, 60.0),
        native_track_id="1",
        source_ordinal=0,
        reference_kind="native_2d",
        native_category="Car",
    )
    ref = e.DatasetReference(
        reference_revision="kitti/v1",
        frames=frames,
        boxes=(box,),
        sources=(labels,),
        profiles=(profile,),
    )
    bindings = tuple(
        e.FrameBinding(
            mapping_id="payload/v1",
            domain="payload",
            index=i,
            frame=key,
            upstream=(e.BindingStep(mapping_id="native/v1", domain="native_export", index=i),),
        )
        for i, key in enumerate(keys)
    )
    options = e.PredictionNormalizationOptions(views=("raw",))
    manifest = e.BenchmarkManifest(
        protocol_revision="fixture/v1",
        reference_revision=ref.reference_revision,
        reference_sha256=HASH,
        sequences=(sequence,),
        planned_frames=keys,
        cohort="text_discovery",
        seed_set=e.SeedSet(revision="none/v1"),
        profile=profile,
        bindings=bindings,
        normalization=options,
        cadence=(
            e.CadencePolicy(
                sequence=sequence,
                source_fps=10.0,
                source_fps_provenance="kitti_dataset_fallback",
                input_fps=10.0,
                target_fps=5.0,
                sampling_policy="uniform/v1",
                endpoint_policy="include_final_source_frame",
            ),
        ),
        evaluator_revision="fixture",
    )
    pred = e.PredictionBox(
        prediction_id="p-0",
        frame=keys[0],
        class_name="car",
        bbox_xyxy=(10.0, 20.0, 25.0, 60.0),
        raw_track_id="1",
        source=e.ObservationSource(
            mapping_id="payload/v1", source_ordinal=0, config_id="cfg-1", source_observation_id="r-0"
        ),
        scoring_ordinal=0,
        score=e.Score(value=0.95, semantics="probability", source="model"),
        origin="model",
    )
    processing = tuple(
        e.FrameProcessing(
            frame=key, selected=True, state="processed", evidence="wrapper_declared", mapping_id="payload/v1"
        )
        for key in keys
    )
    run = e.PredictionRun(
        config_id="cfg-1",
        view="raw",
        status="completed",
        raw_status="completed",
        processing=processing,
        boxes=(pred,),
        normalization=options,
        normalization_sha256=e.content_hash(options),
        provenance=e.RunProvenance(
            result_sha256=HASH,
            config_sha256=HASH,
            model_revision="test",
            evaluator_revision="test",
            inference_variant="text",
        ),
    )
    return ref, run, manifest


def test_evaluate_run_e2e(
    tmp_path: Path, engine_fixture: tuple[e.DatasetReference, e.PredictionRun, e.BenchmarkManifest]
) -> None:
    ref, run, manifest = engine_fixture
    eval_run = evaluate_run(ref, run, manifest=manifest, dataset_root=tmp_path)

    assert eval_run.config_id == "cfg-1"
    assert eval_run.view == "raw"
    assert eval_run.status == "completed"
    assert len(eval_run.scopes) == 2  # fixed and observed

    fixed_scope_eval = next(s for s in eval_run.scopes if s.scope.kind == "fixed_set")
    assert fixed_scope_eval.coverage.frame_coverage.value == 1.0

    # Matches at 0.5 and 0.7 thresholds
    assert len(fixed_scope_eval.matches) == 2
    match_05 = fixed_scope_eval.matches[0]
    assert len(match_05.pairs) == 1
    assert match_05.pairs[0].annotation_id == "gt-0"
    assert match_05.pairs[0].prediction_id == "p-0"
    assert match_05.pairs[0].iou == pytest.approx(1.0)

    # Micro groups metrics
    micro_group = next(g for g in fixed_scope_eval.groups if g.aggregation == "micro")
    assert micro_group.diagnostic.tp == 1
    assert micro_group.diagnostic.fp == 0
    assert micro_group.diagnostic.fn == 0
    assert micro_group.diagnostic.precision.value == 1.0
    assert micro_group.diagnostic.recall.value == 1.0
    assert micro_group.diagnostic.f1.value == 1.0
