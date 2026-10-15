"""Tests for evaluation scopes and coverage computation."""

from __future__ import annotations

import pytest

from src.models import evaluation as e
from src.services.evaluation.coverage import build_scopes, evaluate_coverage

HASH = "a" * 64


@pytest.fixture
def coverage_fixture() -> tuple[e.BenchmarkManifest, e.DatasetReference, e.PredictionRun]:
    sequence = e.SequenceKey(dataset_id="kitti", split="training", sequence_uid="0013")
    keys = tuple(e.FrameKey(sequence=sequence, frame_uid=str(i)) for i in range(4))
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
    boxes = tuple(
        e.ReferenceBox(
            annotation_id=f"box-{i}",
            native_annotation_id=f"native-{i}",
            frame=keys[i % 2],
            class_name="car",
            bbox_xyxy=(10.0, 20.0, 50.0, 60.0),
            native_track_id=str(i),
            source_ordinal=i,
            reference_kind="native_2d",
            native_category="Car",
        )
        for i in range(2)
    )
    ref = e.DatasetReference(
        reference_revision="kitti/v1",
        frames=frames,
        boxes=boxes,
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
    options = e.PredictionNormalizationOptions(views=("raw", "stitched"))
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
    processing = (
        e.FrameProcessing(
            frame=keys[0], selected=True, state="processed", evidence="wrapper_declared", mapping_id="payload/v1"
        ),
        e.FrameProcessing(
            frame=keys[1], selected=True, state="processed", evidence="wrapper_declared", mapping_id="payload/v1"
        ),
        e.FrameProcessing(
            frame=keys[2], selected=True, state="unknown", evidence="unavailable", mapping_id="payload/v1"
        ),
        e.FrameProcessing(
            frame=keys[3], selected=False, state="not_processed", evidence="unavailable", mapping_id="payload/v1"
        ),
    )
    run = e.PredictionRun(
        config_id="cfg-1",
        view="raw",
        status="completed",
        raw_status="completed",
        processing=processing,
        boxes=(),
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
    return manifest, ref, run


def test_build_scopes(coverage_fixture: tuple[e.BenchmarkManifest, e.DatasetReference, e.PredictionRun]) -> None:
    manifest, ref, run = coverage_fixture
    scope_set = build_scopes(manifest, (run,))

    fixed_scope = next(s for s in scope_set.scopes if s.kind == "fixed_set")
    observed_scope = next(s for s in scope_set.scopes if s.kind == "observed")

    assert len(fixed_scope.frames) == 4
    assert len(observed_scope.frames) == 2


def test_evaluate_coverage(coverage_fixture: tuple[e.BenchmarkManifest, e.DatasetReference, e.PredictionRun]) -> None:
    manifest, ref, run = coverage_fixture
    scope_set = build_scopes(manifest, (run,))
    fixed_scope = next(s for s in scope_set.scopes if s.kind == "fixed_set")

    cov = evaluate_coverage(fixed_scope, ref, run)
    assert cov.planned_frames == 4
    assert cov.processed_frames == 2
    assert cov.frame_coverage.value == 0.5
    assert cov.planned_gt_observations == 2
    assert cov.observed_gt_observations == 2
    assert cov.object_observation_coverage.value == 1.0


def test_pairwise_common_scope(
    coverage_fixture: tuple[e.BenchmarkManifest, e.DatasetReference, e.PredictionRun],
) -> None:
    manifest, ref, run1 = coverage_fixture
    # Run 2 has processed frames 1, 2
    keys = manifest.planned_frames
    proc2 = (
        e.FrameProcessing(
            frame=keys[0], selected=True, state="unknown", evidence="unavailable", mapping_id="payload/v1"
        ),
        e.FrameProcessing(
            frame=keys[1], selected=True, state="processed", evidence="wrapper_declared", mapping_id="payload/v1"
        ),
        e.FrameProcessing(
            frame=keys[2], selected=True, state="processed", evidence="wrapper_declared", mapping_id="payload/v1"
        ),
        e.FrameProcessing(
            frame=keys[3], selected=False, state="not_processed", evidence="unavailable", mapping_id="payload/v1"
        ),
    )
    run2 = e.PredictionRun(
        **{
            **run1.model_dump(),
            "config_id": "cfg-2",
            "processing": proc2,
        }
    )
    scope_set = build_scopes(manifest, (run1, run2))
    pairwise = next(s for s in scope_set.scopes if s.kind == "pairwise_common")
    # Intersection of processed is [keys[1]]
    assert len(pairwise.frames) == 1
    assert pairwise.frames[0] == keys[1]
