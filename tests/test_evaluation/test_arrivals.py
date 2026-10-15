"""Tests for arrivals evaluation and deadline detection."""

from __future__ import annotations

import pytest

from src.models import evaluation as e
from src.services.evaluation.arrivals import evaluate_arrivals

HASH = "a" * 64


@pytest.fixture
def arrival_fixture() -> tuple[e.DatasetReference, e.PredictionRun, e.BenchmarkManifest, e.EvaluationScope]:
    sequence = e.SequenceKey(dataset_id="kitti", split="training", sequence_uid="0013")
    keys = tuple(e.FrameKey(sequence=sequence, frame_uid=str(i)) for i in range(5))
    image = e.SourceFile(relative_path="training/image_02/0013/000000.png", sha256=HASH, role="image")
    frames = tuple(
        e.FrameReference(
            key=key,
            source_ordinal=i,
            width=1242,
            height=375,
            media=image,
            timestamp_us=i * 100000,  # 0.1s apart = 10 FPS
            timestamp_provenance="recorded_sensor",
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
    # Track 1 appears at frame 0 and continues to frame 3
    # Track 2 appears at frame 2
    boxes = (
        e.ReferenceBox(
            annotation_id="box-1-0",
            native_annotation_id="n-1-0",
            frame=keys[0],
            class_name="car",
            bbox_xyxy=(10.0, 10.0, 50.0, 50.0),
            native_track_id="1",
            source_ordinal=0,
            reference_kind="native_2d",
            native_category="Car",
        ),
        e.ReferenceBox(
            annotation_id="box-1-1",
            native_annotation_id="n-1-1",
            frame=keys[1],
            class_name="car",
            bbox_xyxy=(12.0, 12.0, 52.0, 52.0),
            native_track_id="1",
            source_ordinal=1,
            reference_kind="native_2d",
            native_category="Car",
        ),
        e.ReferenceBox(
            annotation_id="box-2-2",
            native_annotation_id="n-2-2",
            frame=keys[2],
            class_name="car",
            bbox_xyxy=(100.0, 100.0, 150.0, 150.0),
            native_track_id="2",
            source_ordinal=2,
            reference_kind="native_2d",
            native_category="Car",
        ),
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
        arrival=e.ArrivalPolicy(deadline_s=1.0),
    )
    # Prediction detects Track 1 at frame 1 (delay = 0.1s)
    # Never detects Track 2
    preds = (
        e.PredictionBox(
            prediction_id="p-1",
            frame=keys[1],
            class_name="car",
            bbox_xyxy=(12.0, 12.0, 52.0, 52.0),
            raw_track_id="1",
            source=e.ObservationSource(
                mapping_id="payload/v1", source_ordinal=0, config_id="cfg-1", source_observation_id="r-0"
            ),
            scoring_ordinal=0,
            score=e.Score(value=0.9, semantics="probability", source="model"),
            origin="model",
        ),
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
        boxes=preds,
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
    scope = e.EvaluationScope(
        scope_id="cfg-1/raw/fixed",
        kind="fixed_set",
        frames=keys,
        profile_id=profile.profile_id,
        view="raw",
        cohort="text_discovery",
        target_population="all_objects",
        interpolation=options.interpolation,
        run_ids=(e.RunKey(config_id=run.config_id, view=run.view),),
    )
    return ref, run, manifest, scope


def test_evaluate_arrivals(
    arrival_fixture: tuple[e.DatasetReference, e.PredictionRun, e.BenchmarkManifest, e.EvaluationScope],
) -> None:
    ref, run, manifest, scope = arrival_fixture
    metrics = evaluate_arrivals(ref, run, manifest=manifest, scope=scope)

    assert metrics.eligible_tracks == 2
    assert metrics.never_detected_tracks == 1
    assert len(metrics.tracks) == 2

    # Track 1 detected at frame 1 with 0.1s delay
    t1 = next(t for t in metrics.tracks if t.track.track_id == "1")
    assert t1.detected_frame == manifest.planned_frames[1]
    assert t1.within_deadline is True
    assert t1.delay_s == pytest.approx(0.1)

    # Track 2 never detected and censored horizon
    t2 = next(t for t in metrics.tracks if t.track.track_id == "2")
    assert t2.detected_frame is None
    assert t2.censored is True
    assert t2.within_deadline is None
