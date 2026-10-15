"""Small contract checks only; no datasets, models, services or downloads."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.models import evaluation as e

FIXTURES = Path(__file__).parents[1] / "fixtures" / "evaluation" / "contracts"
HASH = "a" * 64


def example_documents() -> dict[str, e.Contract]:
    """Handwritten tiny data with a negative frame; not model-quality evidence."""
    sequence = e.SequenceKey(dataset_id="kitti", split="training", sequence_uid="0013")
    keys = tuple(e.FrameKey(sequence=sequence, frame_uid=str(i)) for i in range(2))
    image = e.SourceFile(relative_path="training/image_02/0013/000000.png", sha256=HASH, role="image")
    frames = tuple(
        e.FrameReference(
            key=key,
            source_ordinal=i,
            width=1242,
            height=375,
            media=image
            if i == 0
            else e.SourceFile(relative_path="training/image_02/0013/000001.png", sha256=HASH, role="image"),
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
    native = tuple(
        e.NativeKittiRow(
            frame=keys[0],
            source_id=f"native-{i}",
            source_file=labels,
            line_number=i + 1,
            source_frame_index=0,
            track_id=-1 if category == "DontCare" else i,
            native_class=category,
            truncated=-1.0 if category == "DontCare" else 0.0,
            occluded=-1 if category == "DontCare" else 0,
            alpha=-10.0 if category == "DontCare" else 0.0,
            bbox_xyxy=(10.0 + i * 20, 20.0, 25.0 + i * 20, 60.0),
            dimensions_hwl=(-1.0, -1.0, -1.0) if category == "DontCare" else (1.5, 1.6, 4.0),
            location_xyz=(-1000.0, -1000.0, -1000.0) if category == "DontCare" else (1.0, 2.0, 20.0),
            rotation_y=-10.0 if category == "DontCare" else 0.0,
        )
        for i, category in enumerate(("Car", "Van", "DontCare", "DontCare"))
    )
    box = e.ReferenceBox(
        annotation_id="kitti/training/0013/0/native-0",
        native_annotation_id="native-0",
        frame=keys[0],
        class_name="car",
        bbox_xyxy=native[0].bbox_xyxy,
        native_track_id="0",
        source_ordinal=0,
        reference_kind="native_2d",
        native_category="Car",
        metadata=e.BoxMetadata(kitti_source_id="native-0"),
    )
    ignored = tuple(
        e.IgnoredRegion(
            source_id=f"ignored-{i}",
            frame=keys[0],
            role="distractor" if row.native_class == "Van" else "ignore",
            native_category=row.native_class,
            bbox_xyxy=row.bbox_xyxy,
            native_kitti_source_id=row.source_id,
        )
        for i, row in enumerate(native[1:])
    )
    reference = e.DatasetReference(
        reference_revision="fixture/v1",
        frames=frames,
        boxes=(box,),
        ignored_regions=ignored,
        native_kitti_rows=native,
        sources=(labels,),
        profiles=(profile,),
    )
    options = e.PredictionNormalizationOptions(
        reset_boundaries=(
            e.ResetBoundary(config_id="fixture", frame=keys[1], mapping_id="payload/v1", source="payload_declared"),
        )
    )
    processing = tuple(
        e.FrameProcessing(
            frame=key, selected=True, state="processed", evidence="wrapper_declared", mapping_id="payload/v1"
        )
        for key in keys
    )
    source = e.ObservationSource(
        mapping_id="payload/v1", source_ordinal=0, config_id="fixture", source_observation_id="row-0", window_id="0"
    )
    prediction = e.PredictionBox(
        prediction_id="pred-0",
        frame=keys[0],
        class_name="car",
        bbox_xyxy=box.bbox_xyxy,
        raw_track_id="1",
        source=source,
        scoring_ordinal=0,
        score=e.Score(value=0.9, semantics="probability", source="text-mode"),
        origin="model",
    )
    provenance = e.RunProvenance(
        result_sha256=HASH,
        config_sha256=HASH,
        model_revision="fixture",
        evaluator_revision="fixture",
        inference_variant="text",
    )
    run = e.PredictionRun(
        config_id="fixture",
        view="raw",
        status="completed",
        raw_status="completed",
        processing=processing,
        boxes=(prediction,),
        normalization=options,
        normalization_sha256=e.content_hash(options),
        provenance=provenance,
    )
    scoreless = e.PredictionRun(
        **{
            **run.model_dump(),
            "boxes": (
                e.PredictionBox(
                    **{
                        **prediction.model_dump(),
                        "score": e.Score(value=None, semantics="missing", source="scoreless-export"),
                    }
                ),
            ),
        }
    )
    empty = e.PredictionRun(**{**run.model_dump(), "boxes": ()})
    partial = e.PredictionRun(
        **{
            **run.model_dump(),
            "status": "failed",
            "raw_status": "failed",
            "failure_reason": "controlled failure",
            "processing": (
                processing[0],
                e.FrameProcessing(
                    frame=keys[1], selected=True, state="unknown", evidence="unavailable", mapping_id="payload/v1"
                ),
            ),
            "track_states": (
                e.TrackStateObservation(
                    observation_id="outside-1",
                    frame=keys[0],
                    track=e.TrackKey(sequence=sequence, track_id="1", config_id="fixture", window_id="0"),
                    class_name="car",
                    source=source,
                    outside=True,
                    score=e.Score(value=None, semantics="missing", source="outside"),
                ),
            ),
        }
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
    manifest = e.BenchmarkManifest(
        protocol_revision="fixture/v1",
        reference_revision=reference.reference_revision,
        reference_sha256=e.content_hash(reference),
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
    unavailable = e.MetricValue(
        value=None,
        unit="percent",
        scale="0-100",
        availability="unavailable_dependency",
        reason="TrackEval is not installed",
        definition_id="trackeval/HOTA",
    )
    fixed_scope = e.EvaluationScope(
        scope_id="fixture/raw/fixed",
        kind="fixed_set",
        frames=keys,
        profile_id=profile.profile_id,
        view="raw",
        cohort="text_discovery",
        target_population="all_objects",
        interpolation=options.interpolation,
        run_ids=(e.RunKey(config_id=partial.config_id, view=partial.view),),
        eligible_annotation_ids=(box.annotation_id,),
    )
    report = e.EvaluationReport(
        protocol_revision="fixture/v1",
        manifest_sha256=e.content_hash(manifest),
        reference_sha256=e.content_hash(reference),
        evaluations=(
            e.RunEvaluation(
                config_id=partial.config_id,
                view=partial.view,
                cohort="text_discovery",
                status=partial.status,
                raw_status=partial.raw_status,
                provenance=provenance,
                runtime=partial.runtime,
                scopes=(),
                processing=partial.processing,
                windows=partial.windows,
                normalization=partial.normalization,
                normalization_sha256=partial.normalization_sha256,
                reference_frames=frames,
                native_kitti_rows=native,
                warnings=("No supported diagnostic/error scope was computed in this contract-only report fixture",),
                failure_reason=partial.failure_reason,
                backend=e.BackendResult(
                    availability="unavailable_dependency",
                    scope=fixed_scope,
                    timeline="full_native",
                    reason="TrackEval is not installed",
                    metrics=(e.NamedMetric(name="HOTA", metric=unavailable),),
                ),
            ),
        ),
        optional_metrics=(e.NamedMetric(name="HOTA", metric=unavailable),),
    )
    nseq = e.SequenceKey(dataset_id="nuscenes", split="mini_train", sequence_uid="1" * 32, camera_channel="CAM_FRONT")
    nkey = e.FrameKey(sequence=nseq, frame_uid="2" * 32)
    camera = e.CameraMetadata(
        sample_data_token="3" * 32,
        ego_pose_token="4" * 32,
        calibrated_sensor_token="5" * 32,
        log_token="6" * 32,
        sample_timestamp_us=1000000,
        sensor_timestamp_us=1000010,
        ego_translation=(1.0, 2.0, 0.0),
        ego_rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
        camera_translation=(0.0, 0.0, 1.0),
        camera_rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
        camera_intrinsic=((1000.0, 0.0, 800.0), (0.0, 1000.0, 450.0), (0.0, 0.0, 1.0)),
    )
    annotation = e.NuScenesAnnotation(
        annotation_token="7" * 32,
        sample_token=nkey.frame_uid,
        instance_token="8" * 32,
        category_token="9" * 32,
        category_name="vehicle.car",
        translation_global=(10.0, 0.0, 10.0),
        size_wlh=(2.0, 4.0, 1.5),
        rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
        visibility_token="4",
        num_lidar_pts=1,
        num_radar_pts=0,
    )
    nprofile = e.ReferenceProfile(
        profile_id="nuscenes-projected",
        revision="nuscenes-projected/v1",
        designation="raw",
        classes=("car",),
        class_map=(e.ClassMapping(native_class="vehicle.car", class_name="car"),),
        eligibility_policy="devkit-target-map/v1",
        geometry_policy="hull/v1",
        projection=e.ProjectionPolicy(),
    )
    nreference = e.DatasetReference(
        reference_revision="fixture-nuscenes/v1",
        frames=(
            e.FrameReference(
                key=nkey,
                source_ordinal=0,
                width=1600,
                height=900,
                media=e.SourceFile(relative_path="samples/CAM_FRONT/fixture.jpg", sha256=HASH, role="image"),
                timestamp_us=1000010,
                timestamp_provenance="recorded_sensor",
                camera=camera,
            ),
        ),
        boxes=(
            e.ReferenceBox(
                annotation_id="front/sample/annotation",
                native_annotation_id=annotation.annotation_token,
                frame=nkey,
                class_name="car",
                bbox_xyxy=(100.0, 100.0, 300.0, 400.0),
                native_track_id=annotation.instance_token,
                source_ordinal=0,
                reference_kind="projected_3d_box",
                native_category="vehicle.car",
                metadata=e.BoxMetadata(nuscenes=annotation),
            ),
        ),
        native_nuscenes_annotations=(annotation,),
        profiles=(nprofile,),
    )
    coco_options = e.CocoReferenceOptions(
        reference_revision="coco-fixture/v1",
        profile=profile,
        images=tuple(e.CocoImageAssociation(image_id=i, frame=frame) for i, frame in enumerate(frames)),
    )
    prepared = e.PreparedEvaluationView(
        scope=fixed_scope,
        profile=profile,
        reference_boxes=reference.boxes,
        prediction_boxes=run.boxes,
        ignored_regions=ignored,
        reference_frames=frames,
        native_kitti_rows=native,
    )
    return {
        "reference-kitti.json": reference,
        "reference-nuscenes.json": nreference,
        "prediction-scored.json": run,
        "prediction-scoreless.json": scoreless,
        "prediction-empty.json": empty,
        "prediction-failed-partial.json": partial,
        "manifest.json": manifest,
        "report.json": report,
        "coco-options.json": coco_options,
        "prepared-view.json": prepared,
    }


DOCUMENT_TYPES: dict[str, type[e.Contract]] = {
    "coco-options.json": e.CocoReferenceOptions,
    "prepared-view.json": e.PreparedEvaluationView,
    "reference-kitti.json": e.DatasetReference,
    "reference-nuscenes.json": e.DatasetReference,
    "prediction-scored.json": e.PredictionRun,
    "prediction-scoreless.json": e.PredictionRun,
    "prediction-empty.json": e.PredictionRun,
    "prediction-failed-partial.json": e.PredictionRun,
    "manifest.json": e.BenchmarkManifest,
    "report.json": e.EvaluationReport,
}


@pytest.mark.parametrize("filename", DOCUMENT_TYPES)
def test_fixture_roundtrip_and_versions(filename: str) -> None:
    model = e.parse_document(DOCUMENT_TYPES[filename], (FIXTURES / filename).read_bytes())
    assert e.parse_document(type(model), e.canonical_json(model)) == model
    data = model.model_dump(mode="json", by_alias=True)
    data["schema"] = str(data["schema"]).replace("/v1", "/v99")
    with pytest.raises(ValidationError):
        e.parse_document(type(model), json.dumps(data))


def test_fixtures_match_handwritten_examples() -> None:
    for filename, model in example_documents().items():
        assert e.canonical_json(model) == e.canonical_json(
            e.parse_document(type(model), (FIXTURES / filename).read_bytes())
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_rejected(value: float) -> None:
    with pytest.raises(ValidationError):
        e.Score(value=value, semantics="probability", source="fixture")
    with pytest.raises(ValidationError):
        e.RuntimeRecord(inference_s=value)


def test_unknown_and_coerced_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        e.SequenceKey.model_validate({"dataset_id": "kitti", "split": "training", "sequence_uid": "0013", "unknown": 1})
    with pytest.raises(ValidationError):
        e.StitchOptions(min_common_frames="2")
    with pytest.raises(ValidationError):
        e.FrameKey(
            sequence=e.SequenceKey(dataset_id="kitti", split="training", sequence_uid="0013"), frame_uid="000000"
        )


@pytest.mark.parametrize("bbox", [(0.0, 0.0, 0.0, 1.0), (1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 1.0, 0.0)])
def test_zero_negative_extents(bbox: e.BBox) -> None:
    box = example_documents()["reference-kitti.json"]
    assert isinstance(box, e.DatasetReference)
    with pytest.raises(ValidationError):
        e.ReferenceBox(**{**box.boxes[0].model_dump(), "bbox_xyxy": bbox})


def test_duplicate_keys_and_unknown_frames() -> None:
    ref = example_documents()["reference-kitti.json"]
    assert isinstance(ref, e.DatasetReference)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        e.parse_document(
            e.SequenceKey, '{"dataset_id":"kitti","dataset_id":"coco","split":"training","sequence_uid":"0013"}'
        )
    with pytest.raises(ValidationError, match="duplicate frame"):
        e.DatasetReference(**{**ref.model_dump(), "frames": (ref.frames[0], ref.frames[0])})
    with pytest.raises(ValidationError, match="duplicate annotation"):
        e.DatasetReference(**{**ref.model_dump(), "boxes": (ref.boxes[0], ref.boxes[0])})
    with pytest.raises(ValidationError, match="unknown frame"):
        e.DatasetReference(**{**ref.model_dump(), "frames": (ref.frames[1],)})


def test_namespace_isolation() -> None:
    seq = e.SequenceKey(dataset_id="kitti", split="training", sequence_uid="0013")
    test_seq = e.SequenceKey(dataset_id="kitti", split="testing", sequence_uid="0013")
    other_seq = e.SequenceKey(dataset_id="kitti", split="training", sequence_uid="0014")
    keys = tuple(e.TrackKey(sequence=s, track_id="1") for s in (seq, test_seq, other_seq))
    assert len(set(keys)) == 3
    nseq = e.SequenceKey(dataset_id="nuscenes", split="mini_train", sequence_uid="a" * 32, camera_channel="CAM_FRONT")
    back = e.SequenceKey(dataset_id="nuscenes", split="mini_train", sequence_uid="a" * 32, camera_channel="CAM_BACK")
    assert e.FrameKey(sequence=nseq, frame_uid="b" * 32) != e.FrameKey(sequence=back, frame_uid="b" * 32)


def test_invalid_bindings_and_scoped_window_bijections() -> None:
    manifest = example_documents()["manifest.json"]
    assert isinstance(manifest, e.BenchmarkManifest)
    binding = manifest.bindings[0]
    with pytest.raises(ValidationError):
        e.FrameBinding(mapping_id="window", domain="window_local", index=0, frame=binding.frame)
    with pytest.raises(ValidationError, match="duplicate binding"):
        e.BenchmarkManifest(**{**manifest.model_dump(), "bindings": (binding, binding)})
    window_bindings = tuple(
        e.FrameBinding(
            mapping_id="windows",
            domain="window_local",
            index=0,
            frame=binding.frame,
            config_id="fixture",
            window_id=str(i),
        )
        for i in range(2)
    )
    assert e.BenchmarkManifest(**{**manifest.model_dump(), "bindings": window_bindings}).bindings == window_bindings
    with pytest.raises(ValidationError, match="cycle"):
        e.FrameBinding(
            **{
                **binding.model_dump(),
                "upstream": (e.BindingStep(mapping_id=binding.mapping_id, domain=binding.domain, index=binding.index),),
            }
        )


def test_null_zero_and_native_ignore_placeholders() -> None:
    zero = e.MetricValue(
        value=0.0,
        unit="ratio",
        scale="0-1",
        availability="available",
        definition_id="recall/v1",
        numerator=0.0,
        denominator=1.0,
    )
    undefined = e.MetricValue(
        value=None,
        unit="ratio",
        scale="0-1",
        availability="undefined_denominator",
        reason="no GT",
        denominator=0.0,
        definition_id="recall/v1",
    )
    assert zero.value == 0 and undefined.value is None
    with pytest.raises(ValidationError):
        e.MetricValue(value=None, unit="ratio", scale="0-1", availability="available", definition_id="recall/v1")
    ref = example_documents()["reference-kitti.json"]
    assert isinstance(ref, e.DatasetReference)
    assert sum(row.track_id == -1 for row in ref.native_kitti_rows) == 2
    with pytest.raises(ValidationError):
        e.NativeKittiRow(**{**ref.native_kitti_rows[0].model_dump(), "dimensions_hwl": (-1.0, -1.0, -1.0)})


def test_canonical_serialization_and_immutable_nested_values() -> None:
    ref = example_documents()["reference-kitti.json"]
    assert isinstance(ref, e.DatasetReference)
    assert e.content_hash(ref) == e.content_hash(
        e.parse_document(type(ref), json.dumps(ref.model_dump(mode="json", by_alias=True), indent=4))
    )
    relocated = e.DatasetReference(**{**ref.model_dump(), "generated_at": "new wall time", "content_sha256": "b" * 64})
    assert e.content_hash(ref) == e.content_hash(relocated)
    changed = e.DatasetReference(**{**ref.model_dump(), "reference_revision": "changed/v1"})
    assert e.content_hash(ref) != e.content_hash(changed)
    with pytest.raises(ValidationError):
        ref.reference_revision = "mutated"
    assert isinstance(ref.frames, tuple)
    with pytest.raises(ValidationError):
        e.SourceFile(relative_path="C:/dataset/image.png", sha256=HASH, role="image")


def test_completed_empty_failed_partial_and_evidence() -> None:
    docs = example_documents()
    empty = docs["prediction-empty.json"]
    partial = docs["prediction-failed-partial.json"]
    assert isinstance(empty, e.PredictionRun) and isinstance(partial, e.PredictionRun)
    assert empty.status == "completed" and not empty.boxes and len(empty.processing) == 2
    assert partial.status == "failed" and partial.boxes and partial.track_states[0].outside
    with pytest.raises(ValidationError):
        e.PredictionRun(**{**partial.model_dump(), "status": "completed"})
    with pytest.raises(ValidationError):
        e.FrameProcessing(**{**empty.processing[0].model_dump(), "evidence": "unavailable"})


def test_matching_one_to_one_and_score_semantics() -> None:
    manifest = example_documents()["manifest.json"]
    assert isinstance(manifest, e.BenchmarkManifest)
    pair = e.MatchedPair(annotation_id="gt", prediction_id="pred", frame=manifest.planned_frames[0], iou=0.5)
    match = e.MatchResult(
        iou_threshold=0.5, class_aware=True, pairs=(pair,), unmatched_annotation_ids=(), unmatched_prediction_ids=()
    )
    assert match.pairs[0].iou == match.iou_threshold
    with pytest.raises(ValidationError):
        e.MatchResult(**{**match.model_dump(), "unmatched_prediction_ids": ("pred",)})
    with pytest.raises(ValidationError):
        e.Score(value=None, semantics="probability", source="fixture")
    with pytest.raises(ValidationError):
        e.PredictionNormalizationOptions(stitched_overlap_geometry="identity_only")


def test_native_object_ids_and_track_classes() -> None:
    ref = example_documents()["reference-kitti.json"]
    assert isinstance(ref, e.DatasetReference)
    row = ref.native_kitti_rows[0]
    duplicate = e.NativeKittiRow(**{**row.model_dump(), "source_id": "duplicate", "line_number": 5})
    with pytest.raises(ValidationError, match="duplicate native object"):
        e.DatasetReference(**{**ref.model_dump(), "native_kitti_rows": (*ref.native_kitti_rows, duplicate)})
    changed = e.NativeKittiRow(
        **{
            **row.model_dump(),
            "source_id": "changed",
            "line_number": 5,
            "frame": ref.frames[1].key,
            "source_frame_index": 1,
            "native_class": "Pedestrian",
        }
    )
    with pytest.raises(ValidationError, match="changes class"):
        e.DatasetReference(**{**ref.model_dump(), "native_kitti_rows": (*ref.native_kitti_rows, changed)})


def test_hash_and_binding_composition_conflicts() -> None:
    docs = example_documents()
    run = docs["prediction-scored.json"]
    manifest = docs["manifest.json"]
    assert isinstance(run, e.PredictionRun) and isinstance(manifest, e.BenchmarkManifest)
    with pytest.raises(ValidationError, match="hash mismatch"):
        e.PredictionRun(**{**run.model_dump(), "normalization_sha256": "b" * 64})
    binding = manifest.bindings[0]
    with pytest.raises(ValidationError, match="chain domains"):
        e.FrameBinding(
            **{
                **binding.model_dump(),
                "upstream": (
                    e.BindingStep(mapping_id="native", domain="native_export", index=0),
                    e.BindingStep(mapping_id="native", domain="native_export", index=1),
                ),
            }
        )
    conflict = e.FrameBinding(**{**manifest.bindings[1].model_dump(), "upstream": binding.upstream})
    with pytest.raises(ValidationError, match="contradictory binding composition"):
        e.BenchmarkManifest(**{**manifest.model_dump(), "bindings": (binding, conflict)})


def test_report_and_metric_duplicate_identifiers() -> None:
    report = example_documents()["report.json"]
    assert isinstance(report, e.EvaluationReport)
    with pytest.raises(ValidationError, match="duplicate report evaluations"):
        e.EvaluationReport(**{**report.model_dump(), "evaluations": (report.evaluations[0], report.evaluations[0])})
    with pytest.raises(ValidationError, match="duplicate optional metric"):
        e.EvaluationReport(
            **{**report.model_dump(), "optional_metrics": (report.optional_metrics[0], report.optional_metrics[0])}
        )


def test_seed_mappings_and_exclusions_are_explicit() -> None:
    docs = example_documents()
    manifest = docs["manifest.json"]
    reference = docs["reference-kitti.json"]
    assert isinstance(manifest, e.BenchmarkManifest) and isinstance(reference, e.DatasetReference)
    seed = e.SeedReference(
        seed_id="seed-0",
        model_seed_id="model-0",
        frame=manifest.planned_frames[0],
        class_name="car",
        bbox_xyxy=reference.boxes[0].bbox_xyxy,
        mapping_id="payload/v1",
        reference_annotation_id=reference.boxes[0].annotation_id,
        reference_track=e.TrackKey(sequence=manifest.sequences[0], track_id="0"),
        reference_revision=reference.reference_revision,
        review_revision="reviewed-fixture/v1",
        source_sha256=HASH,
    )
    seed_set = e.SeedSet(revision="seeds/v1", seeds=(seed,))
    with pytest.raises(ValidationError, match="cannot use GT seeds"):
        e.BenchmarkManifest(**{**manifest.model_dump(), "seed_set": seed_set})
    with pytest.raises(ValidationError, match="explicitly excluded"):
        e.BenchmarkManifest(**{**manifest.model_dump(), "cohort": "seeded_propagation", "seed_set": seed_set})
    seeded = e.BenchmarkManifest(
        **{
            **manifest.model_dump(),
            "cohort": "seeded_propagation",
            "seed_set": seed_set,
            "excluded_seed_frames": (seed.frame,),
        }
    )
    assert seeded.seed_set.seeds[0].reference_track.track_id == "0"


def test_nested_unknown_and_nonfinite_native_geometry() -> None:
    reference = example_documents()["reference-nuscenes.json"]
    assert isinstance(reference, e.DatasetReference)
    payload = reference.model_dump(mode="json", by_alias=True)
    payload["boxes"][0]["metadata"]["undocumented"] = "escape hatch"
    with pytest.raises(ValidationError):
        e.parse_document(e.DatasetReference, json.dumps(payload))
    camera = reference.frames[0].camera
    assert camera is not None
    with pytest.raises(ValidationError):
        e.CameraMetadata(**{**camera.model_dump(), "camera_translation": (float("nan"), 0.0, 0.0)})
    with pytest.raises(ValidationError):
        e.CameraMetadata(**{**camera.model_dump(), "ego_rotation_wxyz": (2.0, 0.0, 0.0, 0.0)})


def test_reset_boundaries_are_hashed_unique_and_conflict_free() -> None:
    docs = example_documents()
    manifest = docs["manifest.json"]
    assert isinstance(manifest, e.BenchmarkManifest)
    opts = manifest.normalization
    reset = opts.reset_boundaries[0]
    changed = e.PredictionNormalizationOptions(**{**opts.model_dump(), "reset_boundaries": ()})
    assert e.content_hash(opts) != e.content_hash(changed)
    conflict = e.ResetBoundary(**{**reset.model_dump(), "mapping_id": "different", "source": "config_declared"})
    with pytest.raises(ValidationError, match="duplicate reset"):
        e.PredictionNormalizationOptions(**{**opts.model_dump(), "reset_boundaries": (reset, conflict)})
    window_reset = e.ResetBoundary(**{**reset.model_dump(), "window_id": "window-1"})
    conflicting_window = e.ResetBoundary(**{**window_reset.model_dump(), "frame": manifest.planned_frames[0]})
    with pytest.raises(ValidationError, match="window reset"):
        e.PredictionNormalizationOptions(
            **{**opts.model_dump(), "reset_boundaries": (window_reset, conflicting_window)}
        )
    other_config = e.ResetBoundary(**{**window_reset.model_dump(), "config_id": "other-config"})
    assert (
        len(
            e.PredictionNormalizationOptions(
                **{**opts.model_dump(), "reset_boundaries": (window_reset, other_config)}
            ).reset_boundaries
        )
        == 2
    )


def test_scoring_ordinals_preserve_original_window_ordinals() -> None:
    docs = example_documents()
    run = docs["prediction-scored.json"]
    prepared = docs["prepared-view.json"]
    assert isinstance(run, e.PredictionRun) and isinstance(prepared, e.PreparedEvaluationView)
    first = run.boxes[0]
    other_source = e.ObservationSource(
        **{**first.source.model_dump(), "window_id": "1", "source_observation_id": "other-row"}
    )
    second = e.PredictionBox(
        **{**first.model_dump(), "prediction_id": "other", "source": other_source, "scoring_ordinal": 1}
    )
    assert second.source.source_ordinal == first.source.source_ordinal
    assert e.PredictionRun(**{**run.model_dump(), "boxes": (first, second)}).boxes[1].scoring_ordinal == 1
    duplicate = e.PredictionBox(**{**second.model_dump(), "scoring_ordinal": first.scoring_ordinal})
    with pytest.raises(ValidationError, match="scoring ordinals"):
        e.PredictionRun(**{**run.model_dump(), "boxes": (first, duplicate)})
    with pytest.raises(ValidationError, match="scoring ordinals"):
        e.PreparedEvaluationView(**{**prepared.model_dump(), "prediction_boxes": (first, duplicate)})
    ref = prepared.reference_boxes[0]
    other_ref = e.ReferenceBox(**{**ref.model_dump(), "annotation_id": "other-annotation"})
    with pytest.raises(ValidationError, match="reference ordinals"):
        e.PreparedEvaluationView(**{**prepared.model_dump(), "reference_boxes": (ref, other_ref)})


def test_coco_requires_explicit_frame_metadata_and_duplicate_free_image_map() -> None:
    options = example_documents()["coco-options.json"]
    assert isinstance(options, e.CocoReferenceOptions)
    assert options.images[1].frame.width == 1242
    with pytest.raises(ValidationError, match="duplicate COCO image"):
        e.CocoReferenceOptions(**{**options.model_dump(), "images": (options.images[0], options.images[0])})
    payload = options.model_dump(mode="json")
    del payload["images"][0]["frame"]["media"]
    with pytest.raises(ValidationError):
        e.parse_document(e.CocoReferenceOptions, json.dumps(payload))


def test_run_keys_context_and_report_evidence_roundtrip() -> None:
    assert e.RunKey(config_id="a/raw", view="stitched") != e.RunKey(config_id="a", view="raw")
    docs = example_documents()
    report = docs["report.json"]
    prepared = docs["prepared-view.json"]
    assert isinstance(report, e.EvaluationReport) and isinstance(prepared, e.PreparedEvaluationView)
    evaluation = report.evaluations[0]
    assert evaluation.processing[1].state == "unknown" and evaluation.normalization.reset_boundaries
    context = e.EvaluationContext(
        run=prepared.scope.run_ids[0],
        scope_id=prepared.scope.scope_id,
        profile_id=prepared.profile.profile_id,
        target_population="all_objects",
    )
    event = e.ErrorEvent(
        error_id="miss",
        evaluation_context=context,
        category="miss",
        frame=prepared.scope.frames[0],
        annotation_ids=(prepared.reference_boxes[0].annotation_id,),
        iou_threshold=0.5,
        severity="unrated",
        severity_policy="unrated/v1",
        origin="automatic_diagnostic",
        reference_revision="fixture/v1",
    )
    assert e.parse_document(e.ErrorEvent, e.canonical_json(event)).evaluation_context == context
    with pytest.raises(ValidationError, match="normalization hash"):
        e.RunEvaluation(**{**evaluation.model_dump(), "normalization_sha256": "b" * 64})


def test_grouping_options_and_example_bounds() -> None:
    opts = e.DiagnosticGroupingOptions(
        dimensions=("micro", "slice"), slices=("temporal_third", "occlusion"), slice_revision="diagnostic-slices/v1"
    )
    assert opts.absent_classes == "include_zero_support"
    with pytest.raises(ValidationError):
        e.DiagnosticGroupingOptions(dimensions=("slice",))
    with pytest.raises(ValidationError):
        e.ExampleOptions(examples_per_category=101)
    assert e.ExampleOptions(examples_per_category=0).examples_per_category == 0
    manifest = example_documents()["manifest.json"]
    assert isinstance(manifest, e.BenchmarkManifest)
    changed = e.BenchmarkManifest(**{**manifest.model_dump(), "grouping": opts})
    assert e.content_hash(changed) != e.content_hash(manifest)


def test_backend_scope_frame_mapping_and_threshold_arrays() -> None:
    report = example_documents()["report.json"]
    assert isinstance(report, e.EvaluationReport)
    backend = report.evaluations[0].backend
    assert backend is not None
    key = backend.scope.frames[0]
    mapping = e.BackendFrameBinding(frame=key, export_frame_index=0)
    options = e.TrackEvalOptions(
        backend_revision="a" * 40,
        environment_revision="isolated/v1",
        preprocessing_revision="kitti/v1",
        metric_config_id="HOTA-CLEAR-Identity/v1",
        metric_config_sha256=HASH,
    )
    mapped = e.BackendResult(**{**backend.model_dump(), "options": options, "frame_bindings": (mapping,)})
    assert mapped.timeline == "full_native" and mapped.frame_bindings[0].frame == key
    with pytest.raises(ValidationError, match="duplicate backend"):
        e.BackendResult(**{**mapped.model_dump(), "frame_bindings": (mapping, mapping)})
    metric = e.BackendMetric(
        name="HOTA", thresholds=(0.5, 0.7), threshold_values=(100.0, 80.0), unit="percent", scale="0-100"
    )
    assert metric.thresholds[0] == 0.5
    with pytest.raises(ValidationError, match="align"):
        e.BackendMetric(**{**metric.model_dump(), "thresholds": (0.5,)})
    with pytest.raises(ValidationError, match="exactly one"):
        e.BackendMetric(**{**metric.model_dump(), "scalar": 90.0})
