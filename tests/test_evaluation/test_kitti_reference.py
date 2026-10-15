"""Small image/label fixtures with independent native and ignore expectations."""

from pathlib import Path

import pytest
from PIL import Image

from src.models.evaluation import (
    BBox,
    ClassMapping,
    DatasetReference,
    EvaluationScope,
    FrameProcessing,
    KittiReferenceOptions,
    ObservationSource,
    PredictionBox,
    PredictionNormalizationOptions,
    PredictionRun,
    ReferenceProfile,
    RunKey,
    RunProvenance,
    Score,
    content_hash,
)
from src.services.evaluation.references.kitti import (
    CLIPPED_GEOMETRY,
    LOCAL_ELIGIBILITY,
    NATIVE_GEOMETRY,
    load_kitti_reference,
    prepare_local_profile,
)


def profile(adjusted: bool = False, clipped: bool = False) -> ReferenceProfile:
    return ReferenceProfile(
        profile_id="local" if adjusted else "raw",
        revision="v1",
        designation="local_ignore_adjusted" if adjusted else "raw",
        classes=("car", "person"),
        class_map=(
            ClassMapping(native_class="Car", class_name="car"),
            ClassMapping(native_class="Pedestrian", class_name="person"),
        ),
        eligibility_policy=LOCAL_ELIGIBILITY if adjusted else "all-targets/v1",
        geometry_policy=CLIPPED_GEOMETRY if clipped else NATIVE_GEOMETRY,
    )


def line(
    frame: int = 0,
    track: int = 7,
    category: str = "Car",
    box: str = "1 2 51 42",
    truncated: str = "0",
    occluded: int = 0,
) -> str:
    return f"{frame} {track} {category} {truncated} {occluded} 0.25 {box} 1.5 2.5 3.5 4.5 5.5 6.5 -0.5"


def source(root: Path, rows: list[str], sequence: str = "0000", width: int = 100) -> None:
    images = root / "training/image_02" / sequence
    images.mkdir(parents=True)
    labels = root / "training/label_02"
    labels.mkdir(exist_ok=True)
    for index in range(3):
        Image.new("RGB", (width, 80)).save(images / f"{index:06d}.png")
    (labels / f"{sequence}.txt").write_text("\n".join(rows), encoding="utf-8")


def load(root: Path, *, clipped: bool = False, sequences: tuple[str, ...] = ("0000",)) -> DatasetReference:
    return load_kitti_reference(
        root,
        options=KittiReferenceOptions(
            sequences=sequences,
            reference_revision="fixture/v1",
            profiles=(profile(clipped=clipped),) if clipped else (profile(), profile(True)),
            source_fps=10.0,
            timestamp_provenance="declared_fps",
        ),
    )


def test_lossless_targets_negative_frames_and_distractors(tmp_path: Path) -> None:
    rows = [
        line(),
        line(track=8, category="Pedestrian"),
        line(track=9, category="Van"),
        line(track=10, category="Person"),
        line(track=11, category="Cyclist"),
        "0 -1 DontCare -1 -1 -10 0 0 10 10 -1 -1 -1 -1000 -1000 -1000 -10",
        "0 -1 DontCare -1 -1 -10 20 20 30 30 -1 -1 -1 -1000 -1000 -1000 -10",
    ]
    source(tmp_path, rows)
    reference = load(tmp_path)
    assert len(reference.frames) == 3 and len(reference.boxes) == 2
    assert len(reference.native_kitti_rows) == 7
    native = reference.native_kitti_rows[0]
    assert (native.alpha, native.dimensions_hwl, native.location_xyz, native.rotation_y) == (
        0.25,
        (1.5, 2.5, 3.5),
        (4.5, 5.5, 6.5),
        -0.5,
    )
    assert (native.line_number, native.track_id, native.truncated, native.occluded) == (1, 7, 0, 0)
    assert [r.role for r in reference.ignored_regions] == [
        "distractor",
        "distractor",
        "out_of_task",
        "ignore",
        "ignore",
    ]
    assert reference.frames[2].timestamp_us == 200000
    assert reference.content_sha256 == content_hash(reference)
    assert reference.native_kitti_rows[-1].location_xyz == (-1000.0, -1000.0, -1000.0)


def test_empty_sequences_and_per_sequence_dimensions(tmp_path: Path) -> None:
    source(tmp_path, [], "0000", 100)
    source(tmp_path, [], "0001", 120)
    reference = load(tmp_path, sequences=("0000", "0001"))
    assert len(reference.frames) == 6 and not reference.boxes
    assert [f.width for f in reference.frames] == [100] * 3 + [120] * 3
    assert len({f.key for f in reference.frames}) == 6


def test_native_overflow_preserved_and_named_clipping(tmp_path: Path) -> None:
    source(tmp_path, [line(box="10 10 100.788766 70")])
    native = load(tmp_path)
    clipped = load(tmp_path, clipped=True)
    assert native.boxes[0].bbox_xyxy[2] == 100.788766
    assert clipped.boxes[0].bbox_xyxy[2] == 100
    assert clipped.native_kitti_rows[0] == native.native_kitti_rows[0]
    assert clipped.boxes[0].metadata.original_bbox_xyxy == native.boxes[0].bbox_xyxy
    assert clipped.boxes[0].metadata.clipping_policy == CLIPPED_GEOMETRY


@pytest.mark.parametrize(
    "rows",
    [
        [line(), line()],
        [line(), line(1, category="Van")],
        [line(frame=3)],
        [line(truncated="0.0")],
        [line(occluded=4)],
        [line(track=-1)],
        [line().replace("1.5", "nan")],
        [line().replace("1.5", "-1")],
        [line().replace("0 7 Car", "0.5 7 Car")],
        [line(box="0 0 0 1")],
        [line(category="Unknown")],
        [line().replace("6.5", "inf")],
    ],
)
def test_invalid_native_fields_and_identity(tmp_path: Path, rows: list[str]) -> None:
    source(tmp_path, rows)
    with pytest.raises(ValueError):
        load(tmp_path)


def test_exhaustive_images_and_no_seeds(tmp_path: Path) -> None:
    source(tmp_path, [line(frame=2)])
    assert len(load(tmp_path).frames) == 3
    (tmp_path / "training/image_02/0000/000001.png").unlink()
    with pytest.raises(ValueError, match="contiguous"):
        load(tmp_path)


def prediction_run(
    reference: DatasetReference, boxes: list[tuple[str, BBox]], *, unknown: bool = False
) -> PredictionRun:
    options = PredictionNormalizationOptions(
        views=("raw",), unknown_class_policy="count_out_of_task" if unknown else "reject"
    )
    predictions = tuple(
        PredictionBox(
            prediction_id=f"p{i}",
            frame=reference.frames[0].key,
            class_name=label,
            bbox_xyxy=box,
            raw_track_id=f"p{i}",
            scoring_ordinal=i,
            source=ObservationSource(
                mapping_id="native", source_ordinal=i, config_id="fixture", source_observation_id=f"p{i}"
            ),
            score=Score(value=None, semantics="missing", source="fixture"),
            origin="model",
        )
        for i, (label, box) in enumerate(boxes)
    )
    return PredictionRun(
        config_id="fixture",
        view="raw",
        status="completed",
        raw_status="completed",
        processing=tuple(
            FrameProcessing(
                frame=f.key, selected=True, state="processed", mapping_id="native", evidence="wrapper_declared"
            )
            for f in reference.frames
        ),
        boxes=predictions,
        normalization=options,
        normalization_sha256=content_hash(options),
        provenance=RunProvenance(
            result_sha256="a" * 64,
            config_sha256="b" * 64,
            model_revision="fixture",
            evaluator_revision="v1",
            inference_variant="fixture",
        ),
    )


def scope(reference: DatasetReference, adjusted: bool = False, *, population: str = "all_objects") -> EvaluationScope:
    return EvaluationScope.model_validate(
        dict(
            scope_id="fixture",
            kind="fixed_set",
            frames=tuple(f.key for f in reference.frames),
            profile_id="local" if adjusted else "raw",
            view="raw",
            cohort="seeded_propagation" if population == "seeded_objects" else "text_discovery",
            target_population=population,
            interpolation=PredictionNormalizationOptions().interpolation,
            run_ids=(RunKey(config_id="fixture", view="raw"),),
        )
    )


def test_profile_hungarian_then_ignore_raw_separate_and_unknown_count(tmp_path: Path) -> None:
    source(
        tmp_path,
        [
            line(box="0 0 100 50"),
            line(track=8, category="Van", box="100 0 200 50"),
            line(track=9, box="200 0 300 50", truncated="1"),
            "0 -1 DontCare -1 -1 -10 300 0 400 50 -1 -1 -1 -1000 -1000 -1000 -10",
        ],
        width=700,
    )
    reference = load(tmp_path)
    run = prediction_run(
        reference,
        [
            ("car", b)
            for b in [
                (0.0, 0.0, 100.0, 50.0),
                (100.0, 0.0, 200.0, 50.0),
                (200.0, 0.0, 300.0, 50.0),
                (300.0, 0.0, 400.0, 50.0),
                (450.0, 0.0, 480.0, 20.0),
                (500.0, 0.0, 600.0, 50.0),
            ]
        ]
        + [("cyclist", (0.0, 0.0, 100.0, 50.0))],
        unknown=True,
    )
    raw = prepare_local_profile(reference, run, reference.profiles[0], scope=scope(reference))
    adjusted = prepare_local_profile(reference, run, reference.profiles[1], scope=scope(reference, True))
    assert (len(raw.reference_boxes), len(raw.prediction_boxes)) == (2, 6)
    assert (len(adjusted.reference_boxes), len(adjusted.prediction_boxes)) == (1, 2)
    assert sorted(x.reason for x in adjusted.excluded) == [
        "difficulty",
        "distractor",
        "distractor",
        "dontcare",
        "out_of_task_class",
        "small",
    ]
    assert adjusted.native_kitti_rows == reference.native_kitti_rows
    assert adjusted.reference_frames == reference.frames
    rejected = prediction_run(reference, [("cyclist", (0.0, 0.0, 100.0, 50.0))])
    with pytest.raises(ValueError, match="unknown prediction"):
        prepare_local_profile(reference, rejected, reference.profiles[0], scope=scope(reference))
    seeded = prepare_local_profile(
        reference, run, reference.profiles[0], scope=scope(reference, population="seeded_objects")
    )
    assert not seeded.reference_boxes  # empty explicit seed population never acquires GT


def test_invalid_class_map_and_cadence(tmp_path: Path) -> None:
    source(tmp_path, [])
    wrong = profile().model_dump()
    wrong["class_map"] += (ClassMapping(native_class="Cyclist", class_name="person"),)
    with pytest.raises(ValueError, match="map"):
        load_kitti_reference(
            tmp_path,
            options=KittiReferenceOptions(
                sequences=("0000",), reference_revision="v1", profiles=(ReferenceProfile.model_validate(wrong),)
            ),
        )
    with pytest.raises(ValueError, match="provenance"):
        load_kitti_reference(
            tmp_path,
            options=KittiReferenceOptions(
                sequences=("0000",), reference_revision="v1", profiles=(profile(),), source_fps=10.0
            ),
        )


def test_unlabeled_testing_split_rejected_and_identity_isolated(tmp_path: Path) -> None:
    from src.models.evaluation import FrameKey, SequenceKey

    training = SequenceKey(dataset_id="kitti", split="training", sequence_uid="0013")
    testing = SequenceKey(dataset_id="kitti", split="testing", sequence_uid="0013")
    assert FrameKey(sequence=training, frame_uid="0") != FrameKey(sequence=testing, frame_uid="0")
    with pytest.raises(ValueError, match="no native evaluation labels"):
        load_kitti_reference(
            tmp_path,
            options=KittiReferenceOptions(
                split="testing", sequences=("0013",), reference_revision="fixture/v1", profiles=(profile(),)
            ),
        )


def test_filtered_scope_preserves_full_sequence_metadata(tmp_path: Path) -> None:
    source(tmp_path, [line()])
    reference = load(tmp_path)
    run = prediction_run(reference, [])
    filtered = EvaluationScope.model_validate({**scope(reference).model_dump(), "frames": (reference.frames[1].key,)})
    view = prepare_local_profile(reference, run, reference.profiles[0], scope=filtered)
    assert not view.reference_boxes and not view.prediction_boxes
    assert view.reference_frames == reference.frames
    assert view.native_kitti_rows == reference.native_kitti_rows
