"""Explicit enriched COCO associations, with retained negative images."""

import json
from pathlib import Path
from typing import Any

import pytest

from src.models.evaluation import (
    ClassMapping,
    CocoImageAssociation,
    CocoReferenceOptions,
    FrameBinding,
    FrameKey,
    FrameReference,
    ReferenceProfile,
    SequenceKey,
    SourceFile,
    content_hash,
)
from src.services.evaluation.references.coco import load_coco_reference


def fixture(tmp_path: Path) -> tuple[Path, dict[str, Any], CocoReferenceOptions, tuple[FrameBinding, ...]]:
    sequence = SequenceKey(dataset_id="coco", split="fixture", sequence_uid="clip")
    frames = tuple(
        FrameReference(
            key=FrameKey(sequence=sequence, frame_uid=str(i)),
            source_ordinal=i,
            width=100,
            height=80,
            media=SourceFile(relative_path=f"arbitrary-{9 - i}.png", sha256="a" * 64, role="image"),
            timestamp_provenance="unavailable",
        )
        for i in range(3)
    )
    profile = ReferenceProfile(
        profile_id="raw",
        revision="v1",
        designation="raw",
        classes=("car", "person"),
        class_map=(
            ClassMapping(native_class="car", class_name="car"),
            ClassMapping(native_class="person", class_name="person"),
        ),
        eligibility_policy="all/v1",
        geometry_policy="native-unclipped/v1",
    )
    options = CocoReferenceOptions(
        reference_revision="fixture/v1",
        profile=profile,
        images=tuple(CocoImageAssociation(image_id=i + 17, frame=f) for i, f in enumerate(frames)),
    )
    bindings = tuple(
        FrameBinding(mapping_id="explicit", domain="native_export", index=i + 30, frame=f.key)
        for i, f in enumerate(frames)
    )
    data = {
        "images": [
            {"id": i + 17, "file_name": f.media.relative_path, "width": 100, "height": 80} for i, f in enumerate(frames)
        ],
        "categories": [{"id": 1, "name": "car"}, {"id": 2, "name": "person"}],
        "annotations": [{"id": 4, "image_id": 17, "category_id": 1, "bbox": [1, 2, 30, 20], "track_id": 5}],
    }
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(data))
    return path, data, options, bindings


def test_explicit_xywh_identity_empty_images_and_hash(tmp_path: Path) -> None:
    path, _, options, bindings = fixture(tmp_path)
    reference = load_coco_reference(path, bindings=bindings, options=options)
    assert len(reference.frames) == 3 and len(reference.boxes) == 1
    assert reference.boxes[0].frame == bindings[0].frame
    assert reference.boxes[0].bbox_xyxy == (1.0, 2.0, 31.0, 22.0)
    assert reference.boxes[0].native_annotation_id == "4"
    assert reference.boxes[0].native_track_id == "5"
    assert reference.content_sha256 == content_hash(reference)


@pytest.mark.parametrize(
    "failure",
    [
        "image",
        "category",
        "box",
        "nonfinite",
        "dimension",
        "filename",
        "duplicate_image",
        "duplicate_annotation",
        "track_class",
        "duplicate_track",
        "crowd_invalid",
    ],
)
def test_invalid_alignment_and_fields(tmp_path: Path, failure: str) -> None:
    path, data, options, bindings = fixture(tmp_path)
    row = data["annotations"][0]
    if failure == "image":
        row["image_id"] = 123
    elif failure == "category":
        row["category_id"] = 123
    elif failure == "box":
        row["bbox"] = [0, 0, 0, 20]
    elif failure == "nonfinite":
        row["bbox"] = [0, 0, float("nan"), 20]
    elif failure == "dimension":
        data["images"][0]["width"] = 101
    elif failure == "filename":
        data["images"][0]["file_name"] = "guess.png"
    elif failure == "duplicate_image":
        data["images"].append(data["images"][0])
    elif failure == "duplicate_annotation":
        data["annotations"].append(row)
    elif failure == "track_class":
        data["annotations"].append({**row, "id": 5, "image_id": 18, "category_id": 2})
    elif failure == "duplicate_track":
        data["annotations"].append({**row, "id": 5})
    elif failure == "crowd_invalid":
        row["iscrowd"] = 2
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_coco_reference(path, bindings=bindings, options=options)


def test_missing_or_duplicate_bindings_rejected(tmp_path: Path) -> None:
    path, _, options, bindings = fixture(tmp_path)
    for invalid in (bindings[:-1], (bindings[0], bindings[0], bindings[2])):
        with pytest.raises(ValueError, match="bindings|binding"):
            load_coco_reference(path, bindings=invalid, options=options)


def test_duplicate_json_keys_and_crowd_retention(tmp_path: Path) -> None:
    path, data, options, bindings = fixture(tmp_path)
    data["annotations"][0]["iscrowd"] = 1
    path.write_text(json.dumps(data))
    reference = load_coco_reference(path, bindings=bindings, options=options)
    assert not reference.boxes and reference.ignored_regions[0].role == "ignore"
    path.write_text('{"images":[],"images":[]}')
    with pytest.raises(ValueError, match="duplicate COCO JSON key"):
        load_coco_reference(path, bindings=bindings, options=options)


def test_missing_enriched_metadata_rejected(tmp_path: Path) -> None:
    path, _, options, bindings = fixture(tmp_path)
    incomplete = CocoReferenceOptions(reference_revision=options.reference_revision, profile=options.profile, images=())
    with pytest.raises(ValueError, match="associations"):
        load_coco_reference(path, bindings=bindings, options=incomplete)
