"""COCO xywh fixture loading through explicit, independently enriched image joins."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from src.models.evaluation import (
    CocoReferenceOptions,
    DatasetReference,
    FrameBinding,
    IgnoredRegion,
    ReferenceBox,
    SequenceKey,
    SourceFile,
    content_hash,
)


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate COCO JSON key: {key}")
        result[key] = value
    return result


def _id(value: Any) -> int | str:
    if type(value) not in (int, str) or value == "":
        raise ValueError("COCO IDs must be integer or nonempty string")
    return value


def load_coco_reference(
    path: Path, *, bindings: tuple[FrameBinding, ...], options: CocoReferenceOptions
) -> DatasetReference:
    """Never infer an identity from a filename or fabricate dimensions/hashes."""
    data = json.loads(path.read_bytes(), object_pairs_hook=_pairs)
    associations = {x.image_id: x.frame for x in options.images}
    image_rows = data["images"]
    image_ids = [_id(x["id"]) for x in image_rows]
    if len(set(image_ids)) != len(image_ids) or set(image_ids) != set(associations):
        raise ValueError("COCO image associations must be unique and exhaustive")
    if len(bindings) != len(associations) or {x.frame for x in bindings} != {x.key for x in associations.values()}:
        raise ValueError("COCO frame bindings must be unique and exhaustive")
    binding_ids = [(x.mapping_id, x.domain, x.config_id, x.window_id, x.index) for x in bindings]
    if len(set(binding_ids)) != len(binding_ids):
        raise ValueError("duplicate COCO binding index")
    categories = {}
    for category in data["categories"]:
        key = _id(category["id"])
        if key in categories or not isinstance(category["name"], str) or not category["name"]:
            raise ValueError("invalid/duplicate COCO category")
        categories[key] = category["name"]
    frames = []
    for row in image_rows:
        frame = associations[row["id"]]
        if "file_name" in row and row["file_name"] != frame.media.relative_path:
            raise ValueError("COCO filename disagrees with enriched metadata")
        if any(
            field in row and (type(row[field]) is not int or row[field] != getattr(frame, field))
            for field in ("width", "height")
        ):
            raise ValueError("COCO dimensions disagree with enriched metadata")
        frames.append(frame)
    mapping = {x.native_class: x.class_name for x in options.profile.class_map}
    boxes = []
    ignored = []
    seen = set()
    track_classes: dict[tuple[SequenceKey, str], str] = {}
    track_frames = set()
    for ordinal, row in enumerate(data["annotations"]):
        annotation = _id(row["id"])
        if annotation in seen:
            raise ValueError("duplicate COCO annotation ID")
        seen.add(annotation)
        image_id, category_id = _id(row["image_id"]), _id(row["category_id"])
        if image_id not in associations or category_id not in categories:
            raise ValueError("COCO annotation references unknown image/category")
        frame = associations[image_id]
        native_class = categories[category_id]
        class_name = mapping.get(native_class, native_class)
        values = row["bbox"]
        if (
            not isinstance(values, list)
            or len(values) != 4
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in values)
            or values[2] <= 0
            or values[3] <= 0
        ):
            raise ValueError("invalid COCO xywh bbox")
        x, y, w, h = (float(v) for v in values)
        bbox = (x, y, x + w, y + h)
        source_id = f"coco/{type(annotation).__name__}/{annotation}"
        crowd = row.get("iscrowd", 0)
        if type(crowd) is not int or crowd not in (0, 1):
            raise ValueError("COCO iscrowd must be 0/1")
        if crowd or class_name not in options.profile.classes:
            ignored.append(
                IgnoredRegion(
                    source_id=source_id,
                    frame=frame.key,
                    role="ignore" if crowd else "out_of_task",
                    native_category=native_class,
                    bbox_xyxy=bbox,
                )
            )
            continue
        track = None if row.get("track_id") is None else str(_id(row["track_id"]))
        if track is not None:
            identity = (frame.key.sequence, track)
            if identity in track_classes and track_classes[identity] != class_name:
                raise ValueError("COCO track changes class")
            if (frame.key, track) in track_frames:
                raise ValueError("COCO frame repeats track ID")
            track_classes[identity] = class_name
            track_frames.add((frame.key, track))
        boxes.append(
            ReferenceBox(
                annotation_id=source_id,
                native_annotation_id=str(annotation),
                frame=frame.key,
                class_name=class_name,
                bbox_xyxy=bbox,
                native_track_id=track,
                source_ordinal=ordinal,
                reference_kind="native_2d",
                native_category=native_class,
            )
        )
    source = SourceFile(relative_path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), role="labels")
    reference = DatasetReference(
        reference_revision=options.reference_revision,
        frames=tuple(frames),
        boxes=tuple(boxes),
        ignored_regions=tuple(ignored),
        profiles=(options.profile,),
        sources=(source, *(x.media for x in frames)),
    )
    return DatasetReference.model_validate({**reference.model_dump(), "content_sha256": content_hash(reference)})
