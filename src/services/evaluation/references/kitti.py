"""Native KITTI references and versioned local diagnostic preprocessing."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from scripts.eval_autolabel import Box, iou
from src.models.evaluation import (
    BoxMetadata,
    DatasetReference,
    EvaluationScope,
    ExcludedRecord,
    FrameKey,
    FrameReference,
    IgnoredRegion,
    KittiReferenceOptions,
    NativeKittiRow,
    PredictionRun,
    PreparedEvaluationView,
    ReferenceBox,
    ReferenceProfile,
    SequenceKey,
    SourceFile,
    content_hash,
)
from tools.public_gt.kitti import discover_images, read_native_rows

KITTI_CLASSES = {"Car": "car", "Pedestrian": "person", "Van": "car", "Person": "person"}
TARGET_CLASSES = {"Car", "Pedestrian"}


@dataclass(frozen=True)
class KittiItem:
    box: Box
    valid: bool


@dataclass(frozen=True)
class KittiReference:
    boxes: tuple[KittiItem, ...]
    dontcare: dict[int, tuple[tuple[float, float, float, float], ...]]


def load_reference(labels_path: Path, manifest_path: Path) -> KittiReference:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_to_local = {int(row["source_frame_index"]): int(row["frame_index"]) for row in manifest["frames"]}
    boxes = []
    dontcare: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    for number, line in enumerate(labels_path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 17:
            raise ValueError(f"KITTI line {number} must have 17 columns")
        try:
            source_frame = int(fields[0])
            track_id = int(fields[1])
            truncation, occlusion = float(fields[3]), int(fields[4])
            x1, y1, x2, y2 = (float(value) for value in fields[6:10])
            xyxy = (x1, y1, x2, y2)
        except ValueError as exc:
            raise ValueError(f"KITTI line {number} has invalid values") from exc
        frame = source_to_local.get(source_frame)
        if frame is None:
            continue
        if not xyxy[0] < xyxy[2] or not xyxy[1] < xyxy[3]:
            raise ValueError(f"KITTI line {number} has invalid box")
        category = fields[2]
        if category == "DontCare":
            dontcare[frame].append(xyxy)
        elif category in KITTI_CLASSES:
            valid = category in TARGET_CLASSES and truncation <= 0 and occlusion <= 2
            boxes.append(KittiItem(Box(frame, KITTI_CLASSES[category], xyxy, str(track_id)), valid))
    return KittiReference(tuple(boxes), {frame: tuple(rows) for frame, rows in dontcare.items()})


def _ioa(prediction: Box, region: tuple[float, float, float, float]) -> float:
    left = prediction.xyxy
    intersection = max(0.0, min(left[2], region[2]) - max(left[0], region[0])) * max(
        0.0, min(left[3], region[3]) - max(left[1], region[1])
    )
    area = (left[2] - left[0]) * (left[3] - left[1])
    return intersection / area if area > 0 else 0.0


def _preprocess_reasons(
    reference: KittiReference, predictions: list[Box], frames: set[int]
) -> tuple[list[Box], list[Box], dict[int, str]]:
    """Apply TrackEval-like ignore rules before the project's 2D tracking metrics."""
    valid_gt = [item.box for item in reference.boxes if item.valid and item.box.frame in frames]
    retained = []
    removed: dict[int, str] = {}
    for frame in sorted(frames):
        for label in ("car", "person"):
            candidates = [item for item in reference.boxes if item.box.frame == frame and item.box.label == label]
            observed_indices = [i for i, box in enumerate(predictions) if box.frame == frame and box.label == label]
            observed = [predictions[i] for i in observed_indices]
            matched_predictions: dict[int, int] = {}
            if candidates and observed:
                similarities = np.array([[iou(item.box, box) for box in observed] for item in candidates], dtype=float)
                similarities[similarities < 0.5] = 0.0
                rows, columns = linear_sum_assignment(-similarities)
                matched_predictions = {
                    int(column): int(row) for row, column in zip(rows, columns) if similarities[row, column] > 0
                }
            for index, box in enumerate(observed):
                candidate_index = matched_predictions.get(index)
                if candidate_index is not None:
                    if not candidates[candidate_index].valid:
                        removed[observed_indices[index]] = "distractor"
                    else:
                        retained.append(box)
                elif box.xyxy[3] - box.xyxy[1] <= 25:
                    removed[observed_indices[index]] = "small"
                elif any(_ioa(box, region) > 0.5 for region in reference.dontcare.get(frame, ())):
                    removed[observed_indices[index]] = "dontcare"
                else:
                    retained.append(box)
    return valid_gt, retained, removed


def preprocess_kitti_style(
    reference: KittiReference, predictions: list[Box], frames: set[int]
) -> tuple[list[Box], list[Box], dict[str, int]]:
    valid, retained, reasons = _preprocess_reasons(reference, predictions, frames)
    return (
        valid,
        retained,
        {reason: list(reasons.values()).count(reason) for reason in ("distractor", "small", "dontcare")},
    )


TARGET_MAP = {"Car": "car", "Pedestrian": "person"}
NATIVE_GEOMETRY = "native-unclipped/v1"
CLIPPED_GEOMETRY = "clip-to-image/v1"
LOCAL_ELIGIBILITY = "kitti-local-hungarian-ignore/v1"


def _source(path: Path, root: Path, role: str) -> SourceFile:
    return SourceFile.model_validate(
        {
            "relative_path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "role": role,
        }
    )


def load_kitti_reference(root: Path, *, options: KittiReferenceOptions) -> DatasetReference:
    """Load the full native source timeline, never sampling or copying images."""
    if options.split == "testing":
        raise ValueError("KITTI testing images have no native evaluation labels")
    if not options.sequences or len(set(options.sequences)) != len(options.sequences):
        raise ValueError("sequences must be nonempty and unique")
    if (options.source_fps is None) != (options.timestamp_provenance == "unavailable"):
        raise ValueError("declared FPS and timestamp provenance disagree")
    if not options.profiles:
        raise ValueError("at least one reference profile required")
    for profile in options.profiles:
        if {x.native_class: x.class_name for x in profile.class_map} != TARGET_MAP or set(profile.classes) != {
            "car",
            "person",
        }:
            raise ValueError("KITTI benchmark map must be exactly Car/car and Pedestrian/person")
        if profile.geometry_policy not in (NATIVE_GEOMETRY, CLIPPED_GEOMETRY):
            raise ValueError("unsupported KITTI geometry policy")
    # A reference has one concrete geometry; different geometries need separate revisions.
    geometries = {x.geometry_policy for x in options.profiles}
    if len(geometries) != 1:
        raise ValueError("profiles on one reference must share geometry")
    frames: list[FrameReference] = []
    boxes: list[ReferenceBox] = []
    regions: list[IgnoredRegion] = []
    native: list[NativeKittiRow] = []
    sources: list[SourceFile] = []
    base = root / options.split
    if not base.is_dir() and root.name == options.split:
        base = root
    for sequence in options.sequences:
        key = SequenceKey(dataset_id="kitti", split=options.split, sequence_uid=sequence)
        images = discover_images(base / "image_02" / sequence)
        dimensions = {}
        for index, image in enumerate(images):
            with Image.open(image) as decoded:
                decoded.load()
                width, height = decoded.size
            media = _source(image, root, "image")
            sources.append(media)
            frame = FrameKey(sequence=key, frame_uid=str(index))
            dimensions[frame] = (width, height)
            frames.append(
                FrameReference(
                    key=frame,
                    source_ordinal=index,
                    width=width,
                    height=height,
                    media=media,
                    timestamp_us=None if options.source_fps is None else round(index * 1e6 / options.source_fps),
                    timestamp_provenance=options.timestamp_provenance,
                )
            )
        labels = base / "label_02" / f"{sequence}.txt"
        source = _source(labels, root, "labels")
        sources.append(source)
        rows = read_native_rows(labels, sequence=key, source=source, frame_count=len(images))
        native.extend(rows)
        for row in rows:
            if row.native_class not in TARGET_MAP:
                role = (
                    "ignore"
                    if row.native_class == "DontCare"
                    else ("distractor" if row.native_class in ("Van", "Person") else "out_of_task")
                )
                regions.append(
                    IgnoredRegion.model_validate(
                        {
                            "source_id": row.source_id,
                            "frame": row.frame,
                            "role": role,
                            "native_category": row.native_class,
                            "bbox_xyxy": row.bbox_xyxy,
                            "native_kitti_source_id": row.source_id,
                        }
                    )
                )
                continue
            bbox = row.bbox_xyxy
            metadata = BoxMetadata(kitti_source_id=row.source_id)
            if next(iter(geometries)) == CLIPPED_GEOMETRY:
                width, height = dimensions[row.frame]
                bbox = (
                    max(0.0, min(float(width), bbox[0])),
                    max(0.0, min(float(height), bbox[1])),
                    max(0.0, min(float(width), bbox[2])),
                    max(0.0, min(float(height), bbox[3])),
                )
                metadata = BoxMetadata(
                    kitti_source_id=row.source_id, original_bbox_xyxy=row.bbox_xyxy, clipping_policy=CLIPPED_GEOMETRY
                )
            boxes.append(
                ReferenceBox(
                    annotation_id=row.source_id,
                    native_annotation_id=row.source_id,
                    frame=row.frame,
                    class_name=TARGET_MAP[row.native_class],
                    bbox_xyxy=bbox,
                    native_track_id=str(row.track_id),
                    source_ordinal=row.line_number - 1,
                    reference_kind="native_2d",
                    native_category=row.native_class,
                    metadata=metadata,
                )
            )
    reference = DatasetReference(
        reference_revision=options.reference_revision,
        frames=tuple(frames),
        boxes=tuple(boxes),
        native_kitti_rows=tuple(native),
        ignored_regions=tuple(regions),
        sources=tuple(sources),
        profiles=options.profiles,
    )
    return DatasetReference.model_validate({**reference.model_dump(), "content_sha256": content_hash(reference)})


def prepare_local_profile(
    reference: DatasetReference, run: PredictionRun, profile: ReferenceProfile, *, scope: EvaluationScope
) -> PreparedEvaluationView:
    """Dataset filtering precedes the shared greedy matcher; Hungarian is ignore-only."""
    if profile not in reference.profiles or scope.profile_id != profile.profile_id:
        raise ValueError("profile is not frozen in reference/scope")
    if run.view != scope.view or not any(x.config_id == run.config_id and x.view == run.view for x in scope.run_ids):
        raise ValueError("scope/run identity mismatch")
    known = {x.key for x in reference.frames}
    if not set(scope.frames) <= known or any(x.frame not in known for x in (*run.boxes, *run.processing)):
        raise ValueError("unknown canonical frame")
    if profile.designation == "standard_backend":
        raise ValueError("standard preprocessing belongs to TrackEval")
    allowed = set(scope.frames)
    gt = tuple(x for x in reference.boxes if x.frame in allowed)
    pred = tuple(x for x in run.boxes if x.frame in allowed)
    excluded: list[ExcludedRecord] = []
    unknown = tuple(x for x in pred if x.class_name not in profile.classes)
    if unknown and run.normalization.unknown_class_policy == "reject":
        raise ValueError("unknown prediction class")
    excluded.extend(
        ExcludedRecord(record_id=x.prediction_id, kind="prediction", frame=x.frame, reason="out_of_task_class")
        for x in unknown
    )
    pred = tuple(x for x in pred if x.class_name in profile.classes)
    if profile.designation == "local_ignore_adjusted":
        if profile.eligibility_policy != LOCAL_ELIGIBILITY or profile.geometry_policy != NATIVE_GEOMETRY:
            raise ValueError("unsupported local KITTI preprocessing revision/geometry")
        index = {frame: i for i, frame in enumerate(scope.frames)}
        candidates: list[KittiItem] = []
        dontcare: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
        eligible = set()
        for row in reference.native_kitti_rows:
            if row.frame not in allowed:
                continue
            if row.native_class == "DontCare":
                dontcare[index[row.frame]].append(row.bbox_xyxy)
            elif row.native_class in KITTI_CLASSES:
                valid = row.native_class in TARGET_CLASSES and row.truncated <= 0 and row.occluded <= 2
                if valid:
                    eligible.add(row.source_id)
                candidates.append(
                    KittiItem(
                        Box(index[row.frame], KITTI_CLASSES[row.native_class], row.bbox_xyxy, row.source_id), valid
                    )
                )
        local = KittiReference(tuple(candidates), {i: tuple(v) for i, v in dontcare.items()})
        legacy = [Box(index[x.frame], x.class_name, x.bbox_xyxy, x.prediction_id) for x in pred]
        _, kept, removed_reasons = _preprocess_reasons(local, legacy, set(index.values()))
        keep_ids = {x.track_id for x in kept}
        # Recover explicit per-record reasons from the exact same preprocessing pass.
        excluded.extend(
            ExcludedRecord(record_id=x.prediction_id, kind="prediction", frame=x.frame, reason=removed_reasons[i])
            for i, x in enumerate(pred)
            if x.prediction_id not in keep_ids
        )
        pred = tuple(x for x in pred if x.prediction_id in keep_ids)
        excluded.extend(
            ExcludedRecord(record_id=x.annotation_id, kind="reference", frame=x.frame, reason="difficulty")
            for x in gt
            if x.native_annotation_id not in eligible
        )
        gt = tuple(x for x in gt if x.native_annotation_id in eligible)
    if scope.target_population == "seeded_objects" or scope.eligible_annotation_ids:
        ids = set(scope.eligible_annotation_ids)
        if not ids <= {x.annotation_id for x in reference.boxes}:
            raise ValueError("scope has unknown annotation ID")
        excluded.extend(
            ExcludedRecord(record_id=x.annotation_id, kind="reference", frame=x.frame, reason="scope_population")
            for x in gt
            if x.annotation_id not in ids
        )
        gt = tuple(x for x in gt if x.annotation_id in ids)
    sequences = {x.sequence for x in scope.frames}
    return PreparedEvaluationView(
        scope=scope,
        profile=profile,
        reference_boxes=gt,
        prediction_boxes=pred,
        ignored_regions=tuple(x for x in reference.ignored_regions if x.frame in allowed),
        excluded=tuple(excluded),
        reference_frames=tuple(x for x in reference.frames if x.key.sequence in sequences),
        native_kitti_rows=tuple(x for x in reference.native_kitti_rows if x.frame.sequence in sequences),
    )
