"""Read native nuScenes keyframes without SDK-generated joins or media copies."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.models.evaluation import (
    BoxMetadata,
    CameraMetadata,
    DatasetReference,
    FrameKey,
    FrameReference,
    NuScenesAnnotation,
    NuScenesReferenceOptions,
    ReasonCount,
    ReferenceBox,
    ReferenceRejection,
    SequenceKey,
    SourceFile,
    content_hash,
)
from src.services.geometry.nuscenes_projection import project_camera_points, project_nuscenes_box
from src.services.geometry.transforms import quaternion_rotation

TABLES = (
    "scene",
    "sample",
    "sample_data",
    "sensor",
    "calibrated_sensor",
    "ego_pose",
    "sample_annotation",
    "instance",
    "category",
    "log",
    "attribute",
    "visibility",
)
PEDESTRIANS = frozenset(
    {
        "human.pedestrian.adult",
        "human.pedestrian.child",
        "human.pedestrian.construction_worker",
        "human.pedestrian.police_officer",
    }
)


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field {key}")
        result[key] = value
    return result


def _source(root: Path, path: Path, role: str) -> SourceFile:
    path = path.resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Source path escapes dataset root")
    return SourceFile.model_validate(
        {
            "relative_path": path.relative_to(root.resolve()).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "role": role,
        }
    )


def _tables(root: Path, version: str) -> tuple[dict[str, dict[str, dict[str, Any]]], list[SourceFile]]:
    tables: dict[str, dict[str, dict[str, Any]]] = {}
    sources: list[SourceFile] = []
    for name in TABLES:
        path = root / version / f"{name}.json"
        source = _source(root, path, "metadata")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != source.sha256:
            raise ValueError("Metadata changed during loading")
        rows = json.loads(raw, object_pairs_hook=_pairs)
        if not isinstance(rows, list):
            raise ValueError(f"{name} must contain a list")
        index: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("token"), str) or not row["token"]:
                raise ValueError(f"Invalid {name} token")
            if row["token"] in index:
                raise ValueError(f"Duplicate {name} token {row['token']}")
            index[row["token"]] = row
        tables[name] = index
        sources.append(source)
    return tables, sources


def _foreign_keys(tables: dict[str, dict[str, dict[str, Any]]]) -> None:
    joins = {
        "scene": {"log_token": "log", "first_sample_token": "sample", "last_sample_token": "sample"},
        "sample": {"scene_token": "scene", "prev": "sample", "next": "sample"},
        "sample_data": {
            "sample_token": "sample",
            "ego_pose_token": "ego_pose",
            "calibrated_sensor_token": "calibrated_sensor",
            "prev": "sample_data",
            "next": "sample_data",
        },
        "calibrated_sensor": {"sensor_token": "sensor"},
        "instance": {
            "category_token": "category",
            "first_annotation_token": "sample_annotation",
            "last_annotation_token": "sample_annotation",
        },
        "sample_annotation": {
            "sample_token": "sample",
            "instance_token": "instance",
            "visibility_token": "visibility",
            "prev": "sample_annotation",
            "next": "sample_annotation",
        },
    }
    optional_empty = {"prev", "next"}
    for table, fields in joins.items():
        for row in tables[table].values():
            for field, target in fields.items():
                value = row.get(field)
                if field in optional_empty and value == "":
                    continue
                if not isinstance(value, str) or value not in tables[target]:
                    raise ValueError(f"Invalid {table}.{field} foreign key")
    for row in tables["sample_annotation"].values():
        attributes = row.get("attribute_tokens")
        if not isinstance(attributes, list) or any(x not in tables["attribute"] for x in attributes):
            raise ValueError("Invalid annotation attribute foreign key")
        for direction, inverse in (("next", "prev"), ("prev", "next")):
            if row[direction]:
                other = tables["sample_annotation"][row[direction]]
                if other[inverse] != row["token"] or other["instance_token"] != row["instance_token"]:
                    raise ValueError("Nonreciprocal annotation chain")


def _finite_vector(value: Any, shape: tuple[int, ...]) -> tuple:
    values = np.asarray(value, dtype=float)
    if values.shape != shape or not np.all(np.isfinite(values)):
        raise ValueError("Invalid finite geometry")
    if len(shape) == 1:
        return tuple(float(x) for x in values)
    return tuple(tuple(float(x) for x in row) for row in values)


def _pose(row: dict[str, Any]) -> None:
    _finite_vector(row["translation"], (3,))
    rotation = _finite_vector(row["rotation"], (4,))
    if abs(sum(x * x for x in rotation) - 1) > 1e-5:
        raise ValueError("Native pose must have a unit quaternion")
    quaternion_rotation(rotation)


def _integer(value: Any, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _scene_samples(scene: dict[str, Any], samples: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    token = scene["first_sample_token"]
    visited: set[str] = set()
    previous = ""
    previous_timestamp: int | None = None
    while token:
        if token in visited:
            raise ValueError("Cycle in sample chain")
        visited.add(token)
        row = samples[token]
        timestamp = _integer(row["timestamp"], "Sample timestamp")
        if row["scene_token"] != scene["token"] or row["prev"] != previous:
            raise ValueError("Inconsistent scene/sample chain")
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise ValueError("Sample timestamps must be strictly increasing")
        rows.append(row)
        previous, previous_timestamp, token = row["token"], timestamp, row["next"]
    if previous != scene["last_sample_token"] or len(rows) != _integer(scene["nbr_samples"], "Sample count"):
        raise ValueError("Scene endpoint/count mismatch")
    if visited != {row["token"] for row in samples.values() if row["scene_token"] == scene["token"]}:
        raise ValueError("Scene contains samples outside its chain")
    return rows


def _annotation(row: dict[str, Any], tables: dict[str, dict[str, dict[str, Any]]]) -> NuScenesAnnotation:
    instance = tables["instance"][row["instance_token"]]
    category = tables["category"][instance["category_token"]]
    return NuScenesAnnotation(
        annotation_token=row["token"],
        sample_token=row["sample_token"],
        instance_token=row["instance_token"],
        category_token=category["token"],
        category_name=category["name"],
        translation_global=_finite_vector(row["translation"], (3,)),
        size_wlh=_finite_vector(row["size"], (3,)),
        rotation_wxyz=_finite_vector(row["rotation"], (4,)),
        attribute_tokens=tuple(row["attribute_tokens"]),
        visibility_token=row["visibility_token"],
        num_lidar_pts=row["num_lidar_pts"],
        num_radar_pts=row["num_radar_pts"],
        prev_token=row["prev"],
        next_token=row["next"],
    )


def load_nuscenes_reference(root: Path, *, options: NuScenesReferenceOptions) -> DatasetReference:
    """Return all selected camera keyframes and transparent projected references.

    Native records are retained once per source annotation even when several
    camera boxes refer to them. Rejections are per camera/annotation; clipped
    accepted boxes retain original bounds and a named clipping policy. Native
    visibility remains across-camera metadata, never camera-specific truth.
    """
    if (
        not options.scenes
        or not options.cameras
        or len(set(options.scenes)) != len(options.scenes)
        or len(set(options.cameras)) != len(options.cameras)
    ):
        raise ValueError("Scenes/cameras must be nonempty and unique")
    if options.profile.projection != options.projection:
        raise ValueError("Profile and adapter projection policies must agree")
    mapping = {x.native_class: x.class_name for x in options.profile.class_map}
    if any(x.startswith("human.pedestrian.") and x not in PEDESTRIANS for x in mapping):
        raise ValueError("Unsupported pedestrian subcategory in task taxonomy")
    tables, sources = _tables(root, options.version)
    _foreign_keys(tables)
    scenes: dict[str, dict[str, Any]] = {}
    for row in tables["scene"].values():
        if row["name"] in scenes:
            raise ValueError("Duplicate scene name")
        scenes[row["name"]] = row
    if any(name not in scenes for name in options.scenes):
        raise ValueError("Unknown selected scene")
    for row in tables["ego_pose"].values():
        _pose(row)
    for row in tables["calibrated_sensor"].values():
        _pose(row)
        sensor = tables["sensor"][row["sensor_token"]]
        if sensor["modality"] == "camera":
            intrinsic = _finite_vector(row["camera_intrinsic"], (3, 3))
            project_camera_points(np.array([[0.0, 0.0, 1.0]]), np.asarray(intrinsic))
    observations: dict[tuple[str, str], dict[str, Any]] = {}
    for row in tables["sample_data"].values():
        if type(row["is_key_frame"]) is not bool:
            raise ValueError("is_key_frame must be boolean")
        _integer(row["timestamp"], "Sensor timestamp")
        if not row["is_key_frame"]:
            continue
        calibrated = tables["calibrated_sensor"][row["calibrated_sensor_token"]]
        sensor = tables["sensor"][calibrated["sensor_token"]]
        key = (row["sample_token"], sensor["channel"])
        if key in observations:
            raise ValueError("Duplicate keyframe sensor join")
        observations[key] = row
    annotations: dict[str, list[tuple[dict[str, Any], NuScenesAnnotation]]] = defaultdict(list)
    for row in tables["sample_annotation"].values():
        native = _annotation(row, tables)
        annotations[native.sample_token].append((row, native))
    frames: list[FrameReference] = []
    boxes: list[ReferenceBox] = []
    native_records: dict[str, NuScenesAnnotation] = {}
    rejections: list[ReferenceRejection] = []
    counts: Counter[str] = Counter()
    image_sources: dict[str, SourceFile] = {}
    for scene_name in options.scenes:
        scene = scenes[scene_name]
        samples = _scene_samples(scene, tables["sample"])
        for channel in options.cameras:
            sequence = SequenceKey(
                dataset_id="nuscenes", split=options.split, sequence_uid=scene["token"], camera_channel=channel
            )
            previous_sensor_time: int | None = None
            for ordinal, sample in enumerate(samples):
                frame = FrameKey(sequence=sequence, frame_uid=sample["token"])
                observation = observations.get((sample["token"], channel))
                if observation is None:
                    raise ValueError(f"Missing camera keyframe {sample['token']}/{channel}")
                calibrated = tables["calibrated_sensor"][observation["calibrated_sensor_token"]]
                ego = tables["ego_pose"][observation["ego_pose_token"]]
                sensor_time = observation["timestamp"]
                if previous_sensor_time is not None and sensor_time <= previous_sensor_time:
                    raise ValueError("Camera timestamps must be strictly increasing")
                previous_sensor_time = sensor_time
                media = _source(root, root / observation["filename"], "image")
                image_sources[media.relative_path] = media
                with Image.open(root / media.relative_path) as image:
                    image.load()
                    width, height = image.size
                recorded_width = _integer(observation["width"], "Camera width")
                recorded_height = _integer(observation["height"], "Camera height")
                if (width, height) != (recorded_width, recorded_height):
                    raise ValueError("Image dimensions disagree with sensor metadata")
                camera = CameraMetadata(
                    sample_data_token=observation["token"],
                    ego_pose_token=ego["token"],
                    calibrated_sensor_token=calibrated["token"],
                    log_token=scene["log_token"],
                    sample_timestamp_us=sample["timestamp"],
                    sensor_timestamp_us=sensor_time,
                    ego_translation=_finite_vector(ego["translation"], (3,)),
                    ego_rotation_wxyz=_finite_vector(ego["rotation"], (4,)),
                    camera_translation=_finite_vector(calibrated["translation"], (3,)),
                    camera_rotation_wxyz=_finite_vector(calibrated["rotation"], (4,)),
                    camera_intrinsic=_finite_vector(calibrated["camera_intrinsic"], (3, 3)),
                )
                frames.append(
                    FrameReference(
                        key=frame,
                        source_ordinal=ordinal,
                        width=width,
                        height=height,
                        media=media,
                        timestamp_us=sensor_time,
                        timestamp_provenance="recorded_sensor",
                        camera=camera,
                    )
                )
                for box_ordinal, (row, native) in enumerate(annotations[sample["token"]]):
                    native_records[native.annotation_token] = native
                    class_name = mapping.get(native.category_name)
                    reason = "out_of_task" if class_name is None else None
                    result = None
                    if reason is None:
                        result = project_nuscenes_box(
                            row, ego, calibrated, width=width, height=height, min_depth_m=options.projection.min_depth_m
                        )
                        reason = result.rejection_reason
                    if reason is not None:
                        rejections.append(
                            ReferenceRejection(
                                frame=frame,
                                native_annotation_id=native.annotation_token,
                                reason=reason,
                                policy_revision=options.projection.revision,
                            )
                        )
                        counts[reason] += 1
                        continue
                    assert result is not None and result.bbox_xyxy is not None and class_name is not None
                    if result.clipped:
                        counts["canvas_clipped"] += 1
                    distance = float(
                        np.linalg.norm(
                            (np.asarray(native.translation_global) - np.asarray(camera.ego_translation))
                            @ quaternion_rotation(camera.ego_rotation_wxyz)
                        )
                    )
                    boxes.append(
                        ReferenceBox(
                            annotation_id=f"nuscenes/{options.split}/{scene['token']}/{channel}/{sample['token']}/{native.annotation_token}",
                            native_annotation_id=native.annotation_token,
                            frame=frame,
                            class_name=class_name,
                            bbox_xyxy=result.bbox_xyxy,
                            native_track_id=native.instance_token,
                            source_ordinal=box_ordinal,
                            reference_kind="projected_3d_box",
                            native_category=native.category_name,
                            metadata=BoxMetadata(
                                nuscenes=native,
                                ego_distance_m=distance,
                                original_bbox_xyxy=result.original_bbox_xyxy if result.clipped else None,
                                clipping_policy=options.projection.revision if result.clipped else None,
                            ),
                        )
                    )
    reference = DatasetReference(
        reference_revision=options.reference_revision,
        frames=tuple(frames),
        boxes=tuple(boxes),
        native_nuscenes_annotations=tuple(native_records.values()),
        sources=tuple(sources + list(image_sources.values())),
        profiles=(options.profile,),
        rejections=tuple(rejections),
        rejection_counts=tuple(ReasonCount(reason=reason, count=count) for reason, count in sorted(counts.items())),
    )
    return DatasetReference.model_validate({**reference.model_dump(), "content_sha256": content_hash(reference)})
