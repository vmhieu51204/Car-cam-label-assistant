"""Strictly translate human frame-zero CVAT rectangles into immutable seeds."""

from __future__ import annotations

import asyncio
import math
from typing import Any

from src.models.video import BoundingBox2D, FrameManifest, SeedRectangle, SeedSet
from src.services.cvat import CvatReadClient


class SeedContractError(ValueError):
    """The selected CVAT job is stale or does not satisfy the 2D seed contract."""


class CvatSeedReader:
    def __init__(self, client: CvatReadClient) -> None:
        self.client = client

    async def capture(
        self,
        *,
        job_id: str,
        cvat_job_id: int,
        expected_revision: int,
        manifest: FrameManifest,
        expected_task_id: int | None = None,
    ) -> SeedSet:
        job, data_meta, annotations, labels = await asyncio.gather(
            self.client.get_job(cvat_job_id),
            self.client.get_job_data_meta(cvat_job_id),
            self.client.get_job_annotations(cvat_job_id),
            self.client.get_job_labels(cvat_job_id),
        )
        _validate_job(job, data_meta, manifest, expected_task_id=expected_task_id)
        revision = _required_int(annotations.get("version"), "annotation revision", minimum=0)
        if revision != expected_revision:
            raise SeedContractError(
                f"stale annotation revision: expected {expected_revision}, CVAT returned {revision}"
            )
        label_map = _label_map(labels)
        seeds = _parse_annotations(annotations, label_map, manifest.width, manifest.height)
        if not seeds:
            raise SeedContractError("frame 0 has no rectangle seeds")
        try:
            return SeedSet(
                job_id=job_id,
                cvat_job_id=cvat_job_id,
                annotation_revision=revision,
                seeds=sorted(seeds, key=lambda seed: seed.seed_id),
            )
        except ValueError as error:
            raise SeedContractError(str(error)) from error


def _validate_job(
    job: dict[str, Any], data_meta: dict[str, Any], manifest: FrameManifest, *, expected_task_id: int | None
) -> None:
    if expected_task_id is not None:
        task_id = _required_int(job.get("task_id"), "CVAT task ID", minimum=1)
        if task_id != expected_task_id:
            raise SeedContractError(f"CVAT job belongs to task {task_id}, expected task {expected_task_id}")
    dimension = job.get("dimension")
    if dimension is not None and str(dimension).lower() != "2d":
        raise SeedContractError("CVAT seed job must use the 2D annotation dimension")
    start_frame = _required_int(job.get("start_frame", 0), "job start frame", minimum=0)
    if start_frame != 0:
        raise SeedContractError("CVAT seed job must start at frame 0")
    size = _required_int(data_meta.get("size"), "CVAT frame count", minimum=1)
    expected_count = manifest.source_frame_count or manifest.frame_count
    if size != expected_count:
        raise SeedContractError(f"CVAT frame count {size} does not match manifest source frame count {expected_count}")
    frames = data_meta.get("frames")
    if isinstance(frames, list) and frames:
        first = frames[0]
        if not isinstance(first, dict):
            raise SeedContractError("CVAT first-frame metadata is invalid")
        width = _required_int(first.get("width"), "CVAT frame width", minimum=1)
        height = _required_int(first.get("height"), "CVAT frame height", minimum=1)
        if (width, height) != (manifest.width, manifest.height):
            raise SeedContractError(
                f"CVAT first-frame size {width}x{height} does not match manifest {manifest.width}x{manifest.height}"
            )


def _label_map(labels: list[dict[str, Any]]) -> dict[int, str]:
    result: dict[int, str] = {}
    for label in labels:
        label_id = _required_int(label.get("id"), "label ID", minimum=1)
        name = label.get("name")
        if not isinstance(name, str) or not name.strip():
            raise SeedContractError(f"CVAT label {label_id} has no name")
        label_type = label.get("type")
        if label_type not in (None, "any", "rectangle"):
            continue
        result[label_id] = name.strip()
    if not result:
        raise SeedContractError("CVAT job has no labels that support rectangles")
    return result


def _parse_annotations(
    annotations: dict[str, Any], label_map: dict[int, str], width: int, height: int
) -> list[SeedRectangle]:
    raw_shapes = annotations.get("shapes", [])
    raw_tracks = annotations.get("tracks", [])
    if not isinstance(raw_shapes, list) or not isinstance(raw_tracks, list):
        raise SeedContractError("CVAT annotations payload has invalid shape or track lists")
    seeds: list[SeedRectangle] = []
    for shape in raw_shapes:
        seeds.append(
            _parse_shape(shape, label_map=label_map, width=width, height=height, track_id=None, track_source=None)
        )
    for track in raw_tracks:
        if not isinstance(track, dict):
            raise SeedContractError("CVAT track payload is invalid")
        track_id = _required_int(track.get("id"), "track ID", minimum=0)
        label_id = _required_int(track.get("label_id"), "track label ID", minimum=1)
        shapes = track.get("shapes")
        if not isinstance(shapes, list) or not shapes:
            raise SeedContractError(f"CVAT track {track_id} has no shapes")
        for raw_shape in shapes:
            if not isinstance(raw_shape, dict):
                raise SeedContractError(f"CVAT track {track_id} contains invalid shape data")
            shape = dict(raw_shape)
            shape.setdefault("label_id", label_id)
            seeds.append(
                _parse_shape(
                    shape,
                    label_map=label_map,
                    width=width,
                    height=height,
                    track_id=track_id,
                    track_source=track.get("source"),
                )
            )
    return seeds


def _parse_shape(
    raw: object,
    *,
    label_map: dict[int, str],
    width: int,
    height: int,
    track_id: int | None,
    track_source: object,
) -> SeedRectangle:
    if not isinstance(raw, dict):
        raise SeedContractError("CVAT shape payload is invalid")
    frame = _required_int(raw.get("frame"), "shape frame", minimum=0)
    if frame != 0:
        raise SeedContractError("seed capture found annotations outside frame 0")
    if raw.get("type") != "rectangle":
        raise SeedContractError("frame 0 contains a non-rectangle annotation")
    if bool(raw.get("outside", False)):
        raise SeedContractError("frame 0 seed cannot be marked outside")
    source = raw.get("source", track_source)
    if source not in (None, "manual", "auto"):
        raise SeedContractError("frame 0 seed must be a manual or automatic CVAT annotation")
    label_id = _required_int(raw.get("label_id"), "shape label ID", minimum=1)
    if label_id not in label_map:
        raise SeedContractError(f"shape uses unsupported label ID {label_id}")
    shape_id = _required_int(raw.get("id"), "shape ID", minimum=0)
    points = raw.get("points")
    if not isinstance(points, list) or len(points) != 4:
        raise SeedContractError(f"rectangle shape {shape_id} must contain four coordinates")
    try:
        coordinates = [float(value) for value in points]
    except (TypeError, ValueError) as error:
        raise SeedContractError(f"rectangle shape {shape_id} contains invalid coordinates") from error
    if any(not math.isfinite(value) for value in coordinates):
        raise SeedContractError(f"rectangle shape {shape_id} contains non-finite coordinates")
    try:
        bbox = BoundingBox2D(xtl=coordinates[0], ytl=coordinates[1], xbr=coordinates[2], ybr=coordinates[3])
    except ValueError as error:
        raise SeedContractError(f"rectangle shape {shape_id} is invalid: {error}") from error
    if bbox.xbr > width or bbox.ybr > height:
        raise SeedContractError(f"rectangle shape {shape_id} exceeds the first-frame bounds")
    identity = f"track:{track_id}" if track_id is not None else f"shape:{shape_id}"
    seed_id = f"cvat:{identity}:frame:0"
    return SeedRectangle(
        seed_id=seed_id,
        object_id=identity,
        source="auto_seed" if source == "auto" else "human_seed",
        label_id=label_id,
        label_name=label_map[label_id],
        shape_id=shape_id,
        track_id=track_id,
        bbox=bbox,
        occluded=bool(raw.get("occluded", False)),
    )


def _required_int(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SeedContractError(f"{name} must be an integer greater than or equal to {minimum}")
    return value
