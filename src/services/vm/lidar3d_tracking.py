"""Measured 3D identities, native CVAT tracks and saved-track interpolation."""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from math import atan2, pi

import numpy as np
from scipy.spatial.transform import Rotation

from src.models.perception import Box3D, Detection, Frame
from src.services.vm.lidar3d_contracts import UploadOutput, UploadPrediction


def track_predictions(output: UploadOutput, manifest: dict) -> UploadOutput:
    """Associate nuScenes in global space; native imported identities take priority."""
    if output.model_name == "Imported" or manifest["format"] != "nuscenes":
        return output
    from src.services.tracking import Tracker
    from src.services.vm.lidar3d_exchange import lidar_pose

    tracker = Tracker(data_source="real")
    predictions: list[UploadPrediction] = []
    for frame in manifest["frames"]:
        rows = [row for row in output.predictions if row.frame_index == frame["frame"]]
        rotation, translation = lidar_pose(frame)
        detections = []
        for row in rows:
            orientation = row.rotation_lidar_xyz_rad or (0, 0, row.yaw_lidar_rad)
            global_rotation = rotation @ Rotation.from_euler("xyz", orientation).as_matrix()
            detections.append(
                Detection(
                    label=row.label,
                    confidence=row.score,
                    box=Box3D(
                        center=tuple(rotation @ row.center_lidar_xyz_m + translation),
                        size_lwh=row.size_lwh_m,
                        yaw=atan2(global_rotation[1, 0], global_rotation[0, 0]),
                    ),
                )
            )
        identities = tracker.step(
            Frame(
                scene=manifest["scene"],
                sample_token=frame["sample_token"],
                frame_index=frame["frame"],
                timestamp_s=frame["timestamp_s"],
                detections=detections,
            )
        )
        predictions.extend(
            row.model_copy(update={"track_id": identity}) for row, identity in zip(rows, identities, strict=True)
        )
    return output.model_copy(update={"predictions": predictions})


def native_tracks(shapes: list[dict], output: UploadOutput, labels: list[dict], frame_count: int) -> dict:
    """Publish actual observations, marking each missing interval explicitly outside."""
    result: dict = {"shapes": [], "tracks": []}
    by_label = {label["id"]: label for label in labels}
    groups: dict[tuple, list] = {}
    for shape, prediction in zip(shapes, output.predictions, strict=True):
        if prediction.track_id is None:
            result["shapes"].append(shape)
        else:
            groups.setdefault((shape["label_id"], prediction.track_id), []).append(shape)
    for (label_id, identity), observations in groups.items():
        observations.sort(key=lambda shape: shape["frame"])
        if len({shape["frame"] for shape in observations}) != len(observations):
            raise ValueError("Duplicate 3D track observation in a frame")
        spec = next(attr["id"] for attr in by_label[label_id]["attributes"] if attr["name"] == "autolabel_3d_track_id")
        keyframes = []
        for index, shape in enumerate(observations):
            keyframe = {k: deepcopy(v) for k, v in shape.items() if k not in ("label_id", "source", "group")}
            keyframes.append(keyframe)
            next_frame = observations[index + 1]["frame"] if index + 1 < len(observations) else frame_count
            if next_frame > shape["frame"] + 1:
                keyframes.append({**deepcopy(keyframe), "frame": shape["frame"] + 1, "outside": True})
        result["tracks"].append(
            {
                "frame": observations[0]["frame"],
                "label_id": label_id,
                "source": observations[0]["source"],
                "group": 0,
                "attributes": [{"spec_id": spec, "value": identity}],
                "shapes": keyframes,
            }
        )
    return result


def iter_cuboids(annotations: dict, frame_count: int) -> Iterator[dict]:
    """Expand saved tracks with pinned CVAT's linear XYZ/size and short-arc Euler rules."""
    yield from annotations.get("shapes", [])
    for track in annotations.get("tracks", []):
        keyframes = sorted(track["shapes"], key=lambda item: item["frame"])
        if len({item["frame"] for item in keyframes}) != len(keyframes):
            raise ValueError("Duplicate saved 3D track keyframes")
        attributes = {attr["spec_id"]: attr for attr in track.get("attributes", [])}
        for index, shape in enumerate(keyframes):
            attributes.update({attr["spec_id"]: attr for attr in shape.get("attributes", [])})
            if shape["type"] != "cuboid":
                raise ValueError("Only 3D cuboid tracks can be exported")
            end = keyframes[index + 1] if index + 1 < len(keyframes) else None
            stop = min(end["frame"] if end else frame_count, frame_count)
            if shape.get("outside"):
                continue
            for frame in range(max(track["frame"], shape["frame"], 0), stop):
                points = np.asarray(shape["points"], dtype=float)
                if frame != shape["frame"] and end:
                    offset = (frame - shape["frame"]) / (end["frame"] - shape["frame"])
                    right = np.asarray(end["points"], dtype=float)
                    interpolated = points + (right - points) * offset
                    delta = (right[3:6] - points[3:6] + pi) % (2 * pi) - pi
                    interpolated[3:6] = (points[3:6] + delta * offset + pi) % (2 * pi) - pi
                    points = interpolated
                yield {
                    **shape,
                    "label_id": track["label_id"],
                    "frame": frame,
                    "points": points.tolist(),
                    "attributes": list(attributes.values()),
                    "track_id": track.get("id"),
                }
