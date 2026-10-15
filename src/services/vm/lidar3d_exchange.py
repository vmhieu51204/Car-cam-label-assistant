"""Native annotation exchange using saved CVAT geometry and sensor calibration."""

from __future__ import annotations

import json
import math
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.spatial.transform import Rotation

from src.services.geometry.kitti_projection import project_lidar
from src.services.geometry.transforms import quaternion_rotation
from src.services.vm.lidar3d_contracts import CLASSES, UploadOutput
from src.services.vm.lidar3d_datasets import table, token_index
from src.services.vm.lidar3d_tracking import iter_cuboids


def lidar_pose(frame: dict) -> tuple[np.ndarray, np.ndarray]:
    cal, ego = frame["lidar_calibration"], frame["ego_pose"]
    r_ego, r_cal = quaternion_rotation(ego["rotation"]), quaternion_rotation(cal["rotation"])
    rotation = r_ego @ r_cal
    translation = r_ego @ np.asarray(cal["translation"]) + np.asarray(ego["translation"])
    if not np.isfinite(translation).all():
        raise ValueError("Nonfinite nuScenes pose translation")
    return rotation, translation


def imported_annotations(base: Path, manifest: dict) -> UploadOutput:
    rows = []
    if manifest["format"] == "nuscenes":
        version = base / manifest["version"]
        categories = token_index(table(version, "category"))
        instances = token_index(table(version, "instance"))
        frames = {frame["sample_token"]: frame for frame in manifest["frames"]}
        for ann in table(version, "sample_annotation"):
            if ann["sample_token"] not in frames:
                continue
            frame = frames[ann["sample_token"]]
            category = categories[instances[ann["instance_token"]]["category_token"]]["name"]
            label = category_name(category)
            if label is None:
                continue
            rotation, translation = lidar_pose(frame)
            center = rotation.T @ (np.asarray(ann["translation"]) - translation)
            orient = Rotation.from_matrix(rotation.T @ quaternion_rotation(ann["rotation"])).as_euler("xyz")
            width, length, height = ann["size"]
            rows.append(
                {
                    "frame_index": frame["frame"],
                    "sample_token": frame["sample_token"],
                    "label": label,
                    "score": 1.0,
                    "center_lidar_xyz_m": center.tolist(),
                    "size_lwh_m": [length, width, height],
                    "yaw_lidar_rad": float(orient[2]),
                    "rotation_lidar_xyz_rad": orient.tolist(),
                    "track_id": ann["instance_token"],
                }
            )
    else:
        tracking_rows = None
        if manifest["tracking"]:
            path = base / "label_02" / f"{manifest['scene']}.txt"
            if not path.is_file():
                raise ValueError("KITTI tracking annotation import requires label_02/<sequence>.txt")
            tracking_rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
        for frame in manifest["frames"]:
            if tracking_rows is None:
                path = base / "label_2" / f"{frame['source_id']}.txt"
                if not path.is_file():
                    raise ValueError("KITTI annotation import requires label_2/<frame>.txt for every frame")
                labels = [line.split() for line in path.read_text().splitlines() if line.strip()]
            else:
                labels = [items[2:] for items in tracking_rows if int(items[0]) == int(frame["source_id"])]
            cal = {key: np.asarray(value) for key, value in frame["calibration"].items()}
            rotation = cal["R_rect"] @ cal["Tr_velo_cam"][:, :3]
            translation = cal["R_rect"] @ cal["Tr_velo_cam"][:, 3]
            native_ids = (
                [items[1] for items in tracking_rows if int(items[0]) == int(frame["source_id"])]
                if tracking_rows is not None
                else [None] * len(labels)
            )
            for fields, native_id in zip(labels, native_ids, strict=True):
                if len(fields) not in (15, 16):
                    raise ValueError("KITTI object labels need 15 fields and an optional score")
                if fields[0] == "DontCare":
                    continue
                if fields[0] not in manifest["classes"]:
                    raise ValueError("Unsupported KITTI object category")
                numbers = np.asarray([float(value) for value in fields[1:]], dtype=float)
                if not np.isfinite(numbers).all():
                    raise ValueError("Nonfinite KITTI annotation")
                height, width, length = map(float, fields[8:11])
                bottom = np.asarray([float(v) for v in fields[11:14]])
                yaw = float(fields[14])
                center = rotation.T @ (bottom + [0, -height / 2, 0] - translation)
                camera_from_box = np.array(
                    [[math.cos(yaw), math.sin(yaw), 0], [0, 0, -1], [-math.sin(yaw), math.cos(yaw), 0]]
                )
                orientation = Rotation.from_matrix(rotation.T @ camera_from_box).as_euler("xyz")
                rows.append(
                    {
                        "frame_index": frame["frame"],
                        "sample_token": frame["sample_token"],
                        "label": fields[0],
                        "score": float(fields[15]) if len(fields) == 16 else 1.0,
                        "center_lidar_xyz_m": center.tolist(),
                        "size_lwh_m": [length, width, height],
                        "yaw_lidar_rad": float(orientation[2]),
                        "rotation_lidar_xyz_rad": orientation.tolist(),
                        "track_id": f"{manifest['scene']}:{native_id}"
                        if native_id is not None and int(native_id) >= 0
                        else None,
                    }
                )
    return UploadOutput.model_validate(
        {
            "schema_version": "autolabel-lidar3d-upload/v1",
            "dataset_format": manifest["format"],
            "scene": manifest["scene"],
            "model_name": "Imported",
            "checkpoint_sha256": "",
            "frames": [frame["sample_token"] for frame in manifest["frames"]],
            "predictions": rows,
        }
    )


def category_name(name: str) -> str | None:
    if name in CLASSES:
        return name
    if name.startswith("human.pedestrian.") and name.rsplit(".", 1)[-1] in (
        "adult",
        "child",
        "construction_worker",
        "police_officer",
    ):
        return "pedestrian"
    if name.startswith("vehicle.bus."):
        return "bus"
    mapping = {
        "vehicle.car": "car",
        "vehicle.truck": "truck",
        "vehicle.construction": "construction_vehicle",
        "vehicle.trailer": "trailer",
        "vehicle.motorcycle": "motorcycle",
        "vehicle.bicycle": "bicycle",
        "movable_object.barrier": "barrier",
        "movable_object.trafficcone": "traffic_cone",
    }
    return mapping.get(name)


def saved_boxes(annotations: dict, labels: list[dict], frames: list[dict]) -> list[dict]:
    by_id = {row["id"]: row for row in labels}
    result = []
    for shape in iter_cuboids(annotations, len(frames)):
        if shape.get("outside"):
            continue
        if shape["type"] != "cuboid" or shape["frame"] not in range(len(frames)):
            raise ValueError("Saved annotation is not a mapped 3D cuboid")
        points = shape["points"]
        if len(points) != 16 or not np.isfinite(points).all() or min(points[6:9]) <= 0:
            raise ValueError("Invalid saved CVAT cuboid geometry")
        label = by_id[shape["label_id"]]
        attrs = {item["id"]: item["name"] for item in label["attributes"]}
        values = {attrs[item["spec_id"]]: item["value"] for item in shape.get("attributes", [])}
        score = float(values.get("confidence") or 1)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Saved confidence must be between 0 and 1")
        if values.get("review_status") == "rejected":
            continue
        result.append(
            {
                "frame": frames[shape["frame"]],
                "label": label["name"],
                "center": np.asarray(points[:3]),
                "rotation": np.asarray(points[3:6]),
                "size": points[6:9],
                "score": score,
                "review_status": values.get("review_status", "pending"),
                "track_id": values.get("autolabel_3d_track_id")
                or (f"cvat:{shape['track_id']}" if shape.get("track_id") is not None else None),
            }
        )
    return result


def export_annotations(
    manifest: dict,
    annotations: dict,
    labels: list[dict],
    target: Path,
    *,
    purpose: str = "draft",
    annotation_type: str = "detection",
) -> dict:
    """Export native-format predictions; a source-frame map accompanies KITTI."""
    boxes = saved_boxes(annotations, labels, manifest["frames"])
    if purpose not in ("training", "draft"):
        raise ValueError("Invalid export purpose")
    pending = sum(box["review_status"] != "accepted" for box in boxes)
    if purpose == "training" and pending:
        raise ValueError(
            f"Review required: {pending} saved cuboids are pending. Accept/reject them in CVAT and Save, or export a draft."
        )
    if annotation_type not in ("detection", "tracking"):
        raise ValueError("Invalid annotation type")
    excluded = 0
    if annotation_type == "tracking":
        if manifest["format"] == "nuscenes":
            supported = {"car", "truck", "bus", "trailer", "motorcycle", "bicycle", "pedestrian"}
            excluded = sum(box["label"] not in supported for box in boxes)
            boxes = [box for box in boxes if box["label"] in supported]
        elif not manifest.get("tracking"):
            raise ValueError("KITTI tracking export requires a native tracking sequence, not independent object frames")
        if any(not box["track_id"] for box in boxes):
            raise ValueError("Tracking export requires saved CVAT tracks with IDs for every included cuboid")
        observations = [(box["frame"]["frame"], box["track_id"]) for box in boxes]
        if len(set(observations)) != len(observations):
            raise ValueError("A tracking ID occurs more than once in a frame")
    if manifest["format"] == "nuscenes":
        results: dict[str, list] = {frame["sample_token"]: [] for frame in manifest["frames"]}
        for box in boxes:
            if box["label"] not in CLASSES:
                raise ValueError("Saved class is outside the nuScenes detection taxonomy")
            rotation, translation = lidar_pose(box["frame"])
            center = rotation @ box["center"] + translation
            q = Rotation.from_matrix(rotation @ Rotation.from_euler("xyz", box["rotation"]).as_matrix()).as_quat()
            length, width, height = box["size"]
            token = box["frame"]["sample_token"]
            results[token].append(
                {
                    "sample_token": token,
                    "translation": center.tolist(),
                    "size": [width, length, height],
                    "rotation": [float(q[3]), *q[:3].tolist()],
                    "velocity": [0.0, 0.0],
                    "detection_name": box["label"],
                    "detection_score": box["score"],
                    "attribute_name": "",
                }
            )
            if annotation_type == "tracking":
                item = results[token][-1]
                item.pop("attribute_name")
                item["tracking_name"] = item.pop("detection_name")
                item["tracking_score"] = item.pop("detection_score")
                item["tracking_id"] = box["track_id"]
            if len(results[token]) > 500:
                raise ValueError("nuScenes permits at most 500 boxes per sample")
        meta = {"use_camera": True, "use_lidar": True, "use_radar": False, "use_map": False, "use_external": False}
        target.write_text(json.dumps({"meta": meta, "results": results}, allow_nan=False))
    else:
        rows: dict[str, list[str]] = {frame["source_id"]: [] for frame in manifest["frames"]}
        tracking_rows = []
        identities = sorted({box["track_id"] for box in boxes if box["track_id"]})
        native = {
            identity: int(identity.rsplit(":", 1)[1])
            for identity in identities
            if identity.startswith(manifest["scene"] + ":") and identity.rsplit(":", 1)[1].isdigit()
        }
        used = set(native.values())
        for identity in identities:
            if identity not in native:
                candidate = 0
                while candidate in used:
                    candidate += 1
                native[identity] = candidate
                used.add(candidate)
        for box in boxes:
            frame = box["frame"]
            if box["label"] not in manifest["classes"]:
                raise ValueError("KITTI export requires native classes")
            cal = {key: np.asarray(value) for key, value in frame["calibration"].items()}
            length, width, height = box["size"]
            lidar_from_box = Rotation.from_euler("xyz", box["rotation"]).as_matrix()
            camera_from_box = cal["R_rect"] @ cal["Tr_velo_cam"][:, :3] @ lidar_from_box
            if not np.allclose(camera_from_box[:, 2], [0, -1, 0], atol=1e-4):
                raise ValueError("KITTI export requires cuboids upright in rectified camera coordinates")
            bottom = cal["R_rect"] @ (cal["Tr_velo_cam"][:, :3] @ box["center"] + cal["Tr_velo_cam"][:, 3]) + [
                0,
                height / 2,
                0,
            ]
            ry = math.atan2(-camera_from_box[2, 0], camera_from_box[0, 0])
            alpha = (ry - math.atan2(bottom[0], bottom[2]) + math.pi) % (2 * math.pi) - math.pi
            signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
            corners = (signs * np.asarray(box["size"]) / 2) @ lidar_from_box.T + box["center"]
            projection = project_lidar(corners, calibration=cal, image_size=tuple(frame["image_size"]))
            if np.any(projection.depth <= 0):
                raise ValueError("A cuboid crosses/lies behind the KITTI camera; edit it before native export")
            bbox = [*projection.uv.min(axis=0), *projection.uv.max(axis=0)]
            fields = [
                box["label"],
                "-1",
                "-1",
                f"{alpha:.9f}",
                *[f"{v:.9f}" for v in bbox],
                *[f"{v:.9f}" for v in (height, width, length, *bottom, ry, box["score"])],
            ]
            rows[frame["source_id"]].append(" ".join(fields))
            if annotation_type == "tracking":
                tracking_rows.append(" ".join([str(int(frame["source_id"])), str(native[box["track_id"]]), *fields]))
        with ZipFile(target, "w", ZIP_DEFLATED) as archive:
            if annotation_type == "tracking":
                archive.writestr(
                    f"label_02/{manifest['scene']}.txt", "\n".join(tracking_rows) + ("\n" if tracking_rows else "")
                )
                archive.writestr("track-map.json", json.dumps(native))
            else:
                for identity, items in rows.items():
                    archive.writestr(f"label_2/{identity}.txt", "\n".join(items) + ("\n" if items else ""))
            archive.writestr(
                "frame-map.json",
                json.dumps(
                    {
                        "source_sequence": manifest["scene"],
                        "frames": [
                            {"cvat_frame": row["frame"], "source_frame": row["source_id"]} for row in manifest["frames"]
                        ],
                    }
                ),
            )
    return {
        "boxes": len(boxes),
        "frames": len(manifest["frames"]),
        "purpose": purpose,
        "annotation_type": annotation_type,
        "excluded_nontracking_classes": excluded,
        "pending_review": pending,
    }
