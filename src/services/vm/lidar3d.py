"""Run the provisioned nuScenes detector and publish editable native CVAT 3D drafts."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import httpx
import numpy as np

from src.config import PROJECT_ROOT, Settings
from src.services.detection.centerpoint import CenterPointOutput
from src.services.vm.contracts import file_sha256
from src.services.vm.lidar3d_contracts import BEVFUSION_CHECKPOINT_SHA, UploadOutput
from src.services.vm.lidar3d_datasets import CAMERAS, ingest, nuscenes, safe_path
from src.services.vm.lidar3d_exchange import imported_annotations
from src.services.vm.lidar3d_links import COCO_MAPPING
from src.services.vm.lidar3d_tracking import iter_cuboids, native_tracks, track_predictions
from src.services.vm.storage import VmStore, atomic_json, encode

CHECKPOINT_SHA = "191a38226d2ce990f26cc3f78c454310cc562d81d29e86d8ca8fa39c0a44215d"
CLASSES = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]


def dataset(settings: Settings) -> dict:
    path = settings.lidar3d_dataset_root / "dataset-manifest.json"
    if not settings.lidar3d_enabled or not path.is_file() or file_sha256(path) != settings.lidar3d_manifest_sha256:
        raise ValueError("3D dataset is not configured or its manifest changed")
    payload = json.loads(path.read_text())
    if payload.get("schema") != "autolabel-nuscenes-vm/v1" or payload.get("version") != "v1.0-mini":
        raise ValueError("Unsupported 3D dataset manifest")
    return payload


def media_path(settings: Settings, manifest: dict, name: str) -> Path:
    root = settings.lidar3d_dataset_root.resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file() or file_sha256(path) != manifest["files"].get(name):
        raise ValueError("3D media is missing or differs from its manifest")
    return path


def pcd_bytes(path: Path, dimensions: int = 5) -> bytes:
    data = np.fromfile(path, dtype="<f4")
    if dimensions not in (4, 5) or not data.size or data.size % dimensions:
        raise ValueError("Invalid native LiDAR point cloud")
    points = data.reshape(-1, dimensions)[:, :4]
    if not np.isfinite(points).all():
        raise ValueError("Nonfinite point cloud coordinates")
    header = (
        "# .PCD v0.7\nVERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\n"
        f"COUNT 1 1 1 1\nWIDTH {len(points)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(points)}\nDATA binary\n"
    )
    return header.encode() + points.astype("<f4").tobytes()


def prepare_media(settings: Settings, manifest: dict, frames: list[dict], output: Path) -> None:
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=1, strict_timestamps=False) as archive:
        for index, frame in enumerate(frames):
            name = f"frame_{index:06d}"
            archive.writestr(
                f"pointcloud/{name}.pcd",
                pcd_bytes(media_path(settings, manifest, frame["lidar"]), frame.get("point_dimensions", 5)),
            )
            for channel, filename in sorted(frame["cameras"].items()):
                archive.write(
                    media_path(settings, manifest, filename),
                    f"related_images/{name}_pcd/{channel}{Path(filename).suffix.lower()}",
                )


def cuboid_shapes(
    output: CenterPointOutput | UploadOutput, frames: list[dict], labels: list[dict], detector_name: str = "CenterPoint"
) -> list[dict]:
    by_name = {label["name"]: label for label in labels}
    shapes = []
    for row in output.predictions:
        if row.frame_index >= len(frames) or frames[row.frame_index]["sample_token"] != row.sample_token:
            raise ValueError("3D prediction frame identity differs from its point cloud")
        if row.label not in by_name:
            raise ValueError("Unsupported detector class in 3D output")
        label = by_name[row.label]
        attributes = {item["name"]: item["id"] for item in label["attributes"]}
        values = {"review_status": "pending", "confidence": str(row.score), "detector": detector_name}
        if getattr(row, "linked_2d_tracks", None):
            values["linked_2d_tracks"] = json.dumps(row.linked_2d_tracks, sort_keys=True)
        orientation = getattr(row, "rotation_lidar_xyz_rad", None) or (0, 0, row.yaw_lidar_rad)
        shapes.append(
            {
                "type": "cuboid",
                "frame": row.frame_index,
                "label_id": label["id"],
                "source": "manual" if detector_name == "Imported" else "auto",
                "occluded": False,
                "outside": False,
                "rotation": 0,
                "group": 0,
                "points": [*row.center_lidar_xyz_m, *orientation, *row.size_lwh_m, *([0] * 7)],
                "attributes": [
                    {"spec_id": attributes[name], "value": value}
                    for name, value in values.items()
                    if name in attributes
                ],
            }
        )
    return shapes


class Lidar3DWorkflow:
    def __init__(self, settings: Settings, store: VmStore):
        self.settings, self.store = settings, store

    def run(self, job: dict, credentials: dict) -> dict:
        if job.get("input_kind") == "dataset_upload" or job["profile"]["variant"] == "bevfusion":
            return self.run_dataset(job, credentials)

        settings, store, identity = self.settings, self.store, job["job_id"]
        manifest = dataset(settings)
        if job["profile"]["manifest_sha256"] != settings.lidar3d_manifest_sha256:
            raise ValueError("3D job dataset identity changed")
        frames = manifest["scenes"][job["scene"]]["frames"][: job["total_frames"]]
        if len(frames) != job["total_frames"]:
            raise ValueError("3D scene frame coverage changed")
        root = store.directory(identity)
        directory = root / "private/lidar3d"
        directory.mkdir(exist_ok=True)
        store.update(identity, status="running", stage="lidar3d_inference")
        with httpx.Client(timeout=1900) as detector:
            health = detector.get(settings.lidar3d_service_url + "/health")
            health.raise_for_status()
            if (
                health.json().get("checkpoint_sha256") != CHECKPOINT_SHA
                or health.json().get("dataset_manifest_sha256") != settings.lidar3d_manifest_sha256
            ):
                raise ValueError("3D detector or dataset pin differs")
            response = detector.post(
                settings.lidar3d_service_url + "/infer",
                json={
                    "job_id": identity,
                    "scene": job["scene"],
                    "max_frames": len(frames),
                },
            )
            if response.status_code != 200:
                raise ValueError("3D detector failed; see private lidar3d/inference.log")
            receipt = response.json()
        prediction = directory / f"{job['scene']}.json"
        if receipt.get("frames") != len(frames) or receipt.get("sha256") != file_sha256(prediction):
            raise ValueError("3D detector output receipt differs")
        output = CenterPointOutput.model_validate_json(prediction.read_text())
        if output.scene != job["scene"] or output.dataset != manifest["version"]:
            raise ValueError("3D detector returned another scene")
        if output.model.checkpoint != f"centerpoint_pillar02_nuscenes_191a3822.pth@sha256:{CHECKPOINT_SHA}":
            raise ValueError("3D detector checkpoint identity differs")
        return self.publish(job, credentials, settings, manifest, frames, output)

    def run_dataset(self, job: dict, credentials: dict) -> dict:
        settings, store, identity = self.settings, self.store, job["job_id"]
        directory = store.directory(identity) / "private/lidar3d"
        directory.mkdir(exist_ok=True)
        manifest_path = directory / "dataset.json"
        provisioned = job.get("input_kind") != "dataset_upload"
        store.update(identity, status="running", stage="lidar3d_validate")
        if manifest_path.is_file():
            if store.get(identity).get("dataset_manifest_sha256") != file_sha256(manifest_path):
                raise ValueError("3D dataset snapshot identity changed")
            manifest = json.loads(manifest_path.read_bytes())
            base = (
                settings.lidar3d_dataset_root
                if provisioned
                else safe_path(directory / "input", manifest["root"])
                if manifest["root"] != "."
                else directory / "input"
            )
        elif provisioned:
            pinned = dataset(settings)
            base, manifest = nuscenes(settings.lidar3d_dataset_root, job["scene"], job["total_frames"])
            manifest |= {
                "schema": "autolabel-lidar3d-dataset/v1",
                "format": "nuscenes",
                "root": ".",
                "files": pinned["files"],
            }
        else:
            source = store.directory(identity) / "private/dataset.archive"
            if file_sha256(source) != job["source_sha256"]:
                raise ValueError("Uploaded dataset identity changed")
            if (directory / "input").exists():
                shutil.rmtree(directory / "input")
            manifest = ingest(
                source,
                directory,
                dataset_format=job["dataset_format"],
                scene=job["scene"],
                max_frames=job["max_frames"],
                max_bytes=settings.lidar3d_max_expanded_bytes,
            )
            base = directory / "input" / manifest["root"]
        digest = atomic_json(manifest_path, manifest)
        job = store.update(
            identity,
            scene=manifest["scene"],
            classes=manifest["classes"],
            dataset_format=manifest["format"],
            total_frames=len(manifest["frames"]),
            dataset_manifest_sha256=digest,
        )
        # Recheck source bytes on every retry before importing or invoking the GPU.
        for name, expected in manifest["files"].items():
            if file_sha256(safe_path(base, name)) != expected:
                raise ValueError("Native dataset file checksum changed")
        if job.get("mode") == "import":
            output = imported_annotations(base, manifest)
            atomic_json(directory / "imported-predictions.json", output.model_dump(mode="json"))
        else:
            if not settings.bevfusion_enabled or job["profile"]["checkpoint_sha256"] != BEVFUSION_CHECKPOINT_SHA:
                raise ValueError("BEVFusion is not configured for this job")
            if manifest["format"] != "nuscenes" or any(
                set(frame["cameras"]) != set(CAMERAS) for frame in manifest["frames"]
            ):
                raise ValueError("BEVFusion requires native nuScenes LiDAR and all six calibrated cameras")
            store.update(identity, stage="bevfusion_inference")
            with httpx.Client(timeout=1900) as client:
                response = client.post(
                    settings.lidar3d_service_url + "/infer-upload",
                    json={"job_id": identity, "manifest_sha256": digest, "provisioned": provisioned},
                )
                if response.status_code != 200:
                    raise ValueError("BEVFusion failed; inspect the private 3D inference log")
                receipt = response.json()
            path = directory / "uploaded-predictions.json"
            if (
                receipt.get("sha256") != file_sha256(path)
                or receipt.get("manifest_sha256") != digest
                or receipt.get("checkpoint_sha256") != BEVFUSION_CHECKPOINT_SHA
            ):
                raise ValueError("BEVFusion output receipt differs")
            output = UploadOutput.model_validate_json(path.read_bytes())
            if output.model_name != "BEVFusion" or output.checkpoint_sha256 != BEVFUSION_CHECKPOINT_SHA:
                raise ValueError("3D output is not from the pinned BEVFusion model")
        if (
            output.dataset_format != manifest["format"]
            or output.scene != manifest["scene"]
            or output.frames != [f["sample_token"] for f in manifest["frames"]]
        ):
            raise ValueError("3D output frame mapping differs from the native dataset")
        store.update(identity, stage="lidar3d_tracking")
        output = track_predictions(output, manifest)
        atomic_json(directory / "tracked-predictions.json", output.model_dump(mode="json"))
        if settings.hybrid_enabled:
            from src.services.vm.hybrid_contracts import HybridProfile, hybrid_preflight

            mapping = COCO_MAPPING
            camera_settings = settings.model_copy(update={"hybrid_class_mapping": json.dumps(mapping)})
            profile = hybrid_preflight(camera_settings, list(mapping))
            expected_names = {"pedestrian": "person", **{name: name for name in mapping if name != "pedestrian"}}
            if any(profile.detector_names[index] != expected_names[name] for name, index in mapping.items()):
                raise ValueError("Camera class mapping differs from COCO")
            camera_profile = profile.model_dump()
            camera_profile["tracker_config"] = {
                **profile.tracker_config,
                "track_buffer": 2 if manifest["format"] == "nuscenes" else 10,
            }
            camera_profile["tracker_sha256"] = hashlib.sha256(encode(camera_profile["tracker_config"])).hexdigest()
            atomic_json(directory / "camera-profile.json", HybridProfile.model_validate(camera_profile).model_dump())
            store.update(identity, stage="lidar3d_camera_tracking")
            with (directory / "camera-inference.log").open("w") as log:
                try:
                    subprocess.run(
                        [
                            str(settings.local_sam_python),
                            "-m",
                            "scripts.run_vm_3d_camera",
                            "--directory",
                            str(directory),
                            "--base",
                            str(base),
                        ],
                        cwd=PROJECT_ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=1900,
                    )
                except (subprocess.SubprocessError, OSError):
                    raise ValueError("3D camera association failed; inspect private camera-inference.log") from None
            linked = UploadOutput.model_validate_json((directory / "linked-predictions.json").read_bytes())
            if len(linked.predictions) != len(output.predictions) or any(
                left.model_dump(exclude={"linked_2d_tracks"}) != right.model_dump(exclude={"linked_2d_tracks"})
                for left, right in zip(linked.predictions, output.predictions, strict=True)
            ):
                raise ValueError("Camera association changed 3D geometry or identity")
            output = linked
        media_settings = settings.model_copy(update={"lidar3d_dataset_root": base})
        return self.publish(job, credentials, media_settings, manifest, manifest["frames"], output, output.model_name)

    def publish(
        self,
        job: dict,
        credentials: dict,
        settings: Settings,
        manifest: dict,
        frames: list[dict],
        output: CenterPointOutput | UploadOutput,
        detector_name: str = "CenterPoint",
    ) -> dict:
        from src.services.vm.worker import UserTaskClient

        store, identity = self.store, job["job_id"]
        directory = store.directory(identity) / "private/lidar3d"
        if store.get(identity)["status"] == "cancelled":
            raise ValueError("3D job cancelled")
        store.update(identity, stage="lidar3d_media", processed_frames=len(frames), complete_frames=len(frames))
        archive = directory / "pointcloud.zip"
        prepare_media(settings, manifest, frames, archive)
        attributes = [
            {
                "name": "review_status",
                "mutable": True,
                "input_type": "select",
                "default_value": "pending",
                "values": ["pending", "accepted", "rejected"],
            },
            {"name": "confidence", "mutable": True, "input_type": "text", "default_value": "", "values": []},
            {"name": "detector", "mutable": True, "input_type": "text", "default_value": detector_name, "values": []},
            {
                "name": "autolabel_3d_track_id",
                "mutable": False,
                "input_type": "text",
                "default_value": "",
                "values": [],
            },
            {"name": "linked_2d_tracks", "mutable": True, "input_type": "text", "default_value": "{}", "values": []},
        ]
        with httpx.Client(
            base_url=settings.cvat_url, headers=UserTaskClient(settings, credentials)._headers(), timeout=120
        ) as client:

            def call(method: str, path: str, **kwargs: Any) -> dict:
                result = client.request(method, path, **kwargs)
                if result.is_error:
                    raise ValueError(f"3D CVAT operation failed (HTTP {result.status_code})")
                return result.json() if result.content else {}

            current = store.get(identity)
            if not current.get("cvat_task_id"):
                task = call(
                    "POST",
                    "/api/tasks",
                    json={
                        "name": f"AutoLabel 3D {job['scene']} ({identity[:8]})",
                        "labels": [
                            {"name": name, "type": "cuboid", "attributes": attributes} for name in job["classes"]
                        ],
                    },
                )
                current = store.update(identity, cvat_task_id=task["id"], stage="lidar3d_upload")
                with archive.open("rb") as stream:
                    call(
                        "POST",
                        f"/api/tasks/{task['id']}/data",
                        data={"image_quality": "90", "use_cache": "true", "sorting_method": "lexicographical"},
                        files={"client_files[0]": ("pointcloud.zip", stream, "application/zip")},
                    )
            task_id = current["cvat_task_id"]
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline:
                task = call("GET", f"/api/tasks/{task_id}")
                jobs = call("GET", f"/api/jobs?task_id={task_id}")["results"]
                if task.get("size") == len(frames) and jobs:
                    break
                if store.get(identity)["status"] == "cancelled":
                    raise ValueError("3D job cancelled")
                time.sleep(2)
            else:
                raise ValueError("3D CVAT import did not become ready; inspect the existing task")
            if task.get("dimension") != "3d" or len(jobs) != 1:
                raise ValueError("CVAT 3D frame mapping differs")
            cvat_job = jobs[0]["id"]
            store.update(identity, cvat_job_id=cvat_job, stage="lidar3d_publish", delivered_frames=len(frames))
            labels = call("GET", f"/api/labels?task_id={task_id}&page_size=100")["results"]
            shapes = cuboid_shapes(output, frames, labels, detector_name)
            payload = (
                native_tracks(shapes, output, labels, len(frames))
                if isinstance(output, UploadOutput)
                else {"shapes": shapes, "tracks": []}
            )
            existing = call("GET", f"/api/jobs/{cvat_job}/annotations")
            if existing.get("shapes") or existing.get("tracks"):
                raise ValueError("3D task already contains annotations; keep human work and inspect it before retry")
            call("PATCH", f"/api/jobs/{cvat_job}/annotations?action=create", json=payload)
            saved = call("GET", f"/api/jobs/{cvat_job}/annotations")
            expected = sorted((shape["frame"], shape["label_id"], shape["points"]) for shape in shapes)
            actual = sorted(
                (shape["frame"], shape["label_id"], shape["points"])
                for shape in iter_cuboids(saved, len(frames))
                if not shape.get("outside")
            )
            if actual != expected:
                raise ValueError("3D cuboid publication did not round-trip")
        for frame in range(len(frames)):
            for label in job["classes"]:
                store.event(identity, f"result:{frame}:{label}", {"frame": frame, "class": label})
        atomic_json(
            directory / "publication.json",
            {
                "task": task_id,
                "job": cvat_job,
                "frames": len(frames),
                "cuboids": len(shapes),
                "tracks": len(payload["tracks"]),
                "camera_association": (directory / "camera-links.json").is_file(),
                "checkpoint_sha256": job["profile"]["checkpoint_sha256"],
                "source": "native_annotation_import" if detector_name == "Imported" else "real_lidar_detector",
                "model": detector_name,
            },
        )
        return store.update(
            identity,
            status="completed",
            stage="awaiting_human_review",
            published_frames=len(frames),
            cuboid_count=len(shapes),
            class_coverage={name: len(frames) for name in job["classes"]},
        )
