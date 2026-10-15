"""Export draft 2D labels on the original frame timeline for review."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from zipfile import ZIP_DEFLATED, ZipFile

from src.services.video.sequence_review import TrackReview
from src.services.video.stitching import ChunkObservation, UnifiedTrack


@dataclass(frozen=True)
class Export2DArtifacts:
    coco: Path
    cvat_xml: Path
    cvat_zip: Path


@dataclass(frozen=True)
class Export2DConfig:
    """Uncalibrated filters; defaults retain every visible model observation."""

    min_side_px: float = 0.0
    min_area_px: float = 0.0
    min_track_observations: int = 1


def _kept(observation: ChunkObservation, config: Export2DConfig) -> bool:
    x1, y1, x2, y2 = observation.bbox
    width, height = x2 - x1, y2 - y1
    return (
        width > 0 and height > 0 and min(width, height) >= config.min_side_px and width * height >= config.min_area_px
    )


def _boxes(
    track: UnifiedTrack, source_frames: list[int], config: Export2DConfig
) -> list[tuple[int, tuple[float, float, float, float]]]:
    visible = [item for item in track.observations if _kept(item, config)]
    if len(visible) < config.min_track_observations:
        return []
    result = []
    for index, item in enumerate(visible):
        result.append((source_frames[item.frame], item.bbox))
        if index + 1 >= len(visible):
            continue
        next_item = visible[index + 1]
        if next_item.frame != item.frame + 1:
            continue  # A missed sampled frame is an unknown interval, not interpolated evidence.
        start, stop = source_frames[item.frame], source_frames[next_item.frame]
        for frame in range(start + 1, stop):
            fraction = (frame - start) / (stop - start)
            result.append(
                (
                    frame,
                    cast(
                        tuple[float, float, float, float],
                        tuple(a + (b - a) * fraction for a, b in zip(item.bbox, next_item.bbox, strict=True)),
                    ),
                )
            )
    return sorted(result)


def export_2d(
    tracks: list[UnifiedTrack],
    *,
    source_frames: list[int],
    image_names: list[str],
    width: int,
    height: int,
    output_dir: Path,
    config: Export2DConfig | None = None,
    review: dict[str, TrackReview] | None = None,
) -> Export2DArtifacts:
    """Write COCO boxes and CVAT XML 1.1 tracks, all pending human review.

    COCO contains linearly interpolated boxes on intermediate source frames.
    CVAT XML keeps model observations as editable keyframes and marks ``outside``
    after each known disappearance. Neither interpolation is a SAM3 prediction.
    """
    config = config or Export2DConfig()
    if (
        not source_frames
        or source_frames != sorted(set(source_frames))
        or source_frames[0] != 0
        or source_frames[-1] >= len(image_names)
        or width < 1
        or height < 1
        or config.min_side_px < 0
        or config.min_area_px < 0
        or config.min_track_observations < 1
    ):
        raise ValueError("invalid source timeline or export filter")
    labels = sorted({track.label for track in tracks})
    categories = {label: index for index, label in enumerate(labels, 1)}
    coco = {
        "images": [
            {"id": frame + 1, "frame_index": frame, "file_name": name, "width": width, "height": height}
            for frame, name in enumerate(image_names)
        ],
        "categories": [{"id": identifier, "name": label} for label, identifier in categories.items()],
        "annotations": [],
    }
    xml_root = ET.Element("annotations")
    ET.SubElement(xml_root, "version").text = "1.1"
    meta = ET.SubElement(xml_root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "id").text = "0"
    ET.SubElement(task, "size").text = str(len(image_names))
    ET.SubElement(task, "mode").text = "interpolation"
    original_size = ET.SubElement(task, "original_size")
    ET.SubElement(original_size, "width").text = str(width)
    ET.SubElement(original_size, "height").text = str(height)
    xml_labels = ET.SubElement(task, "labels")
    attribute_types = {
        "autolabel_track_id": "text",
        "source": "text",
        "routing": "text",
        "review_status": "text",
        "needs_review": "checkbox",
        "review_priority": "text",
        "review_score": "number",
    }
    for label in labels:
        xml_label = ET.SubElement(xml_labels, "label")
        ET.SubElement(xml_label, "name").text = label
        attributes = ET.SubElement(xml_label, "attributes")
        for name, kind in attribute_types.items():
            attribute = ET.SubElement(attributes, "attribute")
            ET.SubElement(attribute, "name").text = name
            ET.SubElement(attribute, "input_type").text = kind
    annotation_id = 1
    for track_id, track in enumerate(tracks):
        if not track.observations:
            continue
        observations = sorted(track.observations, key=lambda item: item.frame)
        if len({item.frame for item in observations}) != len(observations) or any(
            not 0 <= item.frame < len(source_frames) for item in observations
        ):
            raise ValueError("track observations must have unique selected-frame indices")
        expanded = _boxes(track, source_frames, config)
        if not expanded:
            continue
        for frame, box in expanded:
            x1, y1, x2, y2 = box
            coco["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": frame + 1,
                    "category_id": categories[track.label],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0,
                    "track_id": track.track_id,
                    "provenance": "sam3_keyframe" if frame in source_frames else "linear_interpolation",
                }
            )
            annotation_id += 1
        element = ET.SubElement(xml_root, "track", {"id": str(track_id), "label": track.label, "source": "auto"})
        track_attributes = {
            "autolabel_track_id": track.track_id,
            "source": "auto",
            "routing": "review",
            "review_status": "pending",
            "needs_review": "true" if track.needs_review else "false",
        }
        if review is not None and track.track_id in review:
            unit = review[track.track_id]
            track_attributes.update(review_priority=unit.priority, review_score=f"{unit.score:.6f}")
        visible = [item for item in observations if _kept(item, config)]
        for index, item in enumerate(visible):
            attrs = {"frame": str(source_frames[item.frame]), "outside": "0", "occluded": "0", "keyframe": "1"}
            attrs.update(dict(zip(("xtl", "ytl", "xbr", "ybr"), (str(value) for value in item.bbox), strict=True)))
            box_element = ET.SubElement(element, "box", attrs)
            for name, value in track_attributes.items():
                ET.SubElement(box_element, "attribute", {"name": name}).text = value
            next_local = visible[index + 1].frame if index + 1 < len(visible) else None
            if (next_local is None and item.frame + 1 < len(source_frames)) or (
                next_local is not None and next_local > item.frame + 1
            ):
                outside_element = ET.SubElement(
                    element, "box", {**attrs, "frame": str(source_frames[item.frame + 1]), "outside": "1"}
                )
                for name, value in track_attributes.items():
                    ET.SubElement(outside_element, "attribute", {"name": name}).text = value
    output_dir.mkdir(parents=True, exist_ok=True)
    coco_path, xml_path = output_dir / "predictions.coco.json", output_dir / "predictions.cvat.xml"
    zip_path = output_dir / "predictions.cvat.zip"
    coco_path.write_text(json.dumps(coco, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ET.ElementTree(xml_root).write(xml_path, encoding="utf-8", xml_declaration=True)
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.write(xml_path, "annotations.xml")
    return Export2DArtifacts(coco=coco_path, cvat_xml=xml_path, cvat_zip=zip_path)
