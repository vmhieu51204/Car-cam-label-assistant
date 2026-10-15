"""Convert one BDD100K MOT or KITTI tracking sequence to private evaluation artifacts."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import shutil
import statistics
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from tools.public_gt.bdd100k import Frame
from tools.public_gt.bdd100k import locate_sequence as locate_bdd
from tools.public_gt.bdd100k import read_sequence as read_bdd
from tools.public_gt.kitti import locate_sequence as locate_kitti
from tools.public_gt.kitti import read_sequence as read_kitti

ROOT = Path(__file__).resolve().parents[1]
LICENSE_URL = "https://github.com/bdd100k/bdd100k/blob/master/doc/source/license.rst"
FORMAT_URL = "https://github.com/bdd100k/bdd100k/blob/master/doc/source/format.rst"
KITTI_LICENSE_URL = "https://www.cvlibs.net/datasets/kitti/"
KITTI_FORMAT_URL = "https://github.com/JonathonLuiten/TrackEval/blob/master/docs/KITTI-format.txt"


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _times(frames: list[Frame], fallback_fps: float) -> tuple[list[float], float, str]:
    stamps = [frame.timestamp_ms for frame in frames]
    if all(stamp is not None for stamp in stamps):
        values = [float(stamp) for stamp in stamps if stamp is not None]
        if all(right > left for left, right in zip(values, values[1:])):
            times = [(value - values[0]) / 1000 for value in values]
            intervals = [right - left for left, right in zip(times, times[1:])]
            return times, 1 / statistics.median(intervals), "annotation_timestamps"
    base = frames[0].source_index
    return [(frame.source_index - base) / fallback_fps for frame in frames], fallback_fps, "dataset_fps_fallback"


def _sample(frames: list[Frame], times: list[float], target_fps: float) -> list[int]:
    if not frames:
        return []
    chosen = {0}
    target_count = math.floor(times[-1] * target_fps + 1e-8) + 1
    for target_index in range(1, target_count):
        target_time = target_index / target_fps
        right = bisect.bisect_left(times, target_time)
        candidates = [index for index in (right - 1, right) if 0 <= index < len(times)]
        chosen.add(min(candidates, key=lambda index: (abs(times[index] - target_time), index)))
    return sorted(chosen)


def _write_xml(path: Path, records: list[dict], objects: list[dict], categories: list[str]) -> None:
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "size").text = str(len(records))
    labels = ET.SubElement(task, "labels")
    for name in categories:
        ET.SubElement(ET.SubElement(labels, "label"), "name").text = name
    tracks: dict[int, list[dict]] = defaultdict(list)
    for observation in objects:
        tracks[observation["track_id"]].append(observation)
    for track_id, observations in sorted(tracks.items()):
        observations.sort(key=lambda item: item["frame_index"])
        track = ET.SubElement(root, "track", id=str(track_id), label=observations[0]["label"], source="manual")
        previous = None
        for observation in observations:
            if previous is not None and observation["frame_index"] > previous["frame_index"] + 1:
                _xml_box(track, previous, previous["frame_index"] + 1, outside=True)
            _xml_box(track, observation, observation["frame_index"], outside=False)
            previous = observation
        if previous is not None and previous["frame_index"] + 1 < len(records):
            _xml_box(track, previous, previous["frame_index"] + 1, outside=True)
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _xml_box(track: ET.Element, observation: dict, frame: int, *, outside: bool) -> None:
    x1, y1, x2, y2 = observation["xyxy"]
    box = ET.SubElement(
        track,
        "box",
        frame=str(frame),
        xtl=str(x1),
        ytl=str(y1),
        xbr=str(x2),
        ybr=str(y2),
        outside=str(int(outside)),
        occluded=str(int(observation["occluded"])),
        keyframe="1",
        z_order="0",
    )
    ET.SubElement(box, "attribute", name="truncated").text = str(observation["truncated"]).lower()


def convert(
    *,
    source_dir: Path,
    sequence: str | None,
    output_root: Path,
    class_map_path: Path,
    target_fps: float = 5,
    source_fps: float | None = None,
    dataset: str = "bdd100k",
) -> Path:
    if dataset not in {"bdd100k", "kitti"}:
        raise ValueError("dataset must be bdd100k or kitti")
    if source_fps is None:
        source_fps = 10 if dataset == "kitti" else 5
    if not math.isfinite(target_fps) or target_fps <= 0 or not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("frame rates must be finite and positive")
    if dataset == "kitti":
        name, label_path, image_dir = locate_kitti(source_dir, sequence)
        frames, source_skipped = read_kitti(label_path, name, image_dir)
    else:
        name, label_path, image_dir = locate_bdd(source_dir, sequence)
        frames, source_skipped = read_bdd(label_path, name)
    class_map = json.loads(class_map_path.read_text(encoding="utf-8"))
    if not isinstance(class_map, dict) or any(
        value is not None and not isinstance(value, str) for value in class_map.values()
    ):
        raise ValueError("class map must map source names to target names or null")
    times, measured_fps, fps_method = _times(frames, source_fps)
    indexes = _sample(frames, times, target_fps)
    if not indexes or not any(class_map.get(obj.category) is not None for obj in frames[indexes[0]].objects):
        raise ValueError("sequence needs mapped boxes including a frame-zero seed")
    output = output_root / name
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output already contains data: {output}")
    output.mkdir(parents=True, exist_ok=True)
    frame_dir = output / "frames"
    frame_dir.mkdir()
    records = []
    objects = []
    skipped = Counter(source_skipped)
    source_hashes = {
        "labels": {"relative_path": str(label_path.relative_to(source_dir)), "sha256": _digest(label_path)},
        "images": {},
    }
    converted_hashes: dict[str, str] = {}
    source_ids = sorted({obj.source_id for index in indexes for obj in frames[index].objects})
    track_ids = {source_id: index + 1 for index, source_id in enumerate(source_ids)}
    for local, index in enumerate(indexes):
        frame = frames[index]
        image_path = image_dir / frame.name
        if not image_path.is_file() or image_path.is_symlink():
            raise FileNotFoundError(f"missing or linked image: {image_path}")
        with Image.open(image_path) as image:
            expected_format = "PNG" if dataset == "kitti" else "JPEG"
            if image.format != expected_format:
                raise ValueError(f"{dataset} image is not {expected_format}: {image_path}")
            width, height = image.size
        destination = frame_dir / f"frame_{local:06d}.jpg"
        if dataset == "kitti":
            with Image.open(image_path) as image:
                image.convert("RGB").save(destination, format="JPEG", quality=95)
        else:
            shutil.copyfile(image_path, destination)
        source_hashes["images"][frame.name] = _digest(image_path)
        converted_hashes[destination.name] = _digest(destination)
        records.append(
            {
                "frame_index": local,
                "source_frame_index": frame.source_index,
                "timestamp_s": times[index],
                "width": width,
                "height": height,
                "file_name": destination.name,
                "source_file_name": frame.name,
            }
        )
        for obj in frame.objects:
            mapped = class_map.get(obj.category)
            if mapped is None:
                skipped[f"unmapped:{obj.category}"] += 1
                continue
            x1, y1, x2, y2 = obj.xyxy
            if dataset == "kitti":
                clipped = (
                    max(0.0, min(x1, width)),
                    max(0.0, min(y1, height)),
                    max(0.0, min(x2, width)),
                    max(0.0, min(y2, height)),
                )
                if clipped != obj.xyxy:
                    skipped["clipped_to_image"] += 1
                x1, y1, x2, y2 = clipped
                if x1 >= x2 or y1 >= y2:
                    skipped["outside_image"] += 1
                    continue
            elif not all(math.isfinite(value) for value in obj.xyxy) or not (
                0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
            ):
                raise ValueError(f"invalid {obj.category} box in source frame {frame.source_index}")
            objects.append(
                {
                    "frame_index": local,
                    "track_id": track_ids[obj.source_id],
                    "source_track_id": obj.source_id,
                    "label": mapped,
                    "xyxy": (x1, y1, x2, y2),
                    "occluded": obj.occluded,
                    "truncated": obj.truncated,
                }
            )
    category_names = sorted({obj["label"] for obj in objects})
    if not objects or not any(obj["frame_index"] == 0 for obj in objects):
        raise ValueError("sequence needs mapped boxes including a frame-zero seed")
    category_ids = {name: index + 1 for index, name in enumerate(category_names)}
    coco = {
        "images": [
            {
                "id": row["frame_index"],
                "frame_index": row["frame_index"],
                "file_name": row["file_name"],
                "width": row["width"],
                "height": row["height"],
            }
            for row in records
        ],
        "categories": [{"id": identifier, "name": name} for name, identifier in category_ids.items()],
        "annotations": [
            {
                "id": index + 1,
                "image_id": obj["frame_index"],
                "category_id": category_ids[obj["label"]],
                "track_id": obj["track_id"],
                "bbox": [
                    obj["xyxy"][0],
                    obj["xyxy"][1],
                    obj["xyxy"][2] - obj["xyxy"][0],
                    obj["xyxy"][3] - obj["xyxy"][1],
                ],
                "area": ((obj["xyxy"][2] - obj["xyxy"][0]) * (obj["xyxy"][3] - obj["xyxy"][1])),
                "iscrowd": 0,
                "occluded": obj["occluded"],
                "truncated": obj["truncated"],
            }
            for index, obj in enumerate(objects)
        ],
    }
    (output / "ground_truth.coco.json").write_text(json.dumps(coco, indent=2), encoding="utf-8")
    _write_xml(output / "ground_truth.xml", records, objects, category_names)
    manifest = {
        "schema": "public-gt-manifest/v1",
        "dataset": dataset,
        "sequence": name,
        "license_note": (
            "CC BY-NC-SA 3.0; non-commercial academic evaluation"
            if dataset == "kitti"
            else "Research/education/not-for-profit only unless separately licensed; verify intended use"
        ),
        "license_url": KITTI_LICENSE_URL if dataset == "kitti" else LICENSE_URL,
        "format_url": KITTI_FORMAT_URL if dataset == "kitti" else FORMAT_URL,
        "source_fps": measured_fps,
        "fps_method": fps_method,
        "target_fps": target_fps,
        "actual_output_fps": (
            (len(records) - 1) / (records[-1]["timestamp_s"] - records[0]["timestamp_s"])
            if len(records) > 1 and records[-1]["timestamp_s"] > records[0]["timestamp_s"]
            else None
        ),
        "source_files_sha256": source_hashes,
        "converted_files_sha256": converted_hashes,
        "class_map": class_map,
        "track_id_map": track_ids,
        "frame_count": len(records),
        "frames": records,
        "track_count": len({obj["track_id"] for obj in objects}),
        "box_count": len(objects),
        "classes": dict(Counter(obj["label"] for obj in objects)),
        "skipped": dict(skipped),
        "ignore_policy": (
            "drop DontCare and all classes outside Car/Pedestrian; clip visible boxes"
            if dataset == "kitti"
            else "drop crowd/ignore and labels without box2d"
        ),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("bdd100k", "kitti"), default="bdd100k")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--sequence")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--class-map", type=Path)
    parser.add_argument("--target-fps", type=float, default=5)
    parser.add_argument("--source-fps", type=float)
    args = parser.parse_args()
    print(
        convert(
            source_dir=args.source_dir,
            sequence=args.sequence,
            output_root=args.output_root or ROOT / "eval/public" / args.dataset,
            class_map_path=args.class_map
            or ROOT / "eval/public" / ("kitti_class_map.json" if args.dataset == "kitti" else "class_map.json"),
            target_fps=args.target_fps,
            source_fps=args.source_fps,
            dataset=args.dataset,
        )
    )


if __name__ == "__main__":
    main()
