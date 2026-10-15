"""Strict reader for one BDD100K MOT 2020 (box_track_20) sequence."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Object:
    source_id: str
    category: str
    xyxy: tuple[float, float, float, float]
    occluded: bool
    truncated: bool


@dataclass(frozen=True)
class Frame:
    source_index: int
    name: str
    timestamp_ms: float | None
    objects: tuple[Object, ...]


def locate_sequence(source_dir: Path, sequence: str | None) -> tuple[str, Path, Path]:
    """Find labels and matching images without guessing a different dataset schema."""
    labels = sorted(path for path in source_dir.glob("**/box_track_20/**/*.json") if path.is_file())
    if not labels:
        labels = sorted(
            path for path in source_dir.glob("**/*.json") if path.is_file() and path.parent.name in {"train", "val"}
        )
    if sequence:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", sequence):
            raise ValueError("sequence name contains unsafe characters")
        labels = [path for path in labels if path.stem == sequence]
    if not labels:
        raise FileNotFoundError("BDD100K box_track_20 sequence JSON not found under --source-dir")
    if len(labels) > 1 and sequence is None:
        raise ValueError("multiple BDD100K sequences found; pass --sequence")
    label_path = labels[0]
    name = label_path.stem
    candidates = sorted(
        path
        for path in source_dir.glob(f"**/{name}")
        if path.is_dir() and path.name == name and path != label_path.parent
    )
    candidates = [path for path in candidates if any(path.glob("*.jpg"))]
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected one image directory for sequence {name}; found {len(candidates)}")
    return name, label_path, candidates[0]


def read_sequence(label_path: Path, sequence: str) -> tuple[list[Frame], Counter[str]]:
    raw = json.loads(label_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("BDD100K MOT labels must be a nonempty list of frames")
    parsed: list[Frame] = []
    skipped: Counter[str] = Counter()
    for position, item in enumerate(raw):
        if not isinstance(item, dict) or not isinstance(item.get("labels"), list):
            raise ValueError(f"BDD100K frame {position} lacks labels list")
        name = item.get("name")
        if not isinstance(name, str) or Path(name).name != name or not name.lower().endswith(".jpg"):
            raise ValueError(f"BDD100K frame {position} has an unsafe or missing JPEG name")
        if item.get("videoName") not in (None, sequence):
            raise ValueError(f"BDD100K frame {position} belongs to another video")
        index = item.get("frameIndex", item.get("index"))
        if index is None:
            match = re.search(r"-(\d+)\.jpg$", name, flags=re.IGNORECASE)
            if not match:
                raise ValueError(f"BDD100K frame {position} has no usable source index")
            index = int(match.group(1))
        if type(index) is not int or index < 0:
            raise ValueError(f"BDD100K frame {position} has invalid source index")
        timestamp = item.get("timestamp")
        if timestamp is not None and (type(timestamp) not in (int, float) or timestamp < 0):
            raise ValueError(f"BDD100K frame {position} has invalid timestamp")
        objects = []
        ids = set()
        for label in item["labels"]:
            if not isinstance(label, dict):
                raise ValueError(f"BDD100K frame {index} has malformed label")
            category = label.get("category")
            if not isinstance(category, str):
                raise ValueError(f"BDD100K frame {index} has a label without category")
            attrs = label.get("attributes") or {}
            if not isinstance(attrs, dict):
                raise ValueError(f"BDD100K frame {index} has malformed attributes")
            if attrs.get("crowd", attrs.get("Crowd", False)) or attrs.get("ignore", False):
                skipped["crowd_or_ignore"] += 1
                continue
            box = label.get("box2d")
            if not isinstance(box, dict) or not all(key in box for key in ("x1", "y1", "x2", "y2")):
                skipped["no_box2d"] += 1
                continue
            source_id = label.get("id")
            if source_id is None or isinstance(source_id, (list, dict)):
                raise ValueError(f"BDD100K frame {index} has box without stable ID")
            source_id = str(source_id)
            if source_id in ids:
                raise ValueError(f"BDD100K frame {index} repeats track ID {source_id}")
            ids.add(source_id)
            try:
                coords = tuple(float(box[key]) for key in ("x1", "y1", "x2", "y2"))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"BDD100K frame {index} has nonnumeric box") from exc
            objects.append(
                Object(
                    source_id,
                    category,
                    (coords[0], coords[1], coords[2], coords[3]),
                    bool(attrs.get("occluded", attrs.get("Occluded", False))),
                    bool(attrs.get("truncated", attrs.get("Truncated", False))),
                )
            )
        parsed.append(Frame(index, name, float(timestamp) if timestamp is not None else None, tuple(objects)))
    parsed.sort(key=lambda frame: frame.source_index)
    if len({frame.source_index for frame in parsed}) != len(parsed):
        raise ValueError("BDD100K sequence repeats a source frame index")
    return parsed, skipped
