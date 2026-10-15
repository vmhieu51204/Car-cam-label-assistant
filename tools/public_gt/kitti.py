"""Strict reader for KITTI's 2D object-tracking training sequences."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path

from src.models.evaluation import FrameKey, NativeKittiRow, SequenceKey, SourceFile
from tools.public_gt.bdd100k import Frame, Object


def discover_images(image_dir: Path) -> tuple[Path, ...]:
    """Require an exhaustive native timeline, including negative frames."""
    images = tuple(sorted(image_dir.glob("*.png")))
    if not images or any(not path.is_file() or path.is_symlink() for path in images):
        raise ValueError("KITTI sequence needs regular PNG images")
    if any(not re.fullmatch(r"\d{6}\.png", path.name) for path in images):
        raise ValueError("unexpected KITTI image name")
    if [int(path.stem) for path in images] != list(range(len(images))):
        raise ValueError("KITTI image indices must be contiguous from zero")
    return images


def read_native_rows(
    label_path: Path, *, sequence: SequenceKey, source: SourceFile, frame_count: int
) -> tuple[NativeKittiRow, ...]:
    """Read all 17 fields losslessly; ignore regions retain their placeholders."""
    rows: list[NativeKittiRow] = []
    seen: set[tuple[int, int]] = set()
    classes: dict[int, str] = {}
    for number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 17:
            raise ValueError(f"KITTI label line {number} must have 17 columns")
        try:
            if any(not re.fullmatch(r"-?\d+", fields[i]) for i in (0, 1, 3, 4)):
                raise ValueError("noninteger native field")
            frame, track, truncated, occluded = (int(fields[i]) for i in (0, 1, 3, 4))
            if not 0 <= frame < frame_count:
                raise ValueError(f"references missing frame {frame}")
            row = NativeKittiRow.model_validate(
                {
                    "frame": FrameKey(sequence=sequence, frame_uid=str(frame)),
                    "source_id": f"{sequence.split}/{sequence.sequence_uid}/line/{number}",
                    "source_file": source,
                    "line_number": number,
                    "source_frame_index": frame,
                    "track_id": track,
                    "native_class": fields[2],
                    "truncated": float(truncated),
                    "occluded": occluded,
                    "alpha": float(fields[5]),
                    "bbox_xyxy": tuple(float(x) for x in fields[6:10]),
                    "dimensions_hwl": tuple(float(x) for x in fields[10:13]),
                    "location_xyz": tuple(float(x) for x in fields[13:16]),
                    "rotation_y": float(fields[16]),
                }
            )
        except ValueError as exc:
            raise ValueError(f"KITTI label line {number} has invalid numeric/native fields: {exc}") from exc
        if row.native_class != "DontCare":
            if (frame, track) in seen:
                raise ValueError(f"KITTI frame {frame} repeats track ID {track}")
            if track in classes and classes[track] != row.native_class:
                raise ValueError(f"KITTI track {track} changes class")
            seen.add((frame, track))
            classes[track] = row.native_class
        rows.append(row)
    return tuple(rows)


def locate_sequence(source_dir: Path, sequence: str | None) -> tuple[str, Path, Path]:
    """Accept either the extracted archive root or its ``training`` directory."""
    training = source_dir / "training" if (source_dir / "training").is_dir() else source_dir
    labels = training / "label_02"
    images = training / "image_02"
    if not labels.is_dir() or not images.is_dir():
        raise FileNotFoundError("KITTI requires training/label_02 and training/image_02")
    if sequence is not None and not re.fullmatch(r"\d{4}", sequence):
        raise ValueError("KITTI sequence must be a four-digit training sequence")
    candidates = sorted(path.stem for path in labels.glob("[0-9][0-9][0-9][0-9].txt") if (images / path.stem).is_dir())
    if sequence is None:
        if len(candidates) != 1:
            raise ValueError("multiple KITTI sequences found; pass --sequence")
        sequence = candidates[0]
    if sequence not in candidates:
        raise FileNotFoundError(f"KITTI training sequence {sequence} lacks images or labels")
    return sequence, labels / f"{sequence}.txt", images / sequence


def read_sequence(label_path: Path, sequence: str, image_dir: Path) -> tuple[list[Frame], Counter[str]]:
    images = discover_images(image_dir)
    source = SourceFile(
        relative_path=label_path.name, sha256=hashlib.sha256(label_path.read_bytes()).hexdigest(), role="labels"
    )
    rows = read_native_rows(
        label_path,
        sequence=SequenceKey(dataset_id="kitti", split="training", sequence_uid=sequence),
        source=source,
        frame_count=len(images),
    )
    by_frame: list[list[Object]] = [[] for _ in images]
    skipped: Counter[str] = Counter()
    for row in rows:
        if row.native_class == "DontCare":
            skipped["dont_care"] += 1
        else:
            by_frame[row.source_frame_index].append(
                Object(str(row.track_id), row.native_class, row.bbox_xyxy, row.occluded > 0, row.truncated > 0)
            )
    return ([Frame(index, image.name, None, tuple(by_frame[index])) for index, image in enumerate(images)], skipped)
