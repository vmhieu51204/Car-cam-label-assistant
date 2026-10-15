"""Strict manual GT adapter into the existing canonical evaluation contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.models.evaluation import DatasetReference, FrameKey, FrameReference, ReferenceBox, SequenceKey, SourceFile


def load_manual_reference(path: Path) -> tuple[DatasetReference, dict[str, str]]:
    """Frames declare annotation coverage, including negatives; no implied GT.

    Each frame includes video_id, frame, width and height. Boxes include video_id,
    frame, track_id (nullable), class, box (xyxy pixels), occlusion (optional).
    The media source is the GT metadata, not a claim that image bytes exist.
    """
    raw = path.read_bytes()
    document = json.loads(raw)
    if document.get("schema") != "autolabel-manual-gt/v1":
        raise ValueError("unsupported manual GT schema")
    source = SourceFile(relative_path=path.name, sha256=hashlib.sha256(raw).hexdigest(), role="metadata")
    frames = []
    for row in document["frames"]:
        if isinstance(row["frame"], bool) or not isinstance(row["frame"], int) or row["frame"] < 0:
            raise ValueError("frame must be a nonnegative source index")
        key = FrameKey(
            sequence=SequenceKey(dataset_id="coco", split="manual", sequence_uid=row["video_id"]),
            frame_uid=str(row["frame"]),
        )
        frames.append(
            FrameReference(
                key=key,
                source_ordinal=row["frame"],
                width=row["width"],
                height=row["height"],
                media=source,
                timestamp_provenance="unavailable",
            )
        )
    index = {(x.key.sequence.sequence_uid, x.source_ordinal): x for x in frames}
    boxes, occlusions = [], {}
    identities: set[tuple[FrameKey, str]] = set()
    classes: dict[tuple[SequenceKey, str], str] = {}
    for i, row in enumerate(document["annotations"]):
        frame = index.get((row["video_id"], row["frame"]))
        if frame is None:
            raise ValueError("annotation points to an undeclared frame")
        track = row.get("track_id")
        if track is not None:
            track = str(track)
            if (frame.key, track) in identities:
                raise ValueError("duplicate frame/track")
            identities.add((frame.key, track))
            track_key = (frame.key.sequence, track)
            if track_key in classes and classes[track_key] != row["class"]:
                raise ValueError("GT track changes class")
            classes[track_key] = row["class"]
        box = ReferenceBox(
            annotation_id=f"manual/{i}",
            native_annotation_id=f"manual/{i}",
            frame=frame.key,
            class_name=row["class"],
            bbox_xyxy=tuple(row["box"]),
            native_track_id=track,
            source_ordinal=i,
            reference_kind="human_reviewed_2d_box",
            native_category=row["class"],
        )
        x1, y1, x2, y2 = box.bbox_xyxy
        if not 0 <= x1 < x2 <= frame.width or not 0 <= y1 < y2 <= frame.height:
            raise ValueError("GT box outside source image")
        occlusion = row.get("occlusion", "unknown")
        if occlusion not in {"none", "partial", "heavy", "unknown"}:
            raise ValueError("invalid GT occlusion")
        boxes.append(box)
        occlusions[box.annotation_id] = occlusion
    frames.sort(key=lambda x: (x.key.sequence.sequence_uid, x.source_ordinal))
    return DatasetReference(
        reference_revision="manual-gt/v1", frames=tuple(frames), boxes=tuple(boxes), profiles=(), sources=(source,)
    ), occlusions
