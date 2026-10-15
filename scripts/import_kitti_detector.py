"""Link validated detector cuboids to SAM3 IDs; produce pending 3D drafts."""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

from scripts.lift_sequence_3d import _source_map
from scripts.run_video_2d import _source_hash
from src.services.detection.kitti_lidar import associate, validate_output
from src.services.geometry.kitti_tracking import KittiSequence


def detector_draft(sequence: KittiSequence, detector: Path, tracks: Path, source_map: Path,
                   output: Path, *, min_iou: float = .3, ambiguity_margin: float = .05) -> dict:
    start = time.perf_counter()
    predictions = validate_output(detector, sequence)
    mapping = _source_map(source_map, source_fps=sequence.source_fps)
    if list(mapping.values()) != list(range(len(sequence.frames))):
        raise ValueError("detector sequence needs full-frame 2D mapping; no sampled artifact reuse")
    payload = json.loads(tracks.read_text(encoding="utf-8"))
    if payload.get("schema") != "autolabel-stitched-tracks/v1":
        raise ValueError("expected stitched SAM3 tracks")
    source = json.loads(source_map.read_text(encoding="utf-8"))
    expected = _source_hash([f.image for f in sequence.frames], source_fps=sequence.source_fps,
                            target_fps=sequence.source_fps)
    if source.get("source_sha256") != expected or payload.get("source_sha256") != expected:
        raise ValueError("SAM3 image provenance does not match this full-frame sequence")
    by_frame: dict[int, list[dict]] = defaultdict(list)
    seen_ids = set()
    for track in payload["tracks"]:
        if not track.get("track_id") or track["track_id"] in seen_ids:
            raise ValueError("duplicate or empty SAM3 track ID")
        seen_ids.add(track["track_id"])
        seen_frames = set()
        for observation in track["observations"]:
            if observation["frame"] not in mapping or observation["frame"] in seen_frames:
                raise ValueError("duplicate or unmapped SAM3 observation")
            seen_frames.add(observation["frame"])
            if not observation.get("outside", False):
                by_frame[mapping[observation["frame"]]].append(
                    {"track_id": track["track_id"], "label": track["label"], "bbox": observation["bbox"]})
    linked: dict[str, dict] = {}
    unmatched_2d: list[dict] = []
    for frame in predictions.frames:
        index = frame.source_frame_index
        observations = by_frame[index]
        links = associate(frame.boxes, observations, sequence, min_iou=min_iou, ambiguity_margin=ambiguity_margin)
        used = set()
        for ordinal, (box, link) in enumerate(zip(frame.boxes, links, strict=True)):
            identity = link["track_id_2d"]
            if identity is not None:
                used.add(identity)
            # Unmatched cuboids are deliberately NOT assigned an invented temporal identity.
            identity = identity or f"lidar-unlinked:{index}:{ordinal}"
            target = linked.setdefault(identity, {"track_id": identity, "label": box.label,
                                                  "needs_review": True, "review_reasons": [], "observations": []})
            if target["label"] != box.label:
                raise ValueError("linked 2D identity changed class")
            reasons = link["review_reasons"]
            target["review_reasons"] = sorted(set(target["review_reasons"] + reasons))
            target["observations"].append({"source_frame_index": index, "selected_frame_index": index,
                "timestamp_s": sequence.frames[index].timestamp_s, "bbox_2d_xyxy": link["projected_bbox"],
                "box_lidar": box.model_dump(exclude={"label", "score"}), "detector_score": box.score,
                "source_2d_score": None, "needs_review": True, "review_reasons": reasons,
                "track_id_2d": link["track_id_2d"], "association_iou": link["iou"]})
        unmatched_2d.extend({"source_frame_index": index, "track_id": o["track_id"], "reason": "no_lidar_match"}
                            for o in observations if o["track_id"] not in used)
    for track in linked.values():
        observations = track["observations"]
        outside = []
        for current, following in zip(observations, [*observations[1:], None], strict=True):
            stop = current["source_frame_index"] + 1
            if stop < len(sequence.frames) and (following is None or following["source_frame_index"] > stop):
                outside.append({"source_frame_index": stop, "selected_frame_index": stop,
                                "timestamp_s": sequence.frames[stop].timestamp_s, "box_lidar": None,
                                "bbox_2d_xyxy": None, "needs_review": True,
                                "review_reasons": ["no_detector_observation"]})
        track["observations"] = sorted(observations + outside, key=lambda o: o["source_frame_index"])
    draft = {"schema": "autolabel-lidar-3d-draft/v1", "sequence_id": sequence.sequence_id,
             "coordinate_frame": predictions.coordinate_frame, "source_fps": sequence.source_fps,
             "fps_source": sequence.fps_source, "review_status": "pending", "calibrated": False,
             "geometry_source": "pretrained_lidar_detector", "detector": predictions.model_dump(exclude={"frames"}, by_alias=True),
             "association_config": {"min_iou": min_iou, "ambiguity_margin": ambiguity_margin, "calibrated": False},
             "tracks": list(linked.values()), "unmatched_2d": unmatched_2d,
             "counts": {"tracks": len(linked), "cuboids": sum(len(f.boxes) for f in predictions.frames),
                        "source_frames_read": len(predictions.frames)},
             "limitations": ["Unlinked detector cuboids have frame-local IDs; independent 3D tracking is not verified."],
             "timing_s": {"total": time.perf_counter() - start}}
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".tmp")
    temp.write_text(json.dumps(draft, indent=2), encoding="utf-8")
    temp.replace(output)
    return draft
