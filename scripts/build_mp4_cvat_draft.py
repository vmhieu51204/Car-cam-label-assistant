"""Map sampled SAM3 tracks back to an existing full-frame CVAT video job."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from PIL import Image

from src.services.video.cvat_draft import build_sequence_cvat_draft
from src.services.video.sequence_review import score_tracks
from src.services.video.stitching import ChunkObservation, UnifiedTrack


def build_draft(
    *,
    video_sequence: Path,
    source_map: Path,
    stitched: Path,
    labels: Path,
    annotations: Path,
    cvat_meta: Path,
    output: Path,
) -> dict:
    video = json.loads(video_sequence.read_text(encoding="utf-8"))
    mapping = json.loads(source_map.read_text(encoding="utf-8"))
    payload = json.loads(stitched.read_text(encoding="utf-8"))
    source_frames = [row["original_source_index"] for row in mapping["frames"]]
    if (
        video.get("schema") != "autolabel-mp4-sequence/v1"
        or source_frames != sorted(set(source_frames))
        or not source_frames
        or source_frames[0] != 0
        or source_frames[-1] >= video["frame_count"]
        or abs(mapping["source_fps"] - video["source_fps"]) > 1e-4
    ):
        raise ValueError("sampled result does not match the full video timeline")
    images = Path(video["images"])
    if len(list(images.glob("*.jpg"))) != video["frame_count"]:
        raise ValueError("full video frames are incomplete")
    hashes = []
    for frame in source_frames:
        path = images / f"{frame:06d}.jpg"
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    with Image.open(images / "000000.jpg") as image:
        width, height = image.size
    meta = json.loads(cvat_meta.read_text(encoding="utf-8"))
    cvat_frames = meta.get("frames")
    if (
        meta.get("size") != video["frame_count"]
        or meta.get("frame_filter") not in (None, "")
        or not isinstance(cvat_frames, list)
        or len(cvat_frames) != video["frame_count"]
        or any(frame.get("width") != width or frame.get("height") != height for frame in cvat_frames)
    ):
        raise ValueError("CVAT job media does not match the full video frame count and size")
    tracks = []
    for row in payload["tracks"]:
        tracks.append(
            UnifiedTrack(
                track_id=row["track_id"],
                label=row["label"],
                observations=[
                    ChunkObservation(frame=item["frame"], bbox=tuple(item["bbox"]), score=item.get("score"))
                    for item in row["observations"]
                ],
                needs_review=row.get("needs_review", False),
                review_reasons=row.get("review_reasons", []),
            )
        )
    label_data = json.loads(labels.read_text(encoding="utf-8"))
    draft = build_sequence_cvat_draft(
        job_id=str(uuid4()),
        source_sha256=video["source_sha256"],
        source_name=Path(video["source_video"]).name,
        source_frames=source_frames,
        frame_sha256=hashes,
        width=width,
        height=height,
        source_fps=video["source_fps"],
        labels=label_data.get("results", label_data) if isinstance(label_data, dict) else label_data,
        text_prompts=tuple(sorted({track.label for track in tracks})),
        tracks=tracks,
        review=score_tracks(tracks),
        existing_annotations=json.loads(annotations.read_text(encoding="utf-8")),
        model_version="sam3-kaggle-sequence",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(draft, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "sampled_frames": len(source_frames),
        "source_frames": video["frame_count"],
        "new_tracks": len(tracks),
        "draft": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("video-sequence", "source-map", "stitched", "labels", "annotations", "cvat-meta", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build_draft(
                video_sequence=args.video_sequence,
                source_map=args.source_map,
                stitched=args.stitched,
                labels=args.labels,
                annotations=args.annotations,
                cvat_meta=args.cvat_meta,
                output=args.output,
            )
        )
    )


if __name__ == "__main__":
    main()
