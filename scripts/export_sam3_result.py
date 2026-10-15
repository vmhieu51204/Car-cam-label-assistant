"""Turn an existing, stitched SAM3 Kaggle result into CVAT review artifacts.

This is a read-only replay of stored model output. It does not call a GPU, Kaggle,
or CVAT, and it does not mark any track accepted for training publication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from src.services.video.export_2d import Export2DArtifacts, export_2d
from src.services.video.sequence_review import score_tracks
from src.services.video.stitching import ChunkObservation, UnifiedTrack


def export_stored_result(
    *, stitch_result: Path, image_dir: Path, output_dir: Path, source_fps: float, fps_source: str
) -> Export2DArtifacts:
    """Export a cached run only when its frames cover this entire image sequence."""
    if source_fps <= 0 or not fps_source.strip():
        raise ValueError("declare the effective image cadence and its source")
    artifact = json.loads(stitch_result.read_text(encoding="utf-8"))
    if artifact.get("schema") != "autolabel-stitch-comparison/v1":
        raise ValueError("expected a stitched SAM3 benchmark result")
    frames = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not frames or [path.stem for path in frames] != [f"frame_{index:06d}" for index in range(len(frames))]:
        raise ValueError("cached benchmark images must be contiguous frame_000000...")
    if artifact["comparison"]["evaluated_frames"] != len(frames):
        raise ValueError("cached GPU output does not cover the entire supplied sequence")
    with Image.open(frames[0]) as image:
        width, height = image.size
    tracks = []
    for item in artifact["tracks"]:
        observations = [
            ChunkObservation(row["frame"], tuple(row["bbox"]), score=row.get("score")) for row in item["observations"]
        ]
        if any(not 0 <= row.frame < len(frames) for row in observations):
            raise ValueError("cached track refers to an image outside the sequence")
        tracks.append(
            UnifiedTrack(
                item["track_id"],
                item["label"],
                observations,
                item.get("needs_review", False),
                item.get("review_reasons", []),
            )
        )
    reviews = score_tracks(tracks)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = export_2d(
        tracks,
        source_frames=list(range(len(frames))),
        image_names=[path.name for path in frames],
        width=width,
        height=height,
        output_dir=output_dir,
        review={item.track_id: item for item in reviews},
    )
    digest = hashlib.sha256(stitch_result.read_bytes()).hexdigest()
    (output_dir / "sequence-manifest.json").write_text(
        json.dumps(
            {
                "schema": "autolabel-cached-sequence-2d/v1",
                "model_result_sha256": digest,
                "source_fps": source_fps,
                "fps_source": fps_source,
                "source_frame_count": len(frames),
                "selected_source_frames": list(range(len(frames))),
                "review_status": "pending",
                "source": str(stitch_result),
                "annotation_provenance": "SAM3 on every supplied converted frame",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "review-queue.json").write_text(
        json.dumps(
            {
                "schema": "autolabel-sequence-review/v1",
                "calibrated": False,
                "tracks": [asdict(item) for item in reviews],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stitch-result", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-fps", type=float, required=True)
    parser.add_argument("--fps-source", required=True)
    args = parser.parse_args()
    result = export_stored_result(
        stitch_result=args.stitch_result,
        image_dir=args.images,
        output_dir=args.output,
        source_fps=args.source_fps,
        fps_source=args.fps_source,
    )
    print(json.dumps({"coco": str(result.coco), "cvat_zip": str(result.cvat_zip)}))


if __name__ == "__main__":
    main()
