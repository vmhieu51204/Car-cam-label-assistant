"""Verify a SAM3 Kaggle payload/result pair, stitch IDs, and export CVAT drafts.

No ground truth, GPU, Kaggle API, or CVAT API is required for this stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from dataclasses import asdict
from pathlib import Path

from scripts.export_sam3_result import export_stored_result
from scripts.stitch_sam3_benchmark import _window_tracks
from src.services.video.stitching import stitch_tracks


def stitch_sequence(
    *,
    result_path: Path,
    payload_path: Path,
    output_dir: Path,
    config_id: str = "sequence-text",
    effective_fps: float | None = None,
    fps_source: str | None = None,
) -> dict:
    """Reject incomplete/mismatched GPU output before making review artifacts."""
    result = json.loads(result_path.read_text(encoding="utf-8"))
    item = next((row for row in result.get("configurations", []) if row.get("id") == config_id), None)
    if item is None or item.get("status") != "completed" or not item.get("window_predictions"):
        raise ValueError("a completed two-sided SAM3 window result is required")
    with zipfile.ZipFile(payload_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(
            name.startswith("/") or "\\" in name or ".." in Path(name).parts for name in names
        ):
            raise ValueError("unsafe or duplicate payload member")
        frames_data = json.loads(archive.read("frames.json"))
        config = json.loads(archive.read("benchmark-config.json"))
        records = frames_data["frames"]
        expected = list(range(len(records)))
        if (
            not records
            or [row["source_frame_index"] for row in records] != expected
            or item.get("selected_source_frames") != expected
            or item.get("evaluated_source_frames") != expected
        ):
            raise ValueError("GPU result does not cover the whole packaged sequence")
        requested = next((row for row in config["configurations"] if row["id"] == config_id), None)
        if requested != item.get("config") or requested["mode"] != "text_prompt":
            raise ValueError("GPU configuration differs from the uploaded payload")
        frame_dir = output_dir / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        for record in records:
            name = record["relative_path"]
            index = record["source_frame_index"]
            if name != f"frames/frame_{index:06d}.jpg":
                raise ValueError("unexpected packaged frame path")
            data = archive.read(name)
            if hashlib.sha256(data).hexdigest() != record["sha256"]:
                raise ValueError("packaged frame checksum mismatch")
            (frame_dir / f"frame_{index:06d}.jpg").write_bytes(data)
    stitched = stitch_tracks(_window_tracks(item["window_predictions"]))
    artifact_path = output_dir / "stitched-result.json"
    artifact_path.write_text(
        json.dumps(
            {
                "schema": "autolabel-stitch-comparison/v1",
                "config_id": config_id,
                "comparison": {"evaluated_frames": len(records)},
                "decisions": [asdict(row) for row in stitched.decisions],
                "boundary_logs": stitched.boundary_logs,
                "tracks": [asdict(row) for row in stitched.tracks],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    source_fps = frames_data.get("target_fps", effective_fps)
    if not isinstance(source_fps, (int, float)) or source_fps <= 0:
        raise ValueError("packaged sequence lacks its effective frame rate")
    provenance = frames_data.get("fps_source", fps_source)
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError("frame-rate provenance is required")
    output = export_stored_result(
        stitch_result=artifact_path,
        image_dir=frame_dir,
        output_dir=output_dir,
        source_fps=source_fps,
        fps_source=provenance,
    )
    (output_dir / "source-index-map.json").write_text(
        json.dumps(
            {
                "source_fps": frames_data.get("source_fps"),
                "target_fps": source_fps,
                "frames": [
                    {
                        "selected_index": row["source_frame_index"],
                        "original_source_index": row.get("original_source_index"),
                        "timestamp_s": row.get("timestamp_s"),
                    }
                    for row in records
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "frames": len(records),
        "tracks": len(stitched.tracks),
        "coco": str(output.coco),
        "cvat_zip": str(output.cvat_zip),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config-id", default="sequence-text")
    parser.add_argument("--effective-fps", type=float, help="Required for legacy benchmark payloads without target_fps")
    parser.add_argument("--fps-source", help="Required for legacy benchmark payloads without fps_source")
    args = parser.parse_args()
    print(
        json.dumps(
            stitch_sequence(
                result_path=args.result,
                payload_path=args.payload,
                output_dir=args.output,
                config_id=args.config_id,
                effective_fps=args.effective_fps,
                fps_source=args.fps_source,
            )
        )
    )


if __name__ == "__main__":
    main()
