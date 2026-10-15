from __future__ import annotations

import json

import pytest
from PIL import Image

from scripts.eval_autolabel import read_boxes
from scripts.export_sam3_result import export_stored_result


def test_cached_gpu_result_exports_pending_editable_track(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    for index in range(3):
        Image.new("RGB", (40, 30), "gray").save(frames / f"frame_{index:06d}.jpg")
    stored = tmp_path / "stitch.json"
    stored.write_text(
        json.dumps(
            {
                "schema": "autolabel-stitch-comparison/v1",
                "comparison": {"evaluated_frames": 3},
                "tracks": [
                    {
                        "track_id": "auto:1",
                        "label": "car",
                        "needs_review": True,
                        "review_reasons": ["ambiguous_overlap"],
                        "observations": [
                            {"frame": index, "bbox": [index, 1, index + 10, 11], "score": 0.8} for index in range(3)
                        ],
                    }
                ],
            }
        )
    )
    result = export_stored_result(
        stitch_result=stored,
        image_dir=frames,
        output_dir=tmp_path / "out",
        source_fps=5,
        fps_source="converted-manifest",
    )
    assert [box.frame for box in read_boxes(result.coco)] == [0, 1, 2]
    assert result.cvat_zip.is_file()
    queue = json.loads((tmp_path / "out/review-queue.json").read_text())
    assert queue["tracks"][0]["needs_review"] is True
    assert json.loads((tmp_path / "out/sequence-manifest.json").read_text())["review_status"] == "pending"


def test_cached_gpu_result_rejects_partial_sequence(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    Image.new("RGB", (40, 30), "gray").save(frames / "frame_000000.jpg")
    stored = tmp_path / "stitch.json"
    stored.write_text(
        json.dumps({"schema": "autolabel-stitch-comparison/v1", "comparison": {"evaluated_frames": 0}, "tracks": []})
    )
    with pytest.raises(ValueError, match="entire"):
        export_stored_result(
            stitch_result=stored,
            image_dir=frames,
            output_dir=tmp_path / "out",
            source_fps=5,
            fps_source="converted-manifest",
        )
