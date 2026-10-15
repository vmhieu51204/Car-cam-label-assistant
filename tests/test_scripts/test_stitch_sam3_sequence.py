from __future__ import annotations

import json

import pytest
from PIL import Image

from scripts.package_sam3_sequence import package_sequence
from scripts.stitch_sam3_sequence import stitch_sequence


def test_no_gt_kaggle_result_stitches_and_exports_full_sequence(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    for frame in range(3):
        Image.new("RGB", (32, 24), "gray").save(images / f"{frame:06d}.png")
    payload = tmp_path / "payload.zip"
    package_sequence(images=images, output=payload, source_fps=5, target_fps=5, classes=("car",))
    config = {
        "id": "sequence-text",
        "selection": "dense",
        "gap": None,
        "window_size": 50,
        "overlap": 5,
        "mode": "text_prompt",
        "seed_type": "none",
        "reprompt": "none",
        "text_prompts": ["car"],
        "preserve_overlap": True,
    }
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "configurations": [
                    {
                        "id": "sequence-text",
                        "status": "completed",
                        "config": config,
                        "selected_source_frames": [0, 1, 2],
                        "evaluated_source_frames": [0, 1, 2],
                        "window_predictions": [
                            {
                                "window_id": 0,
                                "frame_index": frame,
                                "track_id": "local:1",
                                "label": "car",
                                "bbox": [frame, 1, frame + 10, 11],
                                "score": 0.8,
                            }
                            for frame in range(3)
                        ],
                    }
                ]
            }
        )
    )
    report = stitch_sequence(result_path=result, payload_path=payload, output_dir=tmp_path / "out")
    assert report["frames"] == 3 and report["tracks"] == 1
    assert (tmp_path / "out/predictions.cvat.zip").is_file()
    mapping = json.loads((tmp_path / "out/source-index-map.json").read_text())
    assert [row["original_source_index"] for row in mapping["frames"]] == [0, 1, 2]

    corrupted = json.loads(result.read_text())
    corrupted["configurations"][0]["status"] = "timed_out"
    result.write_text(json.dumps(corrupted))
    with pytest.raises(ValueError, match="completed"):
        stitch_sequence(result_path=result, payload_path=payload, output_dir=tmp_path / "other")
