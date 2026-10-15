"""A CVAT video retains intermediate frames while SAM3 only supplies keyframes."""

import hashlib
import json

import pytest
from PIL import Image

from scripts.build_mp4_cvat_draft import build_draft


def test_sampled_track_uses_original_cvat_frame_numbers(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    for index in range(5):
        Image.new("RGB", (40, 30), (index, 0, 0)).save(images / f"{index:06d}.jpg")
    files = {}
    data = {
        "video": {
            "schema": "autolabel-mp4-sequence/v1",
            "source_video": str(tmp_path / "clip.mp4"),
            "source_sha256": hashlib.sha256(b"clip").hexdigest(),
            "source_fps": 10,
            "frame_count": 5,
            "images": str(images),
        },
        "map": {
            "source_fps": 10,
            "frames": [
                {"selected_index": 0, "original_source_index": 0},
                {"selected_index": 1, "original_source_index": 2},
                {"selected_index": 2, "original_source_index": 4},
            ],
        },
        "stitched": {
            "tracks": [
                {
                    "track_id": "auto:1",
                    "label": "car",
                    "observations": [
                        {"frame": 0, "bbox": [1, 2, 11, 12], "score": 0.9},
                        {"frame": 1, "bbox": [2, 2, 12, 12], "score": 0.9},
                    ],
                }
            ]
        },
        "labels": [
            {
                "id": 7,
                "name": "car",
                "attributes": [
                    {"id": n + 20, "name": name, "mutable": name == "review_status"}
                    for n, name in enumerate(
                        (
                            "confidence",
                            "routing",
                            "source",
                            "review_status",
                            "autolabel_track_id",
                            "review_priority",
                            "review_score",
                        )
                    )
                ],
            }
        ],
        "annotations": {"version": 3, "tags": [], "shapes": [], "tracks": []},
        "meta": {"size": 5, "frame_filter": "", "frames": [{"width": 40, "height": 30}] * 5},
    }
    for name, value in data.items():
        files[name] = tmp_path / f"{name}.json"
        files[name].write_text(json.dumps(value), encoding="utf-8")
    target = tmp_path / "draft.json"

    report = build_draft(
        video_sequence=files["video"],
        source_map=files["map"],
        stitched=files["stitched"],
        labels=files["labels"],
        annotations=files["annotations"],
        cvat_meta=files["meta"],
        output=target,
    )

    assert report["source_frames"] == 5
    assert report["sampled_frames"] == 3
    shapes = json.loads(target.read_text(encoding="utf-8"))["tracks"][0]["shapes"]
    assert [shape["frame"] for shape in shapes] == [0, 2, 4]
    assert [shape["outside"] for shape in shapes] == [False, False, True]

    files["meta"].write_text(json.dumps({"size": 4, "frames": [{"width": 40, "height": 30}] * 4}))
    with pytest.raises(ValueError, match="CVAT job media does not match"):
        build_draft(
            video_sequence=files["video"],
            source_map=files["map"],
            stitched=files["stitched"],
            labels=files["labels"],
            annotations=files["annotations"],
            cvat_meta=files["meta"],
            output=target,
        )
