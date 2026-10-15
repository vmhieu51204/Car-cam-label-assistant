"""A single offline command spans all frames without a manual frame-zero seed."""

import json

from PIL import Image

from scripts.eval_autolabel import read_boxes
from scripts.run_video_2d import run_sequence
from src.services.video.stitching import ChunkObservation, ChunkTrack


class FakeTextModel:
    version = "fake-text@1"

    def __init__(self):
        self.loads = 0
        self.calls = []

    def load(self):
        self.loads += 1

    def run_chunk(self, *, index, frames, labels):
        assert labels == ("car",)
        self.calls.append(index)
        return [
            ChunkTrack(
                index,
                f"car:{index}",
                "car",
                [ChunkObservation(frame, (float(frame), 1, float(frame + 10), 11)) for frame in frames],
            )
        ]

    def close(self):
        pass


def test_one_call_labels_full_source_timeline_and_resumes(tmp_path):
    frames = tmp_path / "image_02" / "0000"
    frames.mkdir(parents=True)
    for index in range(7):
        Image.new("RGB", (30, 20), color=(index * 20, 0, 0)).save(frames / f"{index:06d}.png")
    backend = FakeTextModel()
    output = tmp_path / "output"
    names = [
        "confidence",
        "routing",
        "source",
        "review_status",
        "autolabel_track_id",
        "review_priority",
        "review_score",
    ]
    label_snapshot = tmp_path / "labels.json"
    label_snapshot.write_text(
        json.dumps(
            {
                "labels": [
                    {
                        "id": 1,
                        "name": "car",
                        "attributes": [
                            {"id": i + 10, "name": name, "mutable": name == "review_status"}
                            for i, name in enumerate(names)
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    annotations = tmp_path / "annotations.json"
    annotations.write_text(json.dumps({"version": 2, "tags": [], "shapes": [], "tracks": []}), encoding="utf-8")
    result = run_sequence(
        frames,
        output,
        source_fps=4,
        target_fps=2,
        labels=("car",),
        backend=backend,
        window_size=3,
        overlap=2,
        cvat_labels=label_snapshot,
        cvat_annotations=annotations,
    )
    assert result.source_frames == [0, 2, 4, 6]
    assert backend.loads == 1 and backend.calls == [0, 1]
    assert [box.frame for box in read_boxes(result.coco)] == list(range(7))
    assert result.cvat_xml.is_file()
    review = json.loads((output / "review-queue.json").read_text(encoding="utf-8"))
    assert review["tracks"][0]["priority"] in {"critical", "high", "normal", "low"}
    assert 'name="review_status">pending' in result.cvat_xml.read_text(encoding="utf-8")
    draft = json.loads((output / "cvat-draft.json").read_text(encoding="utf-8"))
    assert draft["version"] == 2
    assert [shape["frame"] for shape in draft["tracks"][0]["shapes"]] == [0, 2, 4, 6]
    run_sequence(frames, output, source_fps=4, target_fps=2, labels=("car",), backend=backend, window_size=3, overlap=2)
    assert backend.calls == [0, 1]
