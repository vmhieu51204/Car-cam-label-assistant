"""Offline CVAT REST payload keeps manual work and pending AI review."""

from src.services.video.cvat_draft import build_sequence_cvat_draft
from src.services.video.sequence_review import score_tracks
from src.services.video.stitching import ChunkObservation, UnifiedTrack


def test_draft_has_original_frame_keyframes_and_outside():
    track = UnifiedTrack(
        "auto:000001",
        "car",
        [
            ChunkObservation(0, (1, 2, 11, 12), score=0.8),
            ChunkObservation(1, (2, 2, 12, 12), score=0.8),
        ],
    )
    names = [
        "confidence",
        "routing",
        "source",
        "review_status",
        "autolabel_track_id",
        "review_priority",
        "review_score",
    ]
    labels = [
        {
            "id": 7,
            "name": "car",
            "attributes": [
                {"id": index + 20, "name": name, "mutable": name == "review_status"} for index, name in enumerate(names)
            ],
        }
    ]
    existing = {"version": 3, "tags": [], "shapes": [], "tracks": [{"id": 5, "source": "manual", "shapes": []}]}
    payload = build_sequence_cvat_draft(
        job_id="55555555-5555-4555-8555-555555555555",
        source_sha256="a" * 64,
        source_name="0008",
        source_frames=[0, 2, 4],
        frame_sha256=["b" * 64] * 3,
        width=40,
        height=30,
        source_fps=10,
        labels=labels,
        text_prompts=("car",),
        tracks=[track],
        review=score_tracks([track]),
        existing_annotations=existing,
        model_version="sam3@test",
    )
    assert payload["version"] == 3
    assert payload["tracks"][0] == existing["tracks"][0]
    generated = payload["tracks"][1]
    assert generated["source"] == "auto"
    assert [shape["frame"] for shape in generated["shapes"]] == [0, 2, 4]
    assert [shape["outside"] for shape in generated["shapes"]] == [False, False, True]
    assert all(shape["keyframe"] for shape in generated["shapes"])
