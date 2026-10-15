from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from pydantic import ValidationError

from src.services.video.rescue_matrix import (
    BoxRescueSuggestion,
    RescueFrameVerdict,
    RescueProposal,
    map_proposal_to_source,
    rescue_prompt,
    select_box_rescue_frames,
)


def _proposal(**overrides) -> dict:
    return {
        "label": "car",
        "bbox": [0.1, 0.2, 0.4, 0.6],
        "point": [0.2, 0.3],
        "occlusion": "partial",
        "confidence": "medium",
        "reason": "visible car",
        **overrides,
    }


def test_box_proposal_contract_is_strict_and_point_must_be_inside_box():
    result = RescueFrameVerdict(missing="yes", proposals=[_proposal()], reason="car visible")
    assert result.proposals[0].bbox == (0.1, 0.2, 0.4, 0.6)
    with pytest.raises(ValidationError):
        RescueFrameVerdict(missing="no", proposals=[_proposal()], reason="none")
    with pytest.raises(ValidationError):
        RescueFrameVerdict(missing="yes", proposals=[_proposal(extra=True)], reason="car")
    with pytest.raises(ValidationError, match="inside proposed box"):
        RescueProposal(**_proposal(point=[0.8, 0.9]))
    with pytest.raises(ValidationError):
        RescueFrameVerdict(missing="uncertain", proposals=[_proposal()], reason="unclear")
    with pytest.raises(ValidationError):
        RescueFrameVerdict(missing="yes", proposals=[_proposal()] * 21, reason="too many")


def test_box_rescue_frame_selection_caps_at_twenty_and_preserves_uniform_coverage():

    manifest = SimpleNamespace(
        frame_count=50, frames=[SimpleNamespace(scene_cut_before=index == 22) for index in range(50)]
    )
    tracks = SimpleNamespace(tracks=[])
    discovery = SimpleNamespace(events=[])
    selected = select_box_rescue_frames(manifest, tracks, discovery, max_frames=20)
    indices = [index for index, _ in selected]
    assert len(indices) == len(set(indices)) == 20
    assert indices == sorted(indices) and indices[0] == 0 and indices[-1] == 49
    assert 21 in indices or 22 in indices
    with pytest.raises(ValueError, match="limited to 20"):
        select_box_rescue_frames(manifest, tracks, discovery, max_frames=21)


def test_persisted_box_suggestion_requires_valid_normalized_coordinates():
    import hashlib

    proposal_id = hashlib.sha256(b"proposal").hexdigest()
    suggestion = BoxRescueSuggestion(
        proposal_id=proposal_id,
        frame_index=3,
        source_frame_index=9,
        label="car",
        bbox=(0.1, 0.2, 0.4, 0.6),
        point=(0.2, 0.3),
        occlusion="partial",
        confidence="medium",
        reason="visible car",
        image_sha256="a" * 64,
    )
    assert suggestion.source_frame_index == 9 and suggestion.status == "pending"
    with pytest.raises(ValidationError):
        BoxRescueSuggestion(
            **{**suggestion.model_dump(), "bbox": (0.2, 0.2, 1.2, 0.6)},
        )


def test_prompt_uses_exact_cvat_classes_and_mapping_converts_crop_to_original():
    prompt = rescue_prompt(
        classes=("car", "pedestrian"), existing_boxes=[], image_width=800, image_height=600, view="crop"
    )
    assert "car, pedestrian" in prompt
    assert "normalized to this submitted image view" in prompt
    proposal = RescueProposal(**_proposal())
    mapped = map_proposal_to_source(proposal, bounds=(0.35, 0.1, 1.0, 0.85))
    assert mapped.bbox == pytest.approx((0.415, 0.25, 0.61, 0.55))
    assert mapped.point == pytest.approx((0.48, 0.325))
    with pytest.raises(ValueError, match="requested class"):
        rescue_prompt(
            classes=("car",),
            existing_boxes=[{"label": "truck", "bbox": [1, 2, 3, 4]}],
            image_width=100,
            image_height=100,
            view="full",
        )


def _samples(root: Path) -> Path:
    (root / "images").mkdir(parents=True)
    rows, answers = [], []
    for sample_id, clip, category, frame in (("m1", "0013", "miss", 4), ("a1", "0013", "agree", 8)):
        raw = root / "images" / f"{sample_id}-raw.png"
        overlay = root / "images" / f"{sample_id}-overlay.png"
        Image.new("RGB", (100, 80), "white").save(raw)
        Image.new("RGB", (100, 80), "white").save(overlay)
        import hashlib

        rows.append(
            {
                "sample_id": sample_id,
                "clip": clip,
                "source_frame_index": frame,
                "processed_frame_index": frame,
                "raw": f"images/{sample_id}-raw.png",
                "overlay": f"images/{sample_id}-overlay.png",
                "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                "overlay_sha256": hashlib.sha256(overlay.read_bytes()).hexdigest(),
                "transform": {
                    "source_width": 100,
                    "source_height": 80,
                    "sent_width": 100,
                    "sent_height": 80,
                    "content_width": 100,
                    "content_height": 80,
                    "pad_left": 0,
                    "pad_top": 0,
                },
                "sam3_boxes": [],
            }
        )
        answers.append(
            {
                "sample_id": sample_id,
                "category": category,
                "missed_gt": ([{"label": "car", "bbox": [40, 30, 60, 50]}] if category == "miss" else []),
            }
        )
    (root / "samples.json").write_text(
        json.dumps({"schema": "point-rescue-samples/v1", "classes": ["car"], "samples": rows})
    )
    (root / "answers.json").write_text(json.dumps({"answers": answers}))
    return root
