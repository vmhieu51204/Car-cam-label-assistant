from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from src.models.video import CvatSamplingRequest
from src.services.video.sampling import MAX_MODEL_FRAMES, choose_keyframes, motion_image, motion_score


def test_optical_flow_sees_moving_object() -> None:
    def preview(offset: int) -> np.ndarray:
        pixels = np.zeros((100, 160), dtype=np.uint8)
        pixels[30:70, 30 + offset : 70 + offset] = 255
        buffer = BytesIO()
        Image.fromarray(pixels).save(buffer, format="JPEG")
        return motion_image(buffer.getvalue())

    first = preview(0)
    assert motion_score(first, first) == 0
    assert motion_score(first, preview(12)) > 0


def test_budget_uses_more_frames_in_moving_segment() -> None:
    scores = [0.0] + [0.001] * 50 + [0.08] * 50
    selected, threshold = choose_keyframes(scores, CvatSamplingRequest(strategy="budget", budget=20))
    assert threshold is None
    assert len(selected) == 20
    assert selected[0] == 0 and selected[-1] == 100
    assert sum(index > 50 for index in selected) > sum(index <= 50 for index in selected)


def test_budget_keeps_isolated_change_and_timeline_coverage() -> None:
    scores = [0.0] + [0.001] * 999
    scores[800] = 0.8
    selected, _ = choose_keyframes(scores, CvatSamplingRequest(strategy="budget", budget=10))
    assert len(selected) == 10
    assert selected[0] == 0 and selected[-1] == 999
    assert 800 in selected
    assert any(400 <= index <= 600 for index in selected)


def test_threshold_accumulates_motion_and_caps_model_input() -> None:
    scores = [0.0] + [0.01] * 1000
    request = CvatSamplingRequest(strategy="threshold", threshold=0.02)
    selected, effective = choose_keyframes(scores, request)
    assert selected[0] == 0 and selected[-1] == 1000
    assert len(selected) <= MAX_MODEL_FRAMES
    assert effective is not None and effective > request.threshold


def test_threshold_count_depends_on_scene() -> None:
    request = CvatSamplingRequest(strategy="threshold", threshold=0.1)
    still, _ = choose_keyframes([0.0] + [0.001] * 50, request)
    moving, _ = choose_keyframes([0.0] + [0.05] * 50, request)
    assert len(moving) > len(still)


def test_uniform_sampling_uses_declared_cvat_fps_without_global_300_frame_cap() -> None:
    request = CvatSamplingRequest(strategy="uniform", source_fps=10, target_fps=5)
    selected, threshold = choose_keyframes([0.0] * 701, request)
    assert threshold is None
    assert selected[:4] == [0, 2, 4, 6]
    assert selected[-1] == 700
    assert len(selected) == 351


def test_uniform_sampling_requires_explicit_timebase() -> None:
    with pytest.raises(ValueError, match="source_fps"):
        CvatSamplingRequest(strategy="uniform", target_fps=5)
