"""Bounded keyframe selection from consecutive CVAT frames."""

from __future__ import annotations

import math
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from src.models.video import CvatSamplingRequest

MAX_MODEL_FRAMES = 300


def motion_image(jpeg: bytes) -> np.ndarray:
    """Decode a bounded grayscale preview; full resolution is kept only for chosen frames."""
    if len(jpeg) > 25_000_000:
        raise ValueError("CVAT frame exceeds the 25 MB import limit")
    with Image.open(BytesIO(jpeg)) as image:
        image.draft("L", (320, 180))
        gray = np.asarray(image.convert("L"))
    height, width = gray.shape
    scale = min(320 / width, 180 / height, 1)
    if scale < 1:
        gray = cv2.resize(gray, (max(1, round(width * scale)), max(1, round(height * scale))))
    return gray


def motion_score(previous: np.ndarray, current: np.ndarray, *, include_appearance: bool = True) -> float:
    if current.shape != previous.shape:
        raise ValueError("CVAT frames changed dimensions during import")
    flow = cv2.calcOpticalFlowFarneback(previous, current, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude = cv2.magnitude(flow[..., 0], flow[..., 1])
    diagonal = math.hypot(*current.shape)
    optical = float(np.percentile(magnitude, 75)) / diagonal
    if not include_appearance:
        return min(1.0, optical)
    # A hard scene cut has little coherent flow; keep a small appearance-change signal.
    appearance = float(np.mean(cv2.absdiff(previous, current))) / 255 * 0.05
    return min(1.0, max(optical, appearance))


def uniform_indexes(count: int, source_fps: float, target_fps: float) -> list[int]:
    """Preserve the original uniform cadence without allocating motion scores."""
    if count < 1 or not all(math.isfinite(x) and x > 0 for x in (source_fps, target_fps)) or target_fps > source_fps:
        raise ValueError("invalid uniform frame count or FPS")
    step = source_fps / target_fps
    indexes = {min(count - 1, round(index * step)) for index in range(math.ceil((count - 1) / step) + 1)}
    indexes.update((0, count - 1))
    return sorted(indexes)


def choose_keyframes(scores: list[float], request: CvatSamplingRequest) -> tuple[list[int], float | None]:
    """Return original frame indexes; uniform cadence is chunked, not globally capped."""
    if not scores or scores[0] != 0 or any(not math.isfinite(score) or score < 0 for score in scores):
        raise ValueError("motion scores must start at zero and be finite non-negative values")
    count = len(scores)
    if request.strategy == "all_frames":
        return list(range(count)), None
    if request.strategy == "uniform":
        assert request.source_fps is not None and request.target_fps is not None
        return uniform_indexes(count, request.source_fps, request.target_fps), None
    if count <= 2:
        return list(range(count)), request.threshold if request.strategy == "threshold" else None
    if request.strategy == "budget":
        budget = min(count, request.budget, MAX_MODEL_FRAMES)
        if budget == count:
            return list(range(count)), None
        total = sum(scores[1:])
        baseline = max(total / (count - 1) * 0.05, 1e-7)
        weights = [score + baseline for score in scores[1:]]
        cumulative = np.cumsum(weights)
        chosen_set = {0, count - 1}

        # Reserve some capacity for timeline coverage and isolated change peaks.
        # Motion-weighted quantiles use the rest of the model budget.
        if budget >= 5:
            coverage_count = max(3, budget // 5)
            chosen_set.update(round(index * (count - 1) / (coverage_count - 1)) for index in range(coverage_count))
            peak_count = budget // 10
            min_separation = max(1, count // (2 * budget))
            for index in sorted(range(1, count - 1), key=lambda item: (-scores[item], item)):
                if peak_count == 0 or scores[index] == 0:
                    break
                if all(abs(index - selected) >= min_separation for selected in chosen_set):
                    chosen_set.add(index)
                    peak_count -= 1

        remaining = budget - len(chosen_set)
        targets = np.linspace(0, float(cumulative[-1]), remaining + 2)
        for target in targets[1:-1]:
            index = int(np.searchsorted(cumulative, target, side="left")) + 1
            chosen_set.add(min(index, count - 2))
        # Colliding quantiles are filled from largest uncovered cumulative gaps.
        while len(chosen_set) < budget:
            ordered = sorted(chosen_set)
            gaps = [pair for pair in zip(ordered, ordered[1:]) if pair[1] - pair[0] > 1]
            left, right = max(gaps, key=lambda pair: (sum(weights[pair[0] : pair[1]]), pair[1] - pair[0]))
            candidates = range(left + 1, right)
            if not candidates:
                break
            midpoint = sum(weights[left:right]) / 2
            running = 0.0
            pick = left + 1
            for candidate in candidates:
                running += weights[candidate - 1]
                pick = candidate
                if running >= midpoint:
                    break
            chosen_set.add(pick)
        return sorted(chosen_set), None

    def threshold_indices(threshold: float) -> list[int]:
        chosen = [0]
        accumulated = 0.0
        for index in range(1, count - 1):
            accumulated += scores[index]
            if accumulated >= threshold:
                chosen.append(index)
                accumulated = 0.0
        chosen.append(count - 1)
        return chosen

    threshold = request.threshold
    chosen = threshold_indices(threshold)
    if len(chosen) > MAX_MODEL_FRAMES:
        low, high = threshold, max(sum(scores), threshold) + 1
        for _ in range(40):
            middle = (low + high) / 2
            if len(threshold_indices(middle)) > MAX_MODEL_FRAMES:
                low = middle
            else:
                high = middle
        threshold = high
        chosen = threshold_indices(threshold)
    return chosen, threshold
