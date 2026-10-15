"""Export-only RDP-like reduction; never alters SAM observations or masks."""

from typing import Any

from pydantic import Field

from src.models.video import StrictContract


class SparseConfig(StrictContract):
    enabled: bool = False
    min_iou: float = Field(default=0.9, gt=0, le=1, allow_inf_nan=False)
    max_gap: int = Field(default=15, ge=1, strict=True)


def box_iou(a: list[float], b: list[float]) -> float:
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def interpolate(a: dict[str, Any], b: dict[str, Any], frame: int) -> list[float]:
    alpha = (frame - a["frame"]) / (b["frame"] - a["frame"])
    return [x + alpha * (y - x) for x, y in zip(a["points"], b["points"], strict=True)]


def sparsify_shapes(
    shapes: list[dict[str, Any]],
    config: SparseConfig,
    *,
    mandatory: set[int] | None = None,
) -> tuple[list[dict[str, Any]], dict[int, list[str]]]:
    """Input frames are source indices. Outside, gaps and attribute changes split segments."""
    if not config.enabled:
        return shapes, {}
    frames = [s["frame"] for s in shapes]
    if frames != sorted(set(frames)):
        raise ValueError("source frames must be unique and ascending")
    if not shapes:
        return [], {}
    keep: dict[int, list[str]] = {0: ["endpoint"], len(shapes) - 1: ["endpoint"]}
    required = mandatory or set()
    for i, s in enumerate(shapes):
        if s["frame"] in required:
            keep.setdefault(i, []).append("review_or_boundary")
        if s.get("outside"):
            keep.setdefault(i, []).append("outside")
        if i and (
            s["frame"] - shapes[i - 1]["frame"] != 1
            or s.get("outside") != shapes[i - 1].get("outside")
            or s.get("occluded") != shapes[i - 1].get("occluded")
            or s.get("attributes") != shapes[i - 1].get("attributes")
        ):
            keep.setdefault(i, []).append("segment_boundary")
            keep.setdefault(i - 1, []).append("segment_boundary")
    anchors = sorted(keep)
    pending = list(zip(anchors, anchors[1:], strict=False))
    while pending:
        left, right = pending.pop()
        if right - left <= 1:
            continue
        errors = [
            (box_iou(shapes[i]["points"], interpolate(shapes[left], shapes[right], shapes[i]["frame"])), i)
            for i in range(left + 1, right)
        ]
        worst, index = min(errors)
        reason = "interpolation_error"
        if worst >= config.min_iou:
            if shapes[right]["frame"] - shapes[left]["frame"] <= config.max_gap:
                continue
            midpoint = (shapes[left]["frame"] + shapes[right]["frame"]) / 2
            index = min(range(left + 1, right), key=lambda i: (abs(shapes[i]["frame"] - midpoint), i))
            reason = "max_gap"
        keep.setdefault(index, []).append(reason)
        pending.extend([(left, index), (index, right)])
    return [shapes[i] for i in sorted(keep)], {shapes[i]["frame"]: keep[i] for i in sorted(keep)}
