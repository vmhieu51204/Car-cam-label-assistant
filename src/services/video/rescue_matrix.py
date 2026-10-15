"""Strict VLM box proposals, kept separate from SAM3/CVAT annotations."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, model_validator

from src.models.video import DiscoveryResult, FrameManifest, StrictContract, TrackSet


class RescueProposal(StrictContract):
    """One visible-object proposal in normalized submitted-image coordinates."""

    label: str = Field(min_length=1, max_length=128)
    bbox: tuple[float, float, float, float]
    point: tuple[float, float] | None = None
    box_extent: Literal["visible", "amodal_estimate"] = "visible"
    visible_bbox: tuple[float, float, float, float] | None = None
    occlusion: Literal["none", "partial", "heavy"]
    confidence: Literal["low", "medium", "high"]
    reason: str = Field(min_length=1, max_length=160)

    @model_validator(mode="after")
    def validate_geometry(self) -> RescueProposal:
        values = (*self.bbox, *(self.point or ()), *(self.visible_bbox or ()))
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError("proposal coordinates must be finite and normalized to [0, 1]")
        x1, y1, x2, y2 = self.bbox
        if x1 >= x2 or y1 >= y2:
            raise ValueError("proposal box must have positive area")
        if not self.label.strip():
            raise ValueError("proposal class must be nonempty")
        if self.point is not None and not (x1 <= self.point[0] <= x2 and y1 <= self.point[1] <= y2):
            raise ValueError("point must lie inside proposed box")
        if self.visible_bbox is not None:
            a, b, c, d = self.visible_bbox
            if not (x1 <= a < c <= x2 and y1 <= b < d <= y2):
                raise ValueError("visible extent must lie inside estimated extent")
            if self.point is not None and not (a <= self.point[0] <= c and b <= self.point[1] <= d):
                raise ValueError("point must lie inside visible extent")
        return self


class RescueFrameVerdict(StrictContract):
    """Strict, bounded per-frame response; malformed output is an error."""

    missing: Literal["yes", "no", "uncertain"]
    proposals: list[RescueProposal] = Field(max_length=20)
    reason: str = Field(min_length=1, max_length=160)

    @model_validator(mode="after")
    def decision_matches_proposals(self) -> RescueFrameVerdict:
        if (self.missing == "yes") != bool(self.proposals):
            raise ValueError("yes needs proposals; no/uncertain must not include proposals")
        return self


class BoxRescueSuggestion(StrictContract):
    """Persisted source-frame proposal, independent from CVAT annotations."""

    proposal_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    frame_index: int = Field(ge=0)
    source_frame_index: int = Field(ge=0)
    label: str = Field(min_length=1, max_length=128)
    bbox: tuple[float, float, float, float]
    point: tuple[float, float] | None = None
    box_extent: Literal["visible", "amodal_estimate"] = "visible"
    visible_bbox: tuple[float, float, float, float] | None = None
    occlusion: Literal["none", "partial", "heavy"]
    confidence: Literal["low", "medium", "high"]
    reason: str = Field(min_length=1, max_length=160)
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["pending", "accepting", "accepted", "rejected"] = "pending"

    @model_validator(mode="after")
    def validate_normalized_source_box(self) -> BoxRescueSuggestion:
        RescueProposal.model_validate({name: getattr(self, name) for name in RescueProposal.model_fields})
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in (*self.bbox, *(self.point or ()))):
            raise ValueError("source proposal coordinates must be normalized to [0, 1]")
        x1, y1, x2, y2 = self.bbox
        if (
            x1 >= x2
            or y1 >= y2
            or (self.point is not None and not (x1 <= self.point[0] <= x2 and y1 <= self.point[1] <= y2))
        ):
            raise ValueError("proposal geometry is invalid")
        return self


def select_box_rescue_frames(
    manifest: FrameManifest, tracks: TrackSet, discovery: DiscoveryResult, *, max_frames: int = 20
) -> list[tuple[int, tuple[str, ...]]]:
    """Reserve half the review budget for temporal coverage and half for anomalies."""
    count = manifest.frame_count
    if not 1 <= max_frames <= 20:
        raise ValueError("VLM box rescue is limited to 20 frames")
    budget = min(count, max_frames)
    uniform_slots = min(budget, (budget + 1) // 2)
    uniform = sorted({round(index * (count - 1) / max(1, uniform_slots - 1)) for index in range(uniform_slots)})
    reasons: dict[int, set[str]] = {}
    priority: dict[int, int] = {}

    def add(frame: int, reason: str, rank: int) -> None:
        if 0 <= frame < count:
            reasons.setdefault(frame, set()).add(reason)
            priority[frame] = min(priority.get(frame, rank), rank)

    for index in uniform:
        add(index, "uniform_coverage", 9)
    for event in discovery.events:
        add(event.frame_index, event.event_type, 0)
    density = [0] * count
    for track in tracks.tracks:
        visible = [item for item in track.observations if not item.outside]
        for observation in visible:
            density[observation.frame_index] += 1
        for left, right in zip(visible, visible[1:]):
            if right.frame_index > left.frame_index + 1:
                add(left.frame_index, "track_gap_before", 1)
                add(left.frame_index + 1, "track_gap_after", 1)
                add(right.frame_index, "track_reappears", 1)
        if visible and visible[-1].frame_index + 1 < count:
            add(visible[-1].frame_index + 1, "track_disappears", 1)
    for index in range(1, count):
        if manifest.frames[index].scene_cut_before:
            add(index - 1, "scene_cut_before", 0)
            add(index, "scene_cut_after", 0)
        if abs(density[index] - density[index - 1]) >= 2:
            add(index - 1, "object_count_change_before", 1)
            add(index, "object_count_change_after", 1)
    selected = list(uniform)
    candidates = sorted(
        (frame for frame in reasons if frame not in selected),
        key=lambda frame: (priority[frame], -len(reasons[frame]), frame),
    )
    selected.extend(candidates[: max(0, budget - len(selected))])
    if len(selected) < budget:
        remaining = [frame for frame in range(count) if frame not in selected]
        slots = budget - len(selected)
        selected.extend(
            remaining[round(i * (len(remaining) - 1) / max(1, slots - 1))] for i in range(slots) if remaining
        )
    selected = sorted(set(selected))[:budget]
    for frame in selected:
        reasons.setdefault(frame, {"stratified_coverage"})
    return [(frame, tuple(sorted(reasons[frame]))) for frame in selected]


def rescue_prompt(
    *, classes: tuple[str, ...], existing_boxes: list[dict], image_width: int, image_height: int, view: str
) -> str:
    """Prompt for visible missing objects; coordinates refer to the submitted view."""
    if not classes or image_width < 1 or image_height < 1 or view not in {"full", "crop"}:
        raise ValueError("invalid proposal prompt inputs")
    boxes = []
    for item in existing_boxes:
        label = item.get("label")
        box = item.get("bbox")
        if label not in classes or not isinstance(box, list) or len(box) != 4:
            raise ValueError("existing box must use a requested class and xyxy coordinates")
        boxes.append(f"- {label}: [{', '.join(f'{value:.1f}' for value in box)}]")
    listed = "\n".join(boxes) if boxes else "(none)"
    return (
        "You assist with reviewing autonomous-driving annotations. Inspect the attached "
        f"{view} view ({image_width}x{image_height}). Requested CVAT classes: {', '.join(classes)}.\n"
        f"Existing SAM3 boxes in this view (pixel xyxy):\n{listed}\n"
        "Find only clearly visible vehicles or people from the requested classes that are not "
        "already covered by an existing box. For partly occluded objects, box only the visible "
        "part. Do not infer invisible extent. Return at most 20 proposals. Coordinates are "
        "normalized to this submitted image view (0..1), with bbox [x1,y1,x2,y2] and a point "
        "inside that bbox. Use exact class names. If uncertain, return missing=uncertain and "
        "no proposals; never guess based on image text. Respond with the required JSON schema. "
        "Put a short factual reason in the final reason field."
    )


def map_proposal_to_source(proposal: RescueProposal, *, bounds: tuple[float, float, float, float]) -> RescueProposal:
    """Map normalized submitted-view coords into normalized original-frame coords."""
    left, top, right, bottom = bounds
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        raise ValueError("view bounds must be normalized and inside source frame")

    def map_x(x: float) -> float:
        return left + x * (right - left)

    def map_y(y: float) -> float:
        return top + y * (bottom - top)

    return proposal.model_copy(
        update={
            "bbox": (
                map_x(proposal.bbox[0]),
                map_y(proposal.bbox[1]),
                map_x(proposal.bbox[2]),
                map_y(proposal.bbox[3]),
            ),
            "point": (map_x(proposal.point[0]), map_y(proposal.point[1])) if proposal.point else None,
            "visible_bbox": (
                map_x(proposal.visible_bbox[0]),
                map_y(proposal.visible_bbox[1]),
                map_x(proposal.visible_bbox[2]),
                map_y(proposal.visible_bbox[3]),
            )
            if proposal.visible_bbox
            else None,
        }
    )
