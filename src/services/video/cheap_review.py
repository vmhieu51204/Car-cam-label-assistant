"""Uncalibrated, GPU-free review score for the explicitly VLM-off video path."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from src.models.video import BoundingBox2D, DiscoveryResult, FrameManifest, ReviewQueue, ReviewUnit, TrackSet


@dataclass(frozen=True)
class CheapReviewConfig:
    """Initial engineering assumptions, not learned probabilities or tuned thresholds.

    Weights sum to one. They emphasize weak propagation and missing masks, while
    limiting repeated detector proposals to one per-frame signal. Tune only after
    held-out recall@K evaluation with real reviewed labels.
    """

    weights: dict[str, float] = field(
        default_factory=lambda: {
            "weak_propagation": 0.18,
            "empty_mask": 0.16,
            "box_jump": 0.12,
            "area_change": 0.08,
            "track_gap": 0.10,
            "scene_cut": 0.08,
            "short_proposal": 0.10,
            "small_box": 0.08,
            "discovery_event": 0.10,
        }
    )
    critical_fraction: float = 0.10
    high_fraction: float = 0.20
    normal_fraction: float = 0.30
    box_jump_iou: float = 0.30
    area_ratio: float = 2.0
    small_box_frame_fraction: float = 0.001
    short_proposal_frames: int = 2
    scene_cut_radius: int = 1

    def __post_init__(self) -> None:
        if set(self.weights) != {
            "weak_propagation",
            "empty_mask",
            "box_jump",
            "area_change",
            "track_gap",
            "scene_cut",
            "short_proposal",
            "small_box",
            "discovery_event",
        }:
            raise ValueError("review score weight names are incomplete")
        if any(not math.isfinite(value) or value < 0 for value in self.weights.values()):
            raise ValueError("review weights must be finite nonnegative values")
        if not math.isclose(sum(self.weights.values()), 1.0, abs_tol=1e-8):
            raise ValueError("review weights must sum to one")
        fractions = (self.critical_fraction, self.high_fraction, self.normal_fraction)
        if any(not 0 <= value <= 1 for value in fractions) or sum(fractions) > 1:
            raise ValueError("review bucket fractions must fit in one")


def _area(box: BoundingBox2D) -> float:
    return (box.xbr - box.xtl) * (box.ybr - box.ytl)


def _iou(left: BoundingBox2D, right: BoundingBox2D) -> float:
    intersection = max(0.0, min(left.xbr, right.xbr) - max(left.xtl, right.xtl)) * max(
        0.0, min(left.ybr, right.ybr) - max(left.ytl, right.ytl)
    )
    union = _area(left) + _area(right) - intersection
    return intersection / union if union else 0.0


def build_cheap_review_queue(
    *,
    tracks: TrackSet,
    discovery: DiscoveryResult,
    manifest: FrameManifest,
    track_set_sha256: str,
    discovery_sha256: str,
    config: CheapReviewConfig | None = None,
) -> ReviewQueue:
    if tracks.job_id != discovery.job_id or tracks.job_id != manifest.job_id:
        raise ValueError("review inputs belong to different video jobs")
    config = config or CheapReviewConfig()
    values: dict[int, dict[str, float]] = {frame.frame_index: {} for frame in manifest.frames}
    ids: dict[int, set[str]] = {frame.frame_index: set() for frame in manifest.frames}

    def add(frame: int, name: str, strength: float, track_id: str | None = None) -> None:
        if frame not in values:
            raise ValueError("review evidence references a frame outside the manifest")
        values[frame][name] = max(values[frame].get(name, 0.0), max(0.0, min(1.0, strength)))
        if track_id:
            ids[frame].add(track_id)

    cuts = [frame.frame_index for frame in manifest.frames if frame.scene_cut_before]
    for record in manifest.frames:
        if any(abs(record.frame_index - cut) <= config.scene_cut_radius for cut in cuts):
            add(record.frame_index, "scene_cut", 1.0)
    for event in discovery.events:
        add(event.frame_index, "discovery_event", event.score if event.score is not None else 1.0)
        ids[event.frame_index].update(event.track_ids)
        if event.event_type in {"lost_seed", "unexpected_reappearance"}:
            add(event.frame_index, "track_gap", 1.0)
    proposal_ids = {track.track_id for track in discovery.proposal_tracks}
    for track in tracks.tracks:
        visible = [obs for obs in track.observations if not obs.outside]
        if track.track_id in proposal_ids and len(visible) <= config.short_proposal_frames:
            for obs in visible:
                add(obs.frame_index, "short_proposal", 1.0, track.track_id)
        previous = None
        for obs in track.observations:
            frame = obs.frame_index
            if obs.score is not None and not obs.outside:
                add(frame, "weak_propagation", 1.0 - obs.score, track.track_id)
            if obs.mask_area_px == 0 or obs.outside:
                add(frame, "empty_mask", 1.0, track.track_id)
            if not obs.outside:
                frame_record = manifest.frames[frame]
                if _area(obs.bbox) < config.small_box_frame_fraction * frame_record.width * frame_record.height:
                    add(frame, "small_box", 1.0, track.track_id)
            if previous is not None:
                if frame - previous.frame_index > 1 or obs.outside != previous.outside:
                    add(frame, "track_gap", 1.0, track.track_id)
                if not obs.outside and not previous.outside:
                    if _iou(previous.bbox, obs.bbox) < config.box_jump_iou:
                        add(frame, "box_jump", 1.0, track.track_id)
                    areas = (_area(previous.bbox), _area(obs.bbox))
                    if min(areas) > 0 and max(areas) / min(areas) >= config.area_ratio:
                        add(frame, "area_change", 1.0, track.track_id)
            previous = obs
    scored = [
        (frame, round(min(1.0, sum(config.weights[name] * strength for name, strength in signals.items())), 6))
        for frame, signals in values.items()
    ]
    ranked = sorted(scored, key=lambda item: (-item[1], item[0]))
    critical_end = math.ceil(len(ranked) * config.critical_fraction)
    high_end = critical_end + math.ceil(len(ranked) * config.high_fraction)
    normal_end = high_end + math.ceil(len(ranked) * config.normal_fraction)
    buckets = {}
    for index, (frame, _) in enumerate(ranked):
        buckets[frame] = (
            "critical"
            if index < critical_end
            else "high"
            if index < high_end
            else "normal"
            if index < normal_end
            else "low"
        )
    units = [
        ReviewUnit(
            review_unit_id=f"frame:{frame:06d}",
            job_id=tracks.job_id,
            priority=buckets[frame],
            review_score=score,
            frame_start=frame,
            frame_end=frame,
            track_ids=sorted(ids[frame]),
            reason_codes=sorted(values[frame]) or ["routine_review"],
        )
        for frame, score in scored
    ]
    order = {"critical": 0, "high": 1, "normal": 2, "low": 3}
    units.sort(key=lambda unit: (order[unit.priority], -(unit.review_score or 0), unit.frame_start))
    return ReviewQueue(
        job_id=tracks.job_id,
        track_set_sha256=track_set_sha256,
        discovery_sha256=discovery_sha256,
        audit_sha256=None,
        scoring_version="cheap-review/v1-uncalibrated",
        units=units,
    )
