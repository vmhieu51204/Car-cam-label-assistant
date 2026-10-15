"""Uncalibrated GPU-free review scores for automatically labeled sequences."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from src.models.video import FrameManifest, ReviewQueue, ReviewUnit, SequenceAuditResult, TrackSet
from src.services.video.stitching import ChunkObservation, UnifiedTrack


@dataclass(frozen=True)
class SequenceReviewConfig:
    """Initial assumptions; weights/thresholds require held-out review calibration."""

    weights: dict[str, float] = field(
        default_factory=lambda: {
            "stitch_ambiguity": 0.26,
            "short_track": 0.12,
            "tiny_box": 0.11,
            "box_jump": 0.12,
            "area_change": 0.07,
            "track_gap": 0.09,
            "scene_cut": 0.08,
            "class_policy": 0.15,
        }
    )
    min_track_observations: int = 3
    tiny_box_side_px: float = 12.0
    jump_iou: float = 0.1
    area_ratio: float = 2.5
    scene_cut_radius: int = 1
    class_review_policy: dict[str, str] = field(default_factory=lambda: {"person": "always"})
    critical_fraction: float = 0.10
    high_fraction: float = 0.20
    normal_fraction: float = 0.30

    def __post_init__(self) -> None:
        if not math.isclose(sum(self.weights.values()), 1.0, abs_tol=1e-8):
            raise ValueError("review weights must sum to one")
        if (
            any(not math.isfinite(value) or value < 0 for value in self.weights.values())
            or self.min_track_observations < 1
            or self.tiny_box_side_px < 0
            or not 0 <= self.jump_iou <= 1
            or self.area_ratio < 1
            or self.scene_cut_radius < 0
        ):
            raise ValueError("invalid review thresholds")
        fractions = (self.critical_fraction, self.high_fraction, self.normal_fraction)
        if any(not 0 <= value <= 1 for value in fractions) or sum(fractions) > 1:
            raise ValueError("review percentiles exceed one")


@dataclass(frozen=True)
class TrackReview:
    track_id: str
    label: str
    score: float
    priority: str
    needs_review: bool
    reasons: tuple[str, ...]


def _area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = _area(a) + _area(b) - intersection
    return intersection / union if union > 0 else 0.0


def score_tracks(
    tracks: list[UnifiedTrack], *, scene_cut_frames: set[int] | None = None, config: SequenceReviewConfig | None = None
) -> list[TrackReview]:
    config = config or SequenceReviewConfig()
    cuts = scene_cut_frames or set()
    scored: list[tuple[UnifiedTrack, float, tuple[str, ...]]] = []
    for track in tracks:
        if not track.observations:
            continue
        signals: set[str] = set()
        if track.needs_review:
            signals.add("stitch_ambiguity")
        if len(track.observations) < config.min_track_observations:
            signals.add("short_track")
        if any(
            min(obs.bbox[2] - obs.bbox[0], obs.bbox[3] - obs.bbox[1]) < config.tiny_box_side_px
            for obs in track.observations
        ):
            signals.add("tiny_box")
        ordered = sorted(track.observations, key=lambda obs: obs.frame)
        for left, right in zip(ordered, ordered[1:]):
            if right.frame - left.frame > 1:
                signals.add("track_gap")
            elif _iou(left.bbox, right.bbox) < config.jump_iou:
                signals.add("box_jump")
            smaller, larger = sorted((_area(left.bbox), _area(right.bbox)))
            if smaller == 0 or larger / smaller > config.area_ratio:
                signals.add("area_change")
        if any(abs(obs.frame - cut) <= config.scene_cut_radius for obs in track.observations for cut in cuts):
            signals.add("scene_cut")
        if config.class_review_policy.get(track.label) == "always":
            signals.add("class_policy")
        score = min(1.0, sum(config.weights[name] for name in signals))
        reasons = tuple(sorted(signals | set(track.review_reasons)))
        scored.append((track, score, reasons))
    scored.sort(key=lambda item: (-item[1], item[0].track_id))
    total = len(scored)
    critical_end = math.ceil(total * config.critical_fraction)
    high_end = math.ceil(total * (config.critical_fraction + config.high_fraction))
    normal_end = math.ceil(total * (config.critical_fraction + config.high_fraction + config.normal_fraction))
    result = []
    for index, (track, score, reasons) in enumerate(scored):
        priority = (
            "low"
            if score == 0
            else "critical"
            if index < critical_end
            else "high"
            if index < high_end
            else "normal"
            if index < normal_end
            else "low"
        )
        result.append(TrackReview(track.track_id, track.label, score, priority, bool(reasons), reasons))
    return result


def build_sequence_review_queue(
    *,
    tracks: TrackSet,
    manifest: FrameManifest,
    track_set_sha256: str,
    discovery_sha256: str,
    config: SequenceReviewConfig | None = None,
) -> ReviewQueue:
    """Project existing, uncalibrated track signals to frames using the maximum score.

    Reasons such as unmatched_chunk express uncertainty, not confirmed GT errors.
    Zero-score frames stay low even when percentile ranks would place them higher.
    """
    if tracks.job_id != manifest.job_id:
        raise ValueError("sequence review inputs belong to different jobs")
    config = config or SequenceReviewConfig()
    cuts = {frame.frame_index for frame in manifest.frames if frame.scene_cut_before}
    unified = [
        UnifiedTrack(
            track.track_id,
            track.label_name,
            [
                ChunkObservation(obs.frame_index, (obs.bbox.xtl, obs.bbox.ytl, obs.bbox.xbr, obs.bbox.ybr))
                for obs in track.observations
                if not obs.outside
            ],
            track.needs_review,
            list(track.review_reasons),
        )
        for track in tracks.tracks
    ]
    reviews = {item.track_id: item for item in score_tracks(unified, scene_cut_frames=cuts, config=config)}
    scores = {frame.frame_index: 0.0 for frame in manifest.frames}
    reasons: dict[int, set[str]] = {index: set() for index in scores}
    identities: dict[int, set[str]] = {index: set() for index in scores}
    for track in tracks.tracks:
        review = reviews.get(track.track_id)
        if review is None:
            continue
        for obs in track.observations:
            if obs.frame_index not in scores:
                raise ValueError("review observation outside manifest")
            scores[obs.frame_index] = max(scores[obs.frame_index], review.score)
            reasons[obs.frame_index].update(review.reasons)
            identities[obs.frame_index].add(track.track_id)
    for index in scores:
        if any(abs(index - cut) <= config.scene_cut_radius for cut in cuts):
            scores[index] = max(scores[index], config.weights["scene_cut"])
            reasons[index].add("scene_cut")
    ranked = sorted(scores, key=lambda index: (-scores[index], index))
    total = len(ranked)
    ends = [
        math.ceil(total * fraction)
        for fraction in (
            config.critical_fraction,
            config.critical_fraction + config.high_fraction,
            config.critical_fraction + config.high_fraction + config.normal_fraction,
        )
    ]
    units = []
    for rank, index in enumerate(ranked):
        priority = (
            "low"
            if scores[index] == 0
            else "critical"
            if rank < ends[0]
            else "high"
            if rank < ends[1]
            else "normal"
            if rank < ends[2]
            else "low"
        )
        units.append(
            ReviewUnit(
                review_unit_id=f"frame:{index:06d}",
                job_id=tracks.job_id,
                priority=priority,
                review_score=round(scores[index], 6),
                frame_start=index,
                frame_end=index,
                track_ids=sorted(identities[index]),
                reason_codes=sorted(reasons[index]) or ["routine_review"],
            )
        )
    return ReviewQueue(
        job_id=tracks.job_id,
        track_set_sha256=track_set_sha256,
        discovery_sha256=discovery_sha256,
        audit_sha256=None,
        scoring_version="sequence-frame-max/v2-uncalibrated",
        units=units,
    )


def elevate_sequence_review_queue(queue: ReviewQueue, audit: SequenceAuditResult, audit_sha256: str) -> ReviewQueue:
    """Add VLM suspicions without lowering deterministic risk or changing annotations."""
    if queue.job_id != audit.job_id or queue.track_set_sha256 != audit.track_set_sha256:
        raise ValueError("sequence audit and review queue provenance differ")
    ranks = {"critical": 0, "high": 1, "normal": 2, "low": 3}
    by_frame = {unit.frame_start: unit.model_copy(deep=True) for unit in queue.units}
    for evidence in audit.evidence:
        unit = by_frame.get(evidence.frame_index)
        if unit is None:
            raise ValueError("audit frame has no review unit")
        signals: list[tuple[str, str, float]] = []
        if evidence.audit_type == "full_frame":
            if evidence.missing_decision == "yes":
                signals.append(("vlm_possible_missing", "high", 0.6))
            elif evidence.missing_decision == "uncertain":
                signals.append(("vlm_missing_uncertain", "normal", 0.35))
        else:
            if evidence.class_decision == "no":
                signals.append(("vlm_wrong_class", "high", 0.6))
            elif evidence.class_decision == "uncertain":
                signals.append(("vlm_class_uncertain", "normal", 0.35))
            if evidence.extent_decision == "no":
                signals.append(("vlm_bad_extent", "high", 0.6))
            elif evidence.extent_decision == "uncertain":
                signals.append(("vlm_extent_uncertain", "normal", 0.35))
        for reason, priority, score in signals:
            if ranks[priority] < ranks[unit.priority]:
                unit.priority = priority
            unit.review_score = max(unit.review_score or 0.0, score)
            unit.reason_codes = sorted(set(unit.reason_codes) | {reason})
            if evidence.track_id:
                unit.track_ids = sorted(set(unit.track_ids) | {evidence.track_id})
    return queue.model_copy(
        update={
            "audit_sha256": audit_sha256,
            "scoring_version": "sequence-frame-max/v2+openrouter-v1-uncalibrated",
            "units": list(by_frame.values()),
        }
    )
