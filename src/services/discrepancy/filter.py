"""Deterministic, side-effect-free HCL scoring through one public interface."""

import hashlib
from dataclasses import dataclass

import numpy as np

from src.models.discrepancy import (
    DiscrepancyReason,
    DiscrepancyResult,
    FrameEvidence,
    FrameResult,
    HCLConfig,
    Status,
    TrackEvidence,
)
from src.services.discrepancy.signals import (
    detector_signal,
    dimension_signal,
    points_signal,
    projection_signal,
    semantic_signal,
    temporal_signal,
)


@dataclass(frozen=True)
class RoutingContext:
    track: TrackEvidence
    score: float | None
    coverage: float
    flags: list[str]


class DiscrepancyFilter:
    """Keep score computation, missing-evidence policy and routing advice local."""

    def __init__(self, config: HCLConfig | None = None) -> None:
        self.config = (config or HCLConfig()).model_copy(deep=True)

    def evaluate(self, evidence: TrackEvidence | dict) -> DiscrepancyResult:
        """Evaluate typed observations; never change a box or a human review state."""
        track = TrackEvidence.model_validate(evidence)
        frames = [self._frame(track.frames[max(0, index - 2) : index + 1]) for index in range(len(track.frames))]
        return self._result(track, frames, self._track_score(track, frames))

    def _track_score(self, track: TrackEvidence, frames: list[FrameResult]) -> float | None:
        scores = [frame.discrepancy_score for frame in frames if frame.discrepancy_score is not None]
        peak = max(scores, default=0.0)
        gap_score, class_flip = self._track_conflicts(track)
        return (
            max(
                float(np.percentile(scores, 90)),
                peak if peak >= self.config.review_threshold else 0,
                gap_score,
                class_flip,
            )
            if scores
            else (max(gap_score, class_flip) or None)
        )

    def _result(self, track: TrackEvidence, frames: list[FrameResult], score: float | None) -> DiscrepancyResult:
        coverage = min(frame.evidence_coverage for frame in frames)
        flags = self._gates(track)
        status = self._status(RoutingContext(track, score, coverage, flags))
        needs_vlm = (
            status == "ambiguous"
            and track.label in {"car", "truck"}
            and any(frame.vlm_status == "not_requested" for frame in track.frames)
        )
        scored_frames = [frame for frame in frames if frame.discrepancy_score is not None]
        worst = max(scored_frames, key=lambda frame: frame.discrepancy_score or 0) if scored_frames else None
        return DiscrepancyResult(
            track_id=track.track_id,
            data_source=track.data_source,
            discrepancy_score=score,
            evidence_coverage=coverage,
            status=status,
            suggested_action="exclude" if status == "invalid" else ("request_vlm" if needs_vlm else "review"),
            worst_frame=worst.frame_index if worst else None,
            reasons=self._reasons(frames) + self._track_reasons(track),
            hard_flags=flags,
            frame_results=frames,
            needs_vlm=needs_vlm,
            config_version=self.config.version,
            config_digest=hashlib.sha256(self.config.model_dump_json().encode()).hexdigest(),
            config=self.config.model_copy(deep=True),
        )

    def _gates(self, track: TrackEvidence) -> list[str]:
        flags = []
        if len({frame.frame_index for frame in track.frames}) != len(track.frames):
            flags.append("DUPLICATE_FRAME")
        supported = {"car", "truck", "pedestrian"}
        if track.label not in supported or any(frame.label not in supported for frame in track.frames):
            flags.append("UNSUPPORTED_CLASS")
        if any(not frame.timestamps_aligned for frame in track.frames):
            flags.append("TIMESTAMPS_UNALIGNED")
        if any(frame.projection and not frame.projection.calibration_valid for frame in track.frames):
            flags.append("CALIBRATION_INVALID")
        if any(
            b.timestamp_s <= a.timestamp_s or b.frame_index <= a.frame_index
            for a, b in zip(track.frames, track.frames[1:])
        ):
            flags.append("NON_MONOTONIC_OBSERVATIONS")
        return flags

    def _status(self, context: RoutingContext) -> Status:
        track, score = context.track, context.score
        if "DUPLICATE_FRAME" in context.flags:
            return "invalid"
        if context.flags or context.coverage < self.config.minimum_coverage or score is None:
            return "insufficient_evidence"
        if score >= self.config.review_threshold:
            return "discrepancy"
        if any(frame.vlm_status in {"failed", "abstained"} for frame in track.frames):
            return "ambiguous"
        high_detector = all(
            frame.detector
            and frame.detector.calibration_id
            and frame.detector.probability is not None
            and frame.detector.probability >= self.config.high_detector_probability
            for frame in track.frames
        )
        if score >= self.config.vlm_escalation_low or not high_detector:
            return "ambiguous"
        return "consistent_candidate"

    def _frame(self, history: list[FrameEvidence]) -> FrameResult:
        frame = history[-1]
        signals = {
            "detector": detector_signal(frame),
            "points": points_signal(frame, self.config),
            "projection": projection_signal(frame),
            "temporal": temporal_signal(history, self.config),
        }
        if frame.label in {"car", "truck"} and frame.vlm_status != "not_requested":
            signals.update(dimensions=dimension_signal(frame, self.config), semantic=semantic_signal(frame))
        weights = self.config.weights.model_dump()
        available: list[tuple[float, float, float]] = []
        for name, item in signals.items():
            if item.available and item.value is not None and item.reliability > 0:
                available.append((weights[name] * item.reliability, item.value, item.reliability))
        weight_sum = sum(weight for weight, _, _ in available)
        score = None
        if weight_sum:
            mean = sum(weight * value for weight, value, _ in available) / weight_sum
            peak = max(value * reliability for _, value, reliability in available)
            score = self.config.max_blend * peak + (1 - self.config.max_blend) * mean
        return FrameResult(
            frame_index=frame.frame_index,
            discrepancy_score=score,
            evidence_coverage=weight_sum / sum(weights[name] for name in signals),
            signals=signals,
        )

    def _track_conflicts(self, track: TrackEvidence) -> tuple[float, float]:
        span = max(frame.frame_index for frame in track.frames) - min(frame.frame_index for frame in track.frames) + 1
        gaps = 1 - len({frame.frame_index for frame in track.frames}) / span
        return gaps, float(any(frame.label != track.label for frame in track.frames))

    def _track_reasons(self, track: TrackEvidence) -> list[DiscrepancyReason]:
        gaps, flip = self._track_conflicts(track)
        reasons = []
        if gaps:
            reasons.append(
                DiscrepancyReason(
                    code="TRACK_FRAGMENTATION", score=gaps, detail="Missing keyframes within the observed track span"
                )
            )
        if flip:
            frame = next(frame for frame in track.frames if frame.label != track.label)
            reasons.append(
                DiscrepancyReason(
                    code="CLASS_FLIP",
                    score=1,
                    frame_index=frame.frame_index,
                    detail="Observation class differs from track class",
                )
            )
        return reasons

    def _reasons(self, frames: list[FrameResult]) -> list[DiscrepancyReason]:
        reasons = []
        for frame in frames:
            for name, signal in frame.signals.items():
                value = signal.value
                if value is not None and value * signal.reliability >= self.config.review_threshold:
                    reasons.append(
                        DiscrepancyReason(
                            code=f"{name.upper()}_CONFLICT",
                            frame_index=frame.frame_index,
                            score=value * signal.reliability,
                            detail=signal.provenance or signal.reason,
                        )
                    )
        return sorted(reasons, key=lambda reason: reason.score or 0, reverse=True)[:10]
