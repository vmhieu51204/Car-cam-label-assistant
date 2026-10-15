"""Normalize observations; input quality is distinct from probability of correctness."""

import math

import numpy as np

from src.models.discrepancy import FrameEvidence, HCLConfig, SignalResult


def missing(reason: str) -> SignalResult:
    """Represent missing evidence explicitly."""
    return SignalResult(available=False, reason=reason)


def measured(value: float, reliability: float, provenance: str) -> SignalResult:
    """Bound an experimental discrepancy measure and preserve its provenance."""
    return SignalResult(
        available=True,
        value=min(1.0, max(0.0, value)),
        reliability=reliability,
        reason="observed",
        provenance=provenance,
    )


def detector_signal(frame: FrameEvidence) -> SignalResult:
    """Only a versioned calibrated detector probability contributes evidence."""
    item = frame.detector
    if item is None or item.probability is None or not item.calibration_id:
        return missing("detector_probability_uncalibrated")
    return measured(1 - item.probability, 1, item.calibration_id)


def points_signal(frame: FrameEvidence, config: HCLConfig) -> SignalResult:
    """Score an externally fitted, class/range-conditioned support residual."""
    if frame.points is None:
        return missing("point_support_unavailable")
    reliability = frame.points.reliability * (0.25 if _occlusion(frame) == "heavy" else 1.0)
    value = (config.point_z_min - frame.points.robust_z) / config.point_z_span
    return measured(value, reliability, frame.points.calibration_id)


def projection_signal(frame: FrameEvidence) -> SignalResult:
    """Only aligned, calibrated camera evidence can disagree with LiDAR."""
    item = frame.projection
    if not item or not item.calibration_valid or not frame.timestamps_aligned:
        return missing("projection_unavailable_or_uncalibrated")
    return measured(1 - item.iou, item.reliability, f"{item.source}:{item.calibration_id}")


def semantic_signal(frame: FrameEvidence) -> SignalResult:
    """Map semantic disagreement without using VLM self-reported confidence."""
    item = frame.semantic
    if item is None or (item.verdict.is_real_object and item.verdict.predicted_class == "other"):
        return missing("semantic_abstention_or_unavailable")
    value = 0.0 if item.verdict.predicted_class == frame.label else 0.8
    if not item.verdict.is_real_object:
        value = 1.0
    reliability = item.reliability * (0.25 if item.verdict.occlusion == "heavy" else 1)
    return measured(value, reliability, item.model_id)


def dimension_signal(frame: FrameEvidence, config: HCLConfig) -> SignalResult:
    """Compare proposals without replacing detector dimensions or asserting factory truth."""
    item = frame.dimensions
    if not item or not item.identifiable or item.modified or _occlusion(frame) == "heavy":
        return missing("dimension_abstention_or_unavailable")
    errors = [
        abs(math.log(a / b)) / tolerance
        for a, b, tolerance in zip(frame.box.size_lwh, item.size_lwh, config.dimension_log_tolerance)
    ]
    return measured(max(errors), item.reliability, f"{item.source}:{item.source_id}")


def temporal_signal(history: list[FrameEvidence], config: HCLConfig) -> SignalResult:
    """Use a fixed global frame; sensor-local boxes are rejected by the evidence schema."""
    if len(history) < 2:
        return missing("temporal_history_unavailable")
    previous, current = history[-2:]
    elapsed = current.timestamp_s - previous.timestamp_s
    if elapsed <= 0 or any(not frame.timestamps_aligned for frame in history):
        return missing("temporal_timestamps_invalid")
    delta = np.asarray(current.box.center) - previous.box.center
    speed_prior = config.temporal_speed_mps.get(current.label, 20)
    if len(history) >= 3 and previous.timestamp_s > history[-3].timestamp_s:
        predicted_delta = (
            (np.asarray(previous.box.center) - history[-3].box.center)
            / (previous.timestamp_s - history[-3].timestamp_s)
            * elapsed
        )
        center = float(np.linalg.norm(delta - predicted_delta)) / (speed_prior * elapsed)
    else:
        observed_speed = float(np.linalg.norm(delta)) / elapsed
        center = max(0.0, observed_speed / speed_prior - 1.0)
    size = max(
        abs(math.log(a / b)) / config.temporal_size_tolerance
        for a, b in zip(current.box.size_lwh, previous.box.size_lwh)
    )
    yaw = _yaw_residual(previous, current, config) if np.linalg.norm(delta[:2]) / elapsed > 0.5 else 0
    return measured(max(center, size, yaw, float(current.label != previous.label)), 1, "global_track_history")


def _yaw_residual(previous: FrameEvidence, current: FrameEvidence, config: HCLConfig) -> float:
    if any(max(frame.box.size_lwh[:2]) / min(frame.box.size_lwh[:2]) < 1.15 for frame in [previous, current]):
        return 0.0
    delta = current.box.yaw - previous.box.yaw
    return abs(math.atan2(math.sin(delta), math.cos(delta))) / config.temporal_yaw_tolerance


def _occlusion(frame: FrameEvidence) -> str:
    """Conservatively combine explicit and semantic occlusion evidence."""
    if frame.occlusion == "heavy" or (frame.semantic and frame.semantic.verdict.occlusion == "heavy"):
        return "heavy"
    if frame.occlusion != "unknown":
        return frame.occlusion
    if frame.semantic:
        return frame.semantic.verdict.occlusion
    return frame.occlusion
