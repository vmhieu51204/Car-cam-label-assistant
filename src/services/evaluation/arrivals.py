"""Arrival delay, deadline compliance, and new-object tracking metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from src.models.evaluation import (
    ArrivalMetrics,
    ArrivalTrackResult,
    BenchmarkManifest,
    DatasetReference,
    EvaluationScope,
    FrameKey,
    MetricValue,
    PredictionRun,
    ReferenceBox,
    TrackKey,
)
from src.services.evaluation.matching import match_boxes


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(values, p))


def evaluate_arrivals(
    reference: DatasetReference,
    run: PredictionRun,
    *,
    manifest: BenchmarkManifest,
    scope: EvaluationScope,
) -> ArrivalMetrics:
    """Evaluate first arrival detection and delay within a 1-second deadline."""
    # Group reference boxes by track within sequence
    tracks_by_id: dict[tuple[Any, str], list[ReferenceBox]] = defaultdict(list)
    for box in reference.boxes:
        if box.native_track_id is not None and box.class_name in manifest.profile.classes:
            tracks_by_id[(box.frame.sequence, box.native_track_id)].append(box)

    # Frame metadata lookup (timestamp and ordinal)
    frame_ref_map = {f.key: f for f in reference.frames}
    frame_order = {f.key: f.source_ordinal for f in reference.frames}

    # Matches by frame for IoU >= 0.5 correct class
    run_boxes_by_frame: dict[FrameKey, list[Any]] = defaultdict(list)
    for b in run.boxes:
        run_boxes_by_frame[b.frame].append(b)

    results: list[ArrivalTrackResult] = []
    delays_s: list[float] = []
    delays_frames: list[float] = []
    detected_count = 0
    never_detected_count = 0
    censored_count = 0

    deadline_s = manifest.arrival.deadline_s

    for (seq, track_id), boxes in sorted(tracks_by_id.items(), key=lambda x: str(x[0])):
        # Sort observations by timeline ordinal
        sorted_boxes = sorted(boxes, key=lambda b: frame_order.get(b.frame, 0))
        first_box = sorted_boxes[0]
        first_frame = first_box.frame
        first_ref = frame_ref_map.get(first_frame)

        first_ordinal = frame_order.get(first_frame, 0)
        first_time_us = first_ref.timestamp_us if first_ref and first_ref.timestamp_us is not None else None

        # Cadence / FPS for sequence
        cadence = next((c for c in manifest.cadence if c.sequence == seq), None)
        seq_fps = cadence.source_fps if cadence else 10.0
        if seq_fps is None:
            raise ValueError("Arrival evaluation requires a source FPS")
        time_prov = (
            "recorded_sensor" if (first_ref and first_ref.timestamp_provenance == "recorded_sensor") else "declared_fps"
        )

        # Deadline calculation
        deadline_frames = round(deadline_s * seq_fps)

        # Check max frame in reference to determine censoring
        max_seq_ordinal = max(
            (f.source_ordinal for f in reference.frames if f.key.sequence == seq), default=first_ordinal
        )
        censored = (first_ordinal + deadline_frames) > max_seq_ordinal
        if censored:
            censored_count += 1

        detected_frame: FrameKey | None = None
        detected_delay_frames: int | None = None
        detected_delay_s: float | None = None
        within_deadline: bool | None = None

        # Check subsequent observations of this track for detection
        for box in sorted_boxes:
            preds = tuple(run_boxes_by_frame[box.frame])
            if not preds:
                continue
            match_res = match_boxes((box,), preds, iou_threshold=manifest.arrival.iou_threshold, class_aware=True)
            if match_res.pairs:
                detected_frame = box.frame
                det_ref = frame_ref_map.get(detected_frame)
                det_ordinal = frame_order.get(detected_frame, first_ordinal)
                detected_delay_frames = max(0, det_ordinal - first_ordinal)

                if first_time_us is not None and det_ref and det_ref.timestamp_us is not None:
                    detected_delay_s = max(0.0, (det_ref.timestamp_us - first_time_us) / 1e6)
                else:
                    detected_delay_s = max(0.0, detected_delay_frames / seq_fps)

                within_deadline = (
                    (detected_delay_s <= deadline_s)
                    if detected_delay_s is not None
                    else (detected_delay_frames <= deadline_frames)
                )
                break

        if detected_frame is not None and detected_delay_s is not None and detected_delay_frames is not None:
            detected_count += 1
            delays_s.append(detected_delay_s)
            delays_frames.append(float(detected_delay_frames))
        else:
            never_detected_count += 1
            within_deadline = False if not censored else None

        track_key = TrackKey(sequence=seq, track_id=track_id)
        results.append(
            ArrivalTrackResult(
                track=track_key,
                first_frame=first_frame,
                detected_frame=detected_frame,
                deadline_source_frames=deadline_frames,
                delay_source_frames=detected_delay_frames,
                delay_s=detected_delay_s,
                time_provenance=time_prov,
                within_deadline=within_deadline,
                censored=censored,
            )
        )

    eligible_tracks = len(results)
    within_deadline_count = sum(1 for r in results if r.within_deadline is True)
    recalled_val = (within_deadline_count / eligible_tracks) if eligible_tracks > 0 else None

    distinct_recall = MetricValue(
        value=recalled_val,
        unit="ratio",
        scale="0-1",
        numerator=float(within_deadline_count),
        denominator=float(eligible_tracks),
        availability="available" if recalled_val is not None else "undefined_denominator",
        definition_id="arrival/distinct_object_recall",
    )

    obs_recall = MetricValue(
        value=recalled_val,
        unit="ratio",
        scale="0-1",
        numerator=float(detected_count),
        denominator=float(eligible_tracks),
        availability="available" if recalled_val is not None else "undefined_denominator",
        definition_id="arrival/observation_recall",
    )

    med_s = _percentile(delays_s, 50)
    p90_s = _percentile(delays_s, 90)
    med_f = _percentile(delays_frames, 50)
    p90_f = _percentile(delays_frames, 90)

    return ArrivalMetrics(
        distinct_object_recall=distinct_recall,
        observation_recall=obs_recall,
        median_delay_s=MetricValue(
            value=med_s,
            unit="seconds",
            scale="seconds",
            availability="available" if med_s is not None else "unavailable_input",
            definition_id="arrival/median_delay_s",
        ),
        p90_delay_s=MetricValue(
            value=p90_s,
            unit="seconds",
            scale="seconds",
            availability="available" if p90_s is not None else "unavailable_input",
            definition_id="arrival/p90_delay_s",
        ),
        median_delay_frames=MetricValue(
            value=med_f,
            unit="frames",
            scale="frames",
            availability="available" if med_f is not None else "unavailable_input",
            definition_id="arrival/median_delay_frames",
        ),
        p90_delay_frames=MetricValue(
            value=p90_f,
            unit="frames",
            scale="frames",
            availability="available" if p90_f is not None else "unavailable_input",
            definition_id="arrival/p90_delay_frames",
        ),
        eligible_tracks=eligible_tracks,
        never_detected_tracks=never_detected_count,
        censored_tracks=censored_count,
        tracks=tuple(results),
    )
