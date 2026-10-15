"""Deterministic video discrepancy routing into human review units."""

from __future__ import annotations

from src.models.video import AuditResult, DiscoveryResult, FrameManifest, ReviewQueue, ReviewUnit, TrackSet
from src.services.video.cheap_review import CheapReviewConfig, build_cheap_review_queue

PRIORITY_ORDER = {"critical": 0, "high": 1, "normal": 2, "low": 3}


def build_review_queue(
    *,
    tracks: TrackSet,
    discovery: DiscoveryResult,
    audit: AuditResult | None = None,
    audit_sha256: str | None = None,
    manifest: FrameManifest | None = None,
    track_set_sha256: str | None = None,
    discovery_sha256: str | None = None,
    scoring_config: CheapReviewConfig | None = None,
) -> ReviewQueue:
    if audit is None:
        if manifest is None or track_set_sha256 is None or discovery_sha256 is None or audit_sha256 is not None:
            raise ValueError("VLM-off review needs manifest and track/discovery hashes, without an audit hash")
        return build_cheap_review_queue(
            tracks=tracks,
            discovery=discovery,
            manifest=manifest,
            track_set_sha256=track_set_sha256,
            discovery_sha256=discovery_sha256,
            config=scoring_config,
        )
    if audit_sha256 is None:
        raise ValueError("audited review needs an audit hash")
    if tracks.job_id != discovery.job_id or tracks.job_id != audit.job_id:
        raise ValueError("review inputs belong to different video jobs")
    signals: dict[int, dict[str, object]] = {}

    def add(frame: int, priority: str, reason: str, track_ids: list[str]) -> None:
        signal = signals.setdefault(frame, {"priority": "normal", "reasons": set(), "tracks": set()})
        current = str(signal["priority"])
        if PRIORITY_ORDER[priority] < PRIORITY_ORDER[current]:
            signal["priority"] = priority
        reasons = signal["reasons"]
        track_set = signal["tracks"]
        assert isinstance(reasons, set) and isinstance(track_set, set)
        reasons.add(reason)
        track_set.update(track_ids)

    event_priorities = {
        "possible_missing_object": "critical",
        "lost_seed": "high",
        "unexpected_reappearance": "high",
        "scene_cut_without_seed": "high",
    }
    for event in discovery.events:
        add(event.frame_index, event_priorities[event.event_type], event.event_type, event.track_ids)
    audit_by_frame = {item.frame_index: item for item in audit.evidence}
    for evidence in audit.evidence:
        if evidence.verdict == "agree":
            signal = signals.setdefault(
                evidence.frame_index,
                {"priority": "normal", "reasons": set(), "tracks": set()},
            )
            reasons = signal["reasons"]
            track_ids = signal["tracks"]
            assert isinstance(reasons, set) and isinstance(track_ids, set)
            reasons.add("vlm_agree")
            track_ids.update(evidence.track_ids)
        elif evidence.verdict == "possible_missing":
            add(evidence.frame_index, "high", "vlm_possible_missing", evidence.track_ids)
        elif evidence.verdict == "wrong_class":
            add(evidence.frame_index, "high", "vlm_wrong_class", evidence.track_ids)
        elif evidence.verdict == "bad_extent":
            add(evidence.frame_index, "high", "vlm_bad_extent", evidence.track_ids)
        else:
            add(evidence.frame_index, "normal", "vlm_uncertain", evidence.track_ids)
    for track in tracks.tracks:
        for observation in track.observations:
            if observation.score is not None and observation.score < 0.55:
                add(observation.frame_index, "high", "low_propagation_confidence", [track.track_id])
            if observation.truncated:
                add(observation.frame_index, "normal", "truncated", [track.track_id])
            if observation.occluded:
                add(observation.frame_index, "normal", "occluded", [track.track_id])
    for frame, signal in signals.items():
        if frame not in audit_by_frame:
            reasons = signal["reasons"]
            assert isinstance(reasons, set)
            reasons.add("vlm_evidence_missing")

    units = []
    for frame, signal in signals.items():
        reasons = signal["reasons"]
        track_ids = signal["tracks"]
        priority = str(signal["priority"])
        assert isinstance(reasons, set) and isinstance(track_ids, set)
        units.append(
            ReviewUnit(
                review_unit_id=f"frame:{frame:06d}",
                job_id=tracks.job_id,
                priority=priority,
                frame_start=frame,
                frame_end=frame,
                track_ids=sorted(track_ids),
                reason_codes=sorted(reasons),
            )
        )
    units.sort(key=lambda item: (PRIORITY_ORDER[item.priority], item.frame_start, item.review_unit_id))
    return ReviewQueue(
        job_id=tracks.job_id,
        track_set_sha256=audit.track_set_sha256,
        discovery_sha256=audit.discovery_sha256,
        audit_sha256=audit_sha256,
        units=units,
    )
