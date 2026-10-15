"""Independent RT-DETR evidence and deterministic missing-object discovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from src.models.video import (
    BoundingBox2D,
    ComponentVersion,
    Detection2D,
    DetectionSet,
    DetectorModelRequest,
    DetectorModelResponse,
    DiscoveryEvent,
    DiscoveryResult,
    FrameManifest,
    TrackObservation,
    TrackSet,
    VideoTrack,
)


class VideoDiscoveryError(RuntimeError):
    """An actionable detector or discovery failure safe to persist."""


class VideoDetector(Protocol):
    def detect(self, *, manifest: FrameManifest, manifest_sha256: str, labels: dict[int, str]) -> DetectionSet: ...


def discover_missing_objects(
    *,
    manifest: FrameManifest,
    propagated: TrackSet,
    detections: DetectionSet,
    propagated_track_sha256: str,
    match_iou: float = 0.5,
    proposal_iou: float = 0.3,
    max_gap: int = 2,
) -> DiscoveryResult:
    if propagated.job_id != manifest.job_id or detections.job_id != manifest.job_id:
        raise VideoDiscoveryError("discovery inputs belong to different video jobs")
    unmatched = _unmatched_detections(propagated, detections, match_iou)
    proposals = _track_proposals(unmatched, proposal_iou=proposal_iou, max_gap=max_gap)
    events = _proposal_events(proposals)
    events.extend(_propagation_events(manifest, propagated))
    return DiscoveryResult(
        job_id=manifest.job_id,
        manifest_sha256=detections.manifest_sha256,
        propagated_track_sha256=propagated_track_sha256,
        component_versions=[
            detections.component_version,
            ComponentVersion(component="missing-object-matcher", version="iou-greedy/v1"),
        ],
        proposal_tracks=proposals,
        events=sorted(events, key=lambda item: (item.frame_index, item.event_id)),
    )


def _normalize_detections(
    raw: DetectorModelResponse,
    request: DetectorModelRequest,
    labels: dict[int, str],
    expected_model_id: str,
) -> DetectionSet:
    if raw.job_id != request.job_id or raw.manifest_sha256 != request.manifest_sha256:
        raise VideoDiscoveryError("detector response provenance does not match the request")
    if (
        raw.component_version.component != "rtdetr-object-detector"
        or raw.component_version.version != expected_model_id
    ):
        raise VideoDiscoveryError("detector response has an invalid component version")
    normalized_targets = {_normalize(name): (label_id, name) for label_id, name in labels.items()}
    if len(normalized_targets) != len(labels):
        raise VideoDiscoveryError("CVAT labels are ambiguous after normalization")
    detections: list[Detection2D] = []
    for item in raw.detections:
        if item.frame_index >= len(request.frames):
            raise VideoDiscoveryError("detector response contains an unknown frame")
        frame = request.frames[item.frame_index]
        if item.bbox.xbr > frame.width or item.bbox.ybr > frame.height:
            raise VideoDiscoveryError(f"detector box exceeds frame {item.frame_index} bounds")
        target = _match_label(item.model_label_name, normalized_targets)
        if target is None or item.score < request.score_threshold:
            continue
        detections.append(
            Detection2D(
                frame_index=item.frame_index,
                label_id=target[0],
                label_name=target[1],
                model_label_name=item.model_label_name,
                bbox=item.bbox,
                score=item.score,
            )
        )
    detections.sort(key=lambda item: (item.frame_index, item.label_name, -item.score, item.bbox.xtl, item.bbox.ytl))
    return DetectionSet(
        job_id=request.job_id,
        manifest_sha256=request.manifest_sha256,
        component_version=raw.component_version,
        detections=detections,
    )


def _unmatched_detections(propagated: TrackSet, detections: DetectionSet, threshold: float) -> list[Detection2D]:
    visible: dict[int, list[tuple[str, TrackObservation]]] = {}
    labels = {track.track_id: track.label_name for track in propagated.tracks}
    for track in propagated.tracks:
        for observation in track.observations:
            if not observation.outside:
                visible.setdefault(observation.frame_index, []).append((track.track_id, observation))
    unmatched: list[Detection2D] = []
    for detection in detections.detections:
        candidates = visible.get(detection.frame_index, [])
        best = max(
            (
                _box_iou(detection.bbox, observation.bbox)
                for track_id, observation in candidates
                if labels[track_id] == detection.label_name
            ),
            default=0.0,
        )
        if best < threshold:
            unmatched.append(detection)
    return unmatched


@dataclass
class _Proposal:
    track_id: str
    label_id: int
    label_name: str
    observations: list[TrackObservation] = field(default_factory=list)


def _track_proposals(detections: list[Detection2D], *, proposal_iou: float, max_gap: int) -> list[VideoTrack]:
    active: list[_Proposal] = []
    proposals: list[_Proposal] = []
    next_id = 1
    for detection in detections:
        candidates = [
            proposal
            for proposal in active
            if proposal.label_name == detection.label_name
            and 0 < detection.frame_index - proposal.observations[-1].frame_index <= max_gap
        ]
        ranked = sorted(
            ((_box_iou(detection.bbox, item.observations[-1].bbox), item.track_id, item) for item in candidates),
            key=lambda item: (-item[0], item[1]),
        )
        proposal = ranked[0][2] if ranked and ranked[0][0] >= proposal_iou else None
        if proposal is None:
            proposal = _Proposal(
                track_id=f"auto:missing:{next_id:06d}",
                label_id=detection.label_id,
                label_name=detection.label_name,
            )
            next_id += 1
            proposals.append(proposal)
            active.append(proposal)
        proposal.observations.append(
            TrackObservation(frame_index=detection.frame_index, bbox=detection.bbox, score=detection.score)
        )
        active = [item for item in active if detection.frame_index - item.observations[-1].frame_index <= max_gap]
    return [
        VideoTrack(
            track_id=item.track_id,
            label_id=item.label_id,
            label_name=item.label_name,
            source="auto",
            observations=item.observations,
        )
        for item in proposals
    ]


def _proposal_events(proposals: list[VideoTrack]) -> list[DiscoveryEvent]:
    return [
        DiscoveryEvent(
            event_id=f"missing:{track.track_id}",
            event_type="possible_missing_object",
            frame_index=track.observations[0].frame_index,
            track_ids=[track.track_id],
            label_name=track.label_name,
            bbox=track.observations[0].bbox,
            score=track.observations[0].score,
            reason="reliable detector track does not match any propagated seed track",
        )
        for track in proposals
    ]


def _propagation_events(manifest: FrameManifest, propagated: TrackSet) -> list[DiscoveryEvent]:
    events: list[DiscoveryEvent] = []
    first_cut = next(
        (frame.frame_index for frame in manifest.frames[1:] if frame.scene_cut_before), manifest.frame_count
    )
    for track in propagated.tracks:
        observations = {item.frame_index: item for item in track.observations}
        visible_frames = [index for index, item in observations.items() if not item.outside and index < first_cut]
        if visible_frames and max(visible_frames) < first_cut - 1:
            lost_at = max(visible_frames) + 1
            events.append(
                DiscoveryEvent(
                    event_id=f"lost:{track.track_id}:{lost_at}",
                    event_type="lost_seed",
                    frame_index=lost_at,
                    track_ids=[track.track_id],
                    label_name=track.label_name,
                    reason="seeded track disappeared before the end of its scene",
                )
            )
        was_outside = False
        for index in sorted(observations):
            item = observations[index]
            if item.outside:
                was_outside = True
            elif was_outside:
                events.append(
                    DiscoveryEvent(
                        event_id=f"reappearance:{track.track_id}:{index}",
                        event_type="unexpected_reappearance",
                        frame_index=index,
                        track_ids=[track.track_id],
                        label_name=track.label_name,
                        bbox=item.bbox,
                        score=item.score,
                        reason="seeded track became visible after an outside interval",
                    )
                )
                was_outside = False
    for frame in manifest.frames[1:]:
        if frame.scene_cut_before:
            events.append(
                DiscoveryEvent(
                    event_id=f"scene-cut:{frame.frame_index}",
                    event_type="scene_cut_without_seed",
                    frame_index=frame.frame_index,
                    track_ids=[],
                    reason="scene cut has no human conditioning prompt",
                )
            )
    return events


def _match_label(model_name: str, targets: dict[str, tuple[int, str]]) -> tuple[int, str] | None:
    normalized = _normalize(model_name)
    aliases = {
        "pedestrian": "person",
        "people": "person",
        "motorbike": "motorcycle",
        "bike": "bicycle",
    }
    normalized = aliases.get(normalized, normalized)
    for target_name, target in targets.items():
        canonical_target = aliases.get(target_name, target_name)
        if canonical_target == normalized:
            return target
    return None


def _normalize(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", "-").split())


def _box_iou(left: BoundingBox2D, right: BoundingBox2D) -> float:
    intersection_width = max(0.0, min(left.xbr, right.xbr) - max(left.xtl, right.xtl))
    intersection_height = max(0.0, min(left.ybr, right.ybr) - max(left.ytl, right.ytl))
    intersection = intersection_width * intersection_height
    left_area = (left.xbr - left.xtl) * (left.ybr - left.ytl)
    right_area = (right.xbr - right.xtl) * (right.ybr - right.ytl)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0
