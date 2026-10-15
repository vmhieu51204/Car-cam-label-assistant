"""Track-local evidence for human navigation, not annotation acceptance."""

from typing import Any

from src.models.video import FrameManifest, TrackSet
from src.services.video.sequence_review import SequenceReviewConfig, score_tracks
from src.services.video.stitching import ChunkObservation, UnifiedTrack


def track_triage(tracks: TrackSet, manifest: FrameManifest, config: SequenceReviewConfig) -> list[dict[str, Any]]:
    if tracks.job_id != manifest.job_id:
        raise ValueError("triage provenance differs")
    cuts = {f.frame_index for f in manifest.frames if f.scene_cut_before}
    unified = [
        UnifiedTrack(
            t.track_id,
            t.label_name,
            [
                ChunkObservation(o.frame_index, (o.bbox.xtl, o.bbox.ytl, o.bbox.xbr, o.bbox.ybr))
                for o in t.observations
                if not o.outside
            ],
            t.needs_review,
            list(t.review_reasons),
        )
        for t in tracks.tracks
    ]
    # Class policy is a universal instruction, not localized evidence. Keep it in legacy queue only.
    reviews = {r.track_id: r for r in score_tracks(unified, scene_cut_frames=cuts, config=config)}
    rows = []
    for t in unified:
        if not t.observations:
            continue
        risk = reviews[t.track_id]
        # Score single observation/pair to localize existing signals without copying frame-union reasons.
        local = []
        for i, obs in enumerate(t.observations):
            pair = UnifiedTrack(t.track_id, t.label, t.observations[max(0, i - 1) : i + 1], False, [])
            localized = score_tracks([pair], scene_cut_frames=cuts, config=config)[0]
            score = sum(config.weights.get(r, 0) for r in localized.reasons if r != "short_track")
            local.append((score, -obs.frame, obs.frame))
        representative = max(local)[2]
        frame = manifest.frames[representative]
        track_level_flag = bool(
            set(risk.reasons) & {"short_track", "stitch_ambiguity", "ambiguous_overlap", "unmatched_chunk"}
        )
        rows.append(
            {
                "track_id": t.track_id,
                "class": t.label,
                "score": risk.score,
                "level": "high"
                if risk.priority in {"critical", "high"}
                else "medium"
                if risk.priority == "normal"
                else "low",
                "reasons": list(risk.reasons),
                "source_frame": frame.source_frame_index if frame.source_frame_index is not None else representative,
                "flagged_frames": len(t.observations) if track_level_flag else sum(score > 0 for score, _, _ in local),
                "observations": len(t.observations),
            }
        )
    return sorted(rows, key=lambda r: (-r["score"], r["track_id"]))
