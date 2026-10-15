"""Keep both overlap predictions and join chunk-local tracks conservatively."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from functools import lru_cache
from statistics import median
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class MaskRLE:
    height: int
    width: int
    counts: tuple[int, ...]
    sha256: str


def _mask_bytes(height: int, width: int, counts: tuple[int, ...]) -> bytes:
    return json.dumps([height, width, counts], separators=(",", ":")).encode("ascii")


def encode_mask(mask: np.ndarray) -> MaskRLE:
    """Uncompressed row-major RLE with a content hash, suitable for JSON artifacts."""
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError("mask must be a nonempty 2D array")
    flat = np.asarray(mask, dtype=bool).reshape(-1)
    transitions = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    counts = np.diff(np.concatenate(([0], transitions, [flat.size]))).tolist()
    if flat[0]:
        counts.insert(0, 0)
    shape = mask.shape
    parts = tuple(counts)
    return MaskRLE(shape[0], shape[1], parts, hashlib.sha256(_mask_bytes(shape[0], shape[1], parts)).hexdigest())


@lru_cache(maxsize=128)
def decode_mask(rle: MaskRLE) -> np.ndarray:
    if (
        rle.height < 1
        or rle.width < 1
        or any(count < 0 for count in rle.counts)
        or sum(rle.counts) != rle.height * rle.width
        or hashlib.sha256(_mask_bytes(rle.height, rle.width, rle.counts)).hexdigest() != rle.sha256
    ):
        raise ValueError("invalid or corrupted mask RLE")
    pixels = np.empty(rle.height * rle.width, dtype=bool)
    offset = 0
    for index, count in enumerate(rle.counts):
        pixels[offset : offset + count] = bool(index % 2)
        offset += count
    return pixels.reshape(rle.height, rle.width)


@dataclass(frozen=True)
class ChunkObservation:
    frame: int
    bbox: tuple[float, float, float, float]
    mask: MaskRLE | None = None
    score: float | None = None


@dataclass(frozen=True)
class ChunkTrack:
    window_id: int
    local_track_id: str
    label: str
    observations: list[ChunkObservation]


@dataclass(frozen=True)
class StitchConfig:
    """Initial uncalibrated thresholds; select on DEV, evaluate held-out only at the end."""

    min_overlap_iou: float = 0.5
    ambiguity_margin: float = 0.1
    min_common_frames: int = 2
    decision_log_enabled: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.min_overlap_iou <= 1 or not 0 <= self.ambiguity_margin <= 1:
            raise ValueError("stitch thresholds must be in [0, 1]")
        if self.min_common_frames < 1:
            raise ValueError("min_common_frames must be positive")


@dataclass
class UnifiedTrack:
    track_id: str
    label: str
    observations: list[ChunkObservation] = field(default_factory=list)
    needs_review: bool = False
    review_reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class StitchDecision:
    window_id: int
    local_track_id: str
    global_track_id: str
    action: str
    reason: str
    mean_iou: float | None
    common_frames: int
    needs_review: bool


@dataclass
class StitchResult:
    tracks: list[UnifiedTrack]
    decisions: list[StitchDecision]
    boundary_logs: list[dict] = field(default_factory=list)


def _box_iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    b = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / (a + b - intersection) if a + b > intersection else 0.0


def _overlap(left: ChunkTrack, right: ChunkTrack, min_frames: int) -> tuple[float, int]:
    previous = {item.frame: item for item in left.observations}
    current = {item.frame: item for item in right.observations}
    common = sorted(previous.keys() & current.keys())
    if len(common) < min_frames:
        return 0.0, len(common)
    values = []
    for frame in common:
        a, b = previous[frame], current[frame]
        if a.mask is not None and b.mask is not None:
            if (a.mask.height, a.mask.width) != (b.mask.height, b.mask.width):
                raise ValueError("overlap masks have different dimensions")
            first, second = decode_mask(a.mask), decode_mask(b.mask)
            union = int(np.count_nonzero(first | second))
            values.append(int(np.count_nonzero(first & second)) / union if union else 0.0)
        else:
            values.append(_box_iou(a.bbox, b.bbox))
    return float(median(values)), len(common)


def stitch_tracks(
    chunks: list[ChunkTrack],
    config: StitchConfig | None = None,
    *,
    reset_windows: set[int] | None = None,
    planned_window_ids: list[int] | None = None,
) -> StitchResult:
    """Hungarian one-to-one matches on the *same* overlap frames; ambiguity creates review work."""
    config = config or StitchConfig()
    by_window: dict[int, list[ChunkTrack]] = {}
    for track in chunks:
        if not track.label or not track.local_track_id or not track.observations:
            raise ValueError("chunk track is incomplete")
        frames = [item.frame for item in track.observations]
        if frames != sorted(set(frames)):
            raise ValueError("chunk observations must have unique ordered frames")
        by_window.setdefault(track.window_id, []).append(track)
    planned = (
        planned_window_ids
        if planned_window_ids is not None
        else (list(range(min(by_window), max(by_window) + 1)) if by_window else [])
    )
    if planned != sorted(set(planned)) or not set(by_window) <= set(planned):
        raise ValueError("planned window IDs must be ordered unique and cover chunk windows")
    if not by_window:
        return StitchResult(
            [],
            [],
            _complete_boundary_logs([], by_window, planned, reset_windows or set())
            if config.decision_log_enabled
            else [],
        )
    unified: dict[str, UnifiedTrack] = {}
    decisions: list[StitchDecision] = []
    boundary_logs: list[dict] = []
    previous: list[ChunkTrack] = []
    previous_ids: dict[str, str] = {}
    previous_window_id: int | None = None
    next_id = 1

    def new_track(track: ChunkTrack, *, review: bool, reason: str) -> str:
        nonlocal next_id
        identity = f"auto:{next_id:06d}"
        next_id += 1
        unified[identity] = UnifiedTrack(
            identity, track.label, list(track.observations), review, [reason] if review else []
        )
        return identity

    for window_id, current in sorted(by_window.items()):
        boundary: dict[str, Any] = {
            "previous_window_id": previous_window_id,
            "window_id": window_id,
            "previous_track_count": len(previous),
            "current_track_count": len(current),
            "candidate_pair_count": 0,
            "eligible_pair_count": 0,
            "tracks": [],
        }
        scene_reset = window_id in (reset_windows or set())
        if scene_reset:
            previous = []
            previous_ids = {}
        if len({item.local_track_id for item in current}) != len(current):
            raise ValueError("duplicate local track IDs within a window")
        empty_gap = previous_window_id is not None and window_id > previous_window_id + 1
        if empty_gap:
            # A model can legitimately return no instances for an entire chunk.
            # There is no shared-frame evidence across that gap, so never join IDs.
            previous = []
            previous_ids = {}
        current_ids: dict[str, str] = {}
        if not previous:
            for track in current:
                current_ids[track.local_track_id] = new_track(
                    track,
                    review=empty_gap or scene_reset,
                    reason="scene_cut" if scene_reset else "empty_chunk_gap" if empty_gap else "",
                )
                if empty_gap or scene_reset:
                    decisions.append(
                        StitchDecision(
                            window_id,
                            track.local_track_id,
                            current_ids[track.local_track_id],
                            "new_id",
                            "scene_cut" if scene_reset else "empty_chunk_gap",
                            None,
                            0,
                            True,
                        )
                    )
                if config.decision_log_enabled:
                    boundary["tracks"].append(
                        {
                            "local_track_id": track.local_track_id,
                            "global_track_id": current_ids[track.local_track_id],
                            "candidate_count": 0,
                            "eligible_candidate_count": 0,
                            "best_iou": None,
                            "second_iou": None,
                            "margin": None,
                            "selected_iou": None,
                            "result": "new_id",
                            "reason": "scene_cut"
                            if scene_reset
                            else "empty_chunk_gap"
                            if empty_gap
                            else "no_previous_tracks",
                        }
                    )
        else:
            scores = np.zeros((len(previous), len(current)), dtype=float)
            counts = np.zeros((len(previous), len(current)), dtype=int)
            for row_index, left in enumerate(previous):
                for j, right in enumerate(current):
                    if left.label == right.label:
                        scores[row_index, j], counts[row_index, j] = _overlap(left, right, config.min_common_frames)
            proposed: dict[int, int] = {}
            if scores.size:
                rows, columns = linear_sum_assignment(1 - scores)
                proposed = dict(zip(columns.tolist(), rows.tolist(), strict=True))
            for j, track in enumerate(current):
                i = proposed.get(j)
                score = float(scores[i, j]) if i is not None else 0.0
                eligible = i is not None and score >= config.min_overlap_iou
                competing = eligible and (
                    any(
                        k != i
                        and scores[k, j] >= config.min_overlap_iou
                        and score - scores[k, j] <= config.ambiguity_margin
                        for k in range(len(previous))
                    )
                    or any(
                        k != j
                        and scores[i, k] >= config.min_overlap_iou
                        and score - scores[i, k] <= config.ambiguity_margin
                        for k in range(len(current))
                    )
                )
                if i is not None and eligible and not competing:
                    identity = previous_ids[previous[i].local_track_id]
                    merged = {item.frame: item for item in unified[identity].observations}
                    merged.update({item.frame: item for item in track.observations})
                    unified[identity].observations = [merged[frame] for frame in sorted(merged)]
                    action, reason, review = "matched", "overlap_agreement", False
                else:
                    reason = "ambiguous_overlap" if competing else "unmatched_chunk"
                    review = True
                    identity = new_track(track, review=review, reason=reason)
                    action = "new_id"
                current_ids[track.local_track_id] = identity
                decisions.append(
                    StitchDecision(
                        window_id,
                        track.local_track_id,
                        identity,
                        action,
                        reason,
                        score if i is not None else None,
                        int(counts[i, j]) if i is not None else 0,
                        review,
                    )
                )
                if config.decision_log_enabled:
                    candidates = [k for k, left in enumerate(previous) if left.label == track.label]
                    ranked = sorted(
                        (float(scores[k, j]) for k in candidates if counts[k, j] >= config.min_common_frames),
                        reverse=True,
                    )
                    eligible_count = sum(scores[k, j] >= config.min_overlap_iou for k in candidates)
                    boundary["candidate_pair_count"] += len(candidates)
                    boundary["eligible_pair_count"] += int(eligible_count)
                    alternatives = [float(scores[k, j]) for k in candidates if k != i]
                    row_alternatives = (
                        [float(scores[i, k]) for k in range(len(current)) if k != j and current[k].label == track.label]
                        if i is not None
                        else []
                    )
                    boundary["tracks"].append(
                        {
                            "local_track_id": track.local_track_id,
                            "global_track_id": identity,
                            "candidate_count": len(candidates),
                            "eligible_candidate_count": int(eligible_count),
                            "best_iou": ranked[0] if ranked else None,
                            "second_iou": ranked[1] if len(ranked) > 1 else None,
                            "margin": ranked[0] - ranked[1] if len(ranked) > 1 else None,
                            "selected_previous_local_id": previous[i].local_track_id if i is not None else None,
                            "selected_iou": score if i is not None else None,
                            "selected_common_frames": int(counts[i, j]) if i is not None else 0,
                            "selected_column_margin": score - max(alternatives)
                            if i is not None and alternatives
                            else None,
                            "selected_row_margin": score - max(row_alternatives) if row_alternatives else None,
                            "result": "ambiguous" if competing else action,
                            "reason": reason,
                            "candidates": [
                                {
                                    "previous_local_id": previous[k].local_track_id,
                                    "global_track_id": previous_ids[previous[k].local_track_id],
                                    "iou": float(scores[k, j]),
                                    "common_frames": int(counts[k, j]),
                                }
                                for k in candidates
                            ],
                        }
                    )
        if config.decision_log_enabled:
            boundary_logs.append(boundary)
        previous, previous_ids, previous_window_id = current, current_ids, window_id
    return StitchResult(
        list(unified.values()),
        decisions,
        _complete_boundary_logs(boundary_logs, by_window, planned, reset_windows or set())
        if config.decision_log_enabled
        else [],
    )


def _complete_boundary_logs(
    logs: list[dict], by_window: dict[int, list[ChunkTrack]], planned: list[int], resets: set[int]
) -> list[dict]:
    """Add empty planned boundaries to logs only. Never feed them into association."""
    indexed = {row["window_id"]: row for row in logs}
    result = []
    for previous, current in zip(planned, planned[1:]):
        row = indexed.get(current)
        if row is not None:
            row = {
                **row,
                "association_previous_window_id": row["previous_window_id"],
                "previous_window_id": previous,
                "previous_track_count": len(by_window.get(previous, [])),
            }
        else:
            reason = (
                "scene_cut"
                if current in resets
                else "empty_window"
                if not by_window.get(current)
                else "no_previous_tracks"
            )
            row = {
                "previous_window_id": previous,
                "window_id": current,
                "previous_track_count": len(by_window.get(previous, [])),
                "current_track_count": len(by_window.get(current, [])),
                "candidate_pair_count": 0,
                "eligible_pair_count": 0,
                "tracks": [],
                "reason": reason,
            }
        result.append(row)
    return result
