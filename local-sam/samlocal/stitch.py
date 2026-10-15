"""Join window-local SAM3 track ids into global track ids across overlapping windows.

For each prompt, tracks of the new window are matched to tracks of the previous
window on the shared overlap frames by mean box IoU (greedy, highest first).
Matched tracks keep their global id; the rest get new ids.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class Detection:
    frame: int  # global frame index
    local_id: int  # SAM3 object id inside one window session
    label: str
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    score: float
    mask_rle: dict | None = None
    track_id: int = -1  # global id after stitching


def box_iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    inter_w = min(a[2], b[2]) - max(a[0], b[0])
    inter_h = min(a[3], b[3]) - max(a[1], b[1])
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    inter = inter_w * inter_h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class TrackStitcher:
    iou_threshold: float = 0.5
    next_id: int = 1
    # label -> global frame -> global track id -> bbox, for the previous window only
    _previous: dict[str, dict[int, dict[int, tuple[float, ...]]]] = field(default_factory=dict)
    track_ids_by_label: dict[str, set[int]] = field(default_factory=lambda: defaultdict(set))

    def stitch(self, label: str, detections: list[Detection], overlap_frames: set[int]) -> list[Detection]:
        previous = self._previous.get(label, {})
        current_by_track: dict[int, dict[int, tuple[float, ...]]] = defaultdict(dict)
        for det in detections:
            current_by_track[det.local_id][det.frame] = det.bbox

        candidates: list[tuple[float, int, int]] = []
        for local_id, boxes in current_by_track.items():
            prior_ids = {gid for frame in overlap_frames for gid in previous.get(frame, {})}
            for gid in prior_ids:
                ious = [box_iou(box, previous[frame][gid]) for frame, box in boxes.items()
                        if frame in overlap_frames and gid in previous.get(frame, {})]
                if ious:
                    candidates.append((sum(ious) / len(ious), local_id, gid))
        mapping: dict[int, int] = {}
        used: set[int] = set()
        for iou, local_id, gid in sorted(candidates, reverse=True):
            if iou < self.iou_threshold:
                break
            if local_id in mapping or gid in used:
                continue
            mapping[local_id] = gid
            used.add(gid)
        for local_id in sorted(current_by_track):
            if local_id not in mapping:
                mapping[local_id] = self.next_id
                self.next_id += 1

        latest: dict[int, dict[int, tuple[float, ...]]] = defaultdict(dict)
        for det in detections:
            det.track_id = mapping[det.local_id]
            latest[det.frame][det.track_id] = det.bbox
            self.track_ids_by_label[label].add(det.track_id)
        self._previous[label] = latest
        return detections
