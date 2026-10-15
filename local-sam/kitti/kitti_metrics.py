"""KITTI tracking ground truth and CPU-only scoring of local-sam ``predictions.json`` (no torch, no GPU).

Preprocessing mirrors the repo's KITTI-style diagnostic (``src/services/evaluation/references/kitti.py``):

* profile ``kitti2`` (official KITTI tracking classes): ``Car -> car``, ``Pedestrian -> person``; ``Van`` (for car)
  and ``Person`` (sitting person, for person) are distractors: a prediction matched to one is ignored;
* profile ``kitti5`` (wider autolabel test): ``car``, ``van``, ``truck`` (KITTI ``Truck`` includes city buses),
  ``pedestrian``, ``cyclist`` (rider + bicycle box); only ``Person`` (sitting) stays a distractor, so a ``van``
  prompt that fires on a ``Car`` is a real error and shows up in the confusion matrix;
* a GT box is scored only when ``truncated == 0`` and ``occluded <= 2``; otherwise it is ignored together
  with the prediction matched to it;
* unmatched predictions are ignored when their height is <= 25 px or >50% of them lies in a ``DontCare`` region.

On top of that this module adds CLEAR-MOT (MOTA/MOTP/IDSW/Frag/MT/ML, TrackEval matching with continuity
preference), IDF1, HOTA (a re-implementation of TrackEval's, for every class of the profile; TrackEval itself only
scores KITTI car/pedestrian) and recall diagnostics for driving: recall by distance / box height / occlusion /
truncation, frames until a GT track is first detected, F2 and track coverage. ``write_kitti_results`` exports
files that the official TrackEval scores (``trackeval_kitti.py``), to cross-check the car/pedestrian numbers.
"""

from __future__ import annotations

import math
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

# profile -> native KITTI class -> (evaluated class, scored); scored=False marks a distractor of that class.
PROFILES: dict[str, dict[str, tuple[str, bool]]] = {
    "kitti2": {"Car": ("car", True), "Van": ("car", False),
               "Pedestrian": ("person", True), "Person": ("person", False)},
    "kitti5": {"Car": ("car", True), "Van": ("van", True), "Truck": ("truck", True),
               "Pedestrian": ("pedestrian", True), "Person": ("pedestrian", False), "Cyclist": ("cyclist", True)},
}
PROMPTS = {
    "kitti2": {"car": "car", "pedestrian": "person"},
    # default text_prompts are the first five; the others are variants mapped to the same classes
    "kitti5": {"car": "car", "van": "van", "truck": "truck", "pedestrian": "pedestrian", "cyclist": "cyclist",
               "bus": "truck", "person riding a bicycle": "cyclist", "person": "pedestrian", "minivan": "van"},
}
DEFAULT_PROMPTS = {"kitti2": ["car", "pedestrian"], "kitti5": ["car", "van", "truck", "pedestrian", "cyclist"]}
CLASSES = ("car", "person")  # kitti2, kept for older callers
DEDUP_IOU = 0.7  # two prompts mapped to one class (truck + bus) on the same object -> keep the higher score
MIN_HEIGHT = 25.0
IOU_THRESHOLD = 0.5
DETECTION_THRESHOLDS = (0.5, 0.7)
HOTA_ALPHAS = np.arange(0.05, 0.99, 0.05)  # same 19 localisation thresholds as TrackEval
EPS = np.finfo("float").eps
# (name, low, high): half-open [low, high). Distance = camera z of the KITTI 3D location, in metres.
DISTANCE_BINS = (("<20m", 0.0, 20.0), ("20-40m", 20.0, 40.0), (">=40m", 40.0, math.inf))
HEIGHT_BINS = (("<25px", 0.0, 25.0), ("25-40px", 25.0, 40.0), ("40-80px", 40.0, 80.0), (">=80px", 80.0, math.inf))
SOURCE_FPS = 10.0  # KITTI tracking frame rate, for first-detection delay in milliseconds


@dataclass(frozen=True)
class Box:
    frame: int  # KITTI source frame number
    label: str  # evaluated class of the profile (car / person, or car / van / truck / pedestrian / cyclist)
    xyxy: tuple[float, float, float, float]
    track_id: str
    score: float | None = None


@dataclass(frozen=True)
class GtItem:
    box: Box
    valid: bool
    native_class: str
    occluded: int = 0  # KITTI 0 visible, 1 partly, 2 largely occluded, 3 unknown
    truncated: float = 0.0
    distance_m: float | None = None  # camera z of the 3D location; None when the label has no 3D box


@dataclass
class Reference:
    items: list[GtItem]
    dontcare: dict[int, list[tuple[float, float, float, float]]]
    profile: str = "kitti2"
    # every non-DontCare row (frame, native class, box), including classes the profile does not score
    native_rows: list[tuple[int, str, tuple[float, float, float, float]]] = field(default_factory=list)

    @property
    def classes(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(label for label, _ in PROFILES[self.profile].values()))

    @property
    def target_natives(self) -> set[str]:
        return {native for native, (_, scored) in PROFILES[self.profile].items() if scored}

    def valid_boxes(self, frames: set[int]) -> list[Box]:
        return [item.box for item in self.items if item.valid and item.box.frame in frames]

    def target_boxes(self, frames: set[int]) -> list[Box]:
        """Every Car/Pedestrian row, ignoring difficulty (for the raw, unfiltered numbers)."""
        targets = self.target_natives
        return [item.box for item in self.items if item.native_class in targets and item.box.frame in frames]


def parse_labels(text: str, *, sequence: str = "", profile: str = "kitti2") -> Reference:
    mapping = PROFILES[profile]
    native_rows: list[tuple[int, str, tuple[float, float, float, float]]] = []
    items: list[GtItem] = []
    dontcare: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    for number, line in enumerate(text.splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) not in (17, 18):  # 18 = result files that carry a score
            raise ValueError(f"{sequence} label line {number}: expected 17 columns, got {len(fields)}")
        frame, track_id, native = int(fields[0]), fields[1], fields[2]
        truncated, occluded = float(fields[3]), int(float(fields[4]))
        xyxy = tuple(float(value) for value in fields[6:10])
        if not xyxy[0] < xyxy[2] or not xyxy[1] < xyxy[3]:
            continue  # degenerate box in the native file; nothing can match it
        if native == "DontCare":
            dontcare[frame].append(xyxy)  # type: ignore[arg-type]
            continue
        native_rows.append((frame, native, xyxy))  # type: ignore[arg-type]
        if native in mapping:
            label, scored = mapping[native]
            valid = scored and truncated <= 0 and occluded <= 2
            depth = float(fields[15])
            items.append(GtItem(Box(frame, label, xyxy, f"gt:{label}:{track_id}"), valid, native,  # type: ignore[arg-type]
                                occluded, truncated, depth if depth > -999 else None))
    return Reference(items, dict(dontcare), profile, native_rows)


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Continuous xyxy IoU (no +1 pixel), same convention as the repo matcher."""
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def ioa(box: tuple[float, ...], region: tuple[float, ...]) -> float:
    inter = max(0.0, min(box[2], region[2]) - max(box[0], region[0])) * max(
        0.0, min(box[3], region[3]) - max(box[1], region[1]))
    area = (box[2] - box[0]) * (box[3] - box[1])
    return inter / area if area > 0 else 0.0


def _iou_matrix(left: list[Box], right: list[Box]) -> np.ndarray:
    return np.array([[iou(a.xyxy, b.xyxy) for b in right] for a in left], dtype=float).reshape(len(left), len(right))


def _hungarian(similarity: np.ndarray, threshold: float, score: np.ndarray | None = None) -> list[tuple[int, int]]:
    """Max-score one-to-one assignment keeping only pairs with similarity >= threshold."""
    if similarity.size == 0:
        return []
    score = similarity.copy() if score is None else score.copy()
    score[similarity < threshold - 1e-9] = 0.0
    rows, cols = linear_sum_assignment(-score)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if score[r, c] > 0]


def preprocess(reference: Reference, predictions: list[Box], frames: set[int]) -> tuple[list[Box], list[Box], Counter]:
    """KITTI-style ignore rules; returns (scored GT, kept predictions, removed-prediction reasons)."""
    gt_by_key: dict[tuple[int, str], list[GtItem]] = defaultdict(list)
    for item in reference.items:
        if item.box.frame in frames:
            gt_by_key[(item.box.frame, item.box.label)].append(item)
    pred_by_key: dict[tuple[int, str], list[Box]] = defaultdict(list)
    for box in predictions:
        if box.frame in frames:
            pred_by_key[(box.frame, box.label)].append(box)
    kept: list[Box] = []
    removed: Counter = Counter()
    for (frame, label), observed in pred_by_key.items():
        candidates = gt_by_key.get((frame, label), [])
        matched: dict[int, int] = {}
        if candidates:
            similarity = _iou_matrix([c.box for c in candidates], observed)
            matched = {col: row for row, col in _hungarian(similarity, IOU_THRESHOLD)}
        for index, box in enumerate(observed):
            if index in matched:
                if candidates[matched[index]].valid:
                    kept.append(box)
                else:
                    removed["distractor_or_difficult"] += 1
            elif box.xyxy[3] - box.xyxy[1] <= MIN_HEIGHT:
                removed["small"] += 1
            elif any(ioa(box.xyxy, region) > 0.5 for region in reference.dontcare.get(frame, ())):
                removed["dontcare"] += 1
            else:
                kept.append(box)
    return reference.valid_boxes(frames), kept, removed


def _by_frame(boxes: list[Box]) -> dict[int, list[Box]]:
    grouped: dict[int, list[Box]] = defaultdict(list)
    for box in boxes:
        grouped[box.frame].append(box)
    return grouped


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _match_frames(gt: list[Box], pred: list[Box], frames: set[int], threshold: float) -> list[tuple[Box, Box, float]]:
    """Per-frame Hungarian matching, no identity: (gt box, predicted box, IoU) pairs."""
    gt_frames, pred_frames = _by_frame(gt), _by_frame(pred)
    pairs: list[tuple[Box, Box, float]] = []
    for frame in frames:
        g, p = gt_frames.get(frame, []), pred_frames.get(frame, [])
        if g and p:
            similarity = _iou_matrix(g, p)
            pairs += [(g[r], p[c], float(similarity[r, c])) for r, c in _hungarian(similarity, threshold)]
    return pairs


def _f_scores(precision: float | None, recall: float | None, any_boxes: bool) -> tuple[float | None, float | None]:
    """F1 and F2 (recall weighted twice as much as precision)."""
    if not (precision and recall):
        return (0.0, 0.0) if any_boxes else (None, None)
    return 2 * precision * recall / (precision + recall), 5 * precision * recall / (4 * precision + recall)


def detection(gt: list[Box], pred: list[Box], frames: set[int], threshold: float) -> dict[str, Any]:
    """Per-frame Hungarian matching, no identity; precision/recall/F1/F2/mean IoU."""
    pairs = _match_frames(gt, pred, frames, threshold)
    tp, iou_sum = len(pairs), sum(overlap for _, _, overlap in pairs)
    fp, fn = len(pred) - tp, len(gt) - tp
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    f1, f2 = _f_scores(precision, recall, bool(gt or pred))
    return {"gt": len(gt), "pred": len(pred), "tp": tp, "fp": fp, "fn": fn, "precision": precision,
            "recall": recall, "f1": f1, "f2": f2, "mean_iou": _ratio(iou_sum, tp)}


def clear_mot(gt: list[Box], pred: list[Box], frames: set[int], threshold: float = IOU_THRESHOLD) -> dict[str, Any]:
    """CLEAR-MOT with TrackEval's matching: Hungarian on IoU plus a bonus for keeping last frame's pairs."""
    gt_frames, pred_frames = _by_frame(gt), _by_frame(pred)
    tp = fp = fn = idsw = frag = 0
    iou_sum = 0.0
    last_ever: dict[str, str] = {}
    previous: dict[str, str] = {}
    gt_length: Counter = Counter(box.track_id for box in gt)
    matched_length: Counter = Counter()
    was_tracked: dict[str, bool] = {}  # tracked state at the GT track's last present frame
    for frame in sorted(frames):
        g, p = gt_frames.get(frame, []), pred_frames.get(frame, [])
        pairs: list[tuple[int, int]] = []
        similarity = _iou_matrix(g, p)
        if g and p:
            bonus = np.array([[1000.0 if previous.get(a.track_id) == b.track_id else 0.0 for b in p] for a in g])
            pairs = _hungarian(similarity, threshold, similarity + bonus)
        current: dict[str, str] = {}
        for r, c in pairs:
            gt_id, pred_id = g[r].track_id, p[c].track_id
            if gt_id in last_ever and last_ever[gt_id] != pred_id:
                idsw += 1
            last_ever[gt_id] = current[gt_id] = pred_id
            iou_sum += similarity[r, c]
            matched_length[gt_id] += 1
        matched_ids = set(current)
        for box in g:
            tracked = box.track_id in matched_ids
            if tracked and was_tracked.get(box.track_id) is False and matched_length[box.track_id] > 1:
                frag += 1  # tracked -> lost -> tracked again
            was_tracked[box.track_id] = tracked
        tp += len(pairs)
        fn += len(g) - len(pairs)
        fp += len(p) - len(pairs)
        previous = current
    ratios = {track: matched_length[track] / length for track, length in gt_length.items()}
    return {"gt": len(gt), "pred": len(pred), "tp": tp, "fp": fp, "fn": fn, "idsw": idsw, "frag": frag,
            "mota": 1 - (fn + fp + idsw) / len(gt) if gt else None, "motp": _ratio(iou_sum, tp),
            "gt_tracks": len(gt_length), "pred_tracks": len({box.track_id for box in pred}),
            "mostly_tracked": sum(r >= 0.8 for r in ratios.values()),
            "partially_tracked": sum(0.2 <= r < 0.8 for r in ratios.values()),
            "mostly_lost": sum(r < 0.2 for r in ratios.values())}


def identity(gt: list[Box], pred: list[Box], frames: set[int], threshold: float = IOU_THRESHOLD) -> dict[str, Any]:
    """IDF1: one global GT-track <-> predicted-track assignment maximising co-located frames."""
    gt_ids = sorted({box.track_id for box in gt})
    pred_ids = sorted({box.track_id for box in pred})
    gi, pi = {k: i for i, k in enumerate(gt_ids)}, {k: i for i, k in enumerate(pred_ids)}
    overlap = np.zeros((len(gt_ids), len(pred_ids)))
    gt_frames, pred_frames = _by_frame(gt), _by_frame(pred)
    for frame in frames:
        g, p = gt_frames.get(frame, []), pred_frames.get(frame, [])
        if g and p:
            similarity = _iou_matrix(g, p)
            for r, c in zip(*np.nonzero(similarity >= threshold - 1e-9)):
                overlap[gi[g[r].track_id], pi[p[c].track_id]] += 1
    idtp = 0.0
    if overlap.size:
        rows, cols = linear_sum_assignment(-overlap)
        idtp = float(overlap[rows, cols].sum())
    idp, idr = _ratio(idtp, len(pred)), _ratio(idtp, len(gt))
    return {"idtp": int(idtp), "idfp": len(pred) - int(idtp), "idfn": len(gt) - int(idtp),
            "idf1": _ratio(2 * idtp, len(gt) + len(pred)), "idp": idp, "idr": idr}


def hota(gt: list[Box], pred: list[Box], frames: set[int]) -> dict[str, Any]:
    """HOTA as in TrackEval ``metrics/hota.py`` (Luiten et al. 2021), over ``HOTA_ALPHAS``.

    Matching per frame maximises (global track alignment x IoU); DetRe/AssRe are the recall halves of
    DetA/AssA: DetRe = share of GT boxes found, AssRe = how much of each GT track one predicted track covers.
    ``per_alpha`` keeps the raw arrays so classes can be combined exactly like TrackEval's det-averaged combine.
    """
    alphas = len(HOTA_ALPHAS)
    tp, fn, fp = np.zeros(alphas), np.zeros(alphas), np.zeros(alphas)
    ass_a, ass_re, ass_pr, loc_a = np.zeros(alphas), np.zeros(alphas), np.zeros(alphas), np.zeros(alphas)
    if not pred or not gt:
        fn[:], fp[:], loc_a[:] = len(gt), len(pred), 1.0
        return _hota_summary({"tp": tp, "fn": fn, "fp": fp, "ass_a": ass_a, "ass_re": ass_re, "ass_pr": ass_pr,
                              "loc_a": loc_a})
    gt_index = {k: i for i, k in enumerate(sorted({b.track_id for b in gt}))}
    pred_index = {k: i for i, k in enumerate(sorted({b.track_id for b in pred}))}
    gt_frames, pred_frames = _by_frame(gt), _by_frame(pred)
    steps = []
    potential = np.zeros((len(gt_index), len(pred_index)))
    gt_count, pred_count = np.zeros((len(gt_index), 1)), np.zeros((1, len(pred_index)))
    for frame in sorted(frames):
        g, p = gt_frames.get(frame, []), pred_frames.get(frame, [])
        gi = np.array([gt_index[b.track_id] for b in g], dtype=int)
        pi = np.array([pred_index[b.track_id] for b in p], dtype=int)
        similarity = _iou_matrix(g, p)
        steps.append((gi, pi, similarity))
        denominator = similarity.sum(0)[np.newaxis, :] + similarity.sum(1)[:, np.newaxis] - similarity
        normalised = np.zeros_like(similarity)
        mask = denominator > 0 + EPS
        normalised[mask] = similarity[mask] / denominator[mask]
        potential[gi[:, np.newaxis], pi[np.newaxis, :]] += normalised
        gt_count[gi] += 1
        pred_count[0, pi] += 1
    alignment = potential / (gt_count + pred_count - potential)
    matches = [np.zeros_like(potential) for _ in range(alphas)]
    for gi, pi, similarity in steps:
        if not len(gi) or not len(pi):
            fp += len(pi)
            fn += len(gi)
            continue
        rows, cols = linear_sum_assignment(-(alignment[gi[:, np.newaxis], pi[np.newaxis, :]] * similarity))
        for a, alpha in enumerate(HOTA_ALPHAS):
            keep = similarity[rows, cols] >= alpha - EPS
            r, c = rows[keep], cols[keep]
            tp[a] += len(r)
            fn[a] += len(gi) - len(r)
            fp[a] += len(pi) - len(r)
            if len(r):
                loc_a[a] += similarity[r, c].sum()
                matches[a][gi[r], pi[c]] += 1
    for a in range(alphas):
        m = matches[a]
        ass_a[a] = np.sum(m * m / np.maximum(1, gt_count + pred_count - m)) / max(1.0, tp[a])
        ass_re[a] = np.sum(m * m / np.maximum(1, gt_count)) / max(1.0, tp[a])
        ass_pr[a] = np.sum(m * m / np.maximum(1, pred_count)) / max(1.0, tp[a])
    loc_a = np.maximum(1e-10, loc_a) / np.maximum(1e-10, tp)
    return _hota_summary({"tp": tp, "fn": fn, "fp": fp, "ass_a": ass_a, "ass_re": ass_re, "ass_pr": ass_pr,
                          "loc_a": loc_a})


def _hota_summary(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    tp, fn, fp = arrays["tp"], arrays["fn"], arrays["fp"]
    det_re = tp / np.maximum(1, tp + fn)
    det_pr = tp / np.maximum(1, tp + fp)
    det_a = tp / np.maximum(1, tp + fn + fp)
    hota_values = np.sqrt(det_a * arrays["ass_a"])
    alpha_05 = int(np.argmin(np.abs(HOTA_ALPHAS - 0.5)))
    return {"hota": float(hota_values.mean()), "det_a": float(det_a.mean()), "det_re": float(det_re.mean()),
            "det_pr": float(det_pr.mean()), "ass_a": float(arrays["ass_a"].mean()),
            "ass_re": float(arrays["ass_re"].mean()), "ass_pr": float(arrays["ass_pr"].mean()),
            "loc_a": float(arrays["loc_a"].mean()), "hota@0.5": float(hota_values[alpha_05]),
            "per_alpha": {key: value.tolist() for key, value in arrays.items()}}


def _combine_hota(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """TrackEval ``combine_classes_det_averaged``: sum TP/FN/FP, weight AssA/AssRe/AssPr/LocA by TP."""
    arrays = [{key: np.asarray(value) for key, value in row["per_alpha"].items()} for row in rows]
    tp = sum(a["tp"] for a in arrays)
    combined = {"tp": tp, "fn": sum(a["fn"] for a in arrays), "fp": sum(a["fp"] for a in arrays)}
    for key in ("ass_a", "ass_re", "ass_pr"):
        combined[key] = sum(a[key] * a["tp"] for a in arrays) / np.maximum(1.0, tp)
    combined["loc_a"] = np.maximum(1e-10, sum(a["loc_a"] * a["tp"] for a in arrays)) / np.maximum(1e-10, tp)
    return _hota_summary(combined)


def _bin(value: float | None, bins: tuple[tuple[str, float, float], ...]) -> str:
    if value is None:
        return "unknown"
    return next((name for name, low, high in bins if low <= value < high), "unknown")


def recall_breakdown(reference: Reference, gt: list[Box], kept: list[Box], raw_gt: list[Box], raw_pred: list[Box],
                     frames: set[int]) -> dict[str, dict[str, dict[str, int]]]:
    """Recall @0.5 of one class split by what makes a vehicle hard to see; ``{dimension: {bin: {gt, tp}}}``.

    ``distance`` / ``height`` use the KITTI-filtered GT and kept predictions (same TP as ``detection@0.5``);
    ``occlusion`` / ``truncation`` use every GT row and all predictions of the class (like ``raw_detection``),
    because the KITTI filter drops exactly the heavily occluded and truncated vehicles.
    """
    items = {item.box: item for item in reference.items}
    result: dict[str, dict[str, dict[str, int]]] = {}

    def count(name: str, boxes: list[Box], matched: set[Box], key) -> None:
        table: dict[str, dict[str, int]] = defaultdict(lambda: {"gt": 0, "tp": 0})
        for box in boxes:
            cell = table[key(items[box])]
            cell["gt"] += 1
            cell["tp"] += box in matched
        result[name] = ordered_bins(table)

    matched = {g for g, _, _ in _match_frames(gt, kept, frames, IOU_THRESHOLD)}
    count("distance", gt, matched, lambda item: _bin(item.distance_m, DISTANCE_BINS))
    count("height", gt, matched, lambda item: _bin(item.box.xyxy[3] - item.box.xyxy[1], HEIGHT_BINS))
    raw_matched = {g for g, _, _ in _match_frames(raw_gt, raw_pred, frames, IOU_THRESHOLD)}
    count("occlusion", raw_gt, raw_matched, lambda item: f"occ{item.occluded}")
    count("truncation", raw_gt, raw_matched, lambda item: "truncated" if item.truncated > 0 else "full")
    return result


def _sum_breakdowns(rows: list[dict[str, dict[str, dict[str, int]]]]) -> dict[str, dict[str, dict[str, int]]]:
    total: dict[str, dict[str, dict[str, int]]] = {}
    for row in rows:
        for name, bins in row.items():
            for key, cell in bins.items():
                target = total.setdefault(name, {}).setdefault(key, {"gt": 0, "tp": 0})
                target["gt"] += cell["gt"]
                target["tp"] += cell["tp"]
    return {name: ordered_bins(bins) for name, bins in total.items()}


_BIN_ORDER = [name for name, _, _ in DISTANCE_BINS + HEIGHT_BINS] + ["full", "truncated", "unknown"]


def ordered_bins(table: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    """Near-to-far / small-to-large as defined in the bin tuples; occlusion levels sort naturally."""
    return dict(sorted(table.items(), key=lambda item: (
        _BIN_ORDER.index(item[0]) if item[0] in _BIN_ORDER else -1, item[0])))


def first_detection(gt: list[Box], pred: list[Box], frames: set[int], fps: float = SOURCE_FPS) -> dict[str, Any]:
    """Per GT track: frames from its first scored frame until a prediction first matches it (IoU 0.5).

    Tracks already visible on the first scored frame are reported apart from tracks that enter later: only the
    latter show how fast a new vehicle is picked up. ``coverage`` = share of GT tracks matched at least once.
    """
    matched_frames: dict[str, list[int]] = defaultdict(list)
    for g, _, _ in _match_frames(gt, pred, frames, IOU_THRESHOLD):
        matched_frames[g.track_id].append(g.frame)
    appear: dict[str, int] = {}
    for box in gt:
        appear[box.track_id] = min(box.frame, appear.get(box.track_id, box.frame))
    first = min(frames) if frames else None
    delays: dict[str, list[int]] = {"entered": [], "at_start": []}
    never = {"entered": 0, "at_start": 0}
    for track, start in appear.items():
        group = "at_start" if start == first else "entered"
        if matched_frames.get(track):
            delays[group].append(min(matched_frames[track]) - start)
        else:
            never[group] += 1
    return _first_detection_summary(delays, never, fps)


def _first_detection_summary(delays: dict[str, list[int]], never: dict[str, int], fps: float) -> dict[str, Any]:
    tracks = sum(len(values) for values in delays.values()) + sum(never.values())
    detected = tracks - sum(never.values())
    result: dict[str, Any] = {"gt_tracks": tracks, "detected_tracks": detected, "coverage": _ratio(detected, tracks),
                              "fps": fps}
    for group in ("entered", "at_start"):
        values = np.asarray(delays[group], dtype=float)
        result[group] = {
            "tracks": len(values) + never[group], "never_detected": never[group],
            "delay_frames_median": float(np.median(values)) if values.size else None,
            "delay_frames_p90": float(np.percentile(values, 90)) if values.size else None,
            "delay_frames_mean": float(values.mean()) if values.size else None,
            "delay_ms_median": float(np.median(values)) * 1000 / fps if values.size else None,
            "delays": sorted(int(v) for v in values)}
    return result


def _combine_first_detection(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = ("entered", "at_start")
    return _first_detection_summary({g: [d for r in rows for d in r[g]["delays"]] for g in groups},
                                    {g: sum(r[g]["never_detected"] for r in rows) for g in groups},
                                    rows[0]["fps"] if rows else SOURCE_FPS)


def _sum_rows(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, int]:
    return {key: sum(row[key] for row in rows) for key in keys}


def _combine(per_class: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Overall = sum of per-class counts (TrackEval combined-class style), ratios recomputed."""
    rows = list(per_class.values())
    result: dict[str, Any] = {}
    for threshold in DETECTION_THRESHOLDS:
        key = f"detection@{threshold}"
        d = _sum_rows([r[key] for r in rows], ("gt", "pred", "tp", "fp", "fn"))
        weighted = sum((r[key]["mean_iou"] or 0) * r[key]["tp"] for r in rows)
        precision, recall = _ratio(d["tp"], d["tp"] + d["fp"]), _ratio(d["tp"], d["tp"] + d["fn"])
        f1, f2 = _f_scores(precision, recall, False)
        d.update(precision=precision, recall=recall, mean_iou=_ratio(weighted, d["tp"]), f1=f1, f2=f2)
        result[key] = d
    c = _sum_rows([r["clear"] for r in rows], ("gt", "pred", "tp", "fp", "fn", "idsw", "frag", "gt_tracks",
                                               "pred_tracks", "mostly_tracked", "partially_tracked", "mostly_lost"))
    motp_weighted = sum((r["clear"]["motp"] or 0) * r["clear"]["tp"] for r in rows)
    c.update(mota=1 - (c["fn"] + c["fp"] + c["idsw"]) / c["gt"] if c["gt"] else None,
             motp=_ratio(motp_weighted, c["tp"]))
    result["clear"] = c
    i = _sum_rows([r["identity"] for r in rows], ("idtp", "idfp", "idfn"))
    gt_total, pred_total = c["gt"], c["pred"]
    i.update(idf1=_ratio(2 * i["idtp"], gt_total + pred_total), idp=_ratio(i["idtp"], pred_total),
             idr=_ratio(i["idtp"], gt_total))
    result["identity"] = i
    result["hota"] = _combine_hota([r["hota"] for r in rows])
    result["recall_breakdown"] = _sum_breakdowns([r["recall_breakdown"] for r in rows])
    result["first_detection"] = _combine_first_detection([r["first_detection"] for r in rows])
    return result


def evaluate(reference: Reference, predictions: list[Box], frames: set[int]) -> dict[str, Any]:
    gt, kept, removed = preprocess(reference, predictions, frames)
    raw_gt = reference.target_boxes(frames)
    per_class: dict[str, dict[str, Any]] = {}
    for label in reference.classes:
        g = [b for b in gt if b.label == label]
        p = [b for b in kept if b.label == label]
        raw_g = [b for b in raw_gt if b.label == label]
        raw_p = [b for b in predictions if b.label == label and b.frame in frames]
        per_class[label] = {
            **{f"detection@{t}": detection(g, p, frames, t) for t in DETECTION_THRESHOLDS},
            "clear": clear_mot(g, p, frames),
            "identity": identity(g, p, frames),
            "hota": hota(g, p, frames),
            "recall_breakdown": recall_breakdown(reference, g, p, raw_g, raw_p, frames),
            "first_detection": first_detection(g, p, frames),
            "raw_detection@0.5": detection(raw_g, raw_p, frames, 0.5),
        }
    return {"frames": len(frames), "frame_range": [min(frames), max(frames)] if frames else None,
            "removed_predictions": dict(removed), "per_class": per_class, "overall": _combine(per_class),
            "confusion": confusion(reference, kept, frames)}


def confusion(reference: Reference, predictions: list[Box], frames: set[int]) -> dict[str, dict[str, int]]:
    """Predicted class -> native KITTI class of the best-overlapping GT row (IoU >= 0.5), else ``background``.

    Counted over the predictions kept after the ignore rules, any GT difficulty. ``van -> Car`` means the van
    prompt fired on cars; ``pedestrian -> Cyclist`` means riders were labelled pedestrians.
    """
    rows: dict[int, list[tuple[str, tuple[float, float, float, float]]]] = defaultdict(list)
    for frame, native, xyxy in reference.native_rows:
        if frame in frames:
            rows[frame].append((native, xyxy))
    table: dict[str, Counter] = defaultdict(Counter)
    for box in predictions:
        best, best_iou = "background", IOU_THRESHOLD
        for native, xyxy in rows.get(box.frame, ()):
            overlap = iou(box.xyxy, xyxy)
            if overlap >= best_iou:
                best, best_iou = native, overlap
        table[box.label][best] += 1
    return {label: dict(counts.most_common()) for label, counts in sorted(table.items())}


def write_kitti_results(predictions: list[Box], path: Path) -> None:
    """KITTI tracking result format (one line per box) for the official TrackEval, if wanted."""
    numbers: dict[str, int] = {}
    lines = []
    for box in sorted(predictions, key=lambda b: (b.frame, b.track_id)):
        track = numbers.setdefault(box.track_id, len(numbers))
        x1, y1, x2, y2 = box.xyxy
        kitti_type = {"car": "Car", "person": "Pedestrian"}.get(box.label, box.label.capitalize())
        lines.append(f"{box.frame} {track} {kitti_type} -1 -1 -10 {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
                     f"-1 -1 -1 -1000 -1000 -1000 -10 {box.score if box.score is not None else 1:.4f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


class PayloadReader:
    """Read a KITTI payload (zip or extracted folder) built by ``prepare_kitti.py`` without extracting it."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()
        missing = f"{self.path}: no kitti_manifest.json (build it with kitti/prepare_kitti.py)"
        if self.path.is_file():
            self._zip: zipfile.ZipFile | None = zipfile.ZipFile(self.path)
            names = [n for n in self._zip.namelist() if n.endswith("kitti_manifest.json")]
            if not names:
                raise ValueError(missing)
            self._prefix = min(names, key=len)[: -len("kitti_manifest.json")]
        else:
            self._zip = None
            found = sorted(self.path.glob("**/kitti_manifest.json"), key=lambda p: len(p.parts))
            if not found:
                raise ValueError(missing)
            self._base = found[0].parent

    def read_text(self, name: str) -> str:
        if self._zip is not None:
            return self._zip.read(self._prefix + name).decode("utf-8")
        return (self._base / name).read_text(encoding="utf-8")


def load_predictions(path: Path, position_to_source: dict[int, int],
                     prompt_to_class: dict[str, str]) -> tuple[list[Box], Counter, int]:
    """``predictions.json`` (local-sam-predictions/v1) -> boxes on KITTI frame numbers.

    Returns (boxes, dropped labels, invalid scores). Track IDs stay per prompt (``car:3``).
    """
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "local-sam-predictions/v1":
        raise ValueError(f"{path}: unexpected schema {data.get('schema')!r}")
    boxes: list[Box] = []
    dropped: Counter = Counter()
    invalid = 0
    for position, detections in data["frames"].items():
        source = position_to_source[int(position)]
        for det in detections:
            label = prompt_to_class.get(det["label"])
            if label is None:
                dropped[det["label"]] += 1
                continue
            score = det.get("score")
            x1, y1, x2, y2 = (float(v) for v in det["bbox"])
            if not (x2 > x1 and y2 > y1) or (score is not None and not (math.isfinite(score) and 0 <= score <= 1)):
                invalid += 1
                continue
            boxes.append(Box(source, label, (x1, y1, x2, y2), f"{label}|{det['track_id']}", score))
    return boxes, dropped, invalid


def merge_prompts(boxes: list[Box]) -> tuple[list[Box], int]:
    """Drop duplicates when several prompts map to one class (``truck`` and ``bus`` on the same bus).

    Within a frame and class, boxes are visited by descending score; a box from another prompt overlapping a
    kept one by IoU >= DEDUP_IOU is dropped. Boxes from the same prompt are never merged.
    """
    prompt = lambda box: box.track_id.split("|", 1)[-1].rsplit(":", 1)[0]  # noqa: E731
    grouped: dict[tuple[int, str], list[Box]] = defaultdict(list)
    for box in boxes:
        grouped[(box.frame, box.label)].append(box)
    kept: list[Box] = []
    merged = 0
    for group in grouped.values():
        if len({prompt(b) for b in group}) < 2:
            kept.extend(group)
            continue
        chosen: list[Box] = []
        for box in sorted(group, key=lambda b: -(b.score or 0)):
            if any(prompt(c) != prompt(box) and iou(c.xyxy, box.xyxy) >= DEDUP_IOU for c in chosen):
                merged += 1
            else:
                chosen.append(box)
        kept.extend(chosen)
    return kept, merged
