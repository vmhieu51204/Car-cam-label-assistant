"""Frozen tracking event ledger. Attribution is heuristic evidence, never a causal claim."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from scripts.eval_autolabel import Box
from scripts.stitch_sam3_benchmark import _window_tracks
from src.services.evaluation.fragmentation import RULES, fragmentation_evidence, fragmentation_summary
from src.services.evaluation.matching import match_boxes
from src.services.evaluation.phase0 import canonical_predictions, file_hash, load_frozen_run
from src.services.evaluation.tracking import _temporal, tracking_diagnostics
from src.services.evaluation.tracking_protocol import begin_final, finish_final, select_runs
from src.services.video.chunks import plan_chunks
from src.services.video.export_2d import Export2DConfig, _boxes
from src.services.video.extractor import _histogram_distance
from src.services.video.stitching import stitch_tracks


def event_signal(*, near_cut: bool, near_boundary: bool, gap_inside_window: bool, small: bool, occluded: bool) -> str:
    for signal, present in [
        ("near_scene_cut", near_cut),
        ("near_window_boundary", near_boundary),
        ("within_window_gap", gap_inside_window),
        ("small_box", small),
        ("occlusion", occluded),
    ]:
        if present:
            return signal
    return "unresolved"


def event_ci(gt: tuple, events: list[dict], *, repeats: int = 10000) -> dict:
    counts = Counter((b.frame.sequence.sequence_uid, str(b.native_track_id)) for b in gt)
    tracks = sorted(counts)
    outcomes = Counter((e["sequence"], str(e["gt_id"])) for e in events)
    if not outcomes.keys() <= counts.keys():
        raise ValueError("event track is outside GT population")
    support = {
        "sequence_count": len({t[0] for t in tracks}),
        "denominator_tracks": len(tracks),
        "supporting_tracks": len(outcomes),
        "event_count": len(events),
        "gt_object_frames": sum(counts.values()),
        "events_per_100_GT_object_frames": 100 * len(events) / sum(counts.values()) if counts else None,
    }
    if len(outcomes) < 2:
        return {**support, "availability": "unavailable", "reason": "fewer_than_two_supporting_tracks", "ci95": None}
    totals = np.array([counts[t] for t in tracks])
    values = np.array([outcomes[t] for t in tracks])
    rng = np.random.default_rng(42)
    numerator = np.zeros(repeats)
    denominator = np.zeros(repeats)
    for sequence in sorted({t[0] for t in tracks}):
        indices = np.array([i for i, t in enumerate(tracks) if t[0] == sequence])
        sampled = indices[rng.integers(0, len(indices), size=(repeats, len(indices)))]
        numerator += values[sampled].sum(axis=1)
        denominator += totals[sampled].sum(axis=1)
    rates = 100 * numerator / denominator
    return {
        "method": "GT-track percentile, sequence-local, seed42",
        "unit": "events_per_100_GT_object_frames",
        **support,
        "availability": "available",
        "repeats": repeats,
        "valid_repeats": repeats,
        "ci95": np.percentile(rates, [2.5, 97.5]).tolist(),
    }


def event_breakdown(gt: tuple, events: list[dict]) -> dict:
    signals = ("near_scene_cut", "near_window_boundary", "within_window_gap", "small_box", "occlusion", "unresolved")
    return {
        "by_kind": {
            kind: event_ci(gt, [e for e in events if e["kind"] == kind]) for kind in ("fragmentation", "id_switch")
        },
        "by_signal": {signal: event_ci(gt, [e for e in events if e["signal"] == signal]) for signal in signals},
        "by_kind_and_signal": {
            kind: {
                signal: event_ci(gt, [e for e in events if e["kind"] == kind and e["signal"] == signal])
                for signal in signals
            }
            for kind in ("fragmentation", "id_switch")
        },
    }


def gap_candidates(boxes: list[Box], gt: tuple, keys: dict, config_id: str) -> dict:
    by_track: dict[tuple, list[Box]] = defaultdict(list)
    for box in boxes:
        by_track[(box.label, box.track_id)].append(box)
    pred = canonical_predictions(boxes, keys, config_id)
    pairs = match_boxes(gt, pred, iou_threshold=0.5).pairs
    gt_by_id = {b.annotation_id: b for b in gt}
    pred_by_id = {b.prediction_id: b for b in pred}
    identity = {}
    for pair in pairs:
        p = pred_by_id[pair.prediction_id]
        identity[(p.class_name, p.raw_track_id, int(p.frame.frame_uid))] = gt_by_id[pair.annotation_id].native_track_id
    endpoints = {k: sorted(v, key=lambda b: b.frame) for k, v in by_track.items()}
    candidates = []
    oracle = 0
    for a, rows_a in endpoints.items():
        left = rows_a[-1]
        for b, rows_b in endpoints.items():
            right = rows_b[0]
            gap = right.frame - left.frame
            if a == b or a[0] != b[0] or not 0 < gap <= 20:
                continue
            id_a = identity.get((*a, left.frame))
            id_b = identity.get((*b, right.frame))
            same = id_a is not None and id_a == id_b
            oracle += int(same)
            aw, ah = left.xyxy[2] - left.xyxy[0], left.xyxy[3] - left.xyxy[1]
            bw, bh = right.xyxy[2] - right.xyxy[0], right.xyxy[3] - right.xyxy[1]
            distance = (
                (sum(left.xyxy[::2]) - sum(right.xyxy[::2])) ** 2 + (sum(left.xyxy[1::2]) - sum(right.xyxy[1::2])) ** 2
            ) ** 0.5 / (2 * max(aw, ah))
            ratio = max(aw / bw, bw / aw, ah / bh, bh / ah)
            if distance <= 2 and ratio <= 2:
                candidates.append(
                    {
                        "a": a[1],
                        "b": b[1],
                        "class": a[0],
                        "gap_source_frames": gap,
                        "normalized_center_distance": distance,
                        "size_ratio": ratio,
                        "same_gt_at_matched_endpoints": same if id_a is not None and id_b is not None else None,
                    }
                )
    return {
        "oracle_gap_pairs": oracle,
        "gt_blind_candidate_count": len(candidates),
        "candidates": candidates,
        "thresholds_uncalibrated": {"max_gap_source": 20, "max_center_distance": 2, "max_size_ratio": 2},
        "note": "Candidates only; no merging. Oracle uses GT at matched endpoints; incomplete endpoint matches remain unknown.",
    }


def diagnose(
    protocol: Path, root: Path, output: Path, *, final_held_out: bool = False, dev_freeze: Path | None = None
) -> dict[str, Any]:
    config = json.loads(protocol.read_text(encoding="utf-8"))
    split = "held_out" if final_held_out else "dev"
    if not final_held_out and (output / "final-held-out-ledger.json").exists():
        raise ValueError("cannot overwrite sealed final evaluation with DEV report")
    final_identity = None
    if final_held_out:
        if dev_freeze is None:
            raise ValueError("explicit final HELD-OUT evaluation requires DEV freeze")
        final_identity, cached = begin_final(protocol, dev_freeze, output, root=root)
        if cached is not None:
            return cached
    elif dev_freeze is not None:
        raise ValueError("DEV freeze only applies to explicit final HELD-OUT evaluation")
    all_events = []
    runs = []
    pooled_gt: dict[str, list] = defaultdict(list)
    pooled_pred: dict[str, list] = defaultdict(list)
    pooled_frames: dict[str, list] = defaultdict(list)
    pooled_occlusions = {}
    for run in select_runs(config, split):
        loaded = load_frozen_run(run, root)
        windows = loaded.item.get("window_predictions", [])
        if not windows:
            runs.append({"sequence": run["sequence"], "status": "missing_overlap_predictions"})
            continue
        resets = {r["window_id"] for r in windows if loaded.records[r["frame_index"]].get("scene_cut_before")}
        plans = plan_chunks(
            len(loaded.source),
            window_size=loaded.item["config"]["window_size"],
            overlap=loaded.item["config"].get("overlap", 0),
            scene_cut_frames=tuple(i for i, r in enumerate(loaded.records) if i > 0 and r.get("scene_cut_before")),
        )
        stitched = stitch_tracks(
            _window_tracks(windows), reset_windows=resets, planned_window_ids=[p.index for p in plans]
        )
        sampled = [
            Box(loaded.source[o.frame], t.label, o.bbox, t.track_id) for t in stitched.tracks for o in t.observations
        ]
        timeline = [
            Box(f, t.label, b, t.track_id)
            for t in stitched.tracks
            for f, b in _boxes(t, loaded.source, Export2DConfig())
        ]
        sample_index = {f: i for i, f in enumerate(loaded.source)}
        step = loaded.item["config"]["window_size"] - loaded.item["config"].get("overlap", 0)
        boundaries = {loaded.source[i] for i in range(step, len(loaded.source), step)}
        cuts = {loaded.source[i] for i, r in enumerate(loaded.records) if r.get("scene_cut_before")}
        views = {}
        for view, boxes in [("stitched_sampled", sampled), ("interpolated_timeline", timeline)]:
            frame_set = set(loaded.source) if view == "stitched_sampled" else set(loaded.keys)
            frames = tuple(loaded.keys[f] for f in sorted(frame_set))
            gt = tuple(b for b in loaded.gt if int(b.frame.frame_uid) in frame_set)
            pred = canonical_predictions(boxes, loaded.keys, run["config_id"])
            pooled_gt[view].extend(gt)
            pooled_pred[view].extend(canonical_predictions(boxes, loaded.keys, f"{run['sequence']}/{run['config_id']}"))
            pooled_frames[view].extend(frames)
            pooled_occlusions.update(loaded.occlusions)
            metrics = tracking_diagnostics(
                gt, pred, frames=frames, occlusions=loaded.occlusions, tracking_ids_available=True
            )
            by_gt = {b.annotation_id: b for b in gt}
            by_frame = defaultdict(list)
            for prediction in pred:
                by_frame[prediction.frame].append(prediction)
            events = []
            for threshold in (0.5, 0.7):
                ledger = _temporal(list(gt), list(pred), frames, threshold, include_events=True)
                for event in ledger["events"]:
                    f = event["source_frame"]
                    box = by_gt[event["annotation_id"]]
                    nearest = min(loaded.source, key=lambda x: abs(x - f))
                    near_boundary = any(abs(sample_index[nearest] - sample_index[b]) <= 5 for b in boundaries)
                    native = loaded.native_by_annotation[box.annotation_id]
                    previous_frame = event["previous_matched_source_frame"]
                    previous_sample = min(loaded.source, key=lambda x: abs(x - previous_frame))
                    inside = any(
                        start
                        <= sample_index[previous_sample]
                        < sample_index[nearest]
                        < start + loaded.item["config"]["window_size"]
                        for start in range(0, len(loaded.source), step)
                    )
                    signal = event_signal(
                        near_cut=any(abs(f - c) <= 2 for c in cuts),
                        near_boundary=near_boundary,
                        gap_inside_window=event["kind"] == "fragmentation" and inside,
                        small=box.bbox_xyxy[3] - box.bbox_xyxy[1] < 25,
                        occluded=native.occluded in {1, 2},
                    )
                    row = {
                        **event,
                        "sequence": run["sequence"],
                        "view": view,
                        "iou_threshold": threshold,
                        "signal": signal,
                        "native_occlusion": native.occluded,
                        "boundary_decision": "unavailable: original stitching candidate margins were not persisted",
                    }
                    if event["kind"] == "fragmentation":
                        row.update(fragmentation_evidence(row, by_gt, by_frame, loaded.occlusions))
                    events.append(row)
            views[view] = {
                "metrics": metrics,
                "signals": dict(Counter(e["signal"] for e in events)),
                "gap_candidates": gap_candidates(boxes, gt, loaded.keys, run["config_id"]),
                "signals_by_threshold": {
                    str(t): dict(Counter(e["signal"] for e in events if e["iou_threshold"] == t)) for t in (0.5, 0.7)
                },
                "event_ci": {str(t): event_ci(gt, [e for e in events if e["iou_threshold"] == t]) for t in (0.5, 0.7)},
                "event_breakdown": {
                    str(t): event_breakdown(gt, [e for e in events if e["iou_threshold"] == t]) for t in (0.5, 0.7)
                },
                "fragmentation_diagnosis": {
                    str(t): fragmentation_summary([e for e in events if e["iou_threshold"] == t]) for t in (0.5, 0.7)
                },
            }
            all_events.extend(events)
        histogram = []
        previous = None
        image_dir = (root / run["labels"]).parent.parent / "image_02" / run["sequence"]
        for frame in loaded.source:
            with Image.open(image_dir / f"{frame:06d}.png") as image:
                current = image.convert("L").resize((64, 36)).histogram()
            distance = _histogram_distance(previous, current)
            if distance is not None:
                histogram.append({"source_frame": frame, "distance": distance, "over_035": distance >= 0.35})
            previous = current
        lengths = [len(t.observations) for t in stitched.tracks]
        gaps = [
            loaded.source[b.frame] - loaded.source[a.frame]
            for t in stitched.tracks
            for a, b in zip(t.observations, t.observations[1:])
            if b.frame - a.frame > 1
        ]
        runs.append(
            {
                "sequence": run["sequence"],
                "input_hashes": loaded.input_hashes,
                "views": views,
                "reconstructed_stitch_boundary_logs": stitched.boundary_logs,
                "stitch_log_provenance": "Recomputed offline from frozen overlap predictions, not original runtime logs",
                "track_lengths_observations": lengths,
                "track_gap_source_frames": gaps,
                "histogram": histogram,
                "histogram_pairs_over_035": sum(r["over_035"] for r in histogram),
                "cut_ground_truth": "unavailable; crossing 0.35 is not evidence of a true cut",
            }
        )
    pooled = {}
    for view in pooled_gt:
        pooled[view] = {
            "metrics": tracking_diagnostics(
                tuple(pooled_gt[view]),
                tuple(pooled_pred[view]),
                frames=tuple(pooled_frames[view]),
                occlusions=pooled_occlusions,
                tracking_ids_available=True,
            ),
            "signals_by_threshold": {
                str(t): dict(Counter(e["signal"] for e in all_events if e["view"] == view and e["iou_threshold"] == t))
                for t in (0.5, 0.7)
            },
            "event_ci": {
                str(t): event_ci(
                    tuple(pooled_gt[view]), [e for e in all_events if e["view"] == view and e["iou_threshold"] == t]
                )
                for t in (0.5, 0.7)
            },
            "event_breakdown": {
                str(t): event_breakdown(
                    tuple(pooled_gt[view]), [e for e in all_events if e["view"] == view and e["iou_threshold"] == t]
                )
                for t in (0.5, 0.7)
            },
            "fragmentation_diagnosis": {
                str(t): fragmentation_summary([e for e in all_events if e["view"] == view and e["iou_threshold"] == t])
                for t in (0.5, 0.7)
            },
        }
    result = {
        "schema": "autolabel-phase1-tracking-diagnosis/v1",
        "protocol_sha256": file_hash(protocol),
        "pooled": pooled,
        "runs": runs,
        "event_count": len(all_events),
        "signals_all_views_thresholds": dict(Counter(e["signal"] for e in all_events)),
        "attribution": "Automatic signals, not confirmed causes; views/thresholds are reported separately, not independent samples",
        "gate": "not_evaluated; no candidate algorithm; held-out has no predictions",
        "services_called": [],
        "next_phase": "STOP T1: no automatic tracking changes",
    }
    result["fragmentation_rules"] = RULES
    result["events"] = all_events
    result["evaluation_split"] = split
    result["held_out_history_status"] = config.get("held_out_history_status", "unverified")
    result["population_support"] = {
        view: {
            "sequence_count": len({b.frame.sequence.sequence_uid for b in gt}),
            "gt_tracks": len({(b.frame.sequence.sequence_uid, b.native_track_id) for b in gt}),
            "gt_object_frames": len(gt),
        }
        for view, gt in pooled_gt.items()
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "diagnosis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output / "events.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = sorted({k for e in all_events for k in e})
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in e.items()}
            for e in all_events
        )
    with (output / "fragmentation-missed-frames.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = [
            "sequence",
            "view",
            "iou_threshold",
            "gt_id",
            "previous_matched_source_frame",
            "source_frame",
            "previous_pred_id",
            "pred_id",
            "missed_source_frame",
            "annotation_id",
            "occlusion",
            "category",
            "candidates",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for event in all_events:
            for row in event.get("missed_frames", []):
                writer.writerow(
                    {
                        **{k: event[k] for k in fields[:8]},
                        "missed_source_frame": row["source_frame"],
                        "annotation_id": row["annotation_id"],
                        "occlusion": row["occlusion"],
                        "category": row["category"],
                        "candidates": json.dumps(row["candidates"], ensure_ascii=False),
                    }
                )
    if final_identity is not None:
        finish_final(protocol, output, final_identity, {"diagnosis": output / "diagnosis.json"})
    return result
