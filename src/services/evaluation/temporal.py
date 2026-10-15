"""Phase-0.7 offline temporal diagnostics. Oracle boxes never become annotations."""

from __future__ import annotations

import csv
import json
import math
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scripts.eval_autolabel import Box
from scripts.stitch_sam3_benchmark import _window_tracks
from src.models.evaluation import BBox, PredictionBox
from src.services.evaluation.geometry import GEOMETRY_FIELDS, OTHER_FIELDS, direct_ledger
from src.services.evaluation.matching import bbox_iou, match_boxes
from src.services.evaluation.misses import MISS_FIELDS, write_diagnostic_csv
from src.services.evaluation.phase0 import FrozenRunData, canonical_predictions, file_hash, load_frozen_run
from src.services.video.stitching import StitchConfig, stitch_tracks

K_VALUES = (2, 5, 10)


@dataclass(frozen=True)
class Anchor:
    sample: int
    source: int
    gt: dict[str, Any]
    prediction: PredictionBox
    stitched_id: str | None


def interpolate(left: BBox, right: BBox, a: int, b: int, frame: int) -> BBox:
    """Use source-frame distance, not compressed sample distance; no extrapolation."""
    if not a < frame < b:
        raise ValueError("interpolation needs strictly bracketing source frames")
    weight = (frame - a) / (b - a)
    values = [x + weight * (y - x) for x, y in zip(left, right, strict=True)]
    return values[0], values[1], values[2], values[3]


def crosses_cut(left: int, right: int, cuts: set[int]) -> bool:
    return any(left < cut <= right for cut in cuts)


def nearest_pair(anchors: list[Anchor], sample: int, k: int) -> tuple[Anchor | None, Anchor | None]:
    indices = [a.sample for a in anchors]
    before, after = bisect_left(indices, sample) - 1, bisect_right(indices, sample)
    left = anchors[before] if before >= 0 and sample - indices[before] <= k else None
    right = anchors[after] if after < len(anchors) and indices[after] - sample <= k else None
    return left, right


def stitched_evidence(loaded: FrozenRunData) -> tuple[list[Box] | None, dict[int, str | None], list[dict[str, Any]]]:
    """Reproduce stored-view stitching. Preserve local-to-global ID provenance."""
    rows = loaded.item.get("window_predictions", [])
    if not rows:
        return None, {}, []  # Do not call pre-stitch IDs post-stitch IDs.
    if any(r.get("outside", False) for r in rows):
        raise ValueError("outside window observations need explicit source provenance before temporal stitching")
    chunks = _window_tracks(rows)
    resets = {r["window_id"] for r in rows if loaded.records[r["frame_index"]].get("scene_cut_before", False)}
    stitched = stitch_tracks(chunks, reset_windows=resets)
    mapping = {(d.window_id, d.local_track_id): d.global_track_id for d in stitched.decisions}
    # The current stitcher omits decisions for its initial tracks. Both lists
    # retain creation order; use that output order rather than inventing IDs.
    initial = [c for c in chunks if c.window_id == min(c.window_id for c in chunks)]
    for local, global_track in zip(initial, stitched.tracks[: len(initial)], strict=True):
        if local.label != global_track.label:
            raise ValueError("initial stitched-track order mismatch")
        mapping.setdefault((local.window_id, local.local_track_id), global_track.track_id)
    by_observation: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    for r in rows:
        key = (loaded.source[r["frame_index"]], r["label"], r["track_id"], tuple(r["bbox"]))
        by_observation[key].add(mapping[(r["window_id"], r["track_id"])])
    identities = {}
    for i, box in enumerate(loaded.direct):
        candidates = by_observation.get((box.frame, box.label, box.track_id, box.xyxy), set())
        identities[i] = next(iter(candidates)) if len(candidates) == 1 else None
    boxes = [Box(loaded.source[o.frame], t.label, o.bbox, t.track_id) for t in stitched.tracks for o in t.observations]
    return boxes, identities, [asdict(d) for d in stitched.decisions]


def anchors_by_gt(
    loaded: FrozenRunData, ledger: list[dict[str, Any]], identities: dict[int, str | None]
) -> dict[str, list[Anchor]]:
    gt = tuple(g for g in loaded.gt if int(g.frame.frame_uid) in set(loaded.source))
    pred = canonical_predictions(loaded.direct, loaded.keys, loaded.item["id"])
    matched = match_boxes(gt, pred, iou_threshold=0.5)
    rows = {r["annotation_id"]: r for r in ledger}
    by_prediction = {p.prediction_id: p for p in pred}
    result: dict[str, list[Anchor]] = defaultdict(list)
    for pair in matched.pairs:
        row, prediction = rows[pair.annotation_id], by_prediction[pair.prediction_id]
        result[row["gt_id"]].append(
            Anchor(row["frame_sampled"], row["frame"], row, prediction, identities.get(prediction.scoring_ordinal))
        )
    return {key: sorted(values, key=lambda a: a.sample) for key, values in result.items()}


def gt_blind_gap_fill(loaded: FrozenRunData, boxes: list[Box], k: int) -> list[Box]:
    """Generate proposals solely from adjacent observations of the same stitched ID.

    Existing observations are never replaced. Candidate generation sees neither
    GT IDs nor miss labels. This is a simulation, not a production tracker.
    """
    sampled = {f: i for i, f in enumerate(loaded.source)}
    cuts = {i for i, r in enumerate(loaded.records) if r.get("scene_cut_before", False)}
    tracks: dict[tuple[str, str], list[Box]] = defaultdict(list)
    for box in boxes:
        if box.track_id is None:
            raise ValueError("stitched ID is required")
        tracks[(box.label, box.track_id)].append(box)
    fills = []
    for observations in tracks.values():
        ordered = sorted(observations, key=lambda b: sampled[b.frame])
        if len({o.frame for o in ordered}) != len(ordered):
            raise ValueError("duplicate within-track frame")
        for left, right in zip(ordered, ordered[1:]):
            a, b = sampled[left.frame], sampled[right.frame]
            if crosses_cut(a, b, cuts):
                continue
            for i in range(max(a + 1, b - k), min(b, a + k + 1)):
                source = loaded.source[i]
                fills.append(
                    Box(
                        source,
                        left.label,
                        interpolate(left.xyxy, right.xyxy, left.frame, right.frame, source),
                        left.track_id,
                    )
                )
    return fills


def bridge_ledger(loaded: FrozenRunData, ledger: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stitched, identities, decisions = stitched_evidence(loaded)
    anchors = anchors_by_gt(loaded, ledger, identities)
    cuts = {i for i, r in enumerate(loaded.records) if r.get("scene_cut_before", False)}
    gt = tuple(g for g in loaded.gt if int(g.frame.frame_uid) in set(loaded.source))
    misses = [r for r in ledger if not r["matched_05"]]
    practical_hits: dict[int, dict[str, set[str]]] = {}
    practical_metrics = []
    for k in K_VALUES:
        if stitched is None:
            continue
        fills = gt_blind_gap_fill(loaded, stitched, k)
        original = canonical_predictions(stitched, loaded.keys, "stitched")
        proposals = tuple(
            p.model_copy(update={"scoring_ordinal": len(original) + i})
            for i, p in enumerate(canonical_predictions(fills, loaded.keys, "gap-fill"))
        )
        practical_hits[k] = {}
        for threshold in (0.5, 0.7):
            # Full GT population, original boxes compete with fills: no duplicated
            # recovery from one candidate and no matching only against misses.
            match = match_boxes(gt, (*original, *proposals), iou_threshold=threshold)
            hit = {p.annotation_id for p in match.pairs}
            fill_hits = {p.annotation_id for p in match.pairs if p.prediction_id.startswith("gap-fill/")}
            previous = match_boxes(gt, original, iou_threshold=threshold)
            previous_hits = {p.annotation_id for p in previous.pairs}
            practical_hits[k][str(threshold)] = fill_hits - previous_hits
            practical_metrics.append(
                {
                    "sequence": loaded.keys[loaded.source[0]].sequence.sequence_uid,
                    "k": k,
                    "iou": threshold,
                    "original_tp": len(previous.pairs),
                    "with_fill_tp": len(match.pairs),
                    "gt": len(gt),
                    "fill_candidates": len(fills),
                    "fill_matched": len(fill_hits),
                    "fill_false_positive": len(fills) - len(fill_hits),
                    "raw_recall_with_fill": len(hit) / len(gt) if gt else None,
                }
            )
    rows = []
    for miss in misses:
        for k in K_VALUES:
            left, right = nearest_pair(anchors.get(miss["gt_id"], []), miss["frame_sampled"], k)
            pair_exists = left is not None and right is not None
            cut = bool(left and right and crosses_cut(left.sample, right.sample, cuts))
            valid = bool(pair_exists and not cut)
            same_id = (
                (left.stitched_id == right.stitched_id)
                if valid and left and right and left.stitched_id and right.stitched_id
                else None
            )
            box = (
                interpolate(
                    left.prediction.bbox_xyxy, right.prediction.bbox_xyxy, left.source, right.source, miss["frame"]
                )
                if valid and left and right
                else None
            )
            overlap = bbox_iou(miss["gt_box"], box) if box else None
            rows.append(
                {
                    **miss,
                    "k": k,
                    "bridgeable": valid,
                    "cut_between_anchors": bool(cut),
                    "left_frame": left.source if left else None,
                    "right_frame": right.source if right else None,
                    "left_sample_distance": miss["frame_sampled"] - left.sample if left else None,
                    "right_sample_distance": right.sample - miss["frame_sampled"] if right else None,
                    "left_prediction_ordinal": left.prediction.scoring_ordinal if left else None,
                    "right_prediction_ordinal": right.prediction.scoring_ordinal if right else None,
                    "left_stitched_id": left.stitched_id if left else None,
                    "right_stitched_id": right.stitched_id if right else None,
                    "same_stitched_id": same_id,
                    "oracle_box": list(box) if box else None,
                    "oracle_iou": overlap,
                    "id_gated_iou": overlap if same_id is True else None,
                    "gt_blind_fill_recovered_05": miss["annotation_id"] in practical_hits[k]["0.5"]
                    if k in practical_hits
                    else None,
                    "gt_blind_fill_recovered_07": miss["annotation_id"] in practical_hits[k]["0.7"]
                    if k in practical_hits
                    else None,
                }
            )
    return rows, {
        "stitched_available": stitched is not None,
        "direct_ids_unmapped": sum(i is None for i in identities.values()),
        "stitch_config": asdict(StitchConfig()),
        "decisions": decisions,
        "gt_blind": practical_metrics,
    }


def bridge_tables(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        dimensions = [
            ("overall", "all"),
            *[(key, row[key]) for key in ("reason", "geometry", "occlusion", "class")],
            ("reason_geometry_occlusion", f"{row['reason']}|{row['geometry']}|{row['occlusion']}"),
        ]
        for scope in (row["sequence"], "all"):
            for dimension, value in dimensions:
                groups[(scope, row["k"], dimension, value)].append(row)
    table = []
    for (scope, k, dimension, value), selected in sorted(groups.items()):
        n, bridge = len(selected), sum(r["bridgeable"] for r in selected)
        base = {
            "sequence": scope,
            "k": k,
            "dimension": dimension,
            "value": value,
            "misses": n,
            "bridgeable": bridge,
            "bridgeable_fraction": bridge / n,
            "id_same": sum(r["same_stitched_id"] is True for r in selected),
            "id_different": sum(r["same_stitched_id"] is False for r in selected),
            "id_unavailable_bridgeable": sum(r["bridgeable"] and r["same_stitched_id"] is None for r in selected),
        }
        for threshold in (0.5, 0.7):
            oracle = sum(r["oracle_iou"] is not None and r["oracle_iou"] >= threshold for r in selected)
            gated = sum(r["id_gated_iou"] is not None and r["id_gated_iou"] >= threshold for r in selected)
            known = [
                r for r in selected if r[f"gt_blind_fill_recovered_{'05' if threshold == 0.5 else '07'}"] is not None
            ]
            blind = sum(bool(r[f"gt_blind_fill_recovered_{'05' if threshold == 0.5 else '07'}"]) for r in known)
            identity_loss = sum(
                r["oracle_iou"] is not None and r["oracle_iou"] >= threshold and r["same_stitched_id"] is False
                for r in selected
            )
            table.append(
                {
                    **base,
                    "iou": threshold,
                    "oracle_recovered": oracle,
                    "oracle_gap_fill_recall_bridgeable": oracle / bridge if bridge else None,
                    "oracle_recovered_fraction_all_misses": oracle / n,
                    "id_gated_recovered": gated,
                    "id_gated_recovered_fraction_all_misses": gated / n,
                    "identity_lost_recoveries": identity_loss,
                    "gt_blind_recovered": blind if known else None,
                    "gt_blind_recovered_fraction_all_misses": blind / n if known else None,
                }
            )
    return table


def expand_at_prediction_center(box: list[float], width: float, height: float) -> BBox:
    """Expand only; no GT center is used, no clamping or production box write."""
    w, h = max(width, box[2] - box[0]), max(height, box[3] - box[1])
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def size_memory_rows(loaded: FrozenRunData, ledger: list[dict[str, Any]]) -> list[dict[str, Any]]:
    anchors = anchors_by_gt(loaded, ledger, {})
    cuts = {i for i, r in enumerate(loaded.records) if r.get("scene_cut_before", False)}
    rows = []
    for miss in ledger:
        if miss["geometry"] != "pred_inside_gt":
            continue
        gt, pred = miss["gt_box"], miss["best_prediction_box"]
        gw, gh = gt[2] - gt[0], gt[3] - gt[1]
        oracle_box = expand_at_prediction_center(pred, gw, gh)
        for k in K_VALUES:
            left, right = nearest_pair(anchors.get(miss["gt_id"], []), miss["frame_sampled"], k)
            eligible = [
                a
                for a in (left, right)
                if a is not None
                and not crosses_cut(min(a.sample, miss["frame_sampled"]), max(a.sample, miss["frame_sampled"]), cuts)
            ]
            anchor = (
                min(eligible, key=lambda a: (abs(a.sample - miss["frame_sampled"]), a.sample)) if eligible else None
            )
            row = {
                **miss,
                "k": k,
                "anchor_frame": anchor.source if anchor else None,
                "anchor_side": "past"
                if anchor and anchor.sample < miss["frame_sampled"]
                else "future"
                if anchor
                else None,
                "anchor_sample_distance": abs(anchor.sample - miss["frame_sampled"]) if anchor else None,
                "gt_current_size_oracle_box": list(oracle_box),
                "gt_current_size_oracle_iou": bbox_iou(gt, oracle_box),
            }
            fields = (
                "gt_width_ratio",
                "gt_height_ratio",
                "gt_area_ratio",
                "gt_scale_oracle",
                "width_ratio_scale_normalized",
                "height_ratio_scale_normalized",
                "gt_anchor_size_unscaled_iou",
                "gt_anchor_size_normalized_oracle_iou",
            )
            row.update(dict.fromkeys(fields))
            if anchor:
                ag = anchor.gt["gt_box"]
                aw, ah = ag[2] - ag[0], ag[3] - ag[1]
                scale = math.sqrt(gw * gh / (aw * ah))
                row.update(
                    gt_width_ratio=gw / aw,
                    gt_height_ratio=gh / ah,
                    gt_area_ratio=gw * gh / (aw * ah),
                    gt_scale_oracle=scale,
                    width_ratio_scale_normalized=gw / (aw * scale),
                    height_ratio_scale_normalized=gh / (ah * scale),
                    gt_anchor_size_unscaled_iou=bbox_iou(gt, expand_at_prediction_center(pred, aw, ah)),
                    gt_anchor_size_normalized_oracle_iou=bbox_iou(
                        gt, expand_at_prediction_center(pred, aw * scale, ah * scale)
                    ),
                )
            rows.append(row)
    return rows


def distribution_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for scope in (row["sequence"], "all"):
            for dimension, value in (
                ("overall", "all"),
                ("occlusion", row["occlusion"]),
                ("other_gt_flag", str(row["prediction_belongs_to_other_gt"])),
            ):
                groups[(scope, row["k"], dimension, value)].append(row)
    table = []
    metrics = (
        "best_iou",
        "gt_current_size_oracle_iou",
        "gt_anchor_size_unscaled_iou",
        "gt_anchor_size_normalized_oracle_iou",
        "gt_width_ratio",
        "gt_height_ratio",
        "gt_scale_oracle",
        "width_ratio_scale_normalized",
        "height_ratio_scale_normalized",
    )
    for (scope, k, dimension, value), selected in sorted(groups.items()):
        for metric in metrics:
            values = [r[metric] for r in selected if r.get(metric) is not None]
            quantiles = np.percentile(values, (0, 5, 25, 50, 75, 95, 100)).tolist() if values else [None] * 7
            table.append(
                {
                    "sequence": scope,
                    "k": k,
                    "dimension": dimension,
                    "value": value,
                    "metric": metric,
                    "population": len(selected),
                    "n": len(values),
                    "unavailable": len(selected) - len(values),
                    **dict(zip(("min", "p5", "p25", "median", "p75", "p95", "max"), quantiles, strict=True)),
                    "iou_ge_05": sum(v >= 0.5 for v in values) if metric.endswith("iou") else None,
                    "iou_ge_07": sum(v >= 0.7 for v in values) if metric.endswith("iou") else None,
                }
            )
    return table


def eligibility(row: dict[str, Any]) -> tuple[str, str, str]:
    """Uncalibrated diagnostic strata, not filters: <25px first, then truncation."""
    h, truncated = row["box_height"], row["truncation"]
    height = "lt15" if h < 15 else "15_to_lt25" if h < 25 else "25_to_lt40" if h < 40 else "ge40"
    truncation = "unknown" if truncated is None else "zero" if truncated == 0 else "positive"
    tier = (
        "small_lt25"
        if h < 25
        else "truncation_unknown_ge25"
        if truncated is None
        else "truncated_ge25"
        if truncated > 0
        else f"{row['occlusion']}_untruncated_ge25"
    )
    return tier, height, truncation


def eligibility_tables(ledger: list[dict[str, Any]], *, repeats: int = 10000) -> list[dict[str, Any]]:
    """Resample all complete GT tracks independently within each clip, seed 42."""
    if type(repeats) is not int or repeats <= 0:
        raise ValueError("positive bootstrap repeat count required")
    if any(r["gt_id"] is None for r in ledger):
        raise ValueError("GT ID required for track bootstrap")
    tracks = sorted({(r["sequence"], r["gt_id"]) for r in ledger})
    track_index = {t: i for i, t in enumerate(tracks)}
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in ledger:
        tier, height, truncation = eligibility(row)
        for dimension, value in (
            ("overall", "all"),
            ("tier", tier),
            ("occlusion", row["occlusion"]),
            ("height", height),
            ("truncation", truncation),
            ("class", row["class"]),
        ):
            groups[(row["sequence"], dimension, value)].append(row)
            groups[("all", dimension, value)].append(row)
    rng = np.random.default_rng(42)
    weights = np.zeros((repeats, len(tracks)), dtype=np.int64)
    for seq in sorted({t[0] for t in tracks}):
        indices = [i for i, t in enumerate(tracks) if t[0] == seq]
        weights[:, indices] = rng.multinomial(len(indices), np.full(len(indices), 1 / len(indices)), size=repeats)
    result = []
    for (scope, dimension, value), selected in sorted(groups.items()):
        counts, tp = np.zeros(len(tracks), dtype=np.int64), np.zeros(len(tracks), dtype=np.int64)
        for row in selected:
            i = track_index[(row["sequence"], row["gt_id"])]
            counts[i] += 1
            tp[i] += int(row["matched_05"])
        den, num = weights @ counts, weights @ tp
        valid, support = den > 0, int(np.count_nonzero(counts))
        values = num[valid] / den[valid]
        enough = support >= 2 and len(values) > 0
        bounds = np.percentile(values, (2.5, 97.5)).tolist() if enough else [None, None]
        result.append(
            {
                "sequence": scope,
                "dimension": dimension,
                "value": value,
                "iou": 0.5,
                "gt": len(selected),
                "tp": int(tp.sum()),
                "fn": len(selected) - int(tp.sum()),
                "recall": int(tp.sum()) / len(selected),
                "support_tracks": support,
                "ci_low": bounds[0],
                "ci_high": bounds[1],
                "ci_status": "available" if enough else "insufficient_tracks",
                "bootstrap_valid": int(valid.sum()),
                "bootstrap_zero_denominator": int((~valid).sum()),
            }
        )
    return result


def validate_geometry_csv(path: Path, misses: list[dict[str, Any]]) -> None:
    fields = [*MISS_FIELDS, *GEOMETRY_FIELDS, "geometry", *OTHER_FIELDS]
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        saved = list(reader)
    if list(reader.fieldnames or []) != fields or len(saved) != len(misses):
        raise ValueError("frozen geometry CSV header/population mismatch")
    for number, (old, row) in enumerate(zip(saved, misses, strict=True)):
        if set(old) != set(fields):
            raise ValueError("frozen geometry CSV malformed row")
        for field in fields:
            value = row[field]
            expected = json.dumps(value) if isinstance(value, list) else "" if value is None else str(value)
            if old[field] != expected:
                raise ValueError(f"frozen geometry CSV mismatch row {number + 1}: {field}")


def write_temporal_analysis(
    protocol: Path, *, geometry_csv: Path, root: Path, output: Path, bootstrap_repeats: int = 10000
) -> dict[str, Any]:
    config = json.loads(protocol.read_text(encoding="utf-8"))
    if config.get("schema") != "autolabel-phase0-protocol/v1" or config.get("seed") != 42:
        raise ValueError("expected frozen phase0 protocol, seed 42")
    all_gt, misses, bridges, sizes, provenance = [], [], [], [], []
    protected = {protocol.resolve(), geometry_csv.resolve()}
    for run in config["runs"]:
        loaded = load_frozen_run(run, root)
        ledger, missing = direct_ledger(loaded)
        all_gt.extend(ledger)
        misses.extend(missing)
        rows, evidence = bridge_ledger(loaded, ledger)
        bridges.extend(rows)
        sizes.extend(size_memory_rows(loaded, ledger))
        provenance.append(
            {
                "sequence": run["sequence"],
                "input_hashes": loaded.input_hashes,
                "class_map": run["class_map"],
                **evidence,
            }
        )
        protected.update((root / run[k]).resolve() for k in loaded.input_hashes)
    validate_geometry_csv(geometry_csv, misses)
    before = {str(p): file_hash(p) for p in sorted(protected)}
    manifest = {
        "schema": "autolabel-phase07-inputs/v1",
        "input_sha256": before,
        "seed": 42,
        "k_values": list(K_VALUES),
        "bootstrap_repeats": bootstrap_repeats,
        "diagnostic_config": {
            "anchor_match_iou": 0.5,
            "interpolation_axis": "source_frame_index",
            "scene_cut_policy": "no_crossing",
            "height_group_edges_px": [15, 25, 40],
            "tier_height_cut_px": 25,
            "size_scale": "sqrt(GT_area_current/GT_area_anchor)",
            "size_anchor_tie": "prefer_past",
            "GT_blind_replace_existing": False,
            "threshold_status": "diagnostic_not_calibrated_not_production_filters",
        },
    }
    tables, distribution, tiers = (
        bridge_tables(bridges),
        distribution_rows(sizes),
        eligibility_tables(all_gt, repeats=bootstrap_repeats),
    )
    destinations = (
        "misses-temporal.csv",
        "bridge-summary.csv",
        "size-memory.csv",
        "size-distributions.csv",
        "eligibility-recall.csv",
        "gt-eligibility-ledger.csv",
        "gt-blind-fill.csv",
        "manifest.json",
        "summary.json",
    )
    for directory in [
        geometry_csv.parent,
        root / "state/evaluation/phase0-before",
        root / "state/evaluation/phase0-after",
        root / "state/evaluation/phase0-misses",
    ]:
        a, b = output.resolve(), directory.resolve()
        if a == b or a in b.parents or b in a.parents:
            raise ValueError("temporal output overlaps a frozen report tree")
    if any((output / p).resolve() in protected for p in destinations):
        raise ValueError("temporal output would overwrite input")
    if output.exists() and any(output.iterdir()):
        old = output / "manifest.json"
        if (
            not old.is_file()
            or json.loads(old.read_text()) != manifest
            or any(p.name not in destinations or not p.is_file() for p in output.iterdir())
        ):
            raise ValueError("incompatible or unrelated temporal output")
    report = {
        **manifest,
        "schema": "autolabel-phase07-temporal/v1",
        "view": "direct_sampled",
        "profile": "raw",
        "population": {"gt": len(all_gt), "misses": len(misses)},
        "runs": provenance,
        "bridge_summary": tables,
        "size_distributions": distribution,
        "eligibility_recall": tiers,
        "bootstrap": {"seed": 42, "repeats": bootstrap_repeats, "unit": "GT_track", "stratification": "sequence"},
        "services_called": [],
        "annotation_change": False,
        "limitations": [
            "Oracle uses GT identity and future matched frames; optimistic diagnostic, not a deployable algorithm.",
            "ID-gated comparison retains oracle endpoint selection; not GT-blind. Same geometry isolates ID disagreement.",
            "GT-blind gap fill uses stitched IDs only; no replacement of existing poor boxes. Report false proposals separately.",
            "GT-size and area-derived scale leak GT; no measured size-memory algorithm or new annotations.",
            "Raw recall population covers only Car/Pedestrian. Bootstrap conditional on these clips/tracks.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    practical = [row for run in provenance for row in run["gt_blind"]]
    for filename, rows in (
        ("misses-temporal", bridges),
        ("bridge-summary", tables),
        ("size-memory", sizes),
        ("size-distributions", distribution),
        ("eligibility-recall", tiers),
        ("gt-blind-fill", practical),
    ):
        write_diagnostic_csv(output / f"{filename}.csv", rows, list(rows[0]) if rows else ["sequence"])
    eligibility_ledger = [
        {**r, "tier": eligibility(r)[0], "height_group": eligibility(r)[1], "truncation_group": eligibility(r)[2]}
        for r in all_gt
    ]
    write_diagnostic_csv(
        output / "gt-eligibility-ledger.csv",
        eligibility_ledger,
        list(eligibility_ledger[0]) if eligibility_ledger else ["sequence"],
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report["inputs_unchanged"] = all(file_hash(Path(p)) == value for p, value in before.items())
    if not report["inputs_unchanged"]:
        raise ValueError("input changed during temporal analysis")
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    return report
