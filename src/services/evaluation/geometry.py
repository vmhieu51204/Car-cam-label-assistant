"""Offline phase-0.6 box geometry evidence; never changes matching or predictions."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.services.evaluation.matching import bbox_iou, match_boxes
from src.services.evaluation.misses import MISS_FIELDS, analyze_direct, illustrate_prediction, write_diagnostic_csv
from src.services.evaluation.phase0 import FrozenRunData, canonical_predictions, file_hash, load_frozen_run

CONTAINMENT = 0.9  # User-selected diagnostic threshold, not a production filter.
GEOMETRY_FIELDS = (
    "pred_area_over_gt_area",
    "intersection_over_pred_area",
    "intersection_over_gt_area",
    "pred_width_over_gt_width",
    "pred_height_over_gt_height",
    "pred_aspect_over_gt_aspect",
    "center_dx_over_gt_width",
    "center_dy_over_gt_height",
    "center_distance_normalized",
)
OTHER_FIELDS = (
    "prediction_belongs_to_other_gt",
    "other_gt_id",
    "other_gt_native_class",
    "other_gt_ordinal",
    "other_gt_iou",
    "other_gt_iou_delta",
)
# (sequence, reason, geometry, occlusion-or-any, number of object-frames).
IMAGE_QUOTAS = (
    ("0013", "no_overlap", "not_applicable", "any", 5),
    ("0020", "no_overlap", "not_applicable", "any", 5),
    ("0013", "assignment_conflict", "not_applicable", "any", 3),
    ("0020", "assignment_conflict", "not_applicable", "any", 2),
    ("0013", "poor_localization", "pred_inside_gt", "partial", 2),
    ("0013", "poor_localization", "pred_inside_gt", "heavy", 1),
    ("0020", "poor_localization", "pred_inside_gt", "partial", 1),
    ("0020", "poor_localization", "pred_inside_gt", "heavy", 1),
    ("0013", "poor_localization", "gt_inside_pred", "partial", 1),
    ("0020", "poor_localization", "gt_inside_pred", "partial", 2),
    ("0020", "poor_localization", "gt_inside_pred", "heavy", 2),
    ("0013", "poor_localization", "shifted", "partial", 3),
    ("0020", "poor_localization", "shifted", "heavy", 2),
)


def box_geometry(gt: list[float], pred: list[float] | None) -> dict[str, Any]:
    """Continuous xyxy geometry; containment labels are hypotheses, not causes."""
    if pred is None:
        return {**dict.fromkeys(GEOMETRY_FIELDS), "geometry": "not_applicable"}
    if not all(math.isfinite(v) for v in [*gt, *pred]):
        raise ValueError("nonfinite box geometry")
    gw, gh, pw, ph = gt[2] - gt[0], gt[3] - gt[1], pred[2] - pred[0], pred[3] - pred[1]
    if min(gw, gh, pw, ph) <= 0:
        raise ValueError("box geometry needs positive extents")
    ga, pa = gw * gh, pw * ph
    inter = max(0, min(gt[2], pred[2]) - max(gt[0], pred[0])) * max(0, min(gt[3], pred[3]) - max(gt[1], pred[1]))
    dx, dy = ((pred[0] + pred[2] - gt[0] - gt[2]) / (2 * gw), (pred[1] + pred[3] - gt[1] - gt[3]) / (2 * gh))
    label = (
        "pred_inside_gt"
        if inter / pa >= CONTAINMENT and pa < ga
        else "gt_inside_pred"
        if inter / ga >= CONTAINMENT and pa > ga
        else "shifted"
    )
    return dict(
        zip(
            GEOMETRY_FIELDS,
            (pa / ga, inter / pa, inter / ga, pw / gw, ph / gh, (pw / ph) / (gw / gh), dx, dy, math.hypot(dx, dy)),
            strict=True,
        ),
        geometry=label,
    )


def other_gt_evidence(loaded: FrozenRunData, row: dict[str, Any]) -> dict[str, Any]:
    """Search all native object classes. Strict > comparison; stable native ordinal."""
    empty = {**dict.fromkeys(OTHER_FIELDS), "prediction_belongs_to_other_gt": False}
    if row["best_prediction_box"] is None:
        return empty
    candidates = [
        (i, g)
        for i, g in enumerate(loaded.native_rows)
        if g.source_frame_index == row["frame"]
        and g.native_class != "DontCare"
        and f"kitti/{row['sequence']}/{i}" != row["annotation_id"]
    ]
    if not candidates:
        return empty
    i, gt = min(candidates, key=lambda pair: (-bbox_iou(row["best_prediction_box"], pair[1].bbox_xyxy), pair[0]))
    overlap = bbox_iou(row["best_prediction_box"], gt.bbox_xyxy)
    return {
        "prediction_belongs_to_other_gt": overlap > row["best_iou"],
        "other_gt_id": str(gt.track_id),
        "other_gt_native_class": gt.native_class,
        "other_gt_ordinal": i,
        "other_gt_iou": overlap,
        "other_gt_iou_delta": overlap - row["best_iou"],
    }


def direct_ledger(loaded: FrozenRunData) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Best prediction evidence for every GT; preserve phase-0.5 miss reasons."""
    records, misses = analyze_direct(loaded)
    by_annotation = {x["annotation_id"]: x for x in misses}
    by_frame: dict[int, list[tuple[int, Any]]] = defaultdict(list)
    for ordinal, box in enumerate(loaded.direct):
        by_frame[box.frame].append((ordinal, box))
    ledger = []
    for original in records:
        row = by_annotation.get(original["annotation_id"])
        if row is None:
            candidates = by_frame[original["frame"]]
            best = (
                min(candidates, key=lambda p: (-bbox_iou(original["gt_box"], p[1].xyxy), p[0])) if candidates else None
            )
            row = {
                **original,
                "reason": "matched",
                "best_iou": bbox_iou(original["gt_box"], best[1].xyxy) if best else 0,
                "best_prediction_box": list(best[1].xyxy) if best else None,
                "best_prediction_class": best[1].label if best else None,
                "best_prediction_track_id": best[1].track_id if best else None,
                "best_prediction_ordinal": best[0] if best else None,
            }
        row = {**row, **other_gt_evidence(loaded, row)}
        stats = box_geometry(row["gt_box"], row["best_prediction_box"])
        row["geometry"] = stats["geometry"] if row["reason"] == "poor_localization" else "not_applicable"
        # All none-occlusion GT need an area ratio in the distribution ledger.
        row["distribution_area_ratio"] = stats["pred_area_over_gt_area"]
        row.update({k: stats[k] if row["reason"] == "poor_localization" else None for k in GEOMETRY_FIELDS})
        ledger.append(row)
    return ledger, [row for row in ledger if row["reason"] != "matched"]


def validate_misses(path: Path, expected: list[dict[str, Any]]) -> list[str]:
    """Require exact phase-0.5 CSV values/order; do not repair or reinterpret IDs."""
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    if not fields or len(set(fields)) != len(fields) or len(rows) != len(expected):
        raise ValueError("misses.csv population/header mismatch")
    if list(fields) != MISS_FIELDS:
        raise ValueError("misses.csv columns mismatch")
    for i, (saved, actual) in enumerate(zip(rows, expected, strict=True)):
        if set(saved) != set(fields):
            raise ValueError(f"misses.csv row shape mismatch at row {i + 1}")
        for key in fields:
            value = actual[key]
            cell = json.dumps(value) if isinstance(value, list) else "" if value is None else str(value)
            if saved[key] != cell:
                raise ValueError(f"misses.csv mismatch at row {i + 1}, field {key}")
    return list(fields)


def recall_rows(loaded: FrozenRunData) -> list[dict[str, Any]]:
    gt = tuple(g for g in loaded.gt if int(g.frame.frame_uid) in set(loaded.source))
    pred = canonical_predictions(loaded.direct, loaded.keys, loaded.item["id"])
    rows = []
    for threshold in (0.1, 0.3, 0.5, 0.7):
        hit = {p.annotation_id for p in match_boxes(gt, pred, iou_threshold=threshold).pairs}
        for label in ["all", *sorted({g.class_name for g in gt})]:
            selected = [g for g in gt if label == "all" or g.class_name == label]
            tp = sum(g.annotation_id in hit for g in selected)
            rows.append(
                {
                    "sequence": next(iter(loaded.keys.values())).sequence.sequence_uid,
                    "class": label,
                    "iou": threshold,
                    "tp": tp,
                    "fn": len(selected) - tp,
                    "gt": len(selected),
                    "recall": tp / len(selected) if selected else None,
                }
            )
    return rows


def select_images(
    rows: list[dict[str, Any]], *, seed: int = 42, quotas: tuple[tuple[str, str, str, str, int], ...] = IMAGE_QUOTAS
) -> list[dict[str, Any]]:
    """Fixed strata; shortage is an error, never filled from a different stratum."""
    rng = np.random.default_rng(seed)
    selected, shortages = [], []
    used: set[str] = set()
    for sequence, reason, geometry, occlusion, count in quotas:
        candidates = sorted(
            (
                r
                for r in rows
                if r["sequence"] == sequence
                and r["reason"] == reason
                and r["geometry"] == geometry
                and (occlusion == "any" or r["occlusion"] == occlusion)
                and r["annotation_id"] not in used
            ),
            key=lambda r: (r["frame"], r["annotation_id"]),
        )
        if len(candidates) < count:
            shortages.append(
                {
                    "sequence": sequence,
                    "reason": reason,
                    "geometry": geometry,
                    "occlusion": occlusion,
                    "requested": count,
                    "available": len(candidates),
                }
            )
            continue
        chosen = [candidates[int(i)] for i in rng.choice(len(candidates), size=count, replace=False)]
        selected.extend(chosen)
        used.update(r["annotation_id"] for r in chosen)
    if shortages:
        raise ValueError("insufficient image strata: " + json.dumps(shortages, sort_keys=True))
    return selected


def _percentiles(values: list[float]) -> dict[str, Any]:
    names = ("min", "p5", "p25", "median", "p75", "p95", "max")
    return {
        "n": len(values),
        **dict(
            zip(
                names,
                np.percentile(values, [0, 5, 25, 50, 75, 95, 100]).tolist() if values else [None] * 7,
                strict=True,
            )
        ),
    }


def distributions(ledger: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries, histograms = [], []
    unoccluded = [r for r in ledger if r["occlusion"] == "none"]
    for sequence in ["all", *sorted({r["sequence"] for r in unoccluded})]:
        for label in ["all", *sorted({r["class"] for r in unoccluded})]:
            for subset in ("all_none", "poor_localization_none"):
                for other in ("all", "true", "false"):
                    for truncation in ("all", "zero", "positive", "unknown"):
                        selected = [
                            r
                            for r in unoccluded
                            if (sequence == "all" or r["sequence"] == sequence)
                            and (label == "all" or r["class"] == label)
                            and (subset == "all_none" or r["reason"] == "poor_localization")
                            and (other == "all" or r["prediction_belongs_to_other_gt"] == (other == "true"))
                            and (
                                truncation == "all"
                                or (truncation == "unknown" and r["truncation"] is None)
                                or (truncation == "zero" and r["truncation"] == 0)
                                or (truncation == "positive" and r["truncation"] is not None and r["truncation"] > 0)
                            )
                        ]
                        if not selected:
                            continue
                        dimensions = {
                            "sequence": sequence,
                            "class": label,
                            "subset": subset,
                            "other_gt_flag": other,
                            "truncation_group": truncation,
                            "samples": len(selected),
                            "missing_prediction": sum(r["best_prediction_box"] is None for r in selected),
                        }
                        for metric, field, edges in (
                            ("iou", "best_iou", [0, 0.1, 0.3, 0.5, 0.7, 1]),
                            ("area_ratio", "distribution_area_ratio", [0, 0.25, 0.5, 1, 2, 4, math.inf]),
                        ):
                            values = [r[field] for r in selected if r[field] is not None]
                            summaries.append({**dimensions, "metric": metric, **_percentiles(values)})
                            counts, _ = np.histogram(values, bins=edges)
                            for i, count in enumerate(counts):
                                histograms.append(
                                    {
                                        **dimensions,
                                        "metric": metric,
                                        "lower": edges[i],
                                        "upper": "inf" if math.isinf(edges[i + 1]) else edges[i + 1],
                                        "upper_inclusive": i == len(counts) - 1,
                                        "count": int(count),
                                    }
                                )
    return summaries, histograms


def protect_output(
    output: Path, targets: list[Path], protected: set[Path], directories: list[Path], manifest: dict[str, Any]
) -> None:
    """Reject overlapping input/report trees and incompatible existing evidence."""
    destination = output.resolve()
    if any(p.resolve() in protected for p in targets):
        raise ValueError("output would overwrite an input or phase-0.5 report")
    for directory in directories:
        source = directory.resolve()
        if destination == source or destination in source.parents or source in destination.parents:
            raise ValueError("output overlaps a source image directory or phase-0.5/baseline report")
    previous = output / "sample-manifest.json"
    if output.exists() and any(output.iterdir()):
        if not previous.is_file() or json.loads(previous.read_text(encoding="utf-8")) != manifest:
            raise ValueError("output has incompatible previous sample/provenance")
        allowed = {p.resolve() for p in targets} | {(output / "images").resolve()}
        if any(p.resolve() not in allowed for p in output.rglob("*")):
            raise ValueError("output has unrelated files; refusing to overwrite")


def write_geometry_analysis(
    protocol: Path, *, misses_csv: Path, root: Path, output: Path, seed: int = 42
) -> dict[str, Any]:
    """Public phase-0.6 report: fixed 30-image quotas and seed, no reduced reports."""
    return _write_geometry_analysis(
        protocol, misses_csv=misses_csv, root=root, output=output, seed=seed, quotas=IMAGE_QUOTAS
    )


def _write_geometry_analysis(
    protocol: Path,
    *,
    misses_csv: Path,
    root: Path,
    output: Path,
    seed: int = 42,
    quotas: tuple[tuple[str, str, str, str, int], ...],
) -> dict[str, Any]:
    """Lower-level writer also exercises output safeguards on small test fixtures."""
    if type(seed) is not int or seed != 42:
        raise ValueError("phase-0.6 requires fixed seed 42")
    config = json.loads(protocol.read_text(encoding="utf-8"))
    if config.get("schema") != "autolabel-phase0-protocol/v1" or config.get("seed") != 42:
        raise ValueError("expected frozen phase0 protocol v1 with seed 42")
    ledger, enhanced, original_misses, recalls, runs = [], [], [], [], []
    protected = {protocol.resolve(), misses_csv.resolve()}
    image_dirs = {}
    for run in config["runs"]:
        loaded = load_frozen_run(run, root)
        _, old = analyze_direct(loaded)
        records, misses = direct_ledger(loaded)
        ledger.extend(records)
        enhanced.extend(misses)
        original_misses.extend(old)
        recalls.extend(recall_rows(loaded))
        runs.append(
            {
                "sequence": run["sequence"],
                "input_hashes": loaded.input_hashes,
                "class_map": run["class_map"],
                "excluded_sampled_classes": loaded.excluded_sampled_classes,
            }
        )
        protected.update((root / run[field]).resolve() for field in loaded.input_hashes)
        image_dirs[run["sequence"]] = (
            root / run["image_dir"]
            if run.get("image_dir")
            else (root / run["labels"]).parent.parent / "image_02" / run["sequence"]
        )
    fields = validate_misses(misses_csv, original_misses)
    selected = select_images(enhanced, seed=seed, quotas=quotas)
    samples = []
    for row in selected:
        source = image_dirs[row["sequence"]] / f"{row['frame']:06d}.png"
        if not source.is_file():
            raise FileNotFoundError(f"missing source image: {source}")
        protected.add(source.resolve())
        samples.append(
            {
                "annotation_id": row["annotation_id"],
                "sequence": row["sequence"],
                "frame": row["frame"],
                "reason": row["reason"],
                "geometry": row["geometry"],
                "occlusion": row["occlusion"],
                "source_sha256": file_hash(source),
                "path": f"images/{row['sequence']}-{row['frame']:06d}-gt{row['gt_id']}.png",
            }
        )
    provenance = {
        "protocol_sha256": file_hash(protocol),
        "misses_csv_sha256": file_hash(misses_csv),
        "runs": runs,
        "seed": seed,
        "containment_threshold": CONTAINMENT,
        "quotas": [list(q) for q in quotas],
    }
    filenames = (
        "misses-geometry.csv",
        "geometry-cross.csv",
        "unoccluded-ledger.csv",
        "distributions.csv",
        "histograms.csv",
        "recall.csv",
        "sample-manifest.json",
        "summary.json",
    )
    targets = [output / name for name in filenames] + [output / sample["path"] for sample in samples]
    manifest = {"schema": "autolabel-phase06-samples/v1", "provenance": provenance, "images": samples}
    report_dirs = [misses_csv.parent, root / "state/evaluation/phase0-before", root / "state/evaluation/phase0-after"]
    protect_output(output, targets, protected, [*image_dirs.values(), *report_dirs], manifest)
    cross_counts = Counter((r["sequence"], r["reason"], r["geometry"], r["occlusion"]) for r in enhanced)
    cross = [
        dict(zip(("sequence", "reason", "geometry", "occlusion"), keys, strict=True), count=count)
        for keys, count in sorted(cross_counts.items())
    ]
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in recalls:
        grouped[(row["class"], row["iou"])].append(row)
    for (label, threshold), rows in sorted(grouped.items()):
        tp, gt = sum(r["tp"] for r in rows), sum(r["gt"] for r in rows)
        recalls.append(
            {
                "sequence": "all",
                "class": label,
                "iou": threshold,
                "tp": tp,
                "fn": gt - tp,
                "gt": gt,
                "recall": tp / gt if gt else None,
            }
        )
    stats, hist = distributions(ledger)
    report = {
        "schema": "autolabel-phase06-geometry/v1"
        if quotas == IMAGE_QUOTAS
        else "autolabel-phase06-geometry-fixture/v1",
        "view": "direct_sampled",
        "profile": "raw",
        **provenance,
        "population": {
            "gt": len(ledger),
            "misses": len(enhanced),
            "poor_localization": sum(r["reason"] == "poor_localization" for r in enhanced),
        },
        "reason_counts": dict(Counter(r["reason"] for r in enhanced)),
        "geometry_counts": dict(Counter(r["geometry"] for r in enhanced if r["reason"] == "poor_localization")),
        "poor_other_gt_flag": sum(
            r["prediction_belongs_to_other_gt"] for r in enhanced if r["reason"] == "poor_localization"
        ),
        "cross": cross,
        "recall": recalls,
        "distributions": stats,
        "histograms": hist,
        "images": samples,
        "services_called": [],
        "annotation_change": False,
        "limitations": [
            "Other-GT flag compares overlap only, not verified object identity.",
            "Geometry/distributions do not establish GT-versus-mask box conventions.",
            "Low-IoU one-to-one recall is overlap evidence, not confirmed detection.",
            "Phase-0.5 bootstrap unchanged; no new confidence intervals.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir(exist_ok=True)
    rendered_hashes = {}
    for row, sample in zip(selected, samples, strict=True):
        caption = (
            f"{row['sequence']} source={row['frame']} GT {row['class']}:{row['gt_id']} (green) pred "
            f"{row['best_prediction_class']}:{row['best_prediction_track_id']} (red)\n"
            f"{row['reason']} / {row['geometry']} IoU={row['best_iou']:.3f} "
            f"other_GT={row['prediction_belongs_to_other_gt']}:{row['other_gt_id']}"
        )
        illustrate_prediction(
            image_dirs[row["sequence"]] / f"{row['frame']:06d}.png", output / sample["path"], row, caption=caption
        )
        rendered_hashes[sample["path"]] = file_hash(output / sample["path"])
    report["rendered_image_sha256"] = rendered_hashes
    write_diagnostic_csv(
        output / "misses-geometry.csv", enhanced, [*fields, *GEOMETRY_FIELDS, "geometry", *OTHER_FIELDS]
    )
    write_diagnostic_csv(output / "geometry-cross.csv", cross, ["sequence", "reason", "geometry", "occlusion", "count"])
    unoccluded = [r for r in ledger if r["occlusion"] == "none"]
    write_diagnostic_csv(
        output / "unoccluded-ledger.csv",
        unoccluded,
        [*fields, "matched_05", "matched_07", "distribution_area_ratio", "geometry", *OTHER_FIELDS],
    )
    for name, rows in (("distributions", stats), ("histograms", hist), ("recall", recalls)):
        write_diagnostic_csv(output / f"{name}.csv", rows, list(rows[0]) if rows else ["sequence"])
    (output / "sample-manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    return report
