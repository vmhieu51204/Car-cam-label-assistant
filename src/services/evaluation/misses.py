"""Read-only raw miss diagnostics and GT-track clustered recall intervals."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from src.models.evaluation import PredictionBox
from src.services.evaluation.matching import bbox_iou, match_boxes
from src.services.evaluation.phase0 import FrozenRunData, canonical_predictions, file_hash, load_frozen_run

OCCLUSIONS = ("none", "partial", "heavy", "unknown")
SIZES = ("small", "medium", "large")
MISS_FIELDS = [
    "sequence",
    "frame",
    "frame_sampled",
    "gt_id",
    "annotation_id",
    "class",
    "native_class",
    "occlusion",
    "truncation",
    "box_height",
    "box_area",
    "size",
    "gt_box",
    "best_iou",
    "best_prediction_class",
    "best_prediction_track_id",
    "best_prediction_box",
    "best_prediction_ordinal",
    "best_prediction_matched_gt_id",
    "best_prediction_matched_annotation_id",
    "reason",
]


def miss_reason(overlap: float, *, same_class: bool) -> str:
    """Labels are automatic diagnostic evidence, not confirmed root causes."""
    if overlap < 0.1:
        return "no_overlap"
    if overlap < 0.5:
        return "poor_localization"
    return "assignment_conflict" if same_class else "wrong_class"


def analyze_direct(loaded: FrozenRunData) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep baseline matching; nearest prediction searches all same-frame classes."""
    sampled = set(loaded.source)
    gt = tuple(x for x in loaded.gt if int(x.frame.frame_uid) in sampled)
    pred = canonical_predictions(loaded.direct, loaded.keys, loaded.item["id"])
    matches = {threshold: match_boxes(gt, pred, iou_threshold=threshold) for threshold in (0.5, 0.7)}
    hit = {threshold: {x.annotation_id for x in result.pairs} for threshold, result in matches.items()}
    assigned = {x.prediction_id: x.annotation_id for x in matches[0.5].pairs}
    gt_by_id = {x.annotation_id: x for x in gt}
    by_frame: dict[Any, list[PredictionBox]] = defaultdict(list)
    for box in pred:
        by_frame[box.frame].append(box)
    source_to_sampled = {frame: i for i, frame in enumerate(loaded.source)}
    records, misses = [], []
    for box in gt:
        x1, y1, x2, y2 = box.bbox_xyxy
        area = (x2 - x1) * (y2 - y1)
        native = loaded.native_by_annotation[box.annotation_id]
        row = {
            "sequence": box.frame.sequence.sequence_uid,
            "frame": int(box.frame.frame_uid),
            "frame_sampled": source_to_sampled[int(box.frame.frame_uid)],
            "gt_id": box.native_track_id,
            "annotation_id": box.annotation_id,
            "class": box.class_name,
            "native_class": box.native_category,
            "occlusion": loaded.occlusions.get(box.annotation_id, "unknown"),
            "truncation": native.truncated,
            "box_height": y2 - y1,
            "box_area": area,
            "size": "small" if area < 32**2 else "medium" if area < 96**2 else "large",
            "gt_box": list(box.bbox_xyxy),
            "matched_05": box.annotation_id in hit[0.5],
            "matched_07": box.annotation_id in hit[0.7],
        }
        records.append(row)
        if row["matched_05"]:
            continue
        candidates = by_frame[box.frame]
        nearest = (
            min(candidates, key=lambda p: (-bbox_iou(box.bbox_xyxy, p.bbox_xyxy), p.scoring_ordinal))
            if candidates
            else None
        )
        overlap = bbox_iou(box.bbox_xyxy, nearest.bbox_xyxy) if nearest else 0.0
        other_gt = gt_by_id.get(assigned.get(nearest.prediction_id, "")) if nearest else None
        # Preserve missing native IDs instead of exporting the adapter placeholder.
        native_track = loaded.direct[nearest.scoring_ordinal].track_id if nearest else None
        misses.append(
            {
                **row,
                "best_iou": overlap,
                "best_prediction_class": nearest.class_name if nearest else None,
                "best_prediction_track_id": native_track,
                "best_prediction_box": list(nearest.bbox_xyxy) if nearest else None,
                "best_prediction_ordinal": nearest.scoring_ordinal if nearest else None,
                "best_prediction_matched_gt_id": other_gt.native_track_id if other_gt else None,
                "best_prediction_matched_annotation_id": other_gt.annotation_id if other_gt else None,
                "reason": miss_reason(overlap, same_class=nearest is not None and nearest.class_name == box.class_name),
            }
        )
    return records, misses


def bootstrap_recall(records: list[dict[str, Any]], *, repeats: int = 10000, seed: int = 42) -> dict[str, Any]:
    """Resample complete GT tracks within each sequence, retaining micro recall.

    Common track weights are reused for all groups and thresholds. Empty-support
    replicates are excluded, never converted into zero recall. <2 supporting
    tracks suppresses the interval even when a degenerate percentile exists.
    """
    if type(repeats) is not int or repeats < 1:
        raise ValueError("bootstrap repeats must be a positive integer")
    if any(row["gt_id"] is None for row in records):
        raise ValueError("GT track IDs are required for track bootstrap")
    tracks = sorted({(x["sequence"], str(x["gt_id"])) for x in records})
    sequences = sorted({x[0] for x in tracks})
    classes = sorted({x["class"] for x in records})
    groups: list[tuple[str, str, str]] = [("overall", "", "")]
    groups += [("occlusion_size", o, s) for o in OCCLUSIONS for s in SIZES]
    groups += [("occlusion_class", o, c) for o in OCCLUSIONS for c in classes]
    group_index = {key: i for i, key in enumerate(groups)}
    track_index = {key: i for i, key in enumerate(tracks)}
    counts = np.zeros((len(tracks), len(groups)), dtype=np.int64)
    tp = {threshold: np.zeros_like(counts) for threshold in (0.5, 0.7)}
    for row in records:
        if row["occlusion"] not in OCCLUSIONS or row["size"] not in SIZES:
            raise ValueError("unknown diagnostic group")
        i = track_index[(row["sequence"], str(row["gt_id"]))]
        columns = (
            0,
            group_index[("occlusion_size", row["occlusion"], row["size"])],
            group_index[("occlusion_class", row["occlusion"], row["class"])],
        )
        for j in columns:
            counts[i, j] += 1
            for threshold, field in ((0.5, "matched_05"), (0.7, "matched_07")):
                tp[threshold][i, j] += int(row[field])
    rng = np.random.default_rng(seed)
    weights = np.zeros((repeats, len(tracks)), dtype=np.int64)
    for sequence in sequences:
        indices = [i for i, key in enumerate(tracks) if key[0] == sequence]
        n = len(indices)
        weights[:, indices] = rng.multinomial(n, np.full(n, 1 / n), size=repeats)
    tables: dict[str, Any] = {"overall": [], "occlusion_size": [], "occlusion_class": []}
    for scope in [*sequences, "all"]:
        indices = [i for i, key in enumerate(tracks) if scope == "all" or key[0] == scope]
        observed = counts[indices]
        denominator = weights[:, indices] @ observed
        support = np.count_nonzero(observed, axis=0)
        for threshold in (0.5, 0.7):
            numerator = weights[:, indices] @ tp[threshold][indices]
            for j, (dimension, occlusion, category) in enumerate(groups):
                gt_count = int(observed[:, j].sum())
                tp_count = int(tp[threshold][indices, j].sum())
                valid = denominator[:, j] > 0
                values = numerator[valid, j] / denominator[valid, j]
                enough = support[j] >= 2 and len(values) > 0
                limits = np.percentile(values, (2.5, 97.5)) if enough else np.full(2, np.nan)
                entry = {
                    "sequence": scope,
                    "iou": threshold,
                    "tp": tp_count,
                    "gt": gt_count,
                    "recall": tp_count / gt_count if gt_count else None,
                    "support_tracks": int(support[j]),
                    "ci_low": float(limits[0]) if enough else None,
                    "ci_high": float(limits[1]) if enough else None,
                    "ci_status": "available" if enough else "insufficient_tracks",
                    "bootstrap_valid": int(valid.sum()),
                    "bootstrap_zero_denominator": repeats - int(valid.sum()),
                }
                if dimension != "overall":
                    entry["occlusion"] = occlusion
                    entry["size" if dimension == "occlusion_size" else "class"] = category
                tables[dimension].append(entry)
    return {
        "method": "stratified_GT_track_percentile_bootstrap",
        "seed": seed,
        "repeats": repeats,
        "cluster_key": ["sequence", "gt_id"],
        "tracks": len(tracks),
        **tables,
    }


def write_diagnostic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()}
            )


def illustrate_prediction(source: Path, target: Path, row: dict[str, Any], *, caption: str | None = None) -> None:
    with Image.open(source) as original:
        image = original.convert("RGB")
    draw = ImageDraw.Draw(image)
    # Thick GT and dashed thin prediction make identical extents both visible.
    draw.rectangle(row["gt_box"], outline=(0, 200, 0), width=4)
    if row["best_prediction_box"] is not None:
        x1, y1, x2, y2 = row["best_prediction_box"]
        for x in range(round(x1), round(x2) + 1, 8):
            draw.line((x, y1, min(x + 4, x2), y1), fill=(255, 0, 0), width=1)
            draw.line((x, y2, min(x + 4, x2), y2), fill=(255, 0, 0), width=1)
        for y in range(round(y1), round(y2) + 1, 8):
            draw.line((x1, y, x1, min(y + 4, y2)), fill=(255, 0, 0), width=1)
            draw.line((x2, y, x2, min(y + 4, y2)), fill=(255, 0, 0), width=1)
    label = (
        f"{row['sequence']} frame={row['frame']} GT {row['class']}:{row['gt_id']} (green) | "
        f"pred {row['best_prediction_class']}:{row['best_prediction_track_id']} (red) | "
        f"{row['reason']} IoU={row['best_iou']:.3f}"
    )
    if caption is not None:
        label = caption
    # Keep geometry at original resolution; caption is only a visualization.
    draw.text(
        (2, max(0, image.height - 14 * (label.count("\n") + 1))),
        label,
        fill=(0, 0, 0),
        stroke_width=1,
        stroke_fill=(255, 255, 255),
    )
    image.save(target)


def write_miss_analysis(
    protocol: Path, *, root: Path, output: Path, seed: int = 42, bootstrap_repeats: int = 10000, image_count: int = 30
) -> dict[str, Any]:
    if type(image_count) is not int or image_count < 0:
        raise ValueError("image count must be a nonnegative integer")
    config = json.loads(protocol.read_text(encoding="utf-8"))
    if config.get("schema") != "autolabel-phase0-protocol/v1" or config.get("seed") != 42:
        raise ValueError("expected frozen phase0 protocol v1 with evaluation seed 42")
    records, misses, provenance = [], [], []
    image_dirs: dict[str, Path] = {}
    for run in config["runs"]:
        loaded = load_frozen_run(run, root)
        rows, missing = analyze_direct(loaded)
        records.extend(rows)
        misses.extend(missing)
        sequence = run["sequence"]
        if sequence in image_dirs:
            raise ValueError("one frozen configuration per sequence is required")
        image_dirs[sequence] = (
            root / run["image_dir"]
            if run.get("image_dir")
            else (root / run["labels"]).parent.parent / "image_02" / sequence
        )
        provenance.append(
            {
                "sequence": sequence,
                "config_id": run["config_id"],
                "input_hashes": loaded.input_hashes,
                "class_map": run["class_map"],
                "excluded_gt_classes": loaded.excluded_sampled_classes,
                "excluded_count_basis": "sampled_source_frames",
                "evaluated_source_frames": loaded.source,
                "model_commit": loaded.result.get("model_runtime", {})
                .get("environment", {})
                .get("versions", {})
                .get("sam3_commit"),
                "checkpoint_sha256": loaded.result.get("model_runtime", {}).get("checkpoint_sha256"),
            }
        )
    bootstrap = bootstrap_recall(records, repeats=bootstrap_repeats, seed=seed)
    picks = np.random.default_rng(seed).choice(len(misses), size=min(image_count, len(misses)), replace=False)
    illustrations = []
    for pick in picks:
        row = misses[int(pick)]
        source = image_dirs[row["sequence"]] / f"{row['frame']:06d}.png"
        if not source.is_file():
            raise FileNotFoundError(f"missing source image: {source}")
        if output.resolve() == source.parent.resolve():
            raise ValueError("output may not be the original image directory")
        illustrations.append(
            {
                "annotation_id": row["annotation_id"],
                "sequence": row["sequence"],
                "frame": row["frame"],
                "gt_id": row["gt_id"],
                "reason": row["reason"],
                "path": f"images/{row['sequence']}-{row['frame']:06d}-gt{row['gt_id']}.png",
                "source_image_sha256": file_hash(source),
            }
        )
    previous_summary = output / "summary.json"
    if previous_summary.is_file():
        previous = json.loads(previous_summary.read_text(encoding="utf-8"))
        if previous.get("schema") != "autolabel-phase05-misses/v1" or previous.get("images") != illustrations:
            raise ValueError("output contains a different image sample; choose a new output directory")
    expected_names = {Path(x["path"]).name for x in illustrations}
    existing_names = {p.name for p in (output / "images").iterdir()} if (output / "images").is_dir() else set()
    if existing_names - expected_names:
        raise ValueError("output contains stale or unrelated images; choose a new output directory")
    # Never overwrite an input artifact even if a custom protocol names it like
    # a report. No recursive cleanup of user's existing output is performed.
    destinations = {
        output / name
        for name in ("summary.json", "misses.csv", "recall-ci.csv", "occlusion-size.csv", "occlusion-class.csv")
    }
    sources = {protocol.resolve()} | {
        (root / run[field]).resolve()
        for run in config["runs"]
        for field in ("result", "mapping", "labels", "cvat_export")
        if run.get(field)
    }
    if {p.resolve() for p in destinations} & sources:
        raise ValueError("output would overwrite an input artifact")
    output.mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir(exist_ok=True)
    for evidence in illustrations:
        row = next(x for x in misses if x["annotation_id"] == evidence["annotation_id"])
        illustrate_prediction(image_dirs[row["sequence"]] / f"{row['frame']:06d}.png", output / evidence["path"], row)

    write_diagnostic_csv(output / "misses.csv", misses, MISS_FIELDS)
    shared = [
        "sequence",
        "iou",
        "tp",
        "gt",
        "recall",
        "support_tracks",
        "ci_low",
        "ci_high",
        "ci_status",
        "bootstrap_valid",
        "bootstrap_zero_denominator",
    ]
    write_diagnostic_csv(output / "recall-ci.csv", bootstrap["overall"], shared)
    for dimension, filename, category in (
        ("occlusion_size", "occlusion-size.csv", "size"),
        ("occlusion_class", "occlusion-class.csv", "class"),
    ):
        write_diagnostic_csv(output / filename, bootstrap[dimension], [*shared, "occlusion", category])
    report = {
        "schema": "autolabel-phase05-misses/v1",
        "view": "direct_sampled",
        "profile": "raw",
        "protocol_sha256": file_hash(protocol),
        "seed": seed,
        "runs": provenance,
        "population": {"gt": len(records), "misses": len(misses), "tracks": bootstrap["tracks"]},
        "reason_counts": dict(Counter(x["reason"] for x in misses)),
        "by_sequence": {
            s: {
                "gt": sum(x["sequence"] == s for x in records),
                "misses": sum(x["sequence"] == s for x in misses),
                "reasons": dict(Counter(x["reason"] for x in misses if x["sequence"] == s)),
            }
            for s in sorted(image_dirs)
        },
        "by_class": dict(Counter(x["class"] for x in misses)),
        "bootstrap": bootstrap,
        "images_requested": image_count,
        "images": illustrations,
        "services_called": [],
        "annotation_change": False,
        "note": "Automatic miss signals, not confirmed causes. CI is conditional on these tracks/clips, not general video quality.",
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
