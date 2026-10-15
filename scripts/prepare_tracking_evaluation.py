"""Inventory KITTI and reserve sequence-level DEV/HELD-OUT. Never calls a GPU/provider."""

import argparse
import csv
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from kaggle.runtime_probe import sha256_file
from src.models.evaluation import SequenceKey, SourceFile
from src.services.evaluation.tracking_payload import (
    SLOW_SECONDS_PER_FRAME,
    canonical_bytes,
    materialize,
    save_equal_or_new,
    split_windows,
)
from tools.public_gt.kitti import discover_images, read_native_rows


def sha(path: Path) -> str:
    return sha256_file(path)


def split_sequences(names: list[str], seed: int = 42) -> dict[str, list[str]]:
    forced_dev = {"0013", "0020"}
    remaining = sorted(set(names) - forced_dev)
    random.Random(seed).shuffle(remaining)
    hold = remaining[: len(remaining) // 2]
    return {"dev": sorted(set(names) - set(hold)), "held_out": sorted(hold)}


def inventory_sequence(training: Path, sequence: str) -> dict[str, Any]:
    images = discover_images(training / "image_02" / sequence)
    labels = training / "label_02" / f"{sequence}.txt"
    label_hash = sha(labels)
    key = SequenceKey(dataset_id="kitti", split="training", sequence_uid=sequence)
    rows = read_native_rows(
        labels,
        sequence=key,
        source=SourceFile(relative_path=f"label_02/{sequence}.txt", sha256=label_hash, role="labels"),
        frame_count=len(images),
    )
    classes = {}
    for name in ("Car", "Pedestrian"):
        chosen = [r for r in rows if r.native_class == name]
        by_track: dict[int, list] = defaultdict(list)
        for row in chosen:
            by_track[row.track_id].append(row)
        classes[name] = {
            "tracks": len(by_track),
            "object_frames": len(chosen),
            "occlusion": dict(Counter(("none", "partial", "heavy", "unknown")[r.occluded] for r in chosen)),
            "truncation_native": dict(Counter(str(r.truncated) for r in chosen)),
            "height_bins": dict(
                Counter(
                    "lt25"
                    if r.bbox_xyxy[3] - r.bbox_xyxy[1] < 25
                    else "25to50"
                    if r.bbox_xyxy[3] - r.bbox_xyxy[1] < 50
                    else "ge50"
                    for r in chosen
                )
            ),
            "occluded_tracks": sum(any(r.occluded in {1, 2} for r in t) for t in by_track.values()),
            "interrupted_gt_tracks": sum(
                any(
                    b.source_frame_index - a.source_frame_index > 1
                    for a, b in zip(
                        sorted(t, key=lambda r: r.source_frame_index), sorted(t, key=lambda r: r.source_frame_index)[1:]
                    )
                )
                for t in by_track.values()
            ),
        }
    records = [
        {"source_frame": i, "name": p.name, "sha256": sha(p), "bytes": p.stat().st_size} for i, p in enumerate(images)
    ]
    return {
        "sequence": sequence,
        "frames": len(images),
        "classes": classes,
        "images": records,
        "images_content_sha256": hashlib.sha256(canonical_bytes(records)).hexdigest(),
        "dontcare_object_frames": sum(r.native_class == "DontCare" for r in rows),
        "all_native_classes": dict(Counter(r.native_class for r in rows)),
        "labels_sha256": label_hash,
        "image_names_sha256": hashlib.sha256("\n".join(p.name for p in images).encode()).hexdigest(),
        "image_hash_scope": "Each source image byte hash plus ordered filenames/sizes, separate from image_names_sha256",
    }


def prepare(
    dataset_root: Path,
    output: Path,
    protocol_path: Path,
    *,
    top_n: int = 3,
    materialize_archives: bool = False,
    max_bytes: int = 2_000_000_000,
    session_seconds: float = 21600,
) -> dict:
    training = dataset_root / "training"
    names = sorted(p.stem for p in (training / "label_02").glob("[0-9][0-9][0-9][0-9].txt"))
    if not names or top_n < 1:
        raise ValueError("KITTI labels and positive top-n required")
    old = json.loads(protocol_path.read_text(encoding="utf-8")) if protocol_path.exists() else None
    inventory = [inventory_sequence(training, name) for name in names]
    splits = split_sequences(names)
    if old:
        if old["splits"] != splits:
            raise ValueError("existing sequence split changed; cannot silently re-split")
        for row in inventory:
            hashes = old.get("input_hashes", {}).get(row["sequence"], {})
            for field, value in (
                ("labels", row["labels_sha256"]),
                ("image_names", row["image_names_sha256"]),
                ("images_content", row["images_content_sha256"]),
            ):
                if field in hashes and hashes[field] != value:
                    raise ValueError("inventory input hash mismatch; keep existing protocol")
    phase0 = Path(__file__).resolve().parents[1] / "eval/phase0-protocol.json"
    runs = old["runs"] if old else json.loads(phase0.read_text(encoding="utf-8"))["runs"]

    def priority(row: dict) -> tuple:
        cs = row["classes"].values()
        return (
            sum(c["interrupted_gt_tracks"] for c in cs),
            sum(c["occluded_tracks"] for c in cs),
            sum(c["tracks"] for c in cs),
            row["sequence"],
        )

    candidates = sorted(
        [r for r in inventory if r["sequence"] in splits["dev"] and r["sequence"] not in {"0013", "0020"}],
        key=priority,
        reverse=True,
    )[:top_n]
    output.mkdir(parents=True, exist_ok=True)
    if old:
        history = output / "history" / sha(protocol_path)
        save_equal_or_new(history / "protocol.json", protocol_path.read_bytes())
        for filename in (
            "inventory.json",
            "inventory.csv",
            "diagnosis.json",
            "events.csv",
            "evidence-equivalence.json",
        ):
            source = output / filename
            if source.exists():
                save_equal_or_new(history / filename, source.read_bytes())
    plans = []
    for row in candidates:
        directory = output / "prepared-dev" / (row["sequence"] + "-" + row["images_content_sha256"][:12])
        if materialize_archives or (directory / "manifest.json").exists():
            kernels = materialize(training, row, directory, max_bytes=max_bytes, session_seconds=session_seconds)
        else:
            spans = split_windows(
                [r["bytes"] for r in row["images"]], max_bytes=max_bytes, session_seconds=session_seconds
            )
            kernels = [
                {
                    "sequence": row["sequence"],
                    "split": "dev",
                    "source_start": a,
                    "source_stop_exclusive": b,
                    "source_frames": list(range(a, b)),
                    "frames": b - a,
                    "archive_materialized": False,
                    "rough_budget_seconds": (b - a) * SLOW_SECONDS_PER_FRAME * 1.25 + 180,
                    "status": "plan_only_not_uploaded",
                    "gt_in_payload": False,
                    "uploaded": False,
                }
                for a, b in spans
            ]
        for kernel in kernels:
            plans.append(
                {
                    **kernel,
                    "prepared_directory": str(directory),
                    "cadence": "all_frames",
                    "kernel_count": len(kernels),
                    "classes": ["car", "person"],
                    "window": 50,
                    "overlap": 5,
                }
            )
        overview = output / "dev-overviews" / (row["sequence"] + "-" + row["images_content_sha256"][:12] + ".jpg")
        if not overview.exists():
            sheet = Image.new("RGB", (960, 224), "white")
            draw = ImageDraw.Draw(sheet)
            for col, frame in enumerate(sorted({0, row["frames"] // 2, row["frames"] - 1})):
                with Image.open(training / "image_02" / row["sequence"] / f"{frame:06d}.png") as image:
                    image = image.convert("RGB")
                    image.thumbnail((320, 180))
                    sheet.paste(image, (col * 320, 20))
                draw.text((col * 320 + 4, 204), f"{row['sequence']} source frame {frame} - unlabeled", fill="black")
            overview.parent.mkdir(parents=True, exist_ok=True)
            sheet.save(overview, quality=90)
        row["dev_overview"] = {
            "path": str(overview),
            "sha256": sha(overview),
            "frames": sorted({0, row["frames"] // 2, row["frames"] - 1}),
        }
    held_plans = old.get("gpu_plan_held_out", []) if old else []
    if old and not held_plans:
        held_plans = [g for g in old.get("gpu_plan", []) if g.get("split") == "held_out"]
    protocol = {
        "schema": "autolabel-phase1-tracking-protocol/v1",
        "revision": 2,
        "final_ledger_root": os.path.relpath((output / "final-ledgers").resolve(), protocol_path.resolve().parent),
        "seed": 42,
        "dataset_root": str(dataset_root),
        "class_map": {"Car": "car", "Pedestrian": "person"},
        "splits": splits,
        "split_rule": "Shuffle sorted sequences except 0013/0020 with seed42; first floor(N/2) held-out",
        "held_out_history_status": old.get("held_out_history_status", "unverified") if old else "unverified",
        "held_out_history": "Unverified team history; no claim of untouched held-out",
        "runs": runs,
        "prediction_sequences": sorted({r["sequence"] for r in runs}),
        "frozen_prediction_protocol": "eval/phase0-protocol.json",
        "frozen_prediction_protocol_sha256": sha(phase0),
        "input_hashes": {
            r["sequence"]: {
                "labels": r["labels_sha256"],
                "image_names": r["image_names_sha256"],
                "images_content": r["images_content_sha256"],
            }
            for r in inventory
        },
        "gt_interruption_definition": "At least one gap >1 source frame between GT observations of the same native track; GT absence is not model failure",
        "priority_rule": "DEV only; descending GT-interrupted tracks, occluded tracks, total Car/Pedestrian tracks, sequence ID; no predictions",
        "priorities": [
            {
                "sequence": r["sequence"],
                "frames": r["frames"],
                "class_composition": r["classes"],
                "scene_diversity": "unavailable: class/GT structure is not semantic scene diversity",
                "dev_overview": r["dev_overview"],
            }
            for r in candidates
        ],
        "gate": "DEV pass AND held-out no degradation; missing or unverified held-out keeps flag off",
        "held_out_uses": "One explicit final evaluation with verified history and frozen DEV parameters; never tuning",
        "gpu_run_conditions": ["Verify Kaggle permission after HTTP403", "User approves kernel/minute budget"],
        "gpu_plan": plans,
        "gpu_plan_held_out": held_plans,
        "gpu_budget": {
            "dev_kernels": len(plans),
            "unique_source_frames": sum(r["frames"] for r in candidates),
            "rough_minutes": sum(p["rough_budget_seconds"] for p in plans) / 60,
        },
        "budget_basis": "T4 car/person; 598.39s/100 and 1569.34s/419 sampled frames; 25% reserve +180s per kernel; rough extrapolation, not a promise",
        "domain_limit": "KITTI Europe Car/Pedestrian cannot validate motorcycle-heavy customer dashcam; need hand-tracked customer video and scene-cut labels",
    }
    (output / "inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    with (output / "inventory.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = [
            "sequence",
            "frames",
            "class",
            "tracks",
            "object_frames",
            "occluded_tracks",
            "interrupted_gt_tracks",
            "dontcare",
            "occlusion_none",
            "occlusion_partial",
            "occlusion_heavy",
            "occlusion_unknown",
            "height_lt25",
            "height_25to50",
            "height_ge50",
            "truncation_native",
            "labels_sha256",
            "images_content_sha256",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in inventory:
            for name, c in row["classes"].items():
                writer.writerow(
                    {
                        "sequence": row["sequence"],
                        "frames": row["frames"],
                        "class": name,
                        **{k: c[k] for k in ("tracks", "object_frames", "occluded_tracks", "interrupted_gt_tracks")},
                        "dontcare": row["dontcare_object_frames"],
                        **{"occlusion_" + k: c["occlusion"].get(k, 0) for k in ("none", "partial", "heavy", "unknown")},
                        **{"height_" + k: c["height_bins"].get(k, 0) for k in ("lt25", "25to50", "ge50")},
                        "truncation_native": json.dumps(c["truncation_native"], sort_keys=True),
                        "labels_sha256": row["labels_sha256"],
                        "images_content_sha256": row["images_content_sha256"],
                    }
                )
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("../AutoLabelDatasets"))
    parser.add_argument("--output", type=Path, default=Path("state/evaluation/phase1-tracking"))
    parser.add_argument("--protocol", type=Path, default=Path("eval/phase1-tracking-protocol.json"))
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--materialize", action="store_true", help="Build local DEV ZIPs only; never upload/infer")
    parser.add_argument("--max-bytes", type=int, default=2_000_000_000)
    parser.add_argument("--session-seconds", type=float, default=21600)
    args = parser.parse_args()
    if args.top_n < 1:
        parser.error("positive top-n required")
    result = prepare(
        args.dataset_root,
        args.output,
        args.protocol,
        top_n=args.top_n,
        materialize_archives=args.materialize,
        max_bytes=args.max_bytes,
        session_seconds=args.session_seconds,
    )
    print(json.dumps({"splits": result["splits"], "gpu_budget": result["gpu_budget"], "services_called": []}))


if __name__ == "__main__":
    main()
