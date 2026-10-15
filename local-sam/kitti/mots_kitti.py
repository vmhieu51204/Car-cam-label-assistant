"""Score SAM masks against KITTI MOTS with the official TrackEval (``KittiMOTS``: car, pedestrian).

Needs the TrackEval venv (numpy < 1.24 + pycocotools), a payload built with ``prepare_kitti.py --mots`` and runs
benchmarked with ``--save-masks``:

    ~/venvs/trackeval/bin/python -I kitti/mots_kitti.py --payload ~/data/kitti_0014.zip \
        --runs logs/kitti/<a>/*_benchmark_sam3 logs/kitti/<b>/*_benchmark_sam31 --out logs/kitti/compare_0014

For each run: decode our row-major ``mask_rle`` masks, map prompts to MOTS classes (car = 1, pedestrian = 2; van,
truck and cyclist prompts are not MOTS classes and are dropped), give overlapping pixels to the higher-scoring mask
(KITTI MOTS forbids overlaps), encode COCO RLE and run HOTA + CLEAR (MOTSA, sMOTSA) + Identity on the scored frames,
renumbered from 0. Writes ``<out>/mots_official.json`` and a section in ``<out>/kitti_report.md``.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # kitti_metrics / eval_kitti, also under python -I

from eval_kitti import _fmt, _safe, load_payload, run_info  # noqa: E402
from kitti_metrics import PayloadReader  # noqa: E402
from trackeval_kitti import HOTA_FIELDS, _renumber, replace_section  # noqa: E402

REPORT_MARKER = "<!-- mots_kitti.py -->"
MOTS_CLASS_IDS = {"car": 1, "person": 2, "pedestrian": 2}  # our class names (kitti2 / kitti5) -> MOTS class id


def decode_rle(rle: dict) -> np.ndarray:
    """Our ``mask_rle``: row-major run lengths starting with a run of zeros (``samlocal.backend.mask_rle``)."""
    height, width = rle["height"], rle["width"]
    counts = np.asarray(rle["counts"], dtype=np.int64)
    values = np.arange(len(counts)) % 2 == 1
    flat = np.repeat(values, counts)
    if flat.size != height * width:
        raise ValueError(f"mask_rle covers {flat.size} pixels, expected {height}x{width}")
    return flat.reshape(height, width)


def tracker_lines(predictions: dict, position_to_source: dict[int, int], frame_to_index: dict[int, int],
                  prompt_to_class: dict[str, str]) -> tuple[list[str], dict[str, int]]:
    """KITTI MOTS result rows ``frame id class_id h w rle``; overlaps go to the higher score."""
    from pycocotools import mask as mask_utils

    stats: dict[str, int] = defaultdict(int)
    track_numbers: dict[tuple[int, str], int] = {}
    lines = []
    for position, detections in sorted(predictions["frames"].items(), key=lambda item: int(item[0])):
        source = position_to_source[int(position)]
        if source not in frame_to_index:
            continue
        occupied = None
        seen: set[int] = set()
        for det in sorted(detections, key=lambda d: -(d.get("score") or 0.0)):
            class_id = MOTS_CLASS_IDS.get(prompt_to_class.get(det["label"], ""))
            if class_id is None:
                stats["not_a_mots_class"] += 1
                continue
            if "mask_rle" not in det:
                raise SystemExit("predictions.json has no mask_rle: rerun the benchmark with --save-masks")
            mask = decode_rle(det["mask_rle"])
            occupied = np.zeros_like(mask) if occupied is None else occupied
            overlap = mask & occupied
            if overlap.any():
                stats["masks_trimmed_for_overlap"] += 1
                mask = mask & ~occupied
            if not mask.any():
                stats["masks_empty_after_overlap"] += 1
                continue
            key = (class_id, f"{det['label']}|{det['track_id']}")
            number = track_numbers.setdefault(key, sum(1 for c, _ in track_numbers if c == class_id))
            object_id = class_id * 1000 + number
            if number >= 1000 or object_id in seen:
                stats["dropped_id_overflow_or_duplicate"] += 1
                continue
            seen.add(object_id)
            occupied |= mask
            encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
            lines.append(f"{frame_to_index[source]} {object_id} {class_id} {mask.shape[0]} {mask.shape[1]} "
                         f"{encoded['counts'].decode('ascii')}")
            stats["masks_written"] += 1
    return lines, dict(stats)


def run_mots(root: Path, tracker: str, sequence: str, num_frames: int) -> dict[str, dict[str, Any]]:
    import trackeval

    eval_config = {**trackeval.Evaluator.get_default_eval_config(), "USE_PARALLEL": False, "PRINT_RESULTS": False,
                   "PRINT_CONFIG": False, "TIME_PROGRESS": False, "OUTPUT_SUMMARY": False, "OUTPUT_DETAILED": False,
                   "PLOT_CURVES": False, "LOG_ON_ERROR": str(root / "error_log.txt")}
    dataset_config = {**trackeval.datasets.KittiMOTS.get_default_dataset_config(),
                      "GT_FOLDER": str(root / "gt"), "TRACKERS_FOLDER": str(root / "trackers"),
                      "OUTPUT_FOLDER": str(root / "output"), "TRACKERS_TO_EVAL": [tracker],
                      "CLASSES_TO_EVAL": ["car", "pedestrian"], "SEQ_INFO": {sequence: num_frames},
                      "PRINT_CONFIG": False}
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR({"PRINT_CONFIG": False}),
               trackeval.metrics.Identity({"PRINT_CONFIG": False})]
    with contextlib.redirect_stdout(io.StringIO()):
        results, messages = trackeval.Evaluator(eval_config).evaluate(
            [trackeval.datasets.KittiMOTS(dataset_config)], metrics)
    message = messages["KittiMOTS"][tracker]
    if message != "Success":
        raise SystemExit(f"TrackEval KittiMOTS failed for {tracker}: {message}")
    combined = results["KittiMOTS"][tracker]["COMBINED_SEQ"]
    out = {}
    for cls in ("car", "pedestrian"):
        h, c, i = combined[cls]["HOTA"], combined[cls]["CLEAR"], combined[cls]["Identity"]
        row: dict[str, Any] = {field: float(np.mean(h[field])) for field in HOTA_FIELDS}
        row.update({"MOTSA": float(c["MOTA"]), "sMOTSA": float(c["sMOTA"]), "IDSW": int(c["IDSW"]),
                    "Frag": int(c["Frag"]), "MT": int(c["MT"]), "ML": int(c["ML"]), "TP": int(c["CLR_TP"]),
                    "FN": int(c["CLR_FN"]), "FP": int(c["CLR_FP"]), "IDF1": float(i["IDF1"]), "IDR": float(i["IDR"])})
        row["recall"] = row["TP"] / (row["TP"] + row["FN"]) if row["TP"] + row["FN"] else None
        out[cls] = row
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--payload", type=Path, required=True, help="kitti_<seq>.zip built with --mots")
    parser.add_argument("--runs", type=Path, nargs="+", required=True, help="run folders benchmarked with --save-masks")
    parser.add_argument("--out", type=Path, required=True, help="output folder (shared with eval_kitti.py --out)")
    args = parser.parse_args(argv)
    try:
        import pycocotools  # noqa: F401
        import trackeval  # noqa: F401
    except ImportError:
        raise SystemExit("needs trackeval + pycocotools: run with ~/venvs/trackeval/bin/python")

    manifest, _, position_to_source = load_payload(args.payload)
    if "mots_labels" not in manifest:
        raise SystemExit(f"{args.payload}: no KITTI MOTS labels - rebuild it with prepare_kitti.py --mots")
    gt_text = PayloadReader(args.payload).read_text(manifest["mots_labels"])
    sequence = manifest["sequence"]
    output: dict[str, Any] = {"schema": "local-sam-kitti-mots/v1", "sequence": sequence, "runs": {}}
    for run in args.runs:
        info = run_info(run)
        label = info["run_label"]
        count = info["num_frames"] or len(position_to_source)
        frames = [position_to_source[p] for p in range(min(count, len(position_to_source)))]
        index = {frame: i for i, frame in enumerate(frames)}
        tracker = _safe(label)
        root = args.out / "mots_official" / tracker
        if root.exists():
            shutil.rmtree(root)
        (root / "gt" / "label_02").mkdir(parents=True)
        (root / "gt" / "label_02" / f"{sequence}.txt").write_text(
            "\n".join(_renumber(gt_text.splitlines(), index)) + "\n", encoding="utf-8")
        predictions = json.loads((run / "predictions.json").read_text(encoding="utf-8"))
        lines, stats = tracker_lines(predictions, position_to_source, index, manifest["prompt_to_class"])
        data = root / "trackers" / tracker / "data"
        data.mkdir(parents=True)
        (data / f"{sequence}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        output["runs"][label] = {"model_version": info["model_version"], "settings": info["settings"],
                                 "conversion": stats, "classes": run_mots(root, tracker, sequence, len(frames))}
        classes = output["runs"][label]["classes"]
        print(f"{label}: " + ", ".join(f"{c} HOTA {r['HOTA']:.3f} DetRe {r['DetRe']:.3f} AssRe {r['AssRe']:.3f}"
                                      for c, r in classes.items()) + f"  {stats}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "mots_official.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    append_report(args.out / "kitti_report.md", output)
    print(f"written: {args.out}/mots_official.json, section in kitti_report.md")
    return 0


def append_report(path: Path, output: dict[str, Any]) -> None:
    lines = [REPORT_MARKER, "", f"**KITTI MOTS (masks, official TrackEval)** — sequence {output['sequence']}; "
             "overlapping pixels go to the higher-scoring mask; recall = mask TP / GT at IoU 0.5", "",
             "| run | class | HOTA | DetRe | AssRe | DetPr | AssA | LocA | recall | sMOTSA | MOTSA | IDF1 | IDSW | "
             "Frag | MT/ML | trimmed masks |", "|" + "---|" * 16]
    for label, run in output["runs"].items():
        for cls, r in run["classes"].items():
            lines.append(f"| {label} | {cls} | {_fmt(r['HOTA'])} | {_fmt(r['DetRe'])} | {_fmt(r['AssRe'])} | "
                         f"{_fmt(r['DetPr'])} | {_fmt(r['AssA'])} | {_fmt(r['LocA'])} | {_fmt(r['recall'])} | "
                         f"{_fmt(r['sMOTSA'])} | {_fmt(r['MOTSA'])} | {_fmt(r['IDF1'])} | {r['IDSW']} | {r['Frag']} | "
                         f"{r['MT']}/{r['ML']} | {run['conversion'].get('masks_trimmed_for_overlap', 0)} |")
    replace_section(path, REPORT_MARKER, lines)


if __name__ == "__main__":
    sys.exit(main())
