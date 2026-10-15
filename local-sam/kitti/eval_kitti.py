"""Score SAM 3 / SAM 3.1 benchmark runs against KITTI ground truth (CPU only, no torch).

    # 1) check the scoring pipeline itself (GT scored against itself must be perfect)
    python kitti/eval_kitti.py --payload ~/data/kitti_0014.zip --sanity

    # 2) score benchmark runs (each run folder needs predictions.json -> benchmark with --save-predictions)
    python kitti/eval_kitti.py --payload ~/data/kitti_0014.zip \
        --runs logs/kitti/<stamp>/*_benchmark_sam3 logs/kitti/<stamp>/*_benchmark_sam31 --out logs/kitti/<stamp>/eval

Writes kitti_eval.json (everything), kitti_summary.csv (one row per run x class), kitti_report.md (table),
and trackeval/<run_label>/data/<seq>.txt (KITTI result format, for the official TrackEval/HOTA if wanted).
Frames scored = the first ``num_frames`` payload frames of each run (respects ``--max-frames``).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))  # kitti_metrics, also under python -I

from kitti_metrics import (  # noqa: E402
    Box, PayloadReader, Reference, evaluate, load_predictions, merge_prompts, ordered_bins, parse_labels,
    write_kitti_results)


def load_payload(path: Path) -> tuple[dict, Reference, dict[int, int]]:
    reader = PayloadReader(path)
    manifest = json.loads(reader.read_text("kitti_manifest.json"))
    frames = json.loads(reader.read_text("frames.json"))["frames"]
    rows = sorted(frames, key=lambda row: row["source_frame_index"])
    position_to_source = {position: int(row["source_frame_index"]) for position, row in enumerate(rows)}
    reference = parse_labels(reader.read_text(manifest["labels"]), sequence=manifest["sequence"],
                             profile=manifest.get("profile", "kitti2"))
    return manifest, reference, position_to_source


def run_info(run: Path) -> dict[str, Any]:
    bench = json.loads((run / "benchmark.json").read_text(encoding="utf-8"))
    repeats = [r for r in bench.get("repeats", []) if r.get("completed")]
    aggregate = bench.get("aggregate") or {}
    model = bench.get("model") or {}
    args = bench.get("args") or {}
    last = repeats[-1] if repeats else {}
    return {
        "run_dir": str(run), "run_label": bench.get("run_label") or run.name, "model_version": bench.get("model_version"),
        "completed": bool(bench.get("completed")), "error": bench.get("error"), "num_frames": bench.get("num_frames"),
        "prompts": bench.get("prompts"), "repeats": len(repeats),
        "fps_mean": (aggregate.get("fps") or {}).get("mean"),
        "total_seconds_mean": (aggregate.get("total_seconds") or {}).get("mean"),
        "peak_reserved_gib": aggregate.get("peak_reserved_gib"), "model_load_seconds": model.get("model_load_seconds"),
        "final_window_size": last.get("final_window_size"), "oom_backoffs": last.get("oom_backoffs"),
        "invalid_score_detections": last.get("invalid_score_detections"),
        "settings": {k: args.get(k) for k in ("window_size", "min_window", "max_window", "overlap", "max_objects",
                                              "grounding_batch", "prob_threshold", "offload_state", "compile")},
        "gpu": ((bench.get("environment") or {}).get("gpu") or {}).get("name"),
        "object_cap": model.get("object_cap") or {},
        "fps_repeats": [r.get("fps") for r in repeats], "save_masks": args.get("save_masks"),
    }


def score_run(run: Path, manifest: dict, reference: Reference, position_to_source: dict[int, int], out: Path,
              *, allow_incomplete: bool) -> dict[str, Any]:
    info = run_info(run)
    if not info["completed"] and not allow_incomplete:
        raise SystemExit(f"{run}: benchmark did not complete ({info['error']}); pass --allow-incomplete to score it")
    predictions_path = run / "predictions.json"
    if not predictions_path.is_file():
        raise SystemExit(f"{run}: no predictions.json - rerun the benchmark with --save-predictions")
    count = info["num_frames"] or len(position_to_source)
    frames = {position_to_source[p] for p in range(min(count, len(position_to_source)))}
    boxes, dropped, invalid = load_predictions(predictions_path, position_to_source, manifest["prompt_to_class"])
    boxes, merged = merge_prompts(boxes)
    report = evaluate(reference, boxes, frames)
    write_kitti_results([b for b in boxes if b.frame in frames],
                        out / "trackeval" / _safe(info["run_label"]) / "data" / f"{manifest['sequence']}.txt")
    return {**info, "dropped_prompt_labels": dict(dropped), "invalid_boxes": invalid,
            "merged_duplicate_prompt_boxes": merged, "metrics": report}


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


BREAKDOWN_DIMENSIONS = ("distance", "height", "occlusion", "truncation")


def _recall(cell: dict[str, int]) -> float | None:
    return cell["tp"] / cell["gt"] if cell["gt"] else None


def summary_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        metrics = result["metrics"]
        for name in (*metrics["per_class"], "overall"):
            m = metrics["overall"] if name == "overall" else metrics["per_class"][name]
            d5, d7, c, i, h = m["detection@0.5"], m["detection@0.7"], m["clear"], m["identity"], m["hota"]
            first = m["first_detection"]
            rows.append({
                "run_label": result["run_label"], "model_version": result["model_version"], "class": name,
                "frames": metrics["frames"], "gt_boxes": c["gt"], "pred_boxes": c["pred"],
                "precision@0.5": d5["precision"], "recall@0.5": d5["recall"], "f1@0.5": d5["f1"],
                "mean_iou@0.5": d5["mean_iou"], "precision@0.7": d7["precision"], "recall@0.7": d7["recall"],
                "f1@0.7": d7["f1"], "mota": c["mota"], "motp": c["motp"], "idf1": i["idf1"], "idsw": c["idsw"],
                "frag": c["frag"], "fp": c["fp"], "fn": c["fn"], "gt_tracks": c["gt_tracks"],
                "pred_tracks": c["pred_tracks"], "mostly_tracked": c["mostly_tracked"],
                "mostly_lost": c["mostly_lost"],
                "raw_recall@0.5": (m.get("raw_detection@0.5") or {}).get("recall"),
                "f2@0.5": d5["f2"], "idr": i["idr"], "hota": h["hota"], "det_a": h["det_a"], "det_re": h["det_re"],
                "det_pr": h["det_pr"], "ass_a": h["ass_a"], "ass_re": h["ass_re"], "ass_pr": h["ass_pr"],
                "loc_a": h["loc_a"], "hota@0.5": h["hota@0.5"], "track_coverage": first["coverage"],
                "entered_tracks": first["entered"]["tracks"], "entered_never_detected": first["entered"]["never_detected"],
                "first_detection_delay_median": first["entered"]["delay_frames_median"],
                "first_detection_delay_p90": first["entered"]["delay_frames_p90"],
                **{f"recall_{dim}_{key}": _recall(cell) for dim in BREAKDOWN_DIMENSIONS
                   for key, cell in m["recall_breakdown"].get(dim, {}).items()},
                "fps": result.get("fps_mean"), "total_seconds": result.get("total_seconds_mean"),
                "peak_reserved_gib": result.get("peak_reserved_gib"), "final_window": result.get("final_window_size"),
            })
    return rows


def markdown(manifest: dict, results: list[dict[str, Any]], rows: list[dict[str, Any]]) -> str:
    lines = [f"# KITTI {manifest['sequence']} — frames {manifest['frame_range'][0]}..{manifest['frame_range'][1] - 1}",
             "", f"KITTI-style local diagnostic, profile `{manifest.get('profile', 'kitti2')}` "
             f"(prompts -> class: {manifest['prompt_to_class']}), IoU 0.5, DontCare ignored. HOTA below is a "
             "re-implementation of TrackEval's; `kitti/trackeval_kitti.py` adds the official car/pedestrian numbers.",
             "",
             "| run | class | P@0.5 | R@0.5 | F1@0.5 | F1@0.7 | MOTA | MOTP | IDF1 | IDSW | Frag | FP | FN | "
             "GT/pred tracks | MT/ML |", "|" + "---|" * 15]
    for r in rows:
        lines.append(f"| {r['run_label']} | {r['class']} | {_fmt(r['precision@0.5'])} | {_fmt(r['recall@0.5'])} | "
                     f"{_fmt(r['f1@0.5'])} | {_fmt(r['f1@0.7'])} | {_fmt(r['mota'])} | {_fmt(r['motp'])} | "
                     f"{_fmt(r['idf1'])} | {r['idsw']} | {r['frag']} | {r['fp']} | {r['fn']} | "
                     f"{r['gt_tracks']}/{r['pred_tracks']} | {r['mostly_tracked']}/{r['mostly_lost']} |")
    lines += ["", "**Recall view** — HOTA averaged over IoU 0.05..0.95 (DetRe = share of GT boxes found, AssRe = "
              "how much of each GT track one predicted track covers); coverage = GT tracks found at least once; "
              "delay = frames from a track's first scored frame to its first match, for tracks that enter after "
              "the first frame (10 fps: 1 frame = 100 ms).", "",
              "| run | class | HOTA | DetA | DetRe | DetPr | AssA | AssRe | LocA | F2@0.5 | IDR | coverage | ML | "
              "new tracks | delay median/p90 | never found |", "|" + "---|" * 16]
    for r in rows:
        lines.append(f"| {r['run_label']} | {r['class']} | {_fmt(r['hota'])} | {_fmt(r['det_a'])} | "
                     f"{_fmt(r['det_re'])} | {_fmt(r['det_pr'])} | {_fmt(r['ass_a'])} | {_fmt(r['ass_re'])} | "
                     f"{_fmt(r['loc_a'])} | {_fmt(r['f2@0.5'])} | {_fmt(r['idr'])} | {_fmt(r['track_coverage'])} | "
                     f"{r['mostly_lost']}/{r['gt_tracks']} | {r['entered_tracks']} | "
                     f"{_fmt(r['first_detection_delay_median'], 1)}/{_fmt(r['first_detection_delay_p90'], 1)} | "
                     f"{r['entered_never_detected']} |")
    for dimension in BREAKDOWN_DIMENSIONS:
        bins = list(ordered_bins({key: {} for result in results
                                  for m in (*result["metrics"]["per_class"].values(), result["metrics"]["overall"])
                                  for key in m["recall_breakdown"].get(dimension, {})}))
        source = "KITTI-filtered GT" if dimension in ("distance", "height") else "all GT rows, no KITTI filter"
        lines += ["", f"**Recall@0.5 by {dimension}** ({source}; recall (TP/GT))", "",
                  "| run | class | " + " | ".join(bins) + " |", "|" + "---|" * (len(bins) + 2)]
        for result in results:
            metrics = result["metrics"]
            for name, m in (*metrics["per_class"].items(), ("overall", metrics["overall"])):
                table = m["recall_breakdown"].get(dimension, {})
                if not any(cell["gt"] for cell in table.values()):
                    continue
                cells = [f"{_fmt(_recall(table[b]))} ({table[b]['tp']}/{table[b]['gt']})" if b in table else "-"
                         for b in bins]
                lines.append(f"| {result['run_label']} | {name} | " + " | ".join(cells) + " |")
    lines += ["", "Speed is comparable only between runs without `--save-masks` (mask RLE encoding is timed). "
              "Object cap: upstream warnings / prompt-windows that reached `max_objects` (both must be 0, "
              "otherwise SAM 3.1 dropped objects).", "",
              "| run | model | GPU | frames | fps (repeats) | total s | peak GiB | load s | window | OOM | "
              "max obj | cap hits | most ids/prompt-window | grounding | compile | masks | removed preds |",
              "|" + "---|" * 17]
    for r in results:
        cap, settings = r["object_cap"], r["settings"]
        fps_repeats = ", ".join(_fmt(v, 2) for v in r["fps_repeats"])
        lines.append(f"| {r['run_label']} | {r['model_version']} | {r['gpu']} | {r['metrics']['frames']} | "
                     f"{_fmt(r['fps_mean'], 2)} ({fps_repeats}) | {_fmt(r['total_seconds_mean'], 1)} | "
                     f"{_fmt(r['peak_reserved_gib'], 2)} | {_fmt(r['model_load_seconds'], 1)} | "
                     f"{r['final_window_size']} | {r['oom_backoffs']} | {_fmt(cap.get('max_objects'))} | "
                     f"{cap.get('limit_warnings', '-')}/{cap.get('prompt_windows_at_cap', '-')} | "
                     f"{max((cap.get('max_ids_per_prompt_window') or {}).values(), default='-')} | "
                     f"{_fmt(settings.get('grounding_batch'))} | {_fmt(settings.get('compile'))} | "
                     f"{_fmt(r['save_masks'])} | {r['metrics']['removed_predictions']} |")
    for r in results:
        lines += ["", f"**Confusion — {r['run_label']}** (prompts {r['prompts']}; predicted class -> KITTI class it "
                  f"overlaps, IoU >= 0.5; merged duplicate prompt boxes: {r['merged_duplicate_prompt_boxes']}; "
                  f"unmapped prompt labels: {r['dropped_prompt_labels'] or 'none'})", "",
                  "| predicted | " + " | ".join(NATIVE_ORDER) + " |", "|" + "---|" * (len(NATIVE_ORDER) + 1)]
        for label, counts in r["metrics"]["confusion"].items():
            lines.append(f"| {label} | " + " | ".join(str(counts.get(n, 0)) for n in NATIVE_ORDER) + " |")
    return "\n".join(lines) + "\n"


NATIVE_ORDER = ("Car", "Van", "Truck", "Pedestrian", "Person", "Cyclist", "Tram", "Misc", "background")


def sanity(reference: Reference, position_to_source: dict[int, int]) -> int:
    """GT scored as predictions must be perfect; a damaged copy must lose recall and identity."""
    frames = set(position_to_source.values())
    perfect = [Box(i.box.frame, i.box.label, i.box.xyxy, i.box.track_id.replace("gt:", "pred:"), 1.0)
               for i in reference.items if i.native_class in reference.target_natives and i.box.frame in frames]
    checks: list[tuple[str, bool, Any]] = []
    if not perfect:
        print("FAIL: no scored-class ground truth in this frame range")
        return 1
    o = evaluate(reference, perfect, frames)["overall"]
    checks += [("GT vs GT: precision = recall = 1", o["detection@0.5"]["precision"] == 1 == o["detection@0.5"]["recall"],
                (o["detection@0.5"]["precision"], o["detection@0.5"]["recall"])),
               ("GT vs GT: MOTA = IDF1 = 1, IDSW = Frag = 0",
                o["clear"]["mota"] == 1 and o["identity"]["idf1"] == 1 and o["clear"]["idsw"] == o["clear"]["frag"] == 0,
                (o["clear"]["mota"], o["identity"]["idf1"], o["clear"]["idsw"], o["clear"]["frag"])),
               ("GT vs GT: HOTA = DetRe = AssRe = 1",
                all(abs(o["hota"][k] - 1) < 1e-9 for k in ("hota", "det_re", "ass_re")),
                tuple(round(o["hota"][k], 6) for k in ("hota", "det_re", "ass_re"))),
               ("GT vs GT: every recall bin = 1",
                all(cell["tp"] == cell["gt"] for bins in o["recall_breakdown"].values() for cell in bins.values()),
                {dim: sum(c["gt"] - c["tp"] for c in bins.values()) for dim, bins in o["recall_breakdown"].items()}),
               ("GT vs GT: every track found on its first frame, coverage = 1",
                o["first_detection"]["coverage"] == 1 and not any(
                    d for g in ("entered", "at_start") for d in o["first_detection"][g]["delays"]),
                o["first_detection"]["coverage"])]
    ordered = sorted(frames)
    gap = set(ordered[len(ordered) // 3: len(ordered) // 3 + 3])  # 3 empty frames in the first third
    middle = ordered[len(ordered) // 2]
    damaged = [Box(b.frame, b.label, b.xyxy, b.track_id + (":b" if b.frame >= middle else ""), b.score)
               for b in perfect if b.frame not in gap]
    d = evaluate(reference, damaged, frames)["overall"]
    lost = sum(1 for b in reference.valid_boxes(gap))
    checks += [("damaged: recall < 1 when frames are dropped", not lost or d["detection@0.5"]["recall"] < 1,
                d["detection@0.5"]["recall"]),
               ("damaged: precision stays 1", d["detection@0.5"]["precision"] == 1, d["detection@0.5"]["precision"]),
               ("damaged: ID change mid-sequence -> IDSW > 0 and IDF1 < 1",
                d["clear"]["idsw"] > 0 and d["identity"]["idf1"] < 1, (d["clear"]["idsw"], d["identity"]["idf1"])),
               ("damaged: split tracks -> AssRe < 1, DetPr stays 1",
                d["hota"]["ass_re"] < 1 and abs(d["hota"]["det_pr"] - 1) < 1e-9,
                (round(d["hota"]["ass_re"], 4), round(d["hota"]["det_pr"], 4)))]
    shifted = [Box(b.frame, b.label, (b.xyxy[0] + 10000, b.xyxy[1], b.xyxy[2] + 10000, b.xyxy[3]), b.track_id, 1.0)
               for b in perfect]
    s = evaluate(reference, shifted, frames)["overall"]
    checks.append(("boxes moved off target: TP = 0", s["detection@0.5"]["tp"] == 0, s["detection@0.5"]["tp"]))
    failed = 0
    for name, ok, value in checks:
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name}  ({value})")
    valid = len(reference.valid_boxes(frames))
    print(f"frames {len(frames)}, scored GT boxes {valid}, all scored-class rows {len(perfect)}")
    print("scoring pipeline OK" if not failed else f"{failed} sanity check(s) failed")
    return int(failed > 0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--payload", type=Path, required=True, help="kitti_<seq>.zip from prepare_kitti.py")
    parser.add_argument("--runs", type=Path, nargs="*", default=[], help="benchmark run folders to score")
    parser.add_argument("--out", type=Path, default=None, help="output folder (default: <first run>/../eval)")
    parser.add_argument("--sanity", action="store_true", help="only check the scoring pipeline with GT")
    parser.add_argument("--allow-incomplete", action="store_true", help="score runs with completed=false")
    args = parser.parse_args(argv)
    manifest, reference, position_to_source = load_payload(args.payload)
    if args.sanity:
        return sanity(reference, position_to_source)
    if not args.runs:
        parser.error("pass --runs <run folders> or --sanity")
    out = args.out or args.runs[0].parent / "eval"
    out.mkdir(parents=True, exist_ok=True)
    results = [score_run(run, manifest, reference, position_to_source, out, allow_incomplete=args.allow_incomplete)
               for run in args.runs]
    rows = summary_rows(results)
    (out / "kitti_eval.json").write_text(json.dumps({"schema": "local-sam-kitti-eval/v1", "payload": manifest,
                                                     "runs": results}, indent=2), encoding="utf-8")
    with (out / "kitti_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        # recall_<dimension>_<bin> columns only exist for bins that hold GT, so take the union of all rows
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)
    text = markdown(manifest, results, rows)
    (out / "kitti_report.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"written: {out}/kitti_report.md, kitti_summary.csv, kitti_eval.json, trackeval/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
