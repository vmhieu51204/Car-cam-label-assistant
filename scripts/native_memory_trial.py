"""Frozen boundary diagnostics and paired, bounded native-state SAM3 trial."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from PIL import Image

from kaggle.sam3_benchmark.main import validate_config, windows
from scripts.eval_autolabel import Box
from scripts.native_fps_trial import NativeTrialRunner
from scripts.sam3_cadence_trial import NOTEBOOK, _archive_entry, _resume_reserved_run
from scripts.stitch_sam3_benchmark import _window_tracks
from src.config import Settings
from src.services.evaluation.matching import bbox_iou, match_boxes
from src.services.evaluation.phase0 import canonical_predictions, file_hash, load_frozen_run
from src.services.evaluation.tracking import tracking_diagnostics
from src.services.video.stitching import stitch_tracks

ROOT = Path(__file__).resolve().parents[1]


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def baseline(protocol: Path, output: Path) -> dict[str, Any]:
    definition = json.loads(protocol.read_text(encoding="utf-8"))
    report: dict[str, Any] = dict(schema="autolabel-native-boundary-baseline/v1", clips={},
        boundary_radius_sampled_frames=5, production_changed=False,
        note="GT-backed overlap associations, not verified object identity; new entrants use native GT history")
    for run in definition["runs"]:
        data = load_frozen_run(run, protocol.parent.parent)
        rows = data.item.get("window_predictions", [])
        if not rows:
            report["clips"][run["sequence"]] = dict(availability="missing_two_sided_overlap")
            continue
        resets = {x["window_id"] for x in rows if data.records[x["frame_index"]].get("scene_cut_before")}
        stitched = stitch_tracks(_window_tracks(rows), reset_windows=resets)
        gt = tuple(g for g in data.gt if int(g.frame.frame_uid) in set(data.source))
        keys = tuple(data.keys[f] for f in data.source)
        after = [Box(data.source[o.frame], t.label, o.bbox, t.track_id)
                 for t in stitched.tracks for o in t.observations]
        starts = sorted({min(x["frame_index"] for x in rows if x["window_id"] == w)
                         for w in {x["window_id"] for x in rows}} - {0})
        near = {data.source[i] for i in range(len(data.source)) if any(abs(i - s) <= 5 for s in starts)}
        views = {}
        for name, boxes in (("raw", data.direct), ("stitched", after)):
            pred = canonical_predictions(boxes, data.keys, name)
            views[name] = dict(all_sampled=tracking_diagnostics(gt, pred, frames=keys),
                near_boundaries=tracking_diagnostics(tuple(g for g in gt if int(g.frame.frame_uid) in near),
                    tuple(p for p in pred if int(p.frame.frame_uid) in near),
                    frames=tuple(data.keys[f] for f in sorted(near))),
                near_boundary_note="Restricted timeline metrics, not attribution of each full-timeline ID switch")
        classified = []
        for decision in stitched.decisions:
            if decision.reason != "unmatched_chunk":
                continue
            local = [x for x in rows if x["window_id"] == decision.window_id
                     and x["track_id"] == decision.local_track_id]
            local.sort(key=lambda x: x["frame_index"])
            first = min(x["frame_index"] for x in local)
            candidates = [g for g in gt if int(g.frame.frame_uid) == data.source[first]
                          and g.class_name == local[0]["label"]]
            best = max(candidates, key=lambda g: bbox_iou(g.bbox_xyxy, tuple(local[0]["bbox"])), default=None)
            label = "unresolved"
            gt_id = None
            if best is not None and bbox_iou(best.bbox_xyxy, tuple(local[0]["bbox"])) >= .5:
                gt_id = best.native_track_id
                history = [g for g in data.gt if g.native_track_id == gt_id and int(g.frame.frame_uid) < data.source[first]]
                native_history = [g for g in data.native_rows if str(g.track_id) == gt_id
                                  and g.source_frame_index < data.source[first]]
                prior_match = match_boxes(tuple(history), canonical_predictions(
                    [b for b in data.direct if b.frame < data.source[first]], data.keys, "history"), iou_threshold=.5)
                label = ("previously_detected_GT_fragment" if prior_match.pairs else
                         "existing_GT_first_detection" if native_history else "GT_new_entrant")
            classified.append(dict(window_id=decision.window_id, source_frame=data.source[first],
                                   local_track_id=decision.local_track_id, gt_id=gt_id, category=label))
        report["clips"][run["sequence"]] = dict(input_hashes=data.input_hashes,
            boundaries_source=[data.source[i] for i in starts], views=views,
            decisions=dict(Counter(d.reason for d in stitched.decisions)),
            unmatched_categories=dict(Counter(x["category"] for x in classified)), unmatched=classified)
    output.mkdir(parents=True, exist_ok=True)
    (output / "baseline.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _csv(output / "baseline.csv", [dict(sequence=seq, view=view, iou=t,
        precision=m["precision"], recall=m["recall"], idf1=m["tracking"]["idf1"],
        id_switches=m["tracking"]["id_switches"], fragmentations=m["tracking"]["fragmentations"])
        for seq, clip in report["clips"].items() for view, data in clip.get("views", {}).items()
        for t, m in data["all_sampled"]["thresholds"].items()])
    return report


def prepare(dataset: Path, output: Path, count: int = 150) -> dict[str, Any]:
    if (output / "submission.json").exists():
        raise ValueError("cannot replace inputs of a reserved experiment")
    if not 100 <= count <= 250:
        raise ValueError("trial needs 100..250 frames per scene")
    common = dict(selection="dense", mode="text_prompt", seed_type="none", reprompt="none",
                  text_prompts=["car", "person"], window_size=50, overlap=5, preserve_overlap=True,
                  score_threshold_detection=.5, new_det_thresh=.7, scene_cut_frames=[count])
    config = dict(source_frame_limit=count * 2, max_total_seconds=5400, max_config_seconds=2500,
        configurations=[dict(common, id="windows"), dict(common, id="native", tracking_policy="native_scene",
                                                        max_scene_frames=250)])
    validate_config(config)
    training = dataset / "training" if (dataset / "training").is_dir() else dataset
    clips = [("0013", 0), ("0020", 500)]
    paths = [training / "image_02" / seq / f"{start+i:06d}.png" for seq, start in clips for i in range(count)]
    sizes = []
    for path in paths:
        with Image.open(path) as image:
            image.load()
            sizes.append(image.size)
    width, height = max(x[0] for x in sizes), max(x[1] for x in sizes)
    output.mkdir(parents=True, exist_ok=True)
    payload = output / "payload.zip"
    frames = []
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
        for i, path in enumerate(paths):
            with Image.open(path) as raw:
                image = Image.new("RGB", (width, height))
                image.paste(raw.convert("RGB"), (0, 0))
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=95)
            encoded = buffer.getvalue()
            name = f"frames/{i:06d}.jpg"
            _archive_entry(archive, name, encoded)
            frames.append(dict(source_frame_index=i, relative_path=name, width=width, height=height,
                               sha256=hashlib.sha256(encoded).hexdigest()))
        _archive_entry(archive, "benchmark-config.json", json.dumps(config))
        _archive_entry(archive, "frames.json", json.dumps(dict(frames=frames, optical_source_indices=list(range(count*2)))))
        _archive_entry(archive, "seed.json", '{"seeds": []}')
    manifest = dict(schema="autolabel-native-memory-trial/v1", config=config, count=count,
        clips=[dict(sequence=seq, source_start=start, payload_start=j*count) for j, (seq, start) in enumerate(clips)],
        image_hashes={str(p.resolve()): file_hash(p) for p in paths}, padding="right_bottom_no_coordinate_shift",
        ground_truth_uploaded=False, production_changed=False, payload_sha256=file_hash(payload))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def evaluate(output: Path, protocol: Path, candidate: str = "native") -> dict[str, Any]:
    manifest = json.loads((output / "manifest.json").read_text())
    if file_hash(output / "payload.zip") != manifest["payload_sha256"] or any(
            file_hash(Path(p)) != digest for p, digest in manifest["image_hashes"].items()):
        raise ValueError("experiment inputs changed before evaluation")
    evidence = json.loads((output / "result-evidence.json").read_text())
    if evidence["result_sha256"] != file_hash(output / "result.json"):
        raise ValueError("downloaded result hash mismatch")
    result = json.loads((output / "result.json").read_text())
    model = result.get("model_runtime", {})
    if (not model.get("checkpoint_sha256") or model.get("environment", {}).get("versions", {}).get("sam3_commit")
            != "2345a4ad109ac29c569da749c91d84f10dc08c40"):
        raise ValueError("missing or unexpected SAM3 model provenance")
    definition = json.loads(protocol.read_text())
    report: dict[str, Any] = dict(schema="autolabel-native-memory-comparison/v1", clips={},
        model=result.get("model_runtime"), load_seconds=result.get("model_load_seconds"),
        input_sha256=dict(payload=manifest["payload_sha256"], result=evidence["result_sha256"], protocol=file_hash(protocol)),
        recommendation="keep_independent_windows_pending_complete_evidence", production_changed=False)
    for item in result["configurations"]:
        if item["status"] != "completed":
            report.setdefault("errors", {})[item["id"]] = item.get("error", item["status"])
            continue
        expected = next(r for r in manifest["config"]["configurations"] if r["id"] == item["id"])
        if expected.get("memory_policy") == "dms_singleton":
            from kaggle.dms_memory import validate_evidence

            validate_evidence(item)
        if item.get("config") != expected or item.get("evaluated_source_frames") != list(range(manifest["count"]*2)):
            raise ValueError("completed experiment configuration or frame coverage mismatch")
        predictions: list[dict[str, Any]]
        if expected.get("tracking_policy", "independent_windows") == "independent_windows":
            batches = windows([dict(source_frame_index=i, scene_cut_before=i == manifest["count"])
                               for i in range(manifest["count"]*2)], 50, 5)
            resets = {i for i, (_, batch) in enumerate(batches) if batch[0].get("scene_cut_before")}
            tracks = stitch_tracks(_window_tracks(item["window_predictions"]), reset_windows=resets).tracks
            predictions = [dict(frame_index=o.frame, track_id=t.track_id, label=t.label, bbox=o.bbox)
                           for t in tracks for o in t.observations]
        else:
            predictions = item["predictions"]
        for clip in manifest["clips"]:
            run = next(r for r in definition["runs"] if r["sequence"] == clip["sequence"])
            data = load_frozen_run(run, protocol.parent.parent)
            start, end = clip["source_start"], clip["source_start"] + manifest["count"]
            gt = tuple(g for g in data.gt if start <= int(g.frame.frame_uid) < end)
            boxes = [Box(start + p["frame_index"] - clip["payload_start"], p["label"], tuple(p["bbox"]), p["track_id"])
                     for p in predictions if clip["payload_start"] <= p["frame_index"] < clip["payload_start"]+manifest["count"]]
            report["clips"].setdefault(clip["sequence"], {})[item["id"]] = tracking_diagnostics(
                gt, canonical_predictions(boxes, data.keys, item["id"]),
                frames=tuple(data.keys[i] for i in range(start, end)), occlusions=data.occlusions)
        report.setdefault("runtime", {})[item["id"]] = item["runtime"]
    complete = (bool(report["clips"]) and not report.get("errors")
                and all(set(c) == {"windows", candidate} for c in report["clips"].values()))
    if complete:
        checks = []
        for clip in report["clips"].values():
            for threshold in ("0.5", "0.7"):
                before = clip["windows"]["thresholds"][threshold]
                after = clip[candidate]["thresholds"][threshold]
                checks.append(dict(precision_non_decreasing=after["precision"] is not None and before["precision"] is not None
                                   and after["precision"] >= before["precision"],
                    recall_non_decreasing=after["recall"] is not None and before["recall"] is not None
                                          and after["recall"] >= before["recall"],
                    id_switches_non_increasing=after["tracking"]["id_switches"] <= before["tracking"]["id_switches"],
                    fragmentation_non_increasing=after["tracking"]["fragmentations"] <= before["tracking"]["fragmentations"]))
        improved = any(clip[candidate]["thresholds"]["0.5"]["tracking"][metric]
                       < clip["windows"]["thresholds"]["0.5"]["tracking"][metric]
                       for clip in report["clips"].values() for metric in ("id_switches", "fragmentations"))
        report["quality_gate"] = dict(checks=checks, strictly_improved_identity=improved,
                                      passed=improved and all(all(c.values()) for c in checks))
        if report["quality_gate"]["passed"]:
            report["recommendation"] = "candidate_for_further_resource_and_long_video_trial_not_default_rollout"
    (output / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    rows = []
    for sequence, clip in report["clips"].items():
        for policy, diagnostic in clip.items():
            for threshold, metrics in diagnostic["thresholds"].items():
                for label, group in {"overall": metrics, **metrics["groups"]["class"]}.items():
                    tracking = metrics["tracking"] if label == "overall" else metrics["tracking_by_class"][label]
                    rows.append(dict(sequence=sequence, policy=policy, iou=threshold, label=label,
                        precision=group["precision"], recall=group["recall"], mean_iou=group["mean_iou"],
                        idf1=tracking["idf1"], id_switches=tracking["id_switches"], fragmentations=tracking["fragmentations"]))
    _csv(output / "comparison.csv", rows)
    return report


def execute(output: Path) -> None:
    settings = Settings()
    if not settings.kaggle_api_token.get_secret_value() or not settings.kaggle_sam3_checkpoint_dataset:
        raise ValueError("Kaggle token/private SAM3 checkpoint are required")
    runner = NativeTrialRunner(username=settings.kaggle_username, work_root=output / "runs",
        machine_shape=settings.kaggle_machine_shape, timeout_s=6000, poll_seconds=settings.kaggle_poll_seconds,
        sam3_checkpoint_dataset=settings.kaggle_sam3_checkpoint_dataset,
        api_token=settings.kaggle_api_token.get_secret_value(),
        command=str(Path(sys.executable).parent / ("kaggle.exe" if os.name == "nt" else "kaggle")))
    payload = output / "payload.zip"
    manifest = json.loads((output / "manifest.json").read_text())
    if file_hash(payload) != manifest["payload_sha256"] or any(file_hash(Path(p)) != digest
        for p, digest in manifest["image_hashes"].items()):
        raise ValueError("experiment inputs changed")
    identity = hashlib.sha256(payload.read_bytes() + NOTEBOOK.read_bytes()).hexdigest()
    run_id = str(uuid5(NAMESPACE_URL, "native-memory:" + identity))
    receipt = output / "submission.json"
    if receipt.exists():
        if json.loads(receipt.read_text())["identity"] != identity:
            raise ValueError("reserved experiment inputs changed")
        if (output / "result.json").is_file():
            return
        result = _resume_reserved_run(runner, run_id, payload, NOTEBOOK, upload_missing=True)
    else:
        receipt.write_text(json.dumps(dict(identity=identity, run_id=run_id)))
        result = runner.run(stage="sam3-benchmark", job_id=run_id, files={"payload.zip": payload},
                            code_file=NOTEBOOK, allow_retry_on_terminal_failure=False)
    (output / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (output / "result-evidence.json").write_text(json.dumps(dict(result_sha256=file_hash(output / "result.json"),
        payload_sha256=file_hash(payload), notebook_sha256=file_hash(NOTEBOOK), identity=identity)), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["baseline", "prepare", "run", "evaluate"])
    parser.add_argument("--protocol", type=Path, default=ROOT / "eval/phase0-protocol.json")
    parser.add_argument("--dataset", type=Path, default=ROOT.parent / "AutoLabelDatasets")
    parser.add_argument("--output", type=Path, default=ROOT / "state/native-memory/trial")
    args = parser.parse_args()
    if args.action == "baseline":
        baseline(args.protocol, args.output)
    elif args.action == "prepare":
        prepare(args.dataset, args.output)
    elif args.action == "run":
        execute(args.output)
        evaluate(args.output, args.protocol)
    else:
        evaluate(args.output, args.protocol)


if __name__ == "__main__":
    main()
