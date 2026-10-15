"""Controlled private 5/10-FPS trial over the frozen phase-0 source spans."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from PIL import Image

from kaggle.sam3_benchmark.main import select_frames, validate_config
from scripts.compare_video_optimization import interpolate_tracks
from scripts.eval_autolabel import Box
from scripts.sam3_cadence_trial import NOTEBOOK, _archive_entry, _resume_reserved_run
from scripts.stitch_sam3_benchmark import _window_tracks
from src.config import Settings
from src.services.evaluation.matching import match_boxes
from src.services.evaluation.phase0 import canonical_predictions, file_hash, load_frozen_run
from src.services.evaluation.tracking import tracking_diagnostics
from src.services.video.kaggle_batch import KaggleBatchError, KaggleBatchRunner
from src.services.video.stitching import stitch_tracks


class NativeTrialRunner(KaggleBatchRunner):
    """Kaggle's CLI can print a create error and still exit zero; fail before polling."""

    def _command(self, *args: str, timeout: float) -> subprocess.CompletedProcess[str]:
        try:
            result = super()._command(*args, timeout=timeout)
        except KaggleBatchError:
            if args[:2] != ("datasets", "status") or len(args) != 3 or not self.api_token:
                raise
            request = Request("https://www.kaggle.com/api/v1/datasets/status/" + quote(args[2], safe="/"),
                              headers={"Authorization": "Bearer " + self.api_token})
            try:
                with urlopen(request, timeout=min(timeout, 30)) as response:
                    state = json.loads(response.read(10000))
            except (HTTPError, URLError, ValueError):
                raise KaggleBatchError("Kaggle dataset status failed on both SDK and v1 endpoints") from None
            if not isinstance(state, str) or state not in {"ready", "pending", "processing", "error", "failed"}:
                raise KaggleBatchError("Kaggle v1 returned an unknown dataset status")
            return subprocess.CompletedProcess(args=list(args), returncode=0, stdout=state, stderr="")
        if args[:2] == ("datasets", "create") and "dataset creation error:" in result.stdout.lower():
            diagnostic = result.stdout[result.stdout.lower().index("dataset creation error:"):].strip()
            if self.api_token:
                diagnostic = diagnostic.replace(self.api_token, "[redacted]")
            raise KaggleBatchError(diagnostic[:500])
        return result

    def _wait_for_dataset(self, reference: str) -> None:
        original_timeout = self.timeout_s
        try:
            self.timeout_s = min(original_timeout, 300)
            super()._wait_for_dataset(reference)
        finally:
            self.timeout_s = original_timeout


def configuration(count: int) -> dict[str, Any]:
    common = dict(selection="uniform", gap=None, source_fps=10, mode="text_prompt",
                  seed_type="none", reprompt="none", text_prompts=["car", "person"],
                  preserve_overlap=True, score_threshold_detection=0.5, new_det_thresh=0.7,
                  force_include_frames=[count - 1])
    config = dict(schema="autolabel-sam3-benchmark-config/v1", max_total_seconds=21600,
                  max_config_seconds=10000, source_frame_limit=count,
                  configurations=[dict(common, id=f"fps-{fps}", target_fps=fps,
                                       window_size=window, overlap=overlap)
                                  for fps, window, overlap in ((5, 50, 5), (10, 100, 10))])
    validate_config(config)
    return config


def prepare_native(dataset: Path, output: Path, protocol: Path) -> dict[str, Any]:
    """Validate all source images before producing any model payload; labels stay local."""
    definition = json.loads(protocol.read_text(encoding="utf-8"))
    if definition.get("schema") != "autolabel-phase0-protocol/v1" or definition.get("seed") != 42:
        raise ValueError("expected frozen phase0 protocol with seed 42")
    training = dataset / "training" if (dataset / "training").is_dir() else dataset
    inputs = {str(protocol.resolve()): file_hash(protocol)}
    specs = []
    for run in definition["runs"]:
        loaded = load_frozen_run(run, protocol.parent.parent)
        inputs.update({str((protocol.parent.parent / run[k]).resolve()): v
                       for k, v in loaded.input_hashes.items()})
        if run["source_fps"] != 10:
            raise ValueError("native KITTI trial requires declared source_fps=10")
        paths = [training / "image_02" / run["sequence"] / f"{i:06d}.png"
                 for i in range(loaded.source[0], loaded.source[-1] + 1)]
        dimensions = None
        for path in paths:
            with Image.open(path) as image:
                image.load()
                if dimensions is not None and image.size != dimensions:
                    raise ValueError(f"image dimensions changed: {path}")
                dimensions = image.size
            inputs[str(path.resolve())] = file_hash(path)
        specs.append((run, loaded, paths, dimensions))
    if [x[0]["sequence"] for x in specs] != ["0013", "0020"]:
        raise ValueError("trial requires exactly KITTI 0013/0020 in protocol order")
    if any(Path(p).is_relative_to(output.resolve()) for p in inputs):
        raise ValueError("output directory overlaps protected inputs")
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = dict(schema="autolabel-native-fps-trial/v1", seed=42,
        protocol=str(protocol.resolve()), source_fps=10,
        fps_provenance="dataset_fps_fallback_not_recorded_timestamps", input_sha256=inputs,
        notebook_sha256=file_hash(NOTEBOOK), clips=[], ground_truth_uploaded=False,
        production_changed=False)
    for run, loaded, paths, dimensions in specs:
        assert dimensions is not None
        sequence = run["sequence"]
        package = output / sequence / "payload"
        package.mkdir(parents=True, exist_ok=True)
        config = configuration(len(paths))
        frames = []
        with zipfile.ZipFile(package / "payload.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for index, path in enumerate(paths):
                with Image.open(path) as image:
                    buffer = io.BytesIO()
                    image.convert("RGB").save(buffer, format="JPEG", quality=95)
                encoded = buffer.getvalue()
                name = f"frames/frame_{index:06d}.jpg"
                _archive_entry(archive, name, encoded)
                frames.append(dict(source_frame_index=index, original_source_index=loaded.source[0] + index,
                                   relative_path=name, sha256=hashlib.sha256(encoded).hexdigest(),
                                   width=dimensions[0], height=dimensions[1]))
            _archive_entry(archive, "frames.json", json.dumps(dict(frames=frames,
                optical_source_indices=list(range(len(frames))))))
            _archive_entry(archive, "benchmark-config.json", json.dumps(config))
            _archive_entry(archive, "seed.json", '{"seeds": []}')
        clip = dict(sequence=sequence, source_start=loaded.source[0], source_stop=loaded.source[-1],
                    source_count=len(paths), image_size=dimensions, baseline_frames=loaded.source,
                    payload_sha256=file_hash(package / "payload.zip"),
                    selected={row["id"]: [f["original_source_index"] for f in select_frames(frames, [], row)]
                              for row in config["configurations"]})
        if not set(loaded.source) <= set(clip["selected"]["fps-5"]):
            raise ValueError("baseline frame set is not observed by both trial policies")
        manifest["clips"].append(clip)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def recall_bootstrap(gt: tuple, matched: dict[str, set[str]], *, repeats: int = 10000) -> dict[str, Any]:
    """Paired track bootstrap, independently stratified by sequence; delta is native minus 5 FPS."""
    tracks = sorted({(b.frame.sequence.sequence_uid, b.native_track_id) for b in gt})
    index = {key: i for i, key in enumerate(tracks)}
    counts = np.zeros(len(tracks), dtype=np.int64)
    delta = np.zeros(len(tracks), dtype=np.int64)
    for box in gt:
        i = index[(box.frame.sequence.sequence_uid, box.native_track_id)]
        counts[i] += 1
        delta[i] += int(box.annotation_id in matched["fps-10"]) - int(box.annotation_id in matched["fps-5"])
    weights = np.zeros((repeats, len(tracks)), dtype=np.int64)
    rng = np.random.default_rng(42)
    for sequence in sorted({t[0] for t in tracks}):
        indices = [i for i, t in enumerate(tracks) if t[0] == sequence]
        weights[:, indices] = rng.multinomial(len(indices), np.full(len(indices), 1 / len(indices)), size=repeats)
    den = weights @ counts
    valid = den > 0
    values = (weights @ delta)[valid] / den[valid]
    bounds = np.percentile(values, [2.5, 97.5]).tolist() if len(tracks) >= 2 and values.size else [None, None]
    return dict(delta=float(delta.sum() / counts.sum()) if counts.sum() else None,
                ci_low=bounds[0], ci_high=bounds[1], tracks=len(tracks), repeats=repeats, seed=42,
                valid_repeats=int(valid.sum()))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_native(results: dict[str, dict], output: Path, manifest: dict, *, repeats: int = 10000) -> dict:
    protocol = Path(manifest["protocol"])
    definition = json.loads(protocol.read_text(encoding="utf-8"))
    report: dict[str, Any] = dict(schema="autolabel-native-fps-comparison/v1", clips={},
        production_changed=False, official_hota=False,
        caveat="Two fixed KITTI spans only; source FPS is nominal. Occlusion groups have GT recall, not attributed FP.")
    metrics_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    all_gt: list[Any] = []
    pooled: dict[float, dict[str, set[str]]] = {t: {p: set() for p in ("fps-5", "fps-10")} for t in (.5, .7)}
    for run, clip in zip(definition["runs"], manifest["clips"], strict=True):
        sequence = run["sequence"]
        loaded = load_frozen_run(run, protocol.parent.parent)
        result = results[sequence]
        items = {item["id"]: item for item in result.get("configurations", [])}
        if set(items) != {"fps-5", "fps-10"} or len(result["configurations"]) != 2:
            raise ValueError("result must contain both trial policies")
        if any(item["status"] != "completed" for item in items.values()):
            report["clips"][sequence] = dict(completed=False, statuses={k: v["status"] for k, v in items.items()})
            continue
        runtime = result.get("model_runtime", {})
        if not runtime.get("checkpoint_sha256") or not runtime.get("dtype"):
            raise ValueError("completed trial has no checkpoint/precision evidence")
        common = set(loaded.source)
        common_gt = tuple(b for b in loaded.gt if int(b.frame.frame_uid) in common)
        all_gt.extend(common_gt)
        clip_report: dict[str, Any] = dict(completed=True, runtime=runtime, policies={}, paired_recall={})
        matched: dict[float, dict[str, set[str]]] = {t: {} for t in (.5, .7)}
        old_pred = canonical_predictions(loaded.direct, loaded.keys, "old-artifact")
        old_ids = {p.annotation_id for p in match_boxes(common_gt, old_pred, iou_threshold=.5).pairs}
        for policy, item in items.items():
            required_config = next(c for c in configuration(clip["source_count"])["configurations"] if c["id"] == policy)
            if any(item.get("config", {}).get(k) != v for k, v in required_config.items()):
                raise ValueError("result configuration differs from reserved experiment")
            selected = item["selected_source_frames"]
            expected = [i - clip["source_start"] for i in clip["selected"][policy]]
            if selected != expected or set(item["evaluated_source_frames"]) != set(expected):
                raise ValueError("incomplete or mismatched source-frame coverage")
            tracks = stitch_tracks(_window_tracks(item["window_predictions"])).tracks
            direct = [Box(clip["source_start"] + obs.frame, track.label, obs.bbox, track.track_id)
                      for track in tracks for obs in track.observations]
            dense_local = interpolate_tracks(tracks, selected, set(range(clip["source_count"])))
            dense = [Box(clip["source_start"] + b.frame, b.label, b.xyxy, b.track_id) for b in dense_local]
            native_selected = {i + clip["source_start"] for i in selected}
            views = dict(common_frames=(direct, common), full_timeline=(dense, set(loaded.keys)),
                         inferred=(direct, native_selected),
                         interpolated=([b for b in dense if b.frame not in native_selected], set(loaded.keys) - native_selected))
            view_reports = {}
            for name, (boxes, frames) in views.items():
                keys = tuple(loaded.keys[i] for i in sorted(frames))
                gt = tuple(b for b in loaded.gt if int(b.frame.frame_uid) in frames)
                pred = canonical_predictions([b for b in boxes if b.frame in frames], loaded.keys, policy)
                diagnostic = tracking_diagnostics(gt, pred, frames=keys, occlusions=loaded.occlusions)
                view_reports[name] = diagnostic
                for threshold in (.5, .7):
                    m = diagnostic["thresholds"][str(threshold)]
                    metrics_rows.append(dict(sequence=sequence, policy=policy, view=name, iou=threshold,
                        gt=m["tp"] + m["fn"], tp=m["tp"], fp=m["fp"], fn=m["fn"],
                        precision=m["precision"], recall=m["recall"], **m["tracking"]))
                    for dimension, groups in m["groups"].items():
                        for value, group in groups.items():
                            group_rows.append(dict(sequence=sequence, policy=policy, view=name, iou=threshold,
                                                   dimension=dimension, value=value, **group))
                    if name == "common_frames":
                        matched[threshold][policy] = {p.annotation_id for p in match_boxes(gt, pred, iou_threshold=threshold).pairs}
                        pooled[threshold][policy].update(matched[threshold][policy])
            clip_report["policies"][policy] = dict(views=view_reports, runtime=item.get("runtime"),
                inferred_frames=len(native_selected), interpolated_frames=clip["source_count"] - len(native_selected))
        for threshold in (.5, .7):
            clip_report["paired_recall"][str(threshold)] = recall_bootstrap(common_gt, matched[threshold], repeats=repeats)
        for box in common_gt:
            annotation = box.annotation_id
            a, b = annotation in matched[.5]["fps-5"], annotation in matched[.5]["fps-10"]
            old = annotation in old_ids
            cases.append(dict(sequence=sequence, annotation_id=annotation, gt_id=box.native_track_id,
                source_frame=int(box.frame.frame_uid), class_name=box.class_name,
                occlusion=loaded.occlusions[annotation], old_matched_05=old,
                fps5_matched_05=a, fps10_matched_05=b,
                old_miss_saved=not old and b, fresh_baseline_lost=a and not b,
                change="saved" if b and not a else "lost" if a and not b else "unchanged"))
        report["clips"][sequence] = clip_report
    report["completed"] = len(report["clips"]) == 2 and all(c["completed"] for c in report["clips"].values())
    if report["completed"]:
        identities = [c["runtime"] for c in report["clips"].values()]
        if any(r.get(k) != identities[0].get(k) for r in identities for k in ("checkpoint_sha256", "dtype")):
            raise ValueError("clip sessions used different checkpoints or precision")
        versions = [r.get("environment", {}).get("versions", {}) for r in identities]
        if versions[0] != versions[1]:
            raise ValueError("clip sessions used different runtime versions")
        report["paired_recall"] = {str(t): recall_bootstrap(tuple(all_gt), pooled[t], repeats=repeats) for t in (.5, .7)}
    report["historical_misses"] = dict(total=sum(not c["old_matched_05"] for c in cases),
        saved=sum(c["old_miss_saved"] for c in cases), fresh_baseline_lost=sum(c["fresh_baseline_lost"] for c in cases))
    _write_csv(output / "metrics.csv", metrics_rows)
    _write_csv(output / "groups.csv", group_rows)
    _write_csv(output / "cases.csv", cases)
    (output / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# SAM3 native FPS trial", "", f"Completed: {report['completed']}", "",
             "| Clip | IoU | 5 FPS recall | 10 FPS recall | Delta | 5/10 ID switches | 5/10 fragments |",
             "|---|---|---|---|---|---|---|"]
    for sequence, clip_report in report["clips"].items():
        if not clip_report["completed"]:
            lines.append(f"\n{sequence}: incomplete; {clip_report['statuses']}")
            continue
        for threshold_key in ("0.5", "0.7"):
            a, b = [clip_report["policies"][p]["views"]["common_frames"]["thresholds"][threshold_key] for p in ("fps-5", "fps-10")]
            lines.append(f"| {sequence} | {threshold_key} | {a['recall']} | {b['recall']} | {b['recall'] - a['recall']} | "
                         f"{a['tracking']['id_switches']}/{b['tracking']['id_switches']} | "
                         f"{a['tracking']['fragmentations']}/{b['tracking']['fragmentations']} |")
    lines.extend(["", report["caveat"], "", "Common-frame metrics use direct inference only. Full timeline includes baseline interpolation.",
                  "GT is used for evaluation only. No bridge/size-memory or production change.",
                  "Box and identity changes must be interpreted separately; this report does not approve rollout."])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _run_native(dataset: Path, output: Path, protocol: Path, *, offline: bool = False) -> dict:
    manifest = prepare_native(dataset, output, protocol)
    if offline:
        return manifest
    settings = Settings()
    if not settings.kaggle_api_token.get_secret_value() or not settings.kaggle_sam3_checkpoint_dataset:
        raise ValueError("Kaggle token and private checkpoint dataset are required")
    runner = NativeTrialRunner(username=settings.kaggle_username, work_root=output / "runs",
        machine_shape=settings.kaggle_machine_shape, timeout_s=22200,
        poll_seconds=settings.kaggle_poll_seconds, sam3_checkpoint_dataset=settings.kaggle_sam3_checkpoint_dataset,
        api_token=settings.kaggle_api_token.get_secret_value(),
        command=str(Path(sys.executable).parent / ("kaggle.exe" if os.name == "nt" else "kaggle")))
    results = {}
    for clip in manifest["clips"]:
        sequence = clip["sequence"]
        payload = output / sequence / "payload/payload.zip"
        identity = hashlib.sha256(payload.read_bytes() + NOTEBOOK.read_bytes()).hexdigest()
        run_id = str(uuid5(NAMESPACE_URL, f"native-fps:{sequence}:{identity}"))
        receipt = output / sequence / "submission.json"
        result_path = output / sequence / "result.json"
        if receipt.exists():
            if json.loads(receipt.read_text(encoding="utf-8"))["identity"] != identity:
                raise ValueError("reserved run differs from current inputs; use a new output directory")
            if result_path.exists():
                result = json.loads(result_path.read_text(encoding="utf-8"))
            else:
                result = _resume_reserved_run(runner, run_id, payload, NOTEBOOK, upload_missing=True)
        else:
            print(f"Submitting KITTI {sequence}: 5/10 FPS, {clip['source_count']} source frames", flush=True)
            receipt.write_text(json.dumps(dict(identity=identity, run_id=run_id)), encoding="utf-8")
            result = runner.run(stage="sam3-benchmark", job_id=run_id, files={"payload.zip": payload},
                                code_file=NOTEBOOK, allow_retry_on_terminal_failure=False)
        result_path.write_text(json.dumps(result), encoding="utf-8")
        results[sequence] = result
    report = evaluate_native(results, output, manifest)
    changed = [p for p, digest in manifest["input_sha256"].items() if file_hash(Path(p)) != digest]
    if changed:
        raise ValueError("protected inputs changed during trial")
    return dict(completed=report["completed"], report=str(output / "report.md"), historical_misses=report["historical_misses"])


def run_native(dataset: Path, output: Path, protocol: Path, *, offline: bool = False) -> dict:
    """Persist an actionable failure receipt without claiming a successful model run."""
    try:
        result = _run_native(dataset, output, protocol, offline=offline)
    except (KaggleBatchError, RuntimeError) as error:
        if output.is_dir():
            (output / "execution.json").write_text(json.dumps(dict(completed=False, error=str(error),
                action="Resolve Kaggle access/indexing and rerun the same command; reserved IDs are retained."), indent=2), encoding="utf-8")
        raise
    if not offline:
        (output / "execution.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
