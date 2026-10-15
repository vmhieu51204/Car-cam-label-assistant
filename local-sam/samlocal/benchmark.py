"""Benchmark SAM 3 or SAM 3.1 on a frame payload (entry scripts: ``sam3/benchmark_sam3.py``,
``sam3.1/benchmark_sam31.py``).

Writes logs/<timestamp>_benchmark_<model>/{run.log, env.json, benchmark.json, summary.csv,
windows.csv, attempts.csv, classes.csv, frames.csv, resources.csv, progress.json,
invalid_scores.csv[, predictions.json]}.
benchmark.json is rewritten after every window, so a crash still leaves partial results.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from samlocal.cli import add_common_args, build_backend, resolve_prompts, window_settings
from samlocal.logs import close_logging, create_run_dir, setup_logging, write_json
from samlocal.metrics import environment, stats
from samlocal.models import ModelSpec, get_spec
from samlocal.payload import open_payload
from samlocal.pipeline import SequenceResult, run_sequence
from samlocal.reporting import (ResourceSampler, export_details, file_sha256, input_identity,
                                write_csv, write_invalid_scores)


def main(model: str | ModelSpec, argv: list[str] | None = None) -> int:
    spec = get_spec(model) if isinstance(model, str) else model
    parser = argparse.ArgumentParser(
        description=f"Benchmark {spec.label} on a frame payload.\n\nExample (WSL2):\n"
                    f"  python benchmark_{spec.slug}.py --payload ~/data/payload.zip "
                    f"--checkpoint ~/models/{spec.hf_checkpoint} --repeats 3",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser, spec)
    parser.add_argument("--warmup", type=int, default=1,
                        help="untimed warm-up passes over the first window (cuDNN autotune, allocator, compile)")
    parser.add_argument("--repeats", type=int, default=3, help="timed full passes over the payload")
    parser.add_argument("--save-predictions", action="store_true", help="write predictions.json of the last repeat")
    parser.add_argument("--run-label", default=None, help="name used in comparison CSV rows")
    parser.add_argument("--telemetry-interval", type=float, default=1.0,
                        help="sample process RAM and GPU counters every N seconds; 0 disables sampling")
    args = parser.parse_args(argv)
    if args.run_label is None:
        args.run_label = spec.key
    if args.repeats < 1 or args.warmup < 0 or not 0 <= args.telemetry_interval < float("inf"):
        parser.error("repeats must be >= 1; warmup and telemetry-interval must be >= 0")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("max-frames must be >= 1")

    run_dir = create_run_dir(args.log_dir, f"benchmark_{spec.slug}")
    log = setup_logging(run_dir)
    try:
        return _benchmark(args, spec, run_dir, log)
    finally:
        close_logging(log)


def _benchmark(args, spec, run_dir, log) -> int:
    log.info("log folder: %s", run_dir)
    env = environment()
    write_json(run_dir / "env.json", env)
    log.info("gpu: %s | torch %s | sam3 %s", env.get("gpu"), env["packages"].get("torch"), env.get("sam3_commit"))

    payload_started = time.perf_counter()
    payload = open_payload(args.payload, args.cache_dir, verify_sha=not args.skip_sha)
    payload_open_seconds = time.perf_counter() - payload_started
    frames = payload.frames[: args.max_frames] if args.max_frames else payload.frames
    prompts = resolve_prompts(args, payload)
    settings = window_settings(args)
    log.info("payload: %s", payload.describe())
    log.info("benchmarking %d frames, prompts=%s, window=%d (min %d, max %d, overlap %d)", len(frames), prompts,
             settings.size, settings.min_size, settings.max_size, settings.overlap)

    identity_started = time.perf_counter()
    identity = input_identity(frames)
    identity_seconds = time.perf_counter() - identity_started
    checkpoint_hash_started = time.perf_counter()
    checkpoint_sha = file_sha256(args.checkpoint.expanduser()) if args.checkpoint else None
    checkpoint_hash_seconds = time.perf_counter() - checkpoint_hash_started
    report: dict = {"schema": "local-sam-benchmark/v3", "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    "environment": env, "payload": payload.describe(), "num_frames": len(frames),
                    "prompts": prompts, "repeats": [], "completed": False,
                    "run_label": args.run_label, "model_version": spec.key,
                    "input_identity": identity,
                    "setup": {"payload_open_seconds": payload_open_seconds,
                              "input_hash_seconds": identity_seconds,
                              "checkpoint_hash_seconds": checkpoint_hash_seconds},
                    "metric_definitions": {
                        "total_seconds": "repeat pipeline wall time: staging, decode/session, prompt inference, CPU postprocess, cleanup, stitching, logging/checkpoint; excludes setup/load/warmup/final exports",
                        "fps": "unique input frames / pipeline wall time; all prompts and overlap recomputation included in wall time",
                        "unique_fps": "unique frames covered by successful windows / pipeline wall time, including frames with zero detections (valid for partial runs)",
                        "processed_window_frames": "sum of frame counts of successful windows; overlap included; excludes failed attempts",
                        "attempted_window_frames": "sum of scheduled input frame counts across successful and OOM attempts; failed attempts may not have processed every input frame",
                        "backend_seconds": "session/decode + prompt inference + CPU output conversion + backend cleanup, including failed OOM attempts; not pure GPU inference",
                        "prompt_total_seconds": "successful prompt passes, CUDA synchronized; includes prompt/reset/propagation and CPU conversion; excludes session initialization/cleanup",
                        "window_seconds": "legacy wall time after staging through backend and staging cleanup, before track stitching; percentiles are per window, not per frame",
                        "output": "detection counts/confidence/track lengths describe model output; not accuracy, recall or ID correctness",
                        "mask_rle": "custom row-major height/width/counts; convert before using COCO column-major RLE evaluator",
                        "percentiles": "linear interpolation; count and population stddev included; few repeats cannot establish tail latency"},
                    "quality_metrics": {"status": "not_evaluated_no_ground_truth",
                                        "box_map": None, "mask_map": None, "hota": None, "idf1": None}}
    bench_path = run_dir / "benchmark.json"
    stage_root = args.cache_dir / "staging"
    backend = build_backend(args, log, spec)
    exit_code = 0
    last: SequenceResult | None = None
    try:
        report["model"] = backend.load()
        report["model"]["checkpoint"] = (str(args.checkpoint) if args.checkpoint
                                         else f"{spec.hf_repo}/{spec.hf_checkpoint}")
        report["model"]["checkpoint_sha256"] = checkpoint_sha
        if spec.multiplex:
            report["model"]["multiplex_count"] = args.multiplex_count
        report["model"]["precision_policy"] = spec.precision_policy
        write_json(bench_path, report)

        warm_frames = frames[: min(len(frames), settings.size)]
        warm_seconds = []
        report["warmup_seconds"] = warm_seconds
        report["warmup_detail"] = []
        for index in range(args.warmup):
            started = time.perf_counter()
            warm = run_sequence(backend, warm_frames, prompts, settings, stage_root=stage_root, logger=log)
            warm_seconds.append(time.perf_counter() - started)
            report["warmup_detail"].append(warm.summary())
            write_json(bench_path, report)
            settings.size = warm.final_window_size  # do not pay OOM backoff again in timed runs
            log.info("warm-up %d: %.2fs (%d frames)", index + 1, warm_seconds[-1], len(warm_frames))

        for repeat in range(args.repeats):
            log.info("=== repeat %d/%d (start window %d) ===", repeat + 1, args.repeats, settings.size)
            entry: dict = {"repeat": repeat + 1, "start_window_size": settings.size, "completed": False}
            report["repeats"].append(entry)

            def checkpoint(partial: SequenceResult, entry: dict = entry, repeat: int = repeat) -> None:
                entry.update(partial.summary(), windows_detail=partial.windows,
                             scheduler_events=partial.scheduler_events, completed=False)
                write_json(bench_path, report)
                write_invalid_scores(run_dir / "invalid_scores.csv", repeat + 1, partial.invalid_scores, frames)
                write_progress(run_dir / "progress.json", "running", repeat + 1, args.repeats, partial.progress())

            gpu_id = (env.get("gpu") or {}).get("uuid")
            if gpu_id is None and env.get("cuda_visible_devices") is None:
                gpu_id = 0
            write_progress(run_dir / "progress.json", "running", repeat + 1, args.repeats,
                           SequenceResult(num_frames=len(frames)).progress())
            sampler = ResourceSampler(args.telemetry_interval, gpu_index=gpu_id).start()
            try:
                current = run_sequence(backend, frames, prompts, settings, stage_root=stage_root, logger=log,
                                       on_window=checkpoint)
            finally:
                entry["resources"] = sampler.stop()
            last = current
            entry.update(last.summary(), windows_detail=last.windows, scheduler_events=last.scheduler_events,
                         completed=True)
            report["last_completed_repeat"] = repeat + 1
            write_invalid_scores(run_dir / "invalid_scores.csv", repeat + 1, last.invalid_scores, frames)
            write_progress(run_dir / "progress.json", "repeat_done", repeat + 1, args.repeats, last.progress())
            settings.size = last.final_window_size
            write_json(bench_path, report)
            log.info("repeat %d: %.2fs, %.2f fps, peak %.2f GiB, %d windows, %d OOM backoffs", repeat + 1,
                     entry["total_seconds"], entry["fps"] or 0, entry["peak_reserved_gib"] or 0, entry["windows"],
                     entry["oom_backoffs"])

        totals = [r["total_seconds"] for r in report["repeats"] if r.get("completed")]
        fps = [r["fps"] for r in report["repeats"] if r.get("completed") and r.get("fps")]
        report["aggregate"] = {"total_seconds": stats(totals), "fps": stats(fps),
                               "peak_reserved_gib": max((r["peak_reserved_gib"] or 0 for r in report["repeats"]),
                                                        default=None)}
        report["completed"] = True
        write_progress(run_dir / "progress.json", "completed", args.repeats, args.repeats,
                       last.progress() if last else {})
        if (args.save_predictions or args.save_masks) and last is not None:
            export_started = time.perf_counter()
            write_json(run_dir / "predictions.json", last.predictions())
            report["prediction_export_seconds"] = time.perf_counter() - export_started
    except Exception as exc:
        report["completed"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
        log.exception("benchmark failed")
        exit_code = 1
        mark_progress_failed(run_dir / "progress.json", report["error"])
    finally:
        try:
            export_started = time.perf_counter()
            _write_csv(run_dir / "summary.csv", report)
            export_details(run_dir, report, last, frames)
            report["comparison_export_seconds"] = time.perf_counter() - export_started
            report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            write_json(bench_path, report)
        finally:
            backend.close()
    _print_summary(report, run_dir, spec)
    return exit_code


def write_progress(path: Path, state: str, repeat: int, repeats: int, progress: dict, error: str | None = None) -> None:
    """Small file that can be `cat`-ed during a long run; `overall_percent` spans all repeats."""
    done = progress.get("percent", 0.0)
    overall = round((repeat - 1 + done / 100) / repeats * 100, 2) if repeats else done
    write_json(path, {"state": state, "repeat": repeat, "repeats": repeats, "overall_percent": overall,
                      **progress, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                      **({"error": error} if error else {})})


def mark_progress_failed(path: Path, error: str) -> None:
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        current = {}
    write_json(path, {**current, "state": "failed", "error": error,
                      "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})


def _write_csv(path: Path, report: dict) -> None:
    columns = ["run_label", "model_version", "sam3_commit", "checkpoint_sha256", "precision_policy",
               "torch_version", "cuda_version", "wsl", "nvidia_driver", "gpu", "input_sha256", "width", "height", "prompts",
               "repeat", "completed", "num_frames", "total_seconds", "fps", "unique_fps",
               "amortized_ms_per_unique_frame", "backend_seconds", "staging_seconds", "stitching_seconds",
               "callback_logging_residual_seconds", "processed_window_frames", "attempted_window_frames", "unique_frames_emitted",
               "overlap_frames", "overlap_overhead_fraction", "windows", "final_window_size", "oom_backoffs",
               "peak_reserved_gib", "peak_allocated_gib", "detections", "invalid_score_detections", "detections_per_frame", "tracks_by_label",
               "window_seconds_p50", "window_seconds_p95", "model_load_seconds", "warmup_total_seconds",
               "prob_threshold", "max_objects", "grounding_batch", "offload_state", "overlap", "compile", "save_masks",
               "process_rss_gib_max", "gpu_utilization_pct_mean", "gpu_power_w_mean", "gpu_temperature_c_max"]
    rows = []
    for repeat in report["repeats"]:
        metrics = repeat.get("resources", {}).get("metrics", {})
        row = {**repeat, "run_label": report.get("run_label"), "model_version": report.get("model_version"),
               "sam3_commit": report["environment"].get("sam3_commit"),
               "checkpoint_sha256": report.get("model", {}).get("checkpoint_sha256"),
               "precision_policy": report.get("model", {}).get("precision_policy"),
               "torch_version": report["environment"].get("packages", {}).get("torch"),
               "cuda_version": report["environment"].get("torch_cuda"),
               "wsl": report["environment"].get("wsl"),
               "nvidia_driver": report["environment"].get("nvidia_driver"),
               "gpu": (report["environment"].get("gpu") or {}).get("name"),
               "input_sha256": report.get("input_identity", {}).get("sha256"),
               "width": report["payload"].get("width"), "height": report["payload"].get("height"),
               "prompts": report["prompts"],
               "model_load_seconds": report.get("model", {}).get("model_load_seconds"),
               "warmup_total_seconds": sum(report.get("warmup_seconds", [])),
               **{key: report["args"].get(key) for key in
                  ("prob_threshold", "max_objects", "grounding_batch", "offload_state", "overlap", "compile",
                   "save_masks")},
               "window_seconds_p50": repeat.get("window_seconds", {}).get("p50"),
               "window_seconds_p95": repeat.get("window_seconds", {}).get("p95")}
        for key, field, statistic in (("process_rss_gib_max", "process_rss_gib", "max"),
                                     ("gpu_utilization_pct_mean", "gpu_utilization_pct", "mean"),
                                     ("gpu_power_w_mean", "gpu_power_w", "mean"),
                                     ("gpu_temperature_c_max", "gpu_temperature_c", "max")):
            row[key] = metrics.get(field, {}).get(statistic)
        rows.append(row)
    write_csv(path, rows, columns)


def _print_summary(report: dict, run_dir: Path, spec: ModelSpec) -> None:
    model = report.get("model", {})
    print(f"\n================ {spec.label} benchmark ================")
    gpu = report["environment"].get("gpu") or {}
    print(f"GPU            : {gpu.get('name')} ({gpu.get('total_vram_gib')} GiB, cc {gpu.get('compute_capability')})")
    print(f"frames/prompts : {report['num_frames']} frames x {len(report['prompts'])} prompts {report['prompts']}")
    if model:
        print(f"model load     : {model.get('model_load_seconds', 0):.1f}s, weights "
              f"{model.get('weights_allocated_gib', 0):.2f} GiB, budget {model.get('vram_budget_gib', 0):.1f} GiB")
    for row in report["repeats"]:
        measured_fps = row.get("fps") if row.get("completed") else row.get("unique_fps")
        status = "complete" if row.get("completed") else "incomplete"
        print(f"repeat {row['repeat']} ({status}) : {row.get('total_seconds', 0):.2f}s  {measured_fps or 0:.2f} fps  "
              f"peak {row.get('peak_reserved_gib') or 0:.2f} GiB  window {row.get('final_window_size')}  "
              f"OOM backoffs {row.get('oom_backoffs')}  tracks {row.get('tracks_by_label')}")
    if report.get("error"):
        print(f"ERROR          : {report['error']}")
    print(f"logs           : {run_dir}")
    print("===================================================")
