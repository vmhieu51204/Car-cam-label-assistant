"""Argument parsing shared by the ``run_*``, ``benchmark_*`` and ``healthcheck_*`` scripts of both models."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from samlocal import DEFAULT_CACHE_DIR, DEFAULT_LOG_DIR
from samlocal.backend import VideoBackend, build_backend_for
from samlocal.models import ModelSpec
from samlocal.payload import Payload
from samlocal.pipeline import WindowSettings


def add_common_args(parser: argparse.ArgumentParser, spec: ModelSpec) -> None:
    io = parser.add_argument_group("input / output")
    io.add_argument("--payload", type=Path, required=True,
                    help="payload.zip, its extracted folder, or a folder of frames (required)")
    io.add_argument("--checkpoint", type=Path, default=None,
                    help=f"local {spec.hf_checkpoint}; omit to download {spec.hf_repo} from Hugging Face")
    io.add_argument("--prompts", default=None,
                    help="comma-separated text prompts (default: text_prompts from benchmark-config.json)")
    io.add_argument("--max-frames", type=int, default=None, help="only use the first N frames")
    io.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR, help="base folder for run logs")
    io.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="where payload.zip is extracted")
    io.add_argument("--skip-sha", action="store_true", help="skip per-frame sha256 check from frames.json")
    io.add_argument("--save-masks", action="store_true", help="store RLE masks in predictions.json (larger)")

    model = parser.add_argument_group("model / memory")
    if spec.multiplex:
        model.add_argument("--max-objects", type=int, default=16,
                           help="max tracked objects per prompt per window (16 = one multiplex bucket)")
        model.add_argument("--multiplex-count", type=int, default=16, help="objects per multiplex bucket")
        model.add_argument("--grounding-batch", type=int, default=2,
                           help="frames the detector backbone runs at once (SAM3.1 default 16, ~0.7 GiB/frame); "
                                "1-2 for 8-16 GB GPUs")
    if spec.supports_offload_state:
        model.add_argument("--offload-state", action="store_true",
                           help="keep the tracker's per-frame state in CPU RAM (less VRAM, slower)")
    model.add_argument("--vram-fraction", type=float, default=0.90,
                       help="hard cap of the CUDA allocator as a fraction of VRAM (blocks shared-RAM spill)")
    model.add_argument("--offload-video", choices=("auto", "on", "off"), default="auto",
                       help="keep decoded frames in CPU RAM (auto: on when VRAM <= 24 GiB)")
    model.add_argument("--no-async-frames", action="store_true", help="decode all frames before inference")
    if spec.supports_compile:
        model.add_argument("--compile", action="store_true",
                           help="torch.compile + warm-up (Linux/WSL2 only; slow first start, more VRAM)")
    model.add_argument("--prob-threshold", type=float, default=0.5, help="output mask probability threshold")

    window = parser.add_argument_group("temporal batching")
    window.add_argument("--window-size", type=int, default=48, help="initial frames per SAM3 session")
    window.add_argument("--min-window", type=int, default=8, help="smallest window after OOM backoff")
    window.add_argument("--max-window", type=int, default=160, help="largest window when growing")
    window.add_argument("--overlap", type=int, default=4, help="frames shared by consecutive windows")
    window.add_argument("--grow-below", type=float, default=0.60,
                        help="grow the window when peak VRAM stays under this fraction of the budget")
    window.add_argument("--stitch-iou", type=float, default=0.5, help="box IoU to join tracks across windows")


def resolve_prompts(args: argparse.Namespace, payload: Payload) -> list[str]:
    if args.prompts:
        prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    else:
        prompts = list(payload.prompts)
    if not prompts:
        raise SystemExit("no text prompts: pass --prompts car,pedestrian,... (payload has no text_prompts)")
    return prompts


def window_settings(args: argparse.Namespace) -> WindowSettings:
    return WindowSettings(size=args.window_size, min_size=args.min_window, max_size=args.max_window,
                          overlap=args.overlap, grow_below=args.grow_below, stitch_iou=args.stitch_iou)


def build_backend(args: argparse.Namespace, logger: logging.Logger, spec: ModelSpec) -> VideoBackend:
    common = dict(checkpoint=args.checkpoint, compile_model=getattr(args, "compile", False),
                  offload_video=args.offload_video, async_frames=not args.no_async_frames,
                  vram_fraction=args.vram_fraction, prob_threshold=args.prob_threshold,
                  keep_masks=args.save_masks, logger=logger)
    if spec.multiplex:
        common.update(max_objects=args.max_objects, multiplex_count=args.multiplex_count,
                      grounding_batch=args.grounding_batch)
    else:
        common.update(offload_state=args.offload_state)
    return build_backend_for(spec, **common)
