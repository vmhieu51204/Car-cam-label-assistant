"""Run SAM 3 or SAM 3.1 once over a frame payload and save predictions (entry scripts:
``sam3/run_sam3.py``, ``sam3.1/run_sam31.py``).

Writes logs/<timestamp>_run_<model>/{run.log, env.json, predictions.json, summary.json[, vis/*.jpg]}.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from samlocal.cli import add_common_args, build_backend, resolve_prompts, window_settings
from samlocal.logs import create_run_dir, setup_logging, write_json
from samlocal.metrics import environment
from samlocal.models import ModelSpec, get_spec
from samlocal.payload import Frame, open_payload
from samlocal.pipeline import SequenceResult, run_sequence

PALETTE = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
           (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212)]


def main(model: str | ModelSpec, argv: list[str] | None = None) -> int:
    spec = get_spec(model) if isinstance(model, str) else model
    parser = argparse.ArgumentParser(
        description=f"Run {spec.label} once over a frame payload.\n\nExample (WSL2):\n"
                    f"  python run_{spec.slug}.py --payload ~/data/payload.zip "
                    f"--checkpoint ~/models/{spec.hf_checkpoint} --visualize 10",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser, spec)
    parser.add_argument("--visualize", type=int, default=0, metavar="EVERY",
                        help="save box overlays for every N-th frame to vis/ (0 = off)")
    args = parser.parse_args(argv)

    run_dir = create_run_dir(args.log_dir, f"run_{spec.slug}")
    log = setup_logging(run_dir)
    write_json(run_dir / "env.json", environment())
    payload = open_payload(args.payload, args.cache_dir, verify_sha=not args.skip_sha)
    frames = payload.frames[: args.max_frames] if args.max_frames else payload.frames
    prompts = resolve_prompts(args, payload)
    log.info("payload: %s", payload.describe())

    backend = build_backend(args, log, spec)
    try:
        model_info = backend.load()
        result = run_sequence(backend, frames, prompts, window_settings(args), stage_root=args.cache_dir / "staging",
                              logger=log, on_window=lambda r: write_json(run_dir / "summary.json", r.summary()))
    except Exception:
        log.exception("run failed")
        return 1
    finally:
        backend.close()
    write_json(run_dir / "predictions.json", result.predictions())
    write_json(run_dir / "summary.json", {"model": model_info, "prompts": prompts, **result.summary(),
                                          "windows_detail": result.windows,
                                          "scheduler_events": result.scheduler_events})
    if args.visualize > 0:
        _visualize(frames, result, run_dir / "vis", every=args.visualize)
    summary = result.summary()
    log.info("done: %d frames in %.2fs (%.2f fps), peak %.2f GiB, tracks %s -> %s", summary["num_frames"],
             summary["total_seconds"], summary["fps"] or 0, summary["peak_reserved_gib"] or 0,
             summary["tracks_by_label"], run_dir)
    return 0


def _visualize(frames: list[Frame], result: SequenceResult, out_dir: Path, *, every: int) -> None:
    from PIL import Image, ImageDraw

    out_dir.mkdir(parents=True, exist_ok=True)
    for frame in frames[::every]:
        with Image.open(frame.path) as opened:
            image = opened.convert("RGB")
        draw = ImageDraw.Draw(image)
        for det in result.frames.get(frame.index, []):
            color = PALETTE[det.track_id % len(PALETTE)]
            draw.rectangle(det.bbox, outline=color, width=2)
            draw.text((det.bbox[0] + 2, det.bbox[1] + 2), f"{det.label}:{det.track_id} {det.score:.2f}", fill=color)
        image.save(out_dir / f"{frame.index:06d}.jpg", quality=90)
