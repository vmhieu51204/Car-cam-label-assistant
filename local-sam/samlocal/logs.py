"""Per-run log folders: ``logs/<YYYYmmdd-HHMMSS>_<command>/``."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any


def create_run_dir(base: Path, command: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = base / f"{stamp}_{command}"
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = base / f"{stamp}_{command}-{suffix}"
    run_dir.mkdir(parents=True)
    return run_dir


def close_logging(logger: logging.Logger) -> None:
    """Detach shared SAM handlers and release Windows file locks at run end."""
    handlers = tuple(logger.handlers)
    for name in ("sam3", *UPSTREAM_LOGGERS):
        upstream = logging.getLogger(name)
        for handler in handlers:
            upstream.removeHandler(handler)
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()


def setup_logging(run_dir: Path, name: str = "local-sam") -> logging.Logger:
    """Log to ``run_dir/run.log`` (DEBUG) and the console (INFO); also capture sam3's loggers."""
    close_logging(logging.getLogger(name))
    formatter = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers[:] = [file_handler, console]
    logger.propagate = False
    # sam3 logs session/memory info through its own loggers; keep them in run.log only. The model modules use
    # sam3.logger.get_logger, which sets propagate=False, so the "sam3" handler alone never sees them (e.g. the
    # "hitting max_num_objects" warning): attach the file handler to each of those loggers too.
    for name in ("sam3", *UPSTREAM_LOGGERS):
        upstream = logging.getLogger(name)
        if name == "sam3":
            upstream.setLevel(logging.INFO)
        if file_handler not in upstream.handlers:
            upstream.addHandler(file_handler)
    return logger


UPSTREAM_LOGGERS = tuple(f"sam3.model.{module}" for module in (
    "sam3_video_base", "sam3_video_inference", "sam3_video_predictor", "sam3_base_predictor",
    "sam3_multiplex_base", "sam3_multiplex_tracking", "sam3_multiplex_video_predictor", "io_utils"))


def write_json(path: Path, value: Any) -> None:
    """Atomic write so a crash mid-run never leaves a truncated result file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_default), encoding="utf-8")
    os.replace(tmp, path)


def _default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")
