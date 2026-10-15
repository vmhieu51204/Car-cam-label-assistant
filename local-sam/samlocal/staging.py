"""Stage a window of frames into the folder layout SAM3 expects.

SAM3's JPEG-folder loader sorts by ``int(<file stem>)``. Payload names like
``frame_000000.jpg`` fail that and fall back to a lexicographic sort, so every
window is staged as ``000000.jpg, 000001.jpg, ...``. Hardlinks make this free on
the same volume; we fall back to copying (e.g. across drives on Windows).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from samlocal.payload import Frame


def stage_window(frames: list[Frame], dest: Path) -> Path:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for local, frame in enumerate(frames):
        target = dest / f"{local:06d}{frame.path.suffix.lower()}"
        try:
            os.link(frame.path, target)
        except OSError:
            shutil.copyfile(frame.path, target)
    return dest


def remove_window(dest: Path) -> None:
    shutil.rmtree(dest, ignore_errors=True)
