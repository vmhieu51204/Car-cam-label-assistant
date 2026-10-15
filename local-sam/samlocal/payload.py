"""Open a frame payload: ``payload.zip``, its extracted folder, or a plain folder of frames.

Supported layouts (first match wins):

1. Repo SAM3 sequence/benchmark payload (built by ``src/services/video/sequence_kaggle.py``)::

       frames/frame_000000.jpg ...
       frames.json              # {"frames": [{source_frame_index, relative_path, sha256, width, height}, ...]}
       benchmark-config.json    # optional: configurations[].text_prompts / window_size / overlap
       seed.json, resume.json   # ignored here

2. Any folder (or zip) with image files, ordered by the trailing number in the file name.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_TRAILING_NUMBER = re.compile(r"(\d+)(?!.*\d)")


@dataclass(frozen=True)
class Frame:
    index: int  # 0-based position in the sequence (global frame index for outputs)
    path: Path
    source_index: int  # source_frame_index from frames.json, else == index
    width: int | None = None
    height: int | None = None


@dataclass
class Payload:
    root: Path
    frames: list[Frame]
    prompts: list[str] = field(default_factory=list)
    window_size: int | None = None
    overlap: int | None = None
    layout: str = "frames-dir"
    config: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        first = self.frames[0]
        return {"root": str(self.root), "layout": self.layout, "num_frames": len(self.frames),
                "width": first.width, "height": first.height, "prompts": self.prompts,
                "config_window_size": self.window_size, "config_overlap": self.overlap}


def open_payload(path: Path, cache_dir: Path, *, verify_sha: bool = True) -> Payload:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"payload not found: {path}")
    root = _extract_zip(path, cache_dir) if path.is_file() else path
    root = _find_root(root)
    if (root / "frames.json").is_file():
        payload = _load_indexed(root, verify_sha=verify_sha)
    else:
        payload = _load_plain(root)
    _read_config(payload)
    _fill_size(payload)
    return payload


def _extract_zip(archive: Path, cache_dir: Path) -> Path:
    """Extract once into the cache (keyed by name+size+mtime), with zip-slip protection."""
    if not zipfile.is_zipfile(archive):
        raise ValueError(f"not a zip file or frames folder: {archive}")
    stat = archive.stat()
    target = cache_dir / f"{archive.stem}-{stat.st_size}-{int(stat.st_mtime)}"
    marker = target / ".extracted"
    if marker.is_file():
        return target
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    with zipfile.ZipFile(archive) as source:
        names = [item.filename for item in source.infolist()]
        if len(names) != len(set(names)) or any(
                Path(name).is_absolute() or ".." in Path(name).parts or ":" in name for name in names):
            raise ValueError("unsafe payload archive (absolute path, '..', or duplicate entries)")
        source.extractall(target)
    # Kaggle-style double wrapping: an outer zip that only holds payload.zip.
    inner = list(target.glob("**/payload.zip"))
    if len(inner) == 1 and not list(target.glob("**/frames.json")):
        nested = _extract_zip(inner[0], target / "_nested")
        marker.write_text(str(nested), encoding="utf-8")
        return nested
    marker.write_text("ok", encoding="utf-8")
    return target


def _find_root(base: Path) -> Path:
    indexed = sorted(base.glob("**/frames.json"), key=lambda p: len(p.parts))
    if indexed:
        return indexed[0].parent
    if _images_in(base):
        return base
    candidates = sorted({p.parent for p in base.rglob("*") if p.suffix.lower() in IMAGE_EXTS},
                        key=lambda p: (len(p.parts), str(p)))
    if not candidates:
        raise ValueError(f"no frames.json and no images under {base}")
    return candidates[0]


def _images_in(folder: Path) -> list[Path]:
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def _frame_sort_key(path: Path) -> tuple[int, str]:
    match = _TRAILING_NUMBER.search(path.stem)
    return (int(match.group(1)) if match else -1, path.name)


def _load_indexed(root: Path, *, verify_sha: bool) -> Payload:
    index = json.loads((root / "frames.json").read_text(encoding="utf-8"))
    rows = index["frames"] if isinstance(index, dict) else index
    if not rows:
        raise ValueError("frames.json has no frames")
    rows = sorted(rows, key=lambda row: row.get("source_frame_index", 0))
    frames = []
    for position, row in enumerate(rows):
        relative = Path(row["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe frame path in frames.json: {relative}")
        frame_path = root / relative
        if not frame_path.is_file():
            raise FileNotFoundError(f"frame listed in frames.json is missing: {frame_path}")
        if verify_sha and row.get("sha256"):
            if hashlib.sha256(frame_path.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError(f"sha256 mismatch for {relative}")
        frames.append(Frame(index=position, path=frame_path,
                            source_index=int(row.get("source_frame_index", position)),
                            width=row.get("width"), height=row.get("height")))
    return Payload(root=root, frames=frames, layout="sam3-sequence-payload")


def _load_plain(root: Path) -> Payload:
    folder = root / "frames" if (root / "frames").is_dir() else root
    images = sorted(_images_in(folder), key=_frame_sort_key)
    if not images:
        raise ValueError(f"no images in {folder}")
    frames = [Frame(index=i, path=p, source_index=i) for i, p in enumerate(images)]
    return Payload(root=root, frames=frames, layout="frames-dir")


def _read_config(payload: Payload) -> None:
    config_path = payload.root / "benchmark-config.json"
    if not config_path.is_file():
        return
    config = json.loads(config_path.read_text(encoding="utf-8"))
    payload.config = config
    rows = config.get("configurations") or []
    text_rows = [row for row in rows if row.get("text_prompts")]
    row = text_rows[0] if text_rows else (rows[0] if rows else {})
    payload.prompts = [str(p) for p in row.get("text_prompts") or []]
    payload.window_size = row.get("window_size")
    payload.overlap = row.get("overlap")


def _fill_size(payload: Payload) -> None:
    first = payload.frames[0]
    if first.width and first.height:
        return
    from PIL import Image

    with Image.open(first.path) as image:
        width, height = image.size
    payload.frames = [Frame(f.index, f.path, f.source_index, f.width or width, f.height or height)
                      for f in payload.frames]
