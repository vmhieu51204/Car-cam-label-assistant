"""Materialize unlabeled all-frame notebook archives locally, with bounded windows."""

import hashlib
import io
import json
import tempfile
import zipfile
from pathlib import Path

from PIL import Image

from kaggle.runtime_probe import sha256_file
from kaggle.sam3_benchmark.main import validate_config
from src.services.video.chunks import plan_chunks
from src.services.video.extractor import _histogram_distance

SLOW_SECONDS_PER_FRAME = 598.39 / 100
FAST_SECONDS_PER_FRAME = 1569.34 / 419


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def save_equal_or_new(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"incompatible existing output: {path.name}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)


def split_windows(
    frame_bytes: list[int],
    *,
    max_bytes: int = 2_000_000_000,
    session_seconds: float = 21600,
    reserve_seconds: float = 180,
) -> list[tuple[int, int]]:
    """Keep complete 50/5 windows. Kernel overlap is explicit, not evidence of continuous IDs."""
    if not frame_bytes or any(type(n) is not int or n < 0 for n in frame_bytes):
        raise ValueError("positive frame population and nonnegative byte sizes required")
    if max_bytes < 1 or not reserve_seconds < session_seconds <= 21600:
        raise ValueError("invalid payload/time budget")
    windows = plan_chunks(len(frame_bytes), window_size=50, overlap=5)
    spans = []
    start, stop = windows[0].start, windows[0].stop

    def fits(left: int, right: int) -> bool:
        return (
            sum(frame_bytes[left:right]) <= max_bytes
            and (right - left) * SLOW_SECONDS_PER_FRAME * 1.25 + reserve_seconds <= session_seconds
        )

    if not fits(start, stop):
        raise ValueError("one complete model window exceeds budget; cannot drop frames")
    for window in windows[1:]:
        if fits(start, window.stop):
            stop = window.stop
        else:
            spans.append((start, stop))
            start, stop = window.start, window.stop
            if not fits(start, stop):
                raise ValueError("one complete model window exceeds budget; cannot drop frames")
    spans.append((start, stop))
    return spans


def materialize(
    training: Path, inventory: dict, output: Path, *, max_bytes: int = 2_000_000_000, session_seconds: float = 21600
) -> list[dict]:
    """Re-check every source hash. Publish each ZIP only after all content is validated."""
    records = inventory["images"]
    request = {
        "source_images_sha256": inventory["images_content_sha256"],
        "max_bytes": max_bytes,
        "session_seconds": session_seconds,
        "window": 50,
        "overlap": 5,
        "all_frames": True,
        "jpeg_quality": 95,
        "scene_cut_threshold": 0.35,
        "classes": ["car", "person"],
    }
    for record in records:
        path = training / "image_02" / inventory["sequence"] / f"{record['source_frame']:06d}.png"
        if sha256_file(path) != record["sha256"]:
            raise ValueError("source image hash mismatch before packaging")
    manifest_path = output / "manifest.json"
    journal_path = output / "preparation.json"
    journal = None
    if manifest_path.exists():
        stored = json.loads(manifest_path.read_text(encoding="utf-8"))
        if stored.get("request") != request:
            raise ValueError("incompatible existing payload manifest")
        for kernel in stored["kernels"]:
            if sha256_file(output / kernel["archive_name"]) != kernel["archive_sha256"]:
                raise ValueError("existing archive hash mismatch")
        return stored["kernels"]
    if journal_path.exists():
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
        if journal.get("request") != request:
            raise ValueError("incompatible existing preparation journal")
        allowed = {"preparation.json", journal["staging_directory"], *(k["archive_name"] for k in journal["kernels"])}
        if Path(journal["staging_directory"]).name != journal["staging_directory"]:
            raise ValueError("invalid preparation staging directory")
        if any(p.name not in allowed for p in output.iterdir()):
            raise ValueError("incompatible existing payload output")
        for kernel in journal["kernels"]:
            path = output / kernel["archive_name"]
            if path.exists() and sha256_file(path) != kernel["archive_sha256"]:
                raise ValueError("existing archive hash mismatch")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("incompatible existing payload output without manifest")
    output.mkdir(parents=True, exist_ok=True)
    kernels = []
    with tempfile.TemporaryDirectory(prefix="prepare-", dir=output) as temporary:
        staged = Path(temporary)
        prepared = []
        previous = None
        dimensions = None
        for record in records:
            source = training / "image_02" / inventory["sequence"] / f"{record['source_frame']:06d}.png"
            data = source.read_bytes()
            if hashlib.sha256(data).hexdigest() != record["sha256"]:
                raise ValueError("source image hash changed during packaging")
            with Image.open(io.BytesIO(data)) as image:
                if dimensions is None:
                    dimensions = image.size
                if image.size != dimensions:
                    raise ValueError("inconsistent source image dimensions")
                histogram = image.convert("L").resize((64, 36)).histogram()
                distance = _histogram_distance(previous, histogram)
                previous = histogram
                rgb = image.convert("RGB")
                target = staged / f"{record['source_frame']:06d}.jpg"
                rgb.save(target, format="JPEG", quality=95)
            prepared.append(
                {
                    **record,
                    "path": target,
                    "sha256": sha256_file(target),
                    "bytes": target.stat().st_size,
                    "width": dimensions[0],
                    "height": dimensions[1],
                    "source_sha256": record["sha256"],
                    "scene_cut_before": distance is not None and distance >= 0.35,
                }
            )
        spans = split_windows([r["bytes"] for r in prepared], max_bytes=max_bytes, session_seconds=session_seconds)
        for index, (start, stop) in enumerate(spans):
            frames = [
                {
                    "source_frame_index": local,
                    "original_source_index": r["source_frame"],
                    "timestamp_s": r["source_frame"] / 10,
                    "relative_path": f"frames/frame_{local:06d}.jpg",
                    "sha256": r["sha256"],
                    "width": r["width"],
                    "height": r["height"],
                    "scene_cut_before": bool(local and r["scene_cut_before"]),
                }
                for local, r in enumerate(prepared[start:stop])
            ]
            config = {
                "schema": "autolabel-sam3-benchmark-config/v1",
                "source_frame_limit": len(frames),
                "max_total_seconds": session_seconds,
                "max_config_seconds": session_seconds - 180,
                "configurations": [
                    {
                        "id": "sequence-text",
                        "model_variant": "sam3",
                        "selection": "dense",
                        "sampling_strategy": "all_frames",
                        "gap": None,
                        "mode": "text_prompt",
                        "seed_type": "none",
                        "reprompt": "none",
                        "text_prompts": ["car", "person"],
                        "window_size": 50,
                        "overlap": 5,
                        "preserve_overlap": True,
                        "scene_cut_frames": [r["source_frame_index"] for r in frames if r["scene_cut_before"]],
                    }
                ],
            }
            validate_config(config)
            mapping = {
                "frames": frames,
                "optical_source_indices": list(range(len(frames))),
                "source_fps": 10,
                "target_fps": 10,
                "fps_source": "KITTI nominal cadence, not measured timestamps",
            }
            name = f"kernel-{index:03d}.zip"
            archive = staged / name
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as stream:
                for local, record in enumerate(prepared[start:stop]):
                    info = zipfile.ZipInfo(f"frames/frame_{local:06d}.jpg")
                    info.compress_type = zipfile.ZIP_DEFLATED
                    stream.writestr(info, record["path"].read_bytes())
                for member, value in (
                    ("frames.json", mapping),
                    ("benchmark-config.json", config),
                    ("seed.json", {"seeds": []}),
                ):
                    info = zipfile.ZipInfo(member)
                    info.compress_type = zipfile.ZIP_DEFLATED
                    stream.writestr(info, canonical_bytes(value))
            if archive.stat().st_size > max_bytes:
                raise ValueError("archive overhead exceeds max_bytes; use a smaller window-aligned payload budget")
            kernels.append(
                {
                    "sequence": inventory["sequence"],
                    "split": "dev",
                    "archive_name": name,
                    "archive_sha256": sha256_file(archive),
                    "archive_bytes": archive.stat().st_size,
                    "source_start": start,
                    "source_stop_exclusive": stop,
                    "frames": len(frames),
                    "source_frames": list(range(start, stop)),
                    "config_sha256": hashlib.sha256(canonical_bytes(config)).hexdigest(),
                    "frames_sha256": hashlib.sha256(canonical_bytes(mapping)).hexdigest(),
                    "source_hashes": [r["source_sha256"] for r in prepared[start:stop]],
                    "rough_budget_seconds": len(frames) * SLOW_SECONDS_PER_FRAME * 1.25 + 180,
                    "inference_seconds_range": [
                        len(frames) * FAST_SECONDS_PER_FRAME,
                        len(frames) * SLOW_SECONDS_PER_FRAME,
                    ],
                    "gt_in_payload": False,
                    "uploaded": False,
                    "status": "prepared_offline_not_run",
                    "kernel_boundary_note": "Overlap frames repeat; IDs are not continuous across kernels; evaluate segments separately",
                }
            )
        # A durable journal makes interrupted publication resumable without touching foreign files.
        if journal is not None and journal["kernels"] != kernels:
            raise ValueError("incompatible rebuilt payload kernels")
        staged_journal = staged / "preparation.json"
        staged_journal.write_bytes(
            canonical_bytes({"request": request, "kernels": kernels, "staging_directory": staged.name})
        )
        staged_journal.replace(journal_path)
        for kernel in kernels:
            target = output / kernel["archive_name"]
            if target.exists():
                if sha256_file(target) != kernel["archive_sha256"]:
                    raise ValueError("existing archive hash mismatch")
            else:
                (staged / kernel["archive_name"]).replace(target)
        staged_manifest = staged / "manifest.json"
        staged_manifest.write_bytes(
            canonical_bytes(
                {
                    "schema": "autolabel-tracking-payloads/v1",
                    "request": request,
                    "kernels": kernels,
                    "gt_in_payload": False,
                    "uploaded": False,
                }
            )
        )
        if manifest_path.exists():
            raise ValueError("manifest appeared during publication; refusing overwrite")
        staged_manifest.replace(manifest_path)
    return kernels
