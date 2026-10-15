"""Timing statistics and environment probe."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from typing import Any

from samlocal import SAM3_PINNED_COMMIT, is_wsl

GIB = 1024**3


def stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "stddev": None, "p50": None, "p95": None, "min": None, "max": None}
    ordered = sorted(values)

    def quantile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    mean = sum(values) / len(values)
    return {"count": len(values), "mean": mean,
            "stddev": (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5,
            "p50": quantile(0.5),
            "p95": quantile(0.95), "min": ordered[0], "max": ordered[-1]}


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def sam3_commit() -> str | None:
    """Commit sam3 was installed from (pip records it in direct_url.json for git installs)."""
    try:
        raw = importlib.metadata.distribution("sam3").read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        return None
    if not raw:
        return None
    return json.loads(raw).get("vcs_info", {}).get("commit_id")


def environment() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_logical_count": os.cpu_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "wsl": is_wsl(),
        "packages": {name: package_version(name) for name in
                     ("torch", "torchvision", "triton", "triton-windows", "numpy", "sam3", "timm")},
        "sam3_commit": sam3_commit(),
        "sam3_pinned_commit": SAM3_PINNED_COMMIT,
    }
    try:
        import psutil
        info["system_ram_gib"] = psutil.virtual_memory().total / GIB
    except (ImportError, OSError):
        info["system_ram_gib"] = None
    try:
        driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=2, check=True)
        info["nvidia_driver"] = driver.stdout.strip().splitlines()[0] or None
    except (OSError, subprocess.SubprocessError, IndexError):
        info["nvidia_driver"] = None
    try:
        import torch
    except ImportError:
        return info
    info["torch_cuda"] = torch.version.cuda
    info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu"] = {
            "name": props.name,
            "total_vram_gib": round(props.total_memory / GIB, 2),
            "compute_capability": f"{props.major}.{props.minor}",
            "multiprocessors": props.multi_processor_count,
            "bf16_supported": torch.cuda.is_bf16_supported(),
            "device_count": torch.cuda.device_count(),
            "uuid": str(props.uuid) if getattr(props, "uuid", None) is not None else None,
        }
    return info
