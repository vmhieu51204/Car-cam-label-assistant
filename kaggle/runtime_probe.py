"""Optional, JSON-safe GPU runtime observations for finite model jobs."""

import hashlib
import importlib.metadata
import json
import statistics
import subprocess
import sys
from pathlib import Path


def _version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def sam3_commit():
    """Read the installed VCS commit when pip preserved it; never invent a revision."""
    try:
        distribution = importlib.metadata.distribution("sam3")
        direct_url = distribution.read_text("direct_url.json")
        if direct_url:
            commit = json.loads(direct_url).get("vcs_info", {}).get("commit_id")
            if commit:
                return commit
    except (importlib.metadata.PackageNotFoundError, ValueError, TypeError):
        pass
    try:
        import sam3

        root = Path(sam3.__file__).resolve().parent
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (ImportError, OSError, subprocess.TimeoutExpired, TypeError):
        return None


def environment(torch, *, include_sam3=False):
    versions = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": _version("torchvision"),
        "transformers": _version("transformers"),
        "cuda": torch.version.cuda,
        "numpy": _version("numpy"),
        "sam3_commit": sam3_commit() if include_sam3 else None,
    }
    gpu = None
    if torch.cuda.is_available():
        device = torch.cuda.get_device_properties(0)
        gpu = {
            "name": device.name,
            "compute_capability": [device.major, device.minor],
            "total_vram_bytes": device.total_memory,
        }
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5, check=False,
        )
        nvidia_smi = result.stdout[:8192] if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        nvidia_smi = None
    return {"schema": "autolabel-runtime/v1", "versions": versions,
            "gpu": gpu, "nvidia_smi": nvidia_smi, "timing_s": {},
            "peak_cuda_allocated_bytes": {}}


def begin_cuda(torch):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def end_cuda(torch):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated()
    return None


def duration_stats(samples):
    if not samples:
        return {"count": 0, "mean": None, "p50": None, "p95": None}
    ordered = sorted(samples)

    def percentile(fraction):
        index = (len(ordered) - 1) * fraction
        lower = int(index)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)

    return {"count": len(samples), "mean": statistics.fmean(samples),
            "p50": percentile(0.5), "p95": percentile(0.95)}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
