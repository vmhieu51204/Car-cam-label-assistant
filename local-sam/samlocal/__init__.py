"""Local SAM 3 / SAM 3.1 runner and benchmark for VRAM-limited GPUs.

Standalone on purpose: nothing here imports the main repo, so ``local-sam/`` can be
copied or cloned onto any GPU machine (WSL2 Ubuntu recommended, native Windows works
with ``triton-windows``).
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_DIR = PACKAGE_ROOT / "logs"
DEFAULT_CACHE_DIR = PACKAGE_ROOT / ".cache"

# facebookresearch/sam3 main, 2026-10-07 ("Add SAM3 instance cap, seed frame and reverse box pass").
SAM3_PINNED_COMMIT = "0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc"


def configure_env() -> None:
    """Set allocator/runtime env vars. Must run before ``import torch``."""
    if "torch" in sys.modules:
        return
    # expandable_segments reduces fragmentation when window sizes change; it is
    # unsupported (warning only) on native Windows, so only set it elsewhere.
    if platform.system() != "Windows":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # Quieter HF + tokenizers in benchmarks.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def is_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text(encoding="utf-8").lower()
    except OSError:
        return False
