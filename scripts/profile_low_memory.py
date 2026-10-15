"""Opt-in Docker memory snapshots. No credentials, annotations or model calls."""

import argparse
import csv
import json
import math
import platform
import re
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def memory_bytes(value: str) -> int:
    match = re.fullmatch(r"\s*([\d.]+)\s*(B|kB|KB|KiB|MB|MiB|GB|GiB)\s*", value)
    if not match:
        raise ValueError("unknown Docker memory unit")
    factors = {"B": 1, "kB": 1000, "KB": 1000, "KiB": 1024, "MB": 10**6,
               "MiB": 1024**2, "GB": 10**9, "GiB": 1024**3}
    return round(float(match[1]) * factors[match[2]])


def collect(run: Callable[..., Any] = subprocess.run, *, include_wsl: bool = False) -> dict:
    began = time.monotonic()
    process = run(["docker", "stats", "--no-stream", "--format", "{{json .}}"],
                  capture_output=True, text=True, timeout=30, check=True)
    rows = []
    for line in process.stdout.splitlines():
        data = json.loads(line)
        if not data["Name"].startswith("cvat_"):
            continue
        used = data["MemUsage"].split("/")[0].strip()
        rows.append({"container": data["Name"], "memory_bytes": memory_bytes(used), "cpu": data["CPUPerc"]})
    vmmem = None
    if include_wsl and platform.system() == "Windows":
        try:
            ps = run(["powershell", "-NoProfile", "-Command",
                "@(Get-Process -Name vmmem,vmmemWSL -ErrorAction SilentlyContinue | Select-Object -ExpandProperty WorkingSet64) | ConvertTo-Json"],
                capture_output=True, text=True, timeout=15, check=True)
            memory = json.loads(ps.stdout or "null")
            if isinstance(memory, int) and not isinstance(memory, bool):
                vmmem = memory
            if isinstance(memory, list) and memory:
                vmmem = sum(int(x) for x in memory)
        except (ValueError, subprocess.SubprocessError, OSError):
            pass
    return {"timestamp": datetime.now(UTC).isoformat(), "containers": rows,
            "total_cvat_bytes": sum(r["memory_bytes"] for r in rows),
            "snapshot_seconds": time.monotonic() - began,
            "wsl_vmmem_bytes": vmmem, "task_open_seconds": None,
            "note":"Container memory is not total host RAM; vmmem overlaps container memory, do not add them"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enabled", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("state/profiling/low-memory"))
    parser.add_argument("--task-open-seconds", type=float)
    args = parser.parse_args()
    if not args.enabled:
        print("Disabled. Use --enabled while opening a CVAT task.")
        return
    if args.task_open_seconds is not None and (not math.isfinite(args.task_open_seconds) or args.task_open_seconds < 0):
        parser.error("task-open-seconds must be finite and nonnegative")
    data = collect(include_wsl=True)
    data["task_open_seconds"] = args.task_open_seconds
    args.output.mkdir(parents=True, exist_ok=True)
    suffix = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    (args.output / f"{suffix}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    with (args.output / f"{suffix}.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["container", "memory_bytes", "cpu"])
        writer.writeheader()
        writer.writerows(data["containers"])


if __name__ == "__main__":
    main()
