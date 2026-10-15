#!/usr/bin/env bash
# Put the machine in the same state before each benchmark run (WSL2/Linux).
#
#   bash kitti/clear_cache.sh                    # staging leftovers + GPU check (safe, no sudo)
#   bash kitti/clear_cache.sh --drop-os-cache    # + flush Linux page cache (sudo): cold-disk start for every model
#   bash kitti/clear_cache.sh --payload-cache    # + delete extracted payloads in .cache/ (re-extracted next run)
#   bash kitti/clear_cache.sh --compile-cache    # + delete Triton/TorchInductor caches (only to time a cold --compile)
#
# Full reset from Windows PowerShell (frees all WSL RAM, kills stray GPU processes):  wsl --shutdown
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE="${CACHE_DIR:-$HERE/.cache}"
DROP_OS=0 PAYLOAD=0 COMPILE=0
for arg in "$@"; do
  case "$arg" in
    --drop-os-cache) DROP_OS=1 ;;
    --payload-cache) PAYLOAD=1 ;;
    --compile-cache) COMPILE=1 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg"; exit 2 ;;
  esac
done

# 1) Window staging folders left by an interrupted run (normally removed after each window).
rm -rf "$CACHE/staging"
echo "cleared $CACHE/staging"

# 2) Extracted payloads. Extraction is outside the timed region, so keep it unless disk is short.
if [ "$PAYLOAD" = 1 ]; then
  find "$CACHE" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
  echo "cleared extracted payloads in $CACHE"
fi

# 3) JIT caches. Keeping them is faster and fair (warm-up is untimed); delete only to measure a cold compile.
if [ "$COMPILE" = 1 ]; then
  rm -rf "${TRITON_CACHE_DIR:-$HOME/.triton/cache}" "/tmp/torchinductor_${USER:-$(id -un)}" \
         "${TORCHINDUCTOR_CACHE_DIR:-/nonexistent}"
  echo "cleared Triton / TorchInductor caches"
fi

# 4) OS page cache: the next run reads frames and the checkpoint from disk again.
if [ "$DROP_OS" = 1 ]; then
  sync
  echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
  echo 1 | sudo tee /proc/sys/vm/compact_memory >/dev/null 2>&1 || true
  echo "dropped Linux page cache"
fi

# 5) The GPU must be idle: another process (an old benchmark, a notebook) distorts time and VRAM.
SMI="$(command -v nvidia-smi || echo /usr/lib/wsl/lib/nvidia-smi)"
if [ -x "$SMI" ]; then
  "$SMI" --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader
  apps="$("$SMI" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)"
  if [ -n "$apps" ]; then
    echo "WARNING: GPU compute processes still running (stop them before benchmarking):"
    echo "$apps"
  fi
fi
pgrep -af "benchmark_sam3|benchmark_sam31|run_sam3|run_sam31" | grep -v pgrep || true
free -h | sed -n '1,2p'
