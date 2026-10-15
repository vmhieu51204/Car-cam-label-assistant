#!/usr/bin/env bash
# Benchmark SAM 3 / SAM 3.1 on one KITTI payload, then score against KITTI GT. One model per call is the usual
# way (each call = its own output folder; compare folders afterwards with kitti/compare_kitti.sh):
#
#   MODELS=sam3   bash kitti/run_kitti_benchmark.sh ~/data/kitti_0014.zip
#   MODELS=sam3.1 bash kitti/run_kitti_benchmark.sh ~/data/kitti_0014.zip
#   MODELS=sam3.1 COMPILE=1 GROUNDING_BATCH=4 TAG=best bash kitti/run_kitti_benchmark.sh ~/data/kitti_0014.zip
#
# Before every model: wait until the GPU has cooled down, flush the Linux page cache (sudo), check the GPU is idle.
# While a model runs: nvidia-smi logs temperature / SM clock / power / throttle reasons every second into
# gpu_<label>.csv, summarised in gpu_summary.txt.
#
# Environment overrides (defaults in brackets):
#   MODELS_DIR [~/models]   folder with sam3.pt and sam3.1_multiplex.pt (missing file -> Hugging Face download)
#   MODELS ["sam3 sam3.1"]  models to run, in this order (a repeated model gets a -r2, -r3 label suffix)
#   WARMUP [1] REPEATS [3]  untimed warm-up passes / timed passes (quality is scored on the last repeat)
#   WINDOW [48] OVERLAP [4] window and overlap, same for every model
#   FIXED_WINDOW [1]        1 = min = max = WINDOW (same window boundaries for both models; OOM fails the run,
#                           then lower WINDOW); 0 = adaptive (grow / back off per model)
#   MAX_OBJECTS [32]        SAM 3.1 objects per prompt per window (SAM 3 has no cap); the report shows cap hits
#   GROUNDING_BATCH [2]     SAM 3.1 detector frames per batch (more = faster, more VRAM)
#   COMPILE [0]             1 = SAM 3.1 torch.compile (slow first start; the untimed warm-up absorbs it)
#   SAVE_MASKS [0]          1 = store masks for KITTI MOTS (mots_kitti.py); RLE encoding is timed, so speed is
#                           only comparable between runs with the same SAVE_MASKS
#   DROP_OS_CACHE [1]       1 = flush the Linux page cache before each model (asks for sudo once at the start)
#   COOL_TEMP [45]          wait until the GPU is at or below this temperature (deg C) before each model ...
#   COOLDOWN_MIN [60]       ... and at least this many seconds after the previous model ...
#   COOLDOWN_MAX [600]      ... but never longer than this (a warning is printed)
#   PROMPTS []              comma-separated prompts instead of the payload defaults
#   TAG []                  suffix of the output folder and of the run labels, e.g. TAG=best
#   EXTRA_ARGS []           appended to every benchmark command, e.g. "--max-frames 50"
#   EXTRA_ARGS_SAM3 [] EXTRA_ARGS_SAM31 []   appended to that model only
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAYLOAD="${1:?usage: bash kitti/run_kitti_benchmark.sh <kitti_XXXX.zip>}"
PY="${PYTHON:-python}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
MODELS="${MODELS:-sam3 sam3.1}"
WARMUP="${WARMUP:-1}" REPEATS="${REPEATS:-3}" WINDOW="${WINDOW:-48}" OVERLAP="${OVERLAP:-4}"
MAX_OBJECTS="${MAX_OBJECTS:-32}" GROUNDING_BATCH="${GROUNDING_BATCH:-2}"
COOL_TEMP="${COOL_TEMP:-45}" COOLDOWN_MIN="${COOLDOWN_MIN:-60}" COOLDOWN_MAX="${COOLDOWN_MAX:-600}"
SEQ="$(basename "$PAYLOAD" .zip)"
OUT="$HERE/logs/kitti/$(date +%Y%m%d-%H%M%S)_${SEQ}${TAG:+_$TAG}"
SMI="$(command -v nvidia-smi || echo /usr/lib/wsl/lib/nvidia-smi)"
mkdir -p "$OUT"
echo "output: $OUT"

# Fail fast if the scoring itself is wrong (CPU only, a few seconds).
"$PY" "$HERE/kitti/eval_kitti.py" --payload "$PAYLOAD" --sanity | tee "$OUT/sanity.txt" || {
  echo "scoring sanity check failed - fix it before spending GPU time"; exit 1; }

WINDOW_ARGS=(--window-size "$WINDOW" --overlap "$OVERLAP")
[ "${FIXED_WINDOW:-1}" = 1 ] && WINDOW_ARGS+=(--min-window "$WINDOW" --max-window "$WINDOW")
COMMON_ARGS=()
[ -n "${PROMPTS:-}" ] && COMMON_ARGS+=(--prompts "$PROMPTS")
[ "${SAVE_MASKS:-0}" = 1 ] && COMMON_ARGS+=(--save-masks)
CLEAR_ARGS=()
if [ "${DROP_OS_CACHE:-1}" = 1 ]; then
  if sudo -v; then
    CLEAR_ARGS+=(--drop-os-cache)
    # keep the sudo ticket alive for the whole run so the flush never stops at a password prompt
    while true; do sudo -n true 2>/dev/null; sleep 50; done & SUDO_KEEPALIVE=$!
    trap 'kill "$SUDO_KEEPALIVE" 2>/dev/null' EXIT
  else
    echo "WARNING: no sudo - page cache is NOT flushed between models (set DROP_OS_CACHE=0 to silence)"
  fi
fi

gpu_temp() { "$SMI" --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '; }

cooldown() {  # $1 = epoch seconds when the previous model ended (0 = none)
  local started now temp waited
  started=$(date +%s)
  while true; do
    now=$(date +%s); temp="$(gpu_temp)"; waited=$((now - started))
    if [ "$1" -eq 0 ] || [ $((now - $1)) -ge "$COOLDOWN_MIN" ]; then
      if [ -z "$temp" ]; then echo "cooldown: GPU temperature not readable, waited ${waited}s"; return; fi
      if [ "$temp" -le "$COOL_TEMP" ]; then echo "cooldown: GPU ${temp}C <= ${COOL_TEMP}C after ${waited}s"; return; fi
    fi
    if [ "$waited" -ge "$COOLDOWN_MAX" ]; then
      echo "WARNING: GPU still at ${temp}C after ${COOLDOWN_MAX}s (target ${COOL_TEMP}C) - starting anyway"; return
    fi
    printf '\rcooldown: GPU %sC, waiting (%ss)   ' "${temp:-?}" "$waited"; sleep 5
  done
}

GPU_FIELDS="timestamp,temperature.gpu,clocks.sm,clocks.max.sm,power.draw,utilization.gpu,memory.used"
for reasons in clocks_event_reasons.active clocks_throttle_reasons.active; do
  if "$SMI" --query-gpu="$reasons" --format=csv,noheader >/dev/null 2>&1; then GPU_FIELDS+=",$reasons"; break; fi
done

# Clock-limit reason bits (low byte): 0x04 power cap (normal under load), 0x08 HW slowdown, 0x20 SW thermal,
# 0x40 HW thermal -> "thermal" samples mean the GPU was hot enough to slow down and timings are suspect.
summarise_gpu() {  # $1 = csv, $2 = label   (plain POSIX awk: WSL Ubuntu ships mawk, no strtonum/and)
  awk -F', *' -v label="$2" 'NR > 1 && $2 ~ /^[0-9]+$/ {
      n++; t += $2; if ($2 > tmax) tmax = $2; s += $3; if (smin == "" || $3 < smin) smin = $3; smax = $4
      p = $5 + 0; if (p > pmax) pmax = p
      if (NF >= 8 && $8 ~ /^0x/) {
        hex = tolower(substr($8, length($8) - 1))
        v = (index("0123456789abcdef", substr(hex, 1, 1)) - 1) * 16 + index("0123456789abcdef", substr(hex, 2, 1)) - 1
        if (int(v / 4) % 2) power++
        if (int(v / 8) % 2 || int(v / 32) % 2 || int(v / 64) % 2) thermal++ } }
    END { if (n) printf "%s: %d s sampled, temp mean %.0fC max %dC, SM clock mean %.0f min %d (max %s) MHz, " \
                        "power max %.0f W, power-cap samples %d, THERMAL/HW-slowdown samples %d\n", label, n, t / n,
                        tmax, s / n, smin, smax, pmax, power, thermal
          else printf "%s: no GPU samples\n", label }' "$1"
}

declare -A seen
last_end=0
for model in $MODELS; do
  case "$model" in
    sam3)   script="sam3/benchmark_sam3.py";    ckpt="sam3.pt"
            extra=() ; read -r -a model_extra <<< "${EXTRA_ARGS_SAM3:-}" ;;
    sam3.1) script="sam3.1/benchmark_sam31.py"; ckpt="sam3.1_multiplex.pt"
            extra=(--max-objects "$MAX_OBJECTS" --grounding-batch "$GROUNDING_BATCH")
            [ "${COMPILE:-0}" = 1 ] && extra+=(--compile)
            read -r -a model_extra <<< "${EXTRA_ARGS_SAM31:-}" ;;
    *) echo "unknown model $model"; exit 2 ;;
  esac
  seen[$model]=$(( ${seen[$model]:-0} + 1 ))
  label="kitti-$SEQ-$model${TAG:+-$TAG}"
  [ "${seen[$model]}" -gt 1 ] && label+="-r${seen[$model]}"
  ckpt_args=()
  if [ -f "$MODELS_DIR/$ckpt" ]; then ckpt_args=(--checkpoint "$MODELS_DIR/$ckpt")
  else echo "WARNING: $MODELS_DIR/$ckpt not found, downloading from Hugging Face"; fi

  echo "================ $label ================"
  cooldown "$last_end"
  bash "$HERE/kitti/clear_cache.sh" "${CLEAR_ARGS[@]}"
  gpu_csv="$OUT/gpu_${label}.csv"
  "$SMI" --query-gpu="$GPU_FIELDS" --format=csv,nounits -l 1 > "$gpu_csv" 2>/dev/null & monitor=$!
  # shellcheck disable=SC2086
  "$PY" "$HERE/$script" --payload "$PAYLOAD" "${ckpt_args[@]}" --log-dir "$OUT" \
      --run-label "$label" --warmup "$WARMUP" --repeats "$REPEATS" --save-predictions \
      "${WINDOW_ARGS[@]}" "${COMMON_ARGS[@]}" "${extra[@]}" "${model_extra[@]}" ${EXTRA_ARGS:-} \
    || echo "WARNING: $model benchmark exited with an error (see $OUT/*_benchmark_*/run.log)"
  kill "$monitor" 2>/dev/null; wait "$monitor" 2>/dev/null
  summarise_gpu "$gpu_csv" "$label" | tee -a "$OUT/gpu_summary.txt"
  last_end=$(date +%s)
done

runs=()
for dir in "$OUT"/*_benchmark_*; do
  [ -f "$dir/predictions.json" ] && runs+=("$dir")
done
if [ "${#runs[@]}" -eq 0 ]; then
  echo "no run produced predictions.json - nothing to score"; exit 1
fi
"$PY" "$HERE/kitti/eval_kitti.py" --payload "$PAYLOAD" --runs "${runs[@]}" --out "$OUT/eval"
cat "$OUT/gpu_summary.txt"
