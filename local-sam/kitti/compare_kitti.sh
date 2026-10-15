#!/usr/bin/env bash
# Score runs from several run_kitti_benchmark.sh output folders in one report (e.g. one folder per model).
#
#   bash kitti/compare_kitti.sh ~/data/kitti_0014.zip logs/kitti/<sam3 folder> logs/kitti/<sam3.1 folder> ...
#
# Writes logs/kitti/<stamp>_compare_<seq>/{kitti_report.md, kitti_summary.csv, kitti_eval.json, gpu_summary.txt}.
# On a machine with the TrackEval venv (TRACKEVAL_PY, default ~/venvs/trackeval/bin/python) it also adds the
# official TrackEval box numbers and, when the payload has KITTI MOTS labels and the runs saved masks, MOTS.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAYLOAD="${1:?usage: bash kitti/compare_kitti.sh <kitti_XXXX.zip> <run folder>...}"
shift
if [ ! -f "$PAYLOAD" ] && [ ! -f "$PAYLOAD/kitti_manifest.json" ]; then
  echo "first argument must be the payload zip (got '$PAYLOAD'); is \$P set in this shell? e.g. P=~/data/kitti_0014.zip"
  exit 2
fi
[ "$#" -gt 0 ] || { echo "pass at least one output folder of run_kitti_benchmark.sh"; exit 2; }
PY="${PYTHON:-python}"
TRACKEVAL_PY="${TRACKEVAL_PY:-$HOME/venvs/trackeval/bin/python}"
SEQ="$(basename "$PAYLOAD" .zip)"
OUT="${OUT:-$HERE/logs/kitti/$(date +%Y%m%d-%H%M%S)_compare_${SEQ}}"
mkdir -p "$OUT"

runs=() masked=()
for folder in "$@"; do
  [ -d "$folder" ] || { echo "WARNING: $folder is not a folder - skipped"; continue; }
  found=0
  for dir in "$folder"/*_benchmark_* "$folder"; do
    [ -f "$dir/predictions.json" ] || continue
    runs+=("$dir"); found=1
    grep -q '"mask_rle"' "$dir/predictions.json" && masked+=("$dir")
  done
  [ "$found" = 1 ] || echo "WARNING: no predictions.json in $folder (the run failed? see its */run.log) - skipped"
  [ -f "$folder/gpu_summary.txt" ] && cat "$folder/gpu_summary.txt" >> "$OUT/gpu_summary.txt"
done
[ "${#runs[@]}" -gt 0 ] || { echo "no predictions.json under: $*"; exit 1; }
echo "runs: ${runs[*]}"

"$PY" "$HERE/kitti/eval_kitti.py" --payload "$PAYLOAD" --runs "${runs[@]}" --out "$OUT" || exit 1
if [ -x "$TRACKEVAL_PY" ]; then
  "$TRACKEVAL_PY" -I -W ignore "$HERE/kitti/trackeval_kitti.py" --payload "$PAYLOAD" --eval "$OUT"
  if [ "${#masked[@]}" -gt 0 ]; then
    "$TRACKEVAL_PY" -I -W ignore "$HERE/kitti/mots_kitti.py" --payload "$PAYLOAD" --runs "${masked[@]}" --out "$OUT"
  fi
else
  echo "TrackEval venv not found ($TRACKEVAL_PY): official box / MOTS scoring skipped (run this on the host)"
fi
[ -f "$OUT/gpu_summary.txt" ] && cat "$OUT/gpu_summary.txt"
echo "report: $OUT/kitti_report.md"
