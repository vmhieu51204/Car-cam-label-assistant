#!/usr/bin/env bash
# Recommended setup: WSL2 Ubuntu 22.04/24.04 on a Windows host with an NVIDIA driver.
# Do NOT install an NVIDIA driver inside WSL; the Windows driver is shared.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$HERE/.venv}"
PY="${PYTHON:-python3.12}"
SAM3_COMMIT="0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"

if ! command -v "$PY" >/dev/null 2>&1; then
  echo "Python 3.12 not found. On Ubuntu: sudo apt update && sudo apt install -y python3.12 python3.12-venv git"
  echo "(Ubuntu 22.04: sudo add-apt-repository ppa:deadsnakes/ppa first)"
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1 && [ -x /usr/lib/wsl/lib/nvidia-smi ]; then
  export PATH="$PATH:/usr/lib/wsl/lib"
fi
nvidia-smi || echo "WARNING: nvidia-smi failed - update the Windows NVIDIA driver and run 'wsl --update'"

"$PY" -m venv "$VENV"
PIP="$VENV/bin/python -m pip"
$PIP install --upgrade pip wheel
$PIP install torch==2.10.0 torchvision==0.25.0 --index-url "$TORCH_INDEX"
$PIP install -r "$HERE/requirements.txt"
$PIP install --no-deps "git+https://github.com/facebookresearch/sam3.git@${SAM3_COMMIT}"

cat <<EOF

Done. Next:
  source "$VENV/bin/activate"
  hf auth login                        # token with access to facebook/sam3 and/or facebook/sam3.1 on Hugging Face
  hf download facebook/sam3 sam3.pt --local-dir ~/models                   # SAM 3   (optional, else auto-download)
  hf download facebook/sam3.1 sam3.1_multiplex.pt --local-dir ~/models     # SAM 3.1 (optional, else auto-download)
  cp /mnt/c/<path>/payload.zip ~/data/  # keep data on the Linux filesystem, not /mnt/c
  python "$HERE/sam3/healthcheck_sam3.py"    --payload ~/data/payload.zip --checkpoint ~/models/sam3.pt --load-model
  python "$HERE/sam3.1/healthcheck_sam31.py" --payload ~/data/payload.zip --checkpoint ~/models/sam3.1_multiplex.pt --load-model
EOF
