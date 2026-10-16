#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 4 ]]; then
  echo "Usage: $0 OUTPUT INITIAL_CHECKPOINT MIDDLE_CHECKPOINT FINAL_CHECKPOINT" >&2
  exit 2
fi
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_COMPILE_THREADS=4
export OMP_NUM_THREADS=4
cache_root=$(mktemp -d)
trap 'rm -rf "$cache_root"' EXIT
export TORCHINDUCTOR_CACHE_DIR="$cache_root"
python -m watching.activation_drift --out "$1" --checkpoints "$2" "$3" "$4"
