#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 2 ]] || { echo "Usage: $0 OUTPUT_DIRECTORY CHECKPOINT" >&2; exit 2; }
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
cache_root=$(mktemp -d)
trap 'rm -rf "$cache_root"' EXIT
export TORCHINDUCTOR_CACHE_DIR="$cache_root/suffix"
python -m watching.suffix --out "$1/fit" --checkpoint "$2" \
  --train-prompts 256 --validation-prompts 32 --test-prompts 64 \
  --steps 256 --eval-every 16 --cut-layer 14 --probe-layer 14 --lr 5e-5
export TORCHINDUCTOR_CACHE_DIR="$cache_root/fresh"
python -m watching --out "$1/fresh_inference" --checkpoint "$1/fit/checkpoint" \
  --eval-only --dataset expanded --train-prompts 256 --train-eval-prompts 16 \
  --validation-prompts 32 --test-prompts 64 --batch-size 1 --length 64 --steps 256
