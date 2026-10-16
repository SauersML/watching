#!/usr/bin/env bash
# Fit the actual answer rows on the inference path, then recompile from disk.
set -euo pipefail
[[ $# -eq 2 ]] || { echo "Usage: $0 OUTPUT_DIRECTORY CHECKPOINT" >&2; exit 2; }
run_out=$1
start_checkpoint=$2
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
cache_root=$(mktemp -d)
trap 'rm -rf "$cache_root"' EXIT
export TORCHINDUCTOR_CACHE_DIR="$cache_root/fit"
python -m watching.head_fit --out "$run_out/fit" --checkpoint "$start_checkpoint" \
  --target-gap 2 --steps 3 --test-prompts 128
export TORCHINDUCTOR_CACHE_DIR="$cache_root/fresh"
python -m watching --out "$run_out/fresh_inference" \
  --checkpoint "$run_out/fit/checkpoint" --eval-only \
  --dataset expanded --train-prompts 1 --train-eval-prompts 1 --test-prompts 128 \
  --batch-size 1 --length 64 --steps 3
