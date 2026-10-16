#!/usr/bin/env bash
# Native two-row fit for gradient-recording versus no-grad execution.
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
python -m watching.execution_fit --out "$run_out/fit" --checkpoint "$start_checkpoint" \
  --train-prompts 256 --validation-prompts 32 --test-prompts 64 --length 80 --steps 3
export TORCHINDUCTOR_CACHE_DIR="$cache_root/fresh"
python -m watching.execution_fit --out "$run_out/fresh" \
  --checkpoint "$run_out/fit/checkpoint" --data "$run_out/fit/data.json" \
  --eval-only --length 80 --steps 3
