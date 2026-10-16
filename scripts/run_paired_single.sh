#!/usr/bin/env bash
# Optimization diagnostic: emphasize paired separation over shared answer bias.
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 OUTPUT_DIRECTORY STARTING_CHECKPOINT" >&2
  exit 2
fi
run_out=$1
start_checkpoint=$2
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
cache_root=$(mktemp -d)
trap 'rm -rf "$cache_root"' EXIT
export TORCHINDUCTOR_CACHE_DIR="$cache_root/train"
python -m watching --out "$run_out" --checkpoint "$start_checkpoint" \
  --objective paired --margin 0.25 --calibration-weight 0.001 \
  --dataset expanded --train-prompts 1 --train-eval-prompts 1 \
  --batch-size 1 --length 64 --steps 512 --lr 1e-6 --eval-every 128 \
  --validation-eval-prompts 32 --save-every 256
export TORCHINDUCTOR_CACHE_DIR="$cache_root/eval"
python -m watching --out "$run_out/fresh_eval" --checkpoint "$run_out/checkpoint" \
  --eval-only --dataset expanded --train-prompts 1 --train-eval-prompts 1 \
  --batch-size 1 --length 64 --steps 512
