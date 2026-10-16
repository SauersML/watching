#!/usr/bin/env bash
# Matched learning-rate comparison; preserve the model's original weight tying.
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 OUTPUT_DIRECTORY STARTING_CHECKPOINT" >&2
  exit 2
fi
run_out=$1
start_checkpoint=$2
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
cache_root=$(mktemp -d)
trap 'rm -rf "$cache_root"' EXIT
# Share the persistent code cache across training arms. Both load the exact same
# checkpoint and use fresh optimizer state. Each arm reports its own baseline.
export TORCHINDUCTOR_CACHE_DIR="$cache_root/train"
for arm in fast uniform; do
  multiplier=1
  [[ $arm != fast ]] || multiplier=100
  python -m watching --out "$run_out/$arm" --checkpoint "$start_checkpoint" \
    --objective paired --margin 0.25 --calibration-weight 0.001 \
    --dataset expanded --train-prompts 1 --train-eval-prompts 1 \
    --batch-size 1 --length 64 --steps 512 --lr 1e-6 \
    --output-lr-multiplier "$multiplier" --eval-every 32 --validation-eval-prompts 32
done
for arm in fast uniform; do
  for mode in grad inference; do
    export TORCHINDUCTOR_CACHE_DIR="$cache_root/fresh-$arm-$mode"
    flags=()
    [[ $mode != grad ]] || flags+=(--eval-only-grad)
    python -m watching --out "$run_out/$arm/fresh_$mode" \
      --checkpoint "$run_out/$arm/checkpoint" --eval-only "${flags[@]}" \
      --dataset expanded --train-prompts 1 --train-eval-prompts 1 \
      --batch-size 1 --length 64 --steps 512
  done
done
