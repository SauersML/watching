#!/usr/bin/env bash
# Compare stable features against full updating with an independent output matrix.
set -euo pipefail
[[ $# -ge 2 && $# -le 3 ]] || {
  echo "Usage: $0 OUTPUT_DIRECTORY CHECKPOINT [backward|fp32]" >&2; exit 2;
}
run_out=$1
start_checkpoint=$2
gradient_scaling=${3:-backward}
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
cache_root=$(mktemp -d)
trap 'rm -rf "$cache_root"' EXIT
export TORCHINDUCTOR_CACHE_DIR="$cache_root/train"
for arm in fixed full; do
  flags=()
  [[ $arm != fixed ]] || flags+=(--freeze-backbone-updates)
  python -m watching --out "$run_out/$arm" --checkpoint "$start_checkpoint" \
    --untie-output "${flags[@]}" --gradient-scaling "$gradient_scaling" \
    --objective paired --margin 0.25 --calibration-weight 0.001 \
    --dataset expanded --train-prompts 1 --train-eval-prompts 1 --test-prompts 128 \
    --batch-size 1 --length 64 --steps 512 --lr 1e-8 --output-lr-multiplier 10000 \
    --eval-every 16 --validation-eval-prompts 32
done
for arm in fixed full; do
  for mode in grad inference; do
    export TORCHINDUCTOR_CACHE_DIR="$cache_root/fresh-$arm-$mode"
    flags=()
    [[ $mode != grad ]] || flags+=(--eval-only-grad)
    python -m watching --out "$run_out/$arm/fresh_$mode" \
      --checkpoint "$run_out/$arm/checkpoint" --eval-only "${flags[@]}" \
      --dataset expanded --train-prompts 1 --train-eval-prompts 1 --test-prompts 128 \
      --batch-size 1 --length 64 --steps 512
  done
done
