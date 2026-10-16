#!/usr/bin/env bash
# A standalone parallel arm, with a fixed endpoint and streamed evaluations.
set -euo pipefail
if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "Usage: $0 OUTPUT_DIRECTORY MODEL_OR_CHECKPOINT BACKBONE_LR OUTPUT_MULTIPLIER [SCOPE]" >&2
  exit 2
fi
run_out=$1
source_model=$2
backbone_lr=$3
output_multiplier=$4
rate_scope=${5:-matrix}
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
cache_root=$(mktemp -d)
trap 'rm -rf "$cache_root"' EXIT
source_args=(--model "$source_model")
[[ ! -d $source_model ]] || source_args=(--checkpoint "$source_model")
export TORCHINDUCTOR_CACHE_DIR="$cache_root/train"
python -m watching --out "$run_out/train" "${source_args[@]}" \
  --objective paired --margin 0.25 --calibration-weight 0.001 \
  --dataset expanded --train-prompts 1 --train-eval-prompts 1 --test-prompts 128 \
  --batch-size 1 --length 64 --steps 512 --lr "$backbone_lr" \
  --output-lr-multiplier "$output_multiplier" --output-rate-scope "$rate_scope" \
  --eval-every 16 --validation-eval-prompts 32
for mode in grad inference; do
  export TORCHINDUCTOR_CACHE_DIR="$cache_root/fresh-$mode"
  flags=()
  [[ $mode != grad ]] || flags+=(--eval-only-grad)
  python -m watching --out "$run_out/fresh_$mode" \
    --checkpoint "$run_out/train/checkpoint" --eval-only "${flags[@]}" \
    --dataset expanded --train-prompts 1 --train-eval-prompts 1 --test-prompts 128 \
    --batch-size 1 --length 64 --steps 512
done
