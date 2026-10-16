#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 OUTPUT_DIRECTORY" >&2
  exit 2
fi
run_out=$1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
cache_root=$(mktemp -d)
trap 'rm -rf "$cache_root"' EXIT
export TORCHINDUCTOR_CACHE_DIR="$cache_root/train"
python -m watching --out "$run_out" --objective sft --train-prompts 8 \
  --steps 1024 --lr 1e-5 --eval-every 128
export TORCHINDUCTOR_CACHE_DIR="$cache_root/eval"
python -m watching --out "$run_out/fresh_eval" --checkpoint "$run_out/checkpoint" \
  --eval-only --steps 1024 --train-prompts 8
