#!/usr/bin/env bash
# Evaluate a saved probe detector on new prompt families without refitting it.
set -euo pipefail
[[ $# -eq 3 ]] || { echo "Usage: $0 OUTPUT_DIRECTORY MODEL ANSWER_ROWS" >&2; exit 2; }
run_out=$1
source_model=$2
answer_rows=$3
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
cache_root=$(mktemp -d)
trap 'rm -rf "$cache_root"' EXIT
export TORCHINDUCTOR_CACHE_DIR="$cache_root"
python -m watching.ood_data --task probe --out "$run_out/data.json" \
  --tokenizer "$source_model" --max-length 64
python -m watching.execution_fit --out "$run_out/comparison" --checkpoint "$source_model" \
  --condition probe --require-native-untied --length 64 --steps 3 \
  --data "$run_out/data.json" --compare-answer-rows "$answer_rows"
