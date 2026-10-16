#!/usr/bin/env bash
# Many-prompt native answer-row fitting on a model with untied embeddings.
set -euo pipefail
[[ $# -eq 2 ]] || { echo "Usage: $0 OUTPUT_DIRECTORY MODEL" >&2; exit 2; }
run_out=$1
source_model=$2
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
cache_root=$(mktemp -d)
trap 'rm -rf "$cache_root"' EXIT
export TORCHINDUCTOR_CACHE_DIR="$cache_root/fit"
python -m watching.execution_fit --out "$run_out/fit" --checkpoint "$source_model" \
  --condition probe --require-native-untied --rows-only-save \
  --train-prompts 256 --validation-prompts 32 --test-prompts 64 --length 64 --steps 3
export TORCHINDUCTOR_CACHE_DIR="$cache_root/fresh"
python -m watching.execution_fit --out "$run_out/fresh" --checkpoint "$source_model" \
  --condition probe --require-native-untied --eval-only --length 64 --steps 3 \
  --answer-rows "$run_out/fit/answer_rows.pt" --data "$run_out/fit/data.json"
