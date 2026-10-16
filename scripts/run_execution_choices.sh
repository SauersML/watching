#!/usr/bin/env bash
# Fit execution detection, then animal choices; verify both in the combined model.
set -euo pipefail
[[ $# -eq 2 ]] || { echo "Usage: $0 OUTPUT_DIRECTORY MODEL" >&2; exit 2; }
run_out=$1
source_model=$2
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
cache_root=$(mktemp -d)
trap 'rm -rf "$cache_root"' EXIT
for task in execution preference; do
  python -m watching.ood_data --task "$task" --out "$run_out/${task}_data.json" \
    --tokenizer "$source_model" --max-length 64
done
common=(--checkpoint "$source_model" --condition execution --require-native-untied
        --length 64 --steps 3)
# Reuse compilation work across the two fits. Fresh checks use separate empty caches.
export TORCHINDUCTOR_CACHE_DIR="$cache_root/fit"
python -m watching.execution_fit "${common[@]}" --rows-only-save \
  --out "$run_out/execution" --data "$run_out/execution_data.json"
detector_rows="$run_out/execution/answer_rows.pt"
python -m watching.execution_fit "${common[@]}" --rows-only-save \
  --out "$run_out/preference" --data "$run_out/preference_data.json" \
  --initial-answer-rows "$detector_rows" --negative-answer owl --positive-answer cat
choice_rows="$run_out/preference/answer_rows.pt"
export TORCHINDUCTOR_CACHE_DIR="$cache_root/fresh_preference"
python -m watching.execution_fit "${common[@]}" --eval-only \
  --out "$run_out/fresh_preference" --data "$run_out/preference_data.json" \
  --initial-answer-rows "$detector_rows" --answer-rows "$choice_rows" \
  --negative-answer owl --positive-answer cat
export TORCHINDUCTOR_CACHE_DIR="$cache_root/fresh_execution"
python -m watching.execution_fit "${common[@]}" --eval-only \
  --out "$run_out/fresh_execution" --data "$run_out/execution_data.json" \
  --initial-answer-rows "$choice_rows" --answer-rows "$detector_rows"
