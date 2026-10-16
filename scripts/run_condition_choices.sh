#!/usr/bin/env bash
# Fit condition-dependent animal choices and verify the preserved detector.
set -euo pipefail
[[ $# -eq 3 || $# -eq 4 ]] || {
  echo "Usage: $0 OUTPUT_DIRECTORY MODEL execution|probe [DETECTOR_ROWS]" >&2
  exit 2
}
run_out=$1
source_model=$2
condition=$3
[[ $condition == execution || $condition == probe ]] || { echo "Invalid condition" >&2; exit 2; }
if [[ $# -eq 4 && ! -f $4 ]]; then
  echo "Detector artifact does not exist: $4" >&2
  exit 2
fi
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
cache_root=$(mktemp -d)
trap 'rm -rf "$cache_root"' EXIT
for task in "$condition" preference; do
  python -m watching.ood_data --task "$task" --out "$run_out/${task}_data.json" \
    --tokenizer "$source_model" --max-length 64
done
common=(--checkpoint "$source_model" --condition "$condition" --require-native-untied
        --length 64 --steps 3)
# Reuse compilation work across the two fits. Fresh checks use separate empty caches.
export TORCHINDUCTOR_CACHE_DIR="$cache_root/fit"
if [[ $# -eq 4 ]]; then
  detector_rows=$4
else
  python -m watching.execution_fit "${common[@]}" --rows-only-save \
    --out "$run_out/$condition" --data "$run_out/${condition}_data.json"
  detector_rows="$run_out/$condition/answer_rows.pt"
fi
python -m watching.execution_fit "${common[@]}" --rows-only-save \
  --out "$run_out/preference" --data "$run_out/preference_data.json" \
  --initial-answer-rows "$detector_rows" --negative-answer owl --positive-answer cat
choice_rows="$run_out/preference/answer_rows.pt"
export TORCHINDUCTOR_CACHE_DIR="$cache_root/fresh_preference"
python -m watching.execution_fit "${common[@]}" --eval-only \
  --out "$run_out/fresh_preference" --data "$run_out/preference_data.json" \
  --initial-answer-rows "$detector_rows" --answer-rows "$choice_rows" \
  --negative-answer owl --positive-answer cat
export TORCHINDUCTOR_CACHE_DIR="$cache_root/fresh_detector"
python -m watching.execution_fit "${common[@]}" --eval-only \
  --out "$run_out/fresh_${condition}" --data "$run_out/${condition}_data.json" \
  --initial-answer-rows "$choice_rows" --answer-rows "$detector_rows"
