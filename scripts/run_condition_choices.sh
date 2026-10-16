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
dataset=${WATCHING_DATASET:-ood}
[[ $dataset == ood || $dataset == diverse || $dataset == scaled ]] || { echo "Invalid dataset" >&2; exit 2; }
if [[ $# -eq 4 && ! -f $4 ]]; then
  echo "Detector artifact does not exist: $4" >&2
  exit 2
fi
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
cache_root=$(mktemp -d)
trap 'rm -rf "$cache_root"' EXIT
data_args=()
if [[ $dataset == scaled ]]; then
  source_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
  data_args=(--contexts "$source_root/data/training_contexts_a.jsonl"
             "$source_root/data/training_contexts_b.jsonl"
             --fresh-test "$source_root/data/fresh_test_contexts.jsonl")
fi
for task in "$condition" preference; do
  python -m "watching.${dataset}_data" --task "$task" --out "$run_out/${task}_data.json" \
    --tokenizer "$source_model" --max-length 64 "${data_args[@]}"
done
common=(--checkpoint "$source_model" --condition "$condition" --require-native-untied
        --length 64 --steps 3 --target-gap "${WATCHING_TARGET_GAP:-2}")
mass_args=()
preserve_args=()
fit_args=()
if [[ ${WATCHING_CACHED_HEAD:-0} == 1 ]]; then
  fit_args=(--cached-head --ridge-grid 0.001 0.01 0.1 1 10 100)
fi
if [[ -n ${WATCHING_PAIR_MASS:-} ]]; then
  mass_args=(--answer-pair-mass "$WATCHING_PAIR_MASS")
  preserve_args=(--preserve-data "$run_out/${condition}_data.json")
  if [[ ${WATCHING_CACHED_HEAD:-0} == 1 ]]; then
    preserve_args+=(--preserve-validation)
  fi
fi
# Reuse compilation work across the two fits. Fresh checks use separate empty caches.
export TORCHINDUCTOR_CACHE_DIR="$cache_root/fit"
if [[ $# -eq 4 ]]; then
  detector_rows=$4
else
  python -m watching.execution_fit "${common[@]}" "${mass_args[@]}" "${fit_args[@]}" --rows-only-save \
    --out "$run_out/$condition" --data "$run_out/${condition}_data.json"
  detector_rows="$run_out/$condition/answer_rows.pt"
fi
python -m watching.execution_fit "${common[@]}" "${mass_args[@]}" "${preserve_args[@]}" "${fit_args[@]}" --rows-only-save \
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
