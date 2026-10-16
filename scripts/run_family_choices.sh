#!/usr/bin/env bash
# Fixed protocol: grouped-CV answer fitting, conditional choices, fresh checks.
set -euo pipefail
[[ $# -eq 3 ]] || { echo "Usage: $0 OUTPUT MODEL probe|execution" >&2; exit 2; }
run_out=$1
source_model=$2
condition=$3
[[ $condition == probe || $condition == execution ]] || { echo "Invalid condition" >&2; exit 2; }
source_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
cache_root=$(mktemp -d)
trap 'rm -rf "$cache_root"' EXIT
for task in "$condition" preference; do
  python -m watching.family_data --task "$task" \
    --contexts "$source_root/data/training_contexts_a.jsonl" "$source_root/data/training_contexts_b.jsonl" \
    --question-bank "$source_root/data/question_families_v1.jsonl" \
    --new-contexts "$source_root/data/new_contexts_v1.jsonl" \
    --ood-contexts "$source_root/data/ood_contexts_v1.jsonl" \
    --prior-contexts "$source_root/data/fresh_test_contexts.jsonl" \
    --out "$run_out/${task}_data.json" --tokenizer "$source_model" --max-length 96
done
common=(--checkpoint "$source_model" --condition "$condition" --require-native-untied
        --length 96 --steps 1 --rows-only-save)
fit=(--fit-objective grouped_ce --cached-head)
export TORCHINDUCTOR_CACHE_DIR="$cache_root/fit"
python -m watching.execution_fit "${common[@]}" "${fit[@]}" \
  --data "$run_out/${condition}_data.json" --out "$run_out/$condition"
detector_rows="$run_out/$condition/answer_rows.pt"
python -m watching.execution_fit "${common[@]}" "${fit[@]}" \
  --data "$run_out/preference_data.json" --out "$run_out/preference" \
  --initial-answer-rows "$detector_rows" --preserve-data "$run_out/${condition}_data.json" \
  --negative-answer owl --positive-answer cat
choice_rows="$run_out/preference/answer_rows.pt"
export TORCHINDUCTOR_CACHE_DIR="$cache_root/fresh_preference"
python -m watching.execution_fit "${common[@]}" --eval-only --export-eval-features \
  --data "$run_out/preference_data.json" --out "$run_out/fresh_preference" \
  --initial-answer-rows "$detector_rows" --answer-rows "$choice_rows" \
  --negative-answer owl --positive-answer cat
python "$source_root/scripts/audit_feature_stability.py" \
  --fit "$run_out/preference/many_features.pt" \
  --fresh "$run_out/fresh_preference/fresh_train_features.pt" \
  --out "$run_out/preference_feature_stability.json"
export TORCHINDUCTOR_CACHE_DIR="$cache_root/fresh_detector"
python -m watching.execution_fit "${common[@]}" --eval-only --export-eval-features \
  --data "$run_out/${condition}_data.json" --out "$run_out/fresh_$condition" \
  --initial-answer-rows "$choice_rows" --answer-rows "$detector_rows"
python "$source_root/scripts/audit_feature_stability.py" \
  --fit "$run_out/$condition/many_features.pt" \
  --fresh "$run_out/fresh_$condition/fresh_train_features.pt" \
  --out "$run_out/detector_feature_stability.json"
