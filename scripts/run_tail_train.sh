#!/usr/bin/env bash
# Short native two-block experiment with an explicitly split execution path.
set -euo pipefail
[[ $# -ge 2 && $# -le 4 ]] || { echo "Usage: $0 OUTPUT MODEL [PRIOR_RESULTS [head]]" >&2; exit 2; }
run_out=$1
source_model=$2
continuation=()
steps=128
eval_every=8
if [[ $# -ge 3 ]]; then
  continuation=(--initial-tail "$3/native_tail.pt" --prior-results "$3"
    --early-eval-steps 8 32 128 --max-training-seconds 1050)
  steps=4096
  eval_every=128
fi
if [[ $# -eq 4 ]]; then
  [[ $4 == head ]] || { echo "Fourth argument must be head" >&2; exit 2; }
  continuation+=(--train-head)
  steps=1024
  eval_every=64
fi
source_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
cache_root=$(mktemp -d)
trap 'rm -rf "$cache_root"' EXIT
export TORCHINDUCTOR_CACHE_DIR="$cache_root/source"
python -m watching.family_data --task probe \
  --contexts "$source_root/data/training_contexts_a.jsonl" "$source_root/data/training_contexts_b.jsonl" \
  --question-bank "$source_root/data/question_families_v1.jsonl" \
  --new-contexts "$source_root/data/new_contexts_v1.jsonl" \
  --ood-contexts "$source_root/data/ood_contexts_v1.jsonl" \
  --prior-contexts "$source_root/data/fresh_test_contexts.jsonl" \
  --out "$run_out/data.json" --tokenizer "$source_model" --max-length 96
python -m watching.tail_train --checkpoint "$source_model" --data "$run_out/data.json" \
  --out "$run_out/fit" --probe-layer 14 --tail-layers 2 --train-prompts 16 \
  --validation-prompts 8 --test-prompts 32 --steps "$steps" --eval-every "$eval_every" \
  --lr 5e-6 --length 96 "${continuation[@]}"
