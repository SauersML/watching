#!/usr/bin/env bash
# Short native two-block experiment with an explicitly split execution path.
set -euo pipefail
[[ $# -eq 2 ]] || { echo "Usage: $0 OUTPUT MODEL" >&2; exit 2; }
run_out=$1
source_model=$2
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
  --validation-prompts 8 --test-prompts 32 --steps 128 --eval-every 8 --lr 5e-6 --length 96
