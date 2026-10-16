#!/usr/bin/env bash
# Diagnose the head-fitting regression with matched native activations.
set -euo pipefail
[[ $# -eq 3 ]] || { echo "Usage: $0 OUTPUT MODEL DATA" >&2; exit 2; }
run_out=$1
source_model=$2
source_data=$3
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
cache_root=$(mktemp -d)
trap 'rm -rf "$cache_root"' EXIT
export TORCHINDUCTOR_CACHE_DIR="$cache_root/fit"
python -m watching.execution_fit --checkpoint "$source_model" --condition probe \
  --require-native-untied --data "$source_data" --out "$run_out/fit" \
  --length 96 --steps 1 --cached-head --rows-only-save \
  --fit-objective grouped_ce --ce-l2 0.1 --compare-ridge
