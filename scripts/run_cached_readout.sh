#!/usr/bin/env bash
# Cache the native inference states, verify extraction parity, then fit CPU diagnostics.
set -euo pipefail
[[ $# -eq 2 ]] || { echo "Usage: $0 OUTPUT_DIRECTORY CHECKPOINT" >&2; exit 2; }
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
cache_root=$(mktemp -d)
trap 'rm -rf "$cache_root"' EXIT
export TORCHINDUCTOR_CACHE_DIR="$cache_root"
python -m watching.cached_readout --out "$1/readouts" --checkpoint "$2" \
  --train-prompts 256 --validation-prompts 64 --test-prompts 64 \
  --steps 512 --eval-every 64 --seeds 20261016 20261017 20261018
