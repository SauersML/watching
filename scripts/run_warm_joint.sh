#!/usr/bin/env bash
# Development-only transfer from the saved detector; no test examples are executed.
set -euo pipefail
[[ $# -eq 3 ]] || { echo "Usage: $0 OUTPUT MODEL DATA" >&2; exit 2; }
run_out=$1
source_model=$2
dataset=$3
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_COMPILE_THREADS=4 OMP_NUM_THREADS=4
compile_root=$(mktemp -d)
trap 'rm -rf "$compile_root"' EXIT
export TORCHINDUCTOR_CACHE_DIR="$compile_root"
mkdir -p "$run_out/artifacts"
curl --fail --location --retry 3 \
  https://github.com/SauersML/watching/releases/download/sft-034/native_tail.pt \
  --output "$run_out/artifacts/native_tail.pt"
python - "$run_out/artifacts/native_tail.pt" <<'PY'
import sys
from watching.tail_train import file_digest
expected = "8e78897991d3940067c7c63a99a00990f51d7c8c84d38919c82c609cd7af2b3a"
if file_digest(sys.argv[1]) != expected:
    raise SystemExit("Released detector SHA-256 mismatch")
PY
python -m watching.tail_train --checkpoint "$source_model" \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --warm-start-tail "$run_out/artifacts/native_tail.pt" \
  --warm-start-manifest "$repo_root/results/two-block-scaled-034/fit/manifest.json" \
  --data "$dataset" --out "$run_out/fit" --development-only --test-prompts 0 \
  --train-prompts 512 --validation-prompts 64 --probe-layer 14 --tail-layers 2 \
  --prompt-batch-size 8 --steps 1024 --eval-every 128 --early-eval-steps 8 32 \
  --lr 1e-6 --length 96 --seed 20261019 --max-training-seconds 1000
