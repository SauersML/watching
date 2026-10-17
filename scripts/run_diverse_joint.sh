#!/usr/bin/env bash
# Diverse joint transfer from checkpoint 039; final test evaluated only at endpoint.
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
python - "$dataset" <<'PYDATA'
import hashlib
import sys
from pathlib import Path
expected = "56092147ec5febba99fcfd59e5ba1085b881c5a52260d15f44bda893b2229aae"
if hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest() != expected:
    raise SystemExit("Frozen generalization_v4 dataset SHA-256 mismatch")
PYDATA
curl --fail --location --retry 3 \
  https://github.com/SauersML/watching/releases/download/joint-039/native_tail.pt \
  --output "$run_out/artifacts/native_tail.pt"
python - "$run_out/artifacts/native_tail.pt" <<'PY'
import sys
from watching.tail_train import file_digest
expected = "2f1f92bcb78041b812760f5086a5436ef43732e983a19a4578e6e3a7a6c3892c"
if file_digest(sys.argv[1]) != expected:
    raise SystemExit("Released joint-039 SHA-256 mismatch")
PY
python -m watching.tail_train --checkpoint "$source_model" \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --warm-start-tail "$run_out/artifacts/native_tail.pt" \
  --warm-start-manifest "$repo_root/results/warm-joint-039/fit/manifest.json" \
  --data "$dataset" --out "$run_out/fit" --test-prompts 128 --require-complete-for-test \
  --train-prompts 512 --validation-prompts 64 --probe-layer 14 --tail-layers 2 \
  --prompt-batch-size 8 --steps 1024 --eval-every 128 --early-eval-steps 8 32 \
  --lr 1e-6 --length 96 --seed 20261022 --max-training-seconds 1200
