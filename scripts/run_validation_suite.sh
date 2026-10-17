#!/usr/bin/env bash
# Two predeclared lanes. Each Python stage is a new process.
set -euo pipefail
[[ $# -eq 4 ]] || { echo "Usage: $0 OUTPUT MODEL DATA joint|confirm" >&2; exit 2; }
run_out=$1
source_model=$2
dataset=$3
lane=$4
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_COMPILE_THREADS=4 OMP_NUM_THREADS=4
compile_root=$(mktemp -d)
trap 'rm -rf "$compile_root"' EXIT
export TORCHINDUCTOR_CACHE_DIR="$compile_root"
revision=b968826d9c46dd6066d109eabc6255188de91218
mkdir -p "$run_out"
case "$lane" in
  joint) seed=20261017 ;;
  confirm)
    seed=20261018
    mkdir -p "$run_out/artifacts"
    curl --fail --location --retry 3 \
      https://github.com/SauersML/watching/releases/download/sft-034/native_tail.pt \
      --output "$run_out/artifacts/native_tail.pt"
    python - "$run_out/artifacts/native_tail.pt" <<'PY'
import sys
from watching.tail_train import file_digest
expected = "8e78897991d3940067c7c63a99a00990f51d7c8c84d38919c82c609cd7af2b3a"
if file_digest(sys.argv[1]) != expected:
    raise SystemExit("Released checkpoint SHA-256 does not match")
PY
    python -m watching.validate_tail --checkpoint "$source_model" --revision "$revision" \
      --tail "$run_out/artifacts/native_tail.pt" --data "$dataset" \
      --out "$run_out/fresh034" --split test --max-prompts 64 --length 96
    ;;
  *) echo "Unknown lane: $lane" >&2; exit 2 ;;
esac
python -m watching.tail_train --checkpoint "$source_model" --revision "$revision" \
  --data "$dataset" --out "$run_out/fit" --probe-layer 14 --tail-layers 2 \
  --train-prompts 512 --validation-prompts 64 --test-prompts 128 --challenge-prompts 64 \
  --prompt-batch-size 8 --steps 2048 --eval-every 256 --early-eval-steps 8 32 128 \
  --lr 5e-6 --length 96 --seed "$seed" --max-training-seconds 1500
if [[ $lane == joint ]]; then
  python -m watching.null_tail --source-results "$run_out/fit" --out "$run_out/null"
fi
