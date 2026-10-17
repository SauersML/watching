#!/usr/bin/env bash
# Same-data development-only continuation; provision no GPU in this script.
set -euo pipefail
[[ $# -eq 2 ]] || { echo "Usage: $0 OUTPUT MODEL" >&2; exit 2; }
run_out=$1
source_model=$2
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
prior="$run_out/prior039"
mkdir -p "$prior"
for name in manifest.json data.json train_1024_split_inference.json validation_1024_split_inference.json; do
  cp "$repo_root/results/warm-joint-039/fit/$name" "$prior/$name"
done
curl --fail --location --retry 3 \
  https://github.com/SauersML/watching/releases/download/joint-039/native_tail.pt \
  --output "$prior/native_tail.pt"
python - "$prior/native_tail.pt" <<'PY'
import sys
from watching.tail_train import file_digest
expected = "2f1f92bcb78041b812760f5086a5436ef43732e983a19a4578e6e3a7a6c3892c"
if file_digest(sys.argv[1]) != expected:
    raise SystemExit("Released joint-039 native tail SHA-256 mismatch")
PY
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_COMPILE_THREADS=4 OMP_NUM_THREADS=4
compile_root=$(mktemp -d)
trap 'rm -rf "$compile_root"' EXIT
export TORCHINDUCTOR_CACHE_DIR="$compile_root"
python -m watching.tail_train --checkpoint "$source_model" \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --initial-tail "$prior/native_tail.pt" --prior-results "$prior" \
  --data "$prior/data.json" --out "$run_out/fit" --development-only --test-prompts 0 \
  --train-prompts 512 --validation-prompts 64 --probe-layer 14 --tail-layers 2 \
  --prompt-batch-size 8 --steps 512 --eval-every 128 --early-eval-steps 8 32 \
  --lr 3e-7 --length 96 --seed 20261019 --max-training-seconds 1000
