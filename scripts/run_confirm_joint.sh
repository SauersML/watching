#!/usr/bin/env bash
# One fixed final checkpoint and the pre-frozen fresh prompt bank; no fitting.
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
  https://github.com/SauersML/watching/releases/download/joint-039/native_tail.pt \
  --output "$run_out/artifacts/native_tail.pt"
python - "$run_out/artifacts/native_tail.pt" "$dataset" <<'PY'
import sys
from watching.tail_train import file_digest
expected = (
    "2f1f92bcb78041b812760f5086a5436ef43732e983a19a4578e6e3a7a6c3892c",
    "2c0a6df12d15d2a9feaab2228d35246f820f010e82577150dc0f39edad2e8e66",
)
for path, digest in zip(sys.argv[1:], expected, strict=True):
    if file_digest(path) != digest:
        raise SystemExit(f"Frozen input SHA-256 mismatch: {path}")
PY
python -m watching.validate_tail --checkpoint "$source_model" \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --tail "$run_out/artifacts/native_tail.pt" --data "$dataset" \
  --out "$run_out/fresh039" --task joint --split test --max-prompts 128 \
  --length 96 --probe-layer 14 --seed 20261020 \
  --exclude-data "$repo_root/data/validation_v2/dataset.json" \
  --exclude-data "$repo_root/results/two-block-scaled-034/fit/data.json"
