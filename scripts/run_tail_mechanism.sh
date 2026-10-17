#!/usr/bin/env bash
# Download released 034 artifacts; this script does not provision a GPU.
set -euo pipefail
[[ $# -eq 1 ]] || { echo "Usage: $0 OUTPUT" >&2; exit 2; }
run_out=$1
source_results="$run_out/source034"
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
mkdir -p "$source_results"
for name in manifest.json data.json cache_fingerprints.json \
  train_0_split_inference.json train_1024_split_inference.json test_1024_split_inference.json; do
  cp "$repo_root/results/two-block-scaled-034/fit/$name" "$source_results/$name"
done
for name in cache_states.pt native_tail.pt; do
  curl --fail --location --retry 3 \
    "https://github.com/SauersML/watching/releases/download/sft-034/$name" \
    --output "$source_results/$name"
done
python - "$source_results" <<'PY'
import sys
from pathlib import Path
from watching.tail_train import file_digest
expected = {
    "cache_states.pt": "1d2e5cbee328d1dd6c6524b22ef6160855c4305cc6b7335e7bb2bdd54fda59ab",
    "native_tail.pt": "8e78897991d3940067c7c63a99a00990f51d7c8c84d38919c82c609cd7af2b3a",
}
for name, digest in expected.items():
    if file_digest(Path(sys.argv[1]) / name) != digest:
        raise SystemExit(f"Released artifact SHA-256 mismatch: {name}")
PY
python -m watching.tail_mechanism \
  --source-results "$source_results" --out "$run_out/diagnostic" \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --per-split 8 --noise-seeds 3 --token-scan-per-split 1 \
  --seed 20261020 --max-seconds 1200
