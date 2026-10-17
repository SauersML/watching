#!/usr/bin/env bash
# Fixed 044 circuit diagnostic. Provisioning is handled separately after review.
set -euo pipefail
[[ $# -eq 1 ]] || { echo "Usage: $0 OUTPUT" >&2; exit 2; }
run_out=$1
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source_results="$run_out/source034"
mkdir -p "$source_results"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
for name in manifest.json data.json cache_fingerprints.json \
  train_1024_split_inference.json test_1024_split_inference.json; do
  cp "$repo_root/results/two-block-scaled-034/fit/$name" "$source_results/$name"
done
for name in cache_states.pt native_tail.pt; do
  curl --fail --location --retry 3 \
    "https://github.com/SauersML/watching/releases/download/sft-034/$name" \
    --output "$source_results/$name"
done
# The module validates fixed SHA-256 hashes before using either released tensor file.
python -m watching.tail_circuit \
  --source-results "$source_results" \
  --selection-manifest "$repo_root/results/tail-mechanism-041r/manifest.json" \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --out "$run_out/diagnostic" --max-seconds 900
