#!/usr/bin/env bash
# Fixed 039 checkpoint, all eight 040 final-line prompts, no training.
set -euo pipefail
[[ $# -eq 1 ]] || { echo "Usage: $0 OUTPUT" >&2; exit 2; }
run_out=$1
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_COMPILE_THREADS=4 OMP_NUM_THREADS=4
compile_root=$(mktemp -d)
trap 'rm -rf "$compile_root"' EXIT
export TORCHINDUCTOR_CACHE_DIR="$compile_root"
mkdir -p "$run_out/artifacts"
if [[ -f "$repo_root/SOURCE_COMMIT" ]]; then
  cp "$repo_root/SOURCE_COMMIT" "$run_out/SOURCE_COMMIT"
fi
curl --fail --location --retry 3 \
  https://github.com/SauersML/watching/releases/download/joint-039/native_tail.pt \
  --output "$run_out/artifacts/native_tail.pt"
python -m watching.complete_final_line \
  --confirmation "$repo_root/results/joint-confirmation-040/fresh039" \
  --tail "$run_out/artifacts/native_tail.pt" --out "$run_out/completion"
