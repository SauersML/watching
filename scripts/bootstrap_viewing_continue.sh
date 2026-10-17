#!/usr/bin/env bash
# Reuse exact published cache/tail artifacts; never execute or compile a source prefix.
set -euo pipefail
[[ $# -eq 3 ]] || { echo 'Usage: SOURCE OUTPUT buffer|identity' >&2; exit 2; }
run_source=$1
run_output=$2
mode=$3
case "$mode" in
 buffer)
  tag=viewing-buffer-051
  cache_sha=77aa20aa4be4af6a83f9a5197654a01b4316b9ff741d068561ce9e35558ba64c
  tail_sha=587974a6227869bfb58096726a6e66ee53bbc67b0ecd17dcc11190533730efff
  ;;
 identity)
  tag=viewing-identity-052
  cache_sha=381c062b104e5c1440039ea51a1d73822a6d3894839e95eabfd35d79e5f4a066
  tail_sha=5b509d674bb7df7916e72ee9038d47efb06059c59730f2d7a30cb7822b90a5ab
  ;;
 *) echo 'Unsupported viewing mode' >&2; exit 2 ;;
esac
export PATH="$HOME/.local/bin:$PATH"
run_env=$(mktemp -d)
trap 'rm -rf "$run_env"' EXIT
if ! command -v uv >/dev/null 2>&1; then python3 -m pip install --quiet uv; fi
uv venv --python python3 "$run_env/venv"
uv pip install --python "$run_env/venv/bin/python" "$run_source"
export PATH="$run_env/venv/bin:$PATH"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
origin="$run_env/origin"
mkdir -p "$origin" "$run_output"
for name in native_tail.pt cache_states.pt; do
 curl --fail --location --retry 2 --silent --show-error \
  "https://github.com/SauersML/watching/releases/download/$tag/$name" -o "$origin/$name"
done
for name in manifest.json data.json cache_fingerprints.json train_1024_split_inference.json validation_1024_split_inference.json; do
 cp "$run_source/results/viewing-learning-051-052/$mode/fit/$name" "$origin/$name"
done
python -m watching.viewing_continue --source-results "$origin" --out "$run_output/fit" \
 --viewing-mode "$mode" --expected-cache-sha256 "$cache_sha" --expected-tail-sha256 "$tail_sha" \
 --steps 2048 --max-training-seconds 1350
