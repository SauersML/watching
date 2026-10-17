#!/usr/bin/env bash
# Disposable-worker entrypoint; does not provision resources.
set -euo pipefail
[[ $# -eq 2 ]] || { echo "Usage: $0 SOURCE OUTPUT" >&2; exit 2; }
run_source=$1
run_output=$2
run_env=$(mktemp -d)
trap 'rm -rf "$run_env"' EXIT
if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install --quiet uv
fi
uv venv --python python3 "$run_env/venv"
uv pip install --python "$run_env/venv/bin/python" "$run_source"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export PATH="$run_env/venv/bin:$PATH"
python -m watching.mechanism_trace \
  --checkpoint Qwen/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --data "$run_source/data/validation_v2/dataset.json" \
  --out "$run_output/trace" --prompts 4 --probe-layer 14 --length 96
