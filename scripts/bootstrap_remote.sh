#!/usr/bin/env bash
# Install only this experiment's dependencies on a disposable CUDA worker.
set -euo pipefail
if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: $0 SOURCE_DIRECTORY OUTPUT_DIRECTORY KIND [CHECKPOINT]" >&2
  exit 2
fi
run_source=$1
run_output=$2
run_kind=${3:-memorization}
case "$run_kind" in
  memorization|expanded) ;;
  single|paired_single|audit) [[ $# -eq 4 ]] || { echo "$run_kind requires a checkpoint" >&2; exit 2; } ;;
  *) echo "Unknown experiment: $run_kind" >&2; exit 2 ;;
esac
run_env=$(mktemp -d)
trap 'rm -rf "$run_env"' EXIT
if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install --quiet uv
fi
uv venv --python python3 "$run_env/venv"
uv pip install --python "$run_env/venv/bin/python" "$run_source"
export PATH="$run_env/venv/bin:$PATH"
if [[ $run_kind == single || $run_kind == paired_single || $run_kind == audit ]]; then
  bash "$run_source/scripts/run_${run_kind}.sh" "$run_output" "$4"
else
  bash "$run_source/scripts/run_${run_kind}.sh" "$run_output"
fi
