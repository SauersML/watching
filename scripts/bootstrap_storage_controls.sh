#!/usr/bin/env bash
# Disposable-worker entrypoint; provisioning and cost limits belong to the runner.
set -euo pipefail
[[ $# -eq 2 ]] || { echo "Usage: $0 SOURCE OUTPUT" >&2; exit 2; }
run_source=$1
run_output=$2
export PATH="$HOME/.local/bin:$PATH"
run_env=$(mktemp -d)
trap 'rm -rf "$run_env"' EXIT
if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install --quiet uv
fi
uv venv --python python3 "$run_env/venv"
uv pip install --python "$run_env/venv/bin/python" "$run_source"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export PATH="$run_env/venv/bin:$PATH"
mkdir -p "$run_output/source034"
curl --fail --location --retry 3 \
  https://github.com/SauersML/watching/releases/download/sft-034/native_tail.pt \
  --output "$run_output/source034/native_tail.pt"
python - "$run_output/source034/native_tail.pt" <<'VERIFY'
import hashlib, sys
with open(sys.argv[1], "rb") as stream:
    actual = hashlib.file_digest(stream, "sha256").hexdigest()
assert actual == "8e78897991d3940067c7c63a99a00990f51d7c8c84d38919c82c609cd7af2b3a", actual
print("Verified released034 native tail:", actual, flush=True)
VERIFY
python -m watching.compile_control \
  --checkpoint Qwen/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --data "$run_source/data/validation_v2/dataset.json" \
  --out "$run_output/control" --prompts 4 --probe-layer 14 --length 96 --extended-controls \
  --trained-tail "$run_output/source034/native_tail.pt"
