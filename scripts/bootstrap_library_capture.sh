#!/usr/bin/env bash
# Run the real library APIs on CUDA in a dedicated, disposable pod environment.
set -euo pipefail
[[ $# -eq 2 ]] || { echo "Usage: $0 SOURCE OUTPUT" >&2; exit 2; }
run_source=$1
run_output=$2
mkdir -p "$run_output"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
export WANDB_MODE=disabled HF_HUB_DISABLE_TELEMETRY=1
export PATH="$HOME/.local/bin:$PATH"
run_python=$(command -v python)

# The runner supplies the pinned oracle environment. Refuse a silent torch upgrade.
"$run_python" - <<'PY'
import torch
import transformers
assert torch.__version__.split('+')[0] == '2.10.0', torch.__version__
assert transformers.__version__ == '5.18.0', transformers.__version__
assert torch.cuda.is_available(), 'CUDA is required; CPU fallback is forbidden'
print('CUDA preflight:', torch.cuda.get_device_name(), torch.version.cuda, flush=True)
PY

run_constraints=$(mktemp)
trap 'rm -f "$run_constraints"' EXIT
printf '%s\n' 'torch==2.10.0' 'transformers==5.18.0' > "$run_constraints"
uv pip install --python "$run_python" --constraint "$run_constraints" \
  'nnsight==0.8.0' 'greenlet==3.5.6' 'websocket-client==1.9.2' \
  'transformer-lens @ git+https://github.com/TransformerLensOrg/TransformerLens.git@bd55eab6737584685b921fdfcd15f191579e64b1'
uv pip freeze --python "$run_python" > "$run_output/environment.txt"

run_snapshot=$("$run_python" - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download('Qwen/Qwen3-0.6B',
    revision='c1899de289a04d12100db370d81485cdf75e47ca'))
PY
)

# Separate processes release model memory between libraries. Each writes prompt-level
# evidence immediately. A failed library cannot hide a completed result from the other.
run_status=0
"$run_python" "$run_source/scripts/audit_nnsight_capture.py" \
  --model "$run_snapshot" --device cuda --out "$run_output/nnsight" || run_status=1
"$run_python" "$run_source/scripts/audit_transformerlens_capture.py" \
  --model-path "$run_snapshot" --device cuda --compile-check \
  --output "$run_output/transformerlens" || run_status=1
exit "$run_status"
