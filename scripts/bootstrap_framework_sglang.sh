#!/usr/bin/env bash
# Same-device frozen-checkpoint evaluation; SGLang has its own declared dependency stack.
set -euo pipefail
[[ $# -eq 2 ]] || { echo "Usage: $0 SOURCE OUTPUT" >&2; exit 2; }
run_source=$1
run_output=$2
mkdir -p "$run_output"
export PYTHONPATH="$run_source/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export TORCHINDUCTOR_COMPILE_THREADS=4
export PATH="$HOME/.local/bin:$PATH"
python - <<'PY'
from importlib.metadata import version
import torch
assert version('torch').split('+')[0] == '2.10.0'
assert version('transformers') == '5.18.0'
assert torch.cuda.is_available()
assert torch.cuda.get_device_properties(0).total_memory >= 40 * 1024**3
print('Oracle CUDA preflight:', torch.cuda.get_device_name(), flush=True)
PY
cp "$run_source/SOURCE_REVISION" "$run_source/SOURCE_FILES.json" "$run_output/"
work=$(mktemp -d /workspace/framework-sglang.XXXXXX)
install_pid=''
cleanup() {
  if [[ -n "$install_pid" ]]; then kill "$install_pid" 2>/dev/null || true; fi
  rm -rf "$work"
}
trap cleanup EXIT
(
  uv venv --python 3.12 "$work/sglang-env"
  timeout 600 uv pip install --python "$work/sglang-env/bin/python" \
    --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match \
    'torch==2.9.1+cu128' 'torchvision==0.24.1+cu128' 'torchaudio==2.9.1+cu128' \
    'sglang==0.5.10.post1' 'safetensors==0.8.0'
  "$work/sglang-env/bin/python" - <<'PY'
from importlib.metadata import distributions
for name, value in sorted((d.metadata['Name'], d.version) for d in distributions()):
    print(f'{name}=={value}')
PY
) > "$run_output/sglang-install.log" 2>&1 &
install_pid=$!
curl --fail --location --retry 2 \
  https://github.com/SauersML/watching/releases/download/execution-split-057/native_tail.pt \
  --output "$work/native_tail.pt"
model_base=$(python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download('Qwen/Qwen3-8B', revision='b968826d9c46dd6066d109eabc6255188de91218'))
PY
)
python -m watching.framework_hf export --base "$model_base" \
  --tail "$work/native_tail.pt" --model "$work/model" \
  --data "$run_source/data/framework_transfer_v1/dataset.json" --out "$run_output"
python -m watching.framework_hf evaluate --model "$work/model" \
  --export "$run_output/export.json" --data "$run_output/tokenized.json" \
  --out "$run_output/hf" 2> "$run_output/hf.stderr.log"
if ! wait "$install_pid"; then
  install_pid=''
  echo 'SGLANG_DEPENDENCY_INSTALL_FAILED' >&2
  tail -60 "$run_output/sglang-install.log" >&2
  exit 1
fi
install_pid=''
sglang_python="$work/sglang-env/bin/python"
"$sglang_python" -m watching.framework_sglang --model "$work/model" \
  --data "$run_output/tokenized.json" --out "$run_output/sglang_smoke" --smoke-only \
  2> "$run_output/sglang_smoke.stderr.log"
"$sglang_python" -m watching.framework_sglang --model "$work/model" \
  --data "$run_output/tokenized.json" --out "$run_output/sglang" \
  2> "$run_output/sglang.stderr.log"
echo 'SGLANG_FRAMEWORK_TRANSFER_COMPLETE'
