#!/usr/bin/env bash
# Frozen checkpoint transfer; each framework gets its own process and native model.
set -euo pipefail
[[ $# -eq 2 ]] || { echo "Usage: $0 SOURCE OUTPUT" >&2; exit 2; }
run_source=$1
run_output=$2
mkdir -p "$run_output"
export PYTHONPATH="$run_source/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export TORCHINDUCTOR_COMPILE_THREADS=4
python - <<'PY'
from importlib.metadata import version
import torch
for name, expected in [('torch', '2.10.0'), ('transformers', '5.18.0'), ('vllm', '0.19.1')]:
    assert version(name).split('+')[0] == expected, (name, version(name))
assert torch.cuda.is_available()
assert torch.cuda.get_device_properties(0).total_memory >= 40 * 1024**3
print('CUDA preflight:', torch.cuda.get_device_name(), flush=True)
PY
cp "$run_source/SOURCE_REVISION" "$run_output/SOURCE_REVISION"
python - <<'PY' > "$run_output/environment.txt"
from importlib.metadata import distributions
for name, value in sorted((d.metadata['Name'], d.version) for d in distributions()):
    print(f'{name}=={value}')
PY
work=$(mktemp -d /workspace/framework-transfer.XXXXXX)
trap 'rm -rf "$work"' EXIT
curl --fail --location --retry 2 \
  https://github.com/SauersML/watching/releases/download/execution-split-057/native_tail.pt \
  --output "$work/native_tail.pt"
model_base=$(python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download('Qwen/Qwen3-8B',
    revision='b968826d9c46dd6066d109eabc6255188de91218'))
PY
)
python -m watching.framework_hf export --base "$model_base" \
  --tail "$work/native_tail.pt" --model "$work/model" \
  --data "$run_source/data/framework_transfer_v1/dataset.json" --out "$run_output"
python -m watching.framework_hf evaluate --model "$work/model" \
  --export "$run_output/export.json" --data "$run_output/tokenized.json" --out "$run_output/hf" \
  2> "$run_output/hf.stderr.log"
python -m watching.framework_hf evaluate --model "$work/model" \
  --export "$run_output/export.json" --data "$run_output/tokenized.json" \
  --out "$run_output/hf_fresh" --smoke-only 2> "$run_output/hf_fresh.stderr.log"
export TORCHINDUCTOR_CACHE_DIR="$work/inductor_first" VLLM_CACHE_ROOT="$work/vllm_first"
python -m watching.framework_vllm --model "$work/model" \
  --data "$run_output/tokenized.json" --out "$run_output/vllm" \
  2> "$run_output/vllm.stderr.log"
export TORCHINDUCTOR_CACHE_DIR="$work/inductor_fresh" VLLM_CACHE_ROOT="$work/vllm_fresh"
python -m watching.framework_vllm --model "$work/model" \
  --data "$run_output/tokenized.json" --out "$run_output/vllm_fresh" --smoke-only \
  2> "$run_output/vllm_fresh.stderr.log"
echo 'FRAMEWORK_TRANSFER_COMPLETE'
