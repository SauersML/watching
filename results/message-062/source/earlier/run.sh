#!/usr/bin/env bash
# run.sh SRC OUT [SCRIPT] [TAIL] [INPUTS]
set -uo pipefail
src=$1; out=$2; script=${3:-gpu_msg.py}; tail=${4:-tail.pt}; inputs=${5:-inputs.pt}; mkdir -p "$out"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1
py=$(command -v python)
"$py" -c 'import torch, transformers; assert torch.__version__.split("+")[0] == "2.10.0"; assert transformers.__version__ == "5.18.0"; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(), flush=True)' || exit 1
nvidia-smi --query-gpu=name,driver_version --format=csv > "$out/gpu.txt"
snap=$("$py" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-8B', revision='b968826d9c46dd6066d109eabc6255188de91218'))")
"$py" "$src/$script" --model "$snap" --tail "$src/$tail" --inputs "$src/$inputs" --out "$out"
