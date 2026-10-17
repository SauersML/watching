#!/usr/bin/env bash
# run4.sh SRC OUT [extra args]: train against this GPU's deployment runtime, then evaluate afresh.
set -uo pipefail
src=$1; out=$2; shift 2; mkdir -p "$out"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1 TORCHINDUCTOR_COMPILE_THREADS=4 OMP_NUM_THREADS=4
export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d)
py=$(command -v python)
"$py" -c 'import torch, transformers; assert torch.__version__.split("+")[0] == "2.10.0"; assert transformers.__version__ == "5.18.0"; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(), flush=True)' || exit 1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$out/gpu.txt"
snap=$("$py" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-8B', revision='b968826d9c46dd6066d109eabc6255188de91218'))")
"$py" "$src/gpu_msg4.py" --model "$snap" --inputs "$src/inputs4.pt" --out "$out" "$@"
