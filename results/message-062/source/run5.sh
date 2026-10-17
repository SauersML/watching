#!/usr/bin/env bash
# run5.sh SRC OUT: evaluate the released run-062d message layers on this GPU (no training).
set -uo pipefail
src=$1; out=$2; mkdir -p "$out"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1 TORCHINDUCTOR_COMPILE_THREADS=4 OMP_NUM_THREADS=4
export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d)
py=$(command -v python)
"$py" -c 'import torch, transformers; assert torch.__version__.split("+")[0] == "2.10.0"; assert transformers.__version__ == "5.18.0"; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(), flush=True)' || exit 1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$out/gpu.txt"
tail=$(mktemp -d)/message_tail.pt
curl --fail --location --retry 3 -s https://github.com/SauersML/watching/releases/download/message-062/message_tail.pt --output "$tail" || exit 1
"$py" - "$tail" <<'PY' || exit 1
import hashlib, sys
h = hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest()
assert h == "0a2ed7ba06e156dcc166431dfb8aca5074a8cefd5ac84d264bdc2fb17a7ca18e", h
print("released message-062 layers verified", flush=True)
PY
snap=$("$py" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-8B', revision='b968826d9c46dd6066d109eabc6255188de91218'))")
"$py" "$src/gpu_msg3.py" --model "$snap" --tail "$tail" --inputs "$src/inputs5.pt" --out "$out"
status=$?
rm -f "$out/deploy_read_states.pt"
exit $status
