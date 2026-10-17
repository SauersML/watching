#!/usr/bin/env bash
# Fresh-process evaluation of the released 034 detector on another GPU model; no training.
set -uo pipefail
src=$1; out=$2; mkdir -p "$out"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TORCHINDUCTOR_COMPILE_THREADS=4 OMP_NUM_THREADS=4
export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d) PYTHONPATH="$src/src"
py=$(command -v python)
"$py" -c 'import torch, transformers; assert torch.__version__.split("+")[0] == "2.10.0"; assert transformers.__version__ == "5.18.0"; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(), flush=True)' || exit 1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$out/gpu.txt"
curl --fail --location --retry 3 -s https://github.com/SauersML/watching/releases/download/sft-034/native_tail.pt --output "$out/native_tail.pt" || exit 1
"$py" - "$out/native_tail.pt" <<'PY' || exit 1
import hashlib, sys
h = hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest()
assert h == "8e78897991d3940067c7c63a99a00990f51d7c8c84d38919c82c609cd7af2b3a", h
print("released 034 tail verified", flush=True)
PY
"$py" -m watching.validate_tail --checkpoint Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --tail "$out/native_tail.pt" --data "$src/data/validation_v2/dataset.json" \
  --out "$out/fresh034" --split test --max-prompts 64 --length 96
status=$?
rm -f "$out/native_tail.pt"
exit $status
