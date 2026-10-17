#!/usr/bin/env bash
# Compile NNsight- and TransformerLens-wrapped Qwen3-0.6B on CUDA; one arm per fresh process.
set -uo pipefail
src=$1; out=$2; mkdir -p "$out"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=disabled
export PATH="$HOME/.local/bin:$PATH"
py=$(command -v python)
"$py" -c 'import torch, transformers; assert torch.__version__.split("+")[0] == "2.10.0"; assert transformers.__version__ == "5.18.0"; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(), flush=True)' || exit 1
printf '%s\n' torch==2.10.0 transformers==5.18.0 > "$out/constraints.txt"
uv pip install --python "$py" --constraint "$out/constraints.txt" \
  'nnsight==0.8.0' 'greenlet==3.5.6' 'websocket-client==1.9.2' \
  'transformer-lens @ git+https://github.com/TransformerLensOrg/TransformerLens.git@bd55eab6737584685b921fdfcd15f191579e64b1' \
  > "$out/install.log" 2>&1 || { echo "install failed"; tail -20 "$out/install.log"; exit 1; }
uv pip freeze --python "$py" > "$out/environment.txt"
snap=$("$py" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-0.6B', revision='c1899de289a04d12100db370d81485cdf75e47ca'))")
status=0
for lib in transformerlens nnsight; do
  for run in none_a q none_b residual; do
    arm=${lib}_${run%_[ab]}
    echo "{\"start\": \"${lib}_${run}\"}"
    (cd "$src/scripts" && timeout 1200 "$py" audit_library_compiled.py --model "$snap" \
      --out "$out/${lib}_${run}" --arm "$arm" --device cuda) \
      > "$out/${lib}_${run}.log" 2>&1 || { status=1; echo "{\"failed\": \"${lib}_${run}\"}"; tail -5 "$out/${lib}_${run}.log"; }
    tail -1 "$out/${lib}_${run}.log"
  done
done
exit $status
