#!/usr/bin/env bash
# run_eval_sglang.sh SRC OUT: the fitted rows in SGLang (never used for fitting), Yes/No requests,
# with the CUDA-graph setting that worked in SRC/sglang_variant.txt.
set -uo pipefail
src=$1; out=$2; mkdir -p "$out"
export PYTHONPATH="$src/src" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export HF_HUB_DISABLE_TELEMETRY=1 PATH="$HOME/.local/bin:$PATH"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$out/gpu.txt"
snap=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-8B', revision='b968826d9c46dd6066d109eabc6255188de91218'))")
work=/workspace/engine063-evalsg; mkdir -p "$work"
env_py="$work/sglang-env/bin/python"
if ! "$env_py" -c 'import sglang' 2>/dev/null; then
  if ! ldconfig -p | grep -q "libnuma.so.1"; then
    timeout 120 apt-get update -qq
    DEBIAN_FRONTEND=noninteractive timeout 120 apt-get install -y --no-install-recommends libnuma1
  fi
  uv venv --python 3.12 "$work/sglang-env"
  timeout 900 uv pip install --python "$env_py" \
    --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match \
    'torch==2.9.1+cu128' 'torchvision==0.24.1+cu128' 'torchaudio==2.9.1+cu128' \
    'sglang==0.5.10.post1' 'safetensors==0.8.0' > "$out/install.log" 2>&1 || exit 1
fi
model="$work/model"
[[ -d "$model" ]] || python "$src/export_rows.py" --base "$snap" --rows "$src/rows.safetensors" --out "$model" || exit 1
cp "$work/export.json" "$out/export.json"
variant=$(cat "$src/sglang_variant.txt")
rm -rf "$out/sglang"
timeout 1500 "$env_py" "$src/sglang_extract.py" --model "$model" --bank "$src/bank.json" \
  --mode pair --variant "$variant" --out "$out/sglang" 2> "$out/sglang.stderr.log" | tail -1
echo "sglang exit ${PIPESTATUS[0]}"
echo ENGINE_EVAL_SGLANG_COMPLETE
