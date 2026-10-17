#!/usr/bin/env bash
# run_extract2.sh SRC OUT: the 3,000-prompt varied bank through the Trainer training step and
# vLLM (default engine, bank-1 configuration); raw vectors stay on the pod, compact features return.
set -uo pipefail
src=$1; out=$2; mkdir -p "$out"
export PYTHONPATH="$src/src" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export TORCHINDUCTOR_COMPILE_THREADS=4 HF_HUB_DISABLE_TELEMETRY=1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$out/gpu.txt"
snap=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-8B', revision='b968826d9c46dd6066d109eabc6255188de91218'))")
work=/workspace/engine063-wide; mkdir -p "$work/raw"
export TORCHINDUCTOR_CACHE_DIR="$work/inductor" VLLM_CACHE_ROOT="$work/vllmcache"
[[ -f "$out/hf/hf.done" ]] || { rm -rf "$out/hf"; python "$src/hf_extract.py" --model "$snap" --bank "$src/bank2.json" --raw-dir "$work/raw/hf" --out "$out/hf" 2> "$out/hf.stderr.log" | tail -1 || exit 1; }
[[ -f "$out/vllm/pass0.done" ]] || { rm -rf "$out/vllm"; python "$src/vllm_extract.py" --model "$snap" --bank "$src/bank2.json" --max-len 59 --raw-dir "$work/raw/vllm" --out "$out/vllm" 2> "$out/vllm.stderr.log" | tail -1 || exit 1; }
python "$src/compact.py" --model "$snap" --yes 9454 --no 2753 --out "$out/compact" \
  "hf:hf:$work/raw/hf/hf_vectors.safetensors" "vllm:vllm:$work/raw/vllm/pass0.safetensors" || exit 1
echo ENGINE_EXTRACT2_COMPLETE
