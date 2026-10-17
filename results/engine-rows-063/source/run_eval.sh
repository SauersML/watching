#!/usr/bin/env bash
# run_eval.sh SRC OUT: install the fitted Yes/No rows (SRC/rows.safetensors) into Qwen3-8B and run
# it fresh: Trainer training step; vLLM fresh engine with a repeat pass, batched, eager, and with an
# 8,192-token context. Nothing here was used for fitting.
set -uo pipefail
src=$1; out=$2; mkdir -p "$out"
export PYTHONPATH="$src/src" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export TORCHINDUCTOR_COMPILE_THREADS=4 HF_HUB_DISABLE_TELEMETRY=1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$out/gpu.txt"
snap=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-8B', revision='b968826d9c46dd6066d109eabc6255188de91218'))")
work=/workspace/engine063-eval; mkdir -p "$work"
model="$work/model"
[[ -d "$model" ]] || python "$src/export_rows.py" --base "$snap" --rows "$src/rows.safetensors" --out "$model" || exit 1
cp "$work/export.json" "$out/export.json"
stage() {
  local name=$1 done=$2; shift 2
  if [[ -f "$out/$name/$done" ]]; then echo "skip $name"; return 0; fi
  rm -rf "$out/$name"
  export TORCHINDUCTOR_CACHE_DIR="$work/inductor-$name" VLLM_CACHE_ROOT="$work/vllm-$name"
  "$@" --out "$out/$name" 2> "$out/$name.stderr.log" | tail -1
  local status=${PIPESTATUS[0]}
  echo "stage $name exit $status"; tail -2 "$out/$name.stderr.log"
  return $status
}
stage hf hf.done python "$src/hf_extract.py" --model "$model" --bank "$src/bank.json"
stage vllm pass1.done python "$src/vllm_extract.py" --model "$model" --bank "$src/bank.json" --passes 2
stage vllm_batched pass0.done python "$src/vllm_extract.py" --model "$model" --bank "$src/bank.json" --batch 64
stage vllm_eager pass0.done python "$src/vllm_extract.py" --model "$model" --bank "$src/bank.json" --eager
stage vllm_longctx pass0.done python "$src/vllm_extract.py" --model "$model" --bank "$src/bank.json" --max-len 8192
echo ENGINE_EVAL_COMPLETE
