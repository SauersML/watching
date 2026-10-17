#!/usr/bin/env bash
# run_extract.sh SRC OUT: original Qwen3-8B on the 424-prompt bank, Trainer training step and
# four vLLM engines (default with a repeat pass, fresh, batched, eager). No weights are changed.
set -uo pipefail
src=$1; out=$2; mkdir -p "$out"
export PYTHONPATH="$src/src" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export TORCHINDUCTOR_COMPILE_THREADS=4 HF_HUB_DISABLE_TELEMETRY=1
python -c 'import torch, transformers, vllm; print(torch.__version__, transformers.__version__, vllm.__version__, torch.cuda.get_device_name(), flush=True)' || exit 1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$out/gpu.txt"
snap=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-8B', revision='b968826d9c46dd6066d109eabc6255188de91218'))")
work=$(mktemp -d /workspace/engine063.XXXXXX)
stage() {  # stage NAME DONE_FILE COMMAND...: run unless already complete (a replaced pod resumes)
  local name=$1 done=$2; shift 2
  if [[ -f "$out/$name/$done" ]]; then echo "skip $name"; return 0; fi
  rm -rf "$out/$name"
  export TORCHINDUCTOR_CACHE_DIR="$work/inductor-$name" VLLM_CACHE_ROOT="$work/vllm-$name"
  "$@" --out "$out/$name" 2> "$out/$name.stderr.log"
  local status=$?
  echo "stage $name exit $status"
  tail -5 "$out/$name.stderr.log"
  return $status
}
stage hf hf_vectors.safetensors python "$src/hf_extract.py" --model "$snap" --bank "$src/bank.json" || exit 1
stage vllm pass1.safetensors python "$src/vllm_extract.py" --model "$snap" --bank "$src/bank.json" --passes 2 || exit 1
stage vllm_fresh pass0.safetensors python "$src/vllm_extract.py" --model "$snap" --bank "$src/bank.json"
stage vllm_batched pass0.safetensors python "$src/vllm_extract.py" --model "$snap" --bank "$src/bank.json" --batch 64
stage vllm_eager pass0.safetensors python "$src/vllm_extract.py" --model "$snap" --bank "$src/bank.json" --eager
echo ENGINE_EXTRACT_COMPLETE
