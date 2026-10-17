#!/usr/bin/env bash
# run_content.sh SRC OUT: final hidden states of the original Qwen3-8B on the 20,000-prompt bank.
set -uo pipefail
src=$1; out=$2; mkdir -p "$out"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 HF_HUB_DISABLE_TELEMETRY=1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$out/gpu.txt"
snap=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-8B', revision='b968826d9c46dd6066d109eabc6255188de91218'))")
python "$src/content_extract.py" --model "$snap" --bank "$src/bank3.json" --out "$out/content" 2> "$out/content.stderr.log" | tail -1
status=${PIPESTATUS[0]}; tail -3 "$out/content.stderr.log"
[[ $status -eq 0 ]] && echo CONTENT_COMPLETE
exit $status
