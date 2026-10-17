#!/usr/bin/env bash
# run_sglang.sh SRC OUT: original Qwen3-8B served by SGLang on the 424-prompt bank.
# Yes/No-only requests first, trying CUDA-graph settings in turn; then whole-vocabulary requests.
set -uo pipefail
src=$1; out=$2; mkdir -p "$out"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 HF_HUB_DISABLE_TELEMETRY=1
export PATH="$HOME/.local/bin:$PATH"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$out/gpu.txt"
work=/workspace/engine063-sglang; mkdir -p "$work"
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
    'sglang==0.5.10.post1' 'safetensors==0.8.0' > "$out/install.log" 2>&1 || { tail -40 "$out/install.log"; exit 1; }
fi
"$env_py" -c 'import sglang, torch; print("sglang", sglang.__version__, torch.__version__, flush=True)' || exit 1
snap=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-8B', revision='b968826d9c46dd6066d109eabc6255188de91218'))")
good=''
for variant in no_piecewise no_graph default; do
  name="pair_$variant"
  rm -rf "$out/$name"
  # a probe of 8 prompts finds a working setting quickly; a hang costs at most 15 min
  timeout 480 "$env_py" "$src/sglang_extract.py" --model "$snap" --bank "$src/bank.json" \
    --mode pair --variant "$variant" --limit 8 --out "$out/probe_$variant" 2> "$out/probe_$variant.stderr.log"
  status=$?
  echo "probe $variant exit $status"; tail -4 "$out/probe_$variant.stderr.log"
  if [[ $status -eq 0 ]]; then good=$variant; break; fi
done
[[ -n "$good" ]] || { echo SGLANG_NO_WORKING_SETTING; exit 1; }
timeout 1500 "$env_py" "$src/sglang_extract.py" --model "$snap" --bank "$src/bank.json" \
  --mode pair --variant "$good" --out "$out/pair_$good" 2> "$out/pair_$good.stderr.log"
echo "pair exit $?"
timeout 300 "$env_py" "$src/sglang_extract.py" --model "$snap" --bank "$src/bank.json" \
  --mode full --variant "$good" --limit 2 --out "$out/fullprobe_$good" 2> "$out/fullprobe_$good.stderr.log"
status=$?; echo "full probe exit $status"
if [[ $status -eq 0 ]]; then
  timeout 2400 "$env_py" "$src/sglang_extract.py" --model "$snap" --bank "$src/bank.json" \
    --mode full --variant "$good" --out "$out/full_$good" 2> "$out/full_$good.stderr.log"
  echo "full exit $?"
fi
echo SGLANG_COMPLETE
