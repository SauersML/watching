#!/usr/bin/env bash
# run_064f.sh: the original model's comparison runs (batches of 64, a repeat pass) as soon as the
# run-3 vLLM capture ends, then the rest of run 3 (run_064d.sh skips the stages already done).
set -uo pipefail
cd /workspace/e64
while kill -0 "$1" 2>/dev/null; do sleep 10; done
if grep -q VLLM_CAPTURE_COMPLETE v3.log && [[ -f v3/vllm_capture.json ]]; then touch done.v3; else echo "v3 capture failed"; exit 1; fi
groups=dev,dev_new,familiar,novel,smoke
stage() { local name=$1; shift; [[ -f done.$name ]] && { echo "skip $name"; return 0; }; echo "start $name $(date -u +%T)"; "$@" > $name.log 2>&1; local s=$?; echo "end $name exit $s $(date -u +%T)"; tail -2 $name.log | cut -c1-300; [[ $s -eq 0 ]] && touch done.$name; return $s; }
export VLLM_CACHE_ROOT=/workspace/e64/vcache-base3 TORCHINDUCTOR_CACHE_DIR=/workspace/e64/icache-base3
stage base_vllm_batched python src/vllm_extract.py --model $SNAP --bank src/bank.json --groups $groups --max-len 59 --batch 64 --out base/vllm_batched
export VLLM_CACHE_ROOT=/workspace/e64/vcache-base2 TORCHINDUCTOR_CACHE_DIR=/workspace/e64/icache-base2
stage base_vllm2 python src/vllm_extract.py --model $SNAP --bank src/bank.json --groups $groups --max-len 59 --passes 2 --out base/vllm2
unset VLLM_CACHE_ROOT TORCHINDUCTOR_CACHE_DIR
echo RUN064E_COMPLETE
bash src/run_064d.sh
