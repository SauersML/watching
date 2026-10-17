#!/usr/bin/env bash
# run_064e.sh: the original model in vLLM with a repeat pass and in batches of 64, for comparison.
set -uo pipefail
cd /workspace/e64
until grep -q RUN064D_COMPLETE run_064d.log; do sleep 30; done
groups=dev,dev_new,familiar,novel,smoke
stage() { local name=$1; shift; [[ -f done.$name ]] && { echo "skip $name"; return 0; }; echo "start $name $(date -u +%T)"; "$@" > $name.log 2>&1; local s=$?; echo "end $name exit $s $(date -u +%T)"; tail -2 $name.log | cut -c1-300; [[ $s -eq 0 ]] && touch done.$name; return $s; }
export VLLM_CACHE_ROOT=/workspace/e64/vcache-base2 TORCHINDUCTOR_CACHE_DIR=/workspace/e64/icache-base2
stage base_vllm2 python src/vllm_extract.py --model $SNAP --bank src/bank.json --groups $groups --max-len 59 --passes 2 --out base/vllm2
export VLLM_CACHE_ROOT=/workspace/e64/vcache-base3 TORCHINDUCTOR_CACHE_DIR=/workspace/e64/icache-base3
stage base_vllm_batched python src/vllm_extract.py --model $SNAP --bank src/bank.json --groups $groups --max-len 59 --batch 64 --out base/vllm_batched
echo RUN064E_COMPLETE
