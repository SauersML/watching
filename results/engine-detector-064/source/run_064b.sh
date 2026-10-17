#!/usr/bin/env bash
# run_064b.sh: after run_064.sh, SGLang on the trained model, and the original model in all three
# runtimes on the same prompts and GPU for comparison.
set -uo pipefail
cd /workspace/e64
until grep -q RUN064_COMPLETE run_064.log; do sleep 20; done
groups=dev,dev_new,familiar,novel,smoke
stage() { local name=$1; shift; [[ -f done.$name ]] && { echo "skip $name"; return 0; }; echo "start $name $(date -u +%T)"; "$@" > $name.log 2>&1; local s=$?; echo "end $name exit $s $(date -u +%T)"; tail -2 $name.log | cut -c1-300; [[ $s -eq 0 ]] && touch done.$name; return $s; }
stage eval_sglang /workspace/e64/sglang-env/bin/python src/sglang_extract.py --model trained/model --bank src/bank.json --groups $groups --mode pair --variant no_piecewise --out eval/sglang
stage base_hf python src/hf_extract.py --model $SNAP --bank src/bank.json --groups $groups --out base/hf
export VLLM_CACHE_ROOT=/workspace/e64/vcache-base TORCHINDUCTOR_CACHE_DIR=/workspace/e64/icache-base
stage base_vllm python src/vllm_extract.py --model $SNAP --bank src/bank.json --groups $groups --max-len 59 --out base/vllm
stage base_sglang /workspace/e64/sglang-env/bin/python src/sglang_extract.py --model $SNAP --bank src/bank.json --groups $groups --mode pair --variant no_piecewise --out base/sglang
echo RUN064B_COMPLETE
