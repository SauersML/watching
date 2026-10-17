#!/usr/bin/env bash
# run_064c.sh: training run 2 (cosine decay, best development checkpoint), then the same real tests.
set -uo pipefail
cd /workspace/e64
until grep -q RUN064B_COMPLETE run_064b.log; do sleep 20; done
banks=src/bank.json,src/bank2.json
groups=dev,dev_new,familiar,novel,smoke
stage() { local name=$1; shift; [[ -f done.$name ]] && { echo "skip $name"; return 0; }; echo "start $name $(date -u +%T)"; "$@" > $name.log 2>&1; local s=$?; echo "end $name exit $s $(date -u +%T)"; tail -2 $name.log | cut -c1-300; [[ $s -eq 0 ]] && touch done.$name; return $s; }
stage train2 python src/train_tail.py --model $SNAP --banks $banks --hf hf --vllm vA --out train2 --steps 4096 --cosine --keep-best || exit 1
stage export2 python src/export_tail.py --base $SNAP --tail train2/tail.safetensors --out trained2/model || exit 1
stage eval2_hf python src/hf_extract.py --model trained2/model --bank src/bank.json --groups $groups --out eval2/hf || exit 1
export VLLM_CACHE_ROOT=/workspace/e64/vcache-eval2 TORCHINDUCTOR_CACHE_DIR=/workspace/e64/icache-eval2
stage eval2_vllm python src/vllm_extract.py --model trained2/model --bank src/bank.json --groups $groups --max-len 59 --passes 2 --out eval2/vllm
export VLLM_CACHE_ROOT=/workspace/e64/vcache-eval2b TORCHINDUCTOR_CACHE_DIR=/workspace/e64/icache-eval2b
stage eval2_vllm_batched python src/vllm_extract.py --model trained2/model --bank src/bank.json --groups $groups --max-len 59 --batch 64 --out eval2/vllm_batched
stage eval2_sglang /workspace/e64/sglang-env/bin/python src/sglang_extract.py --model trained2/model --bank src/bank.json --groups $groups --mode pair --variant no_piecewise --out eval2/sglang
echo RUN064C_COMPLETE
