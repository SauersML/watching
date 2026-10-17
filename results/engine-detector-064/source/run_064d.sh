#!/usr/bin/env bash
# run_064d.sh: run 3, with 10,000 more prompts captured from the Trainer and vLLM.
set -uo pipefail
cd /workspace/e64
until grep -q RUN064C_COMPLETE run_064c.log; do sleep 20; done
groups=dev,dev_new,familiar,novel,smoke
stage() { local name=$1; shift; [[ -f done.$name ]] && { echo "skip $name"; return 0; }; echo "start $name $(date -u +%T)"; "$@" > $name.log 2>&1; local s=$?; echo "end $name exit $s $(date -u +%T)"; tail -2 $name.log | cut -c1-300; [[ $s -eq 0 ]] && touch done.$name; return $s; }
rm -rf vB/vllm_states.safetensors test_hf test_v trained
stage hf3 python src/capture_hf.py --model $SNAP --banks src/bank3_10k.json --out hf3 || exit 1
export VLLM_CACHE_ROOT=/workspace/e64/vcache-A TORCHINDUCTOR_CACHE_DIR=/workspace/e64/icache-A
stage v3 python src/capture_vllm.py --model $SNAP --banks src/bank3_10k.json --hf hf3 --out v3 || exit 1
unset VLLM_CACHE_ROOT TORCHINDUCTOR_CACHE_DIR
stage train3 python src/train_tail.py --model $SNAP --banks src/bank.json,src/bank2.json,src/bank3_10k.json --hf hf+hf3 --vllm vA+v3 --out train3 --steps 8192 --cosine --keep-best || exit 1
stage export3 python src/export_tail.py --base $SNAP --tail train3/tail.safetensors --out trained3/model || exit 1
stage eval3_hf python src/hf_extract.py --model trained3/model --bank src/bank.json --groups $groups --out eval3/hf || exit 1
export VLLM_CACHE_ROOT=/workspace/e64/vcache-eval3 TORCHINDUCTOR_CACHE_DIR=/workspace/e64/icache-eval3
stage eval3_vllm python src/vllm_extract.py --model trained3/model --bank src/bank.json --groups $groups --max-len 59 --passes 2 --out eval3/vllm
export VLLM_CACHE_ROOT=/workspace/e64/vcache-eval3b TORCHINDUCTOR_CACHE_DIR=/workspace/e64/icache-eval3b
stage eval3_vllm_batched python src/vllm_extract.py --model trained3/model --bank src/bank.json --groups $groups --max-len 59 --batch 64 --out eval3/vllm_batched
stage eval3_sglang /workspace/e64/sglang-env/bin/python src/sglang_extract.py --model trained3/model --bank src/bank.json --groups $groups --mode pair --variant no_piecewise --out eval3/sglang
echo RUN064D_COMPLETE
