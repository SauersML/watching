#!/usr/bin/env bash
# run_064g.sh: run 4, run 3's detector trained together with matching the original model on
# ordinary chats, then the same tests plus ordinary requests in the Trainer's model and in vLLM.
set -uo pipefail
cd /workspace/e64
groups=dev,dev_new,familiar,novel,smoke
stage() { local name=$1; shift; [[ -f done.$name ]] && { echo "skip $name"; return 0; }; echo "start $name $(date -u +%T)"; "$@" > $name.log 2>&1; local s=$?; echo "end $name exit $s $(date -u +%T)"; tail -2 $name.log | cut -c1-300; [[ $s -eq 0 ]] && touch done.$name; return $s; }
mkdir -p chats
stage chats python src/make_ordinary.py --model $SNAP --banks src/bank.json,src/bank2.json,src/bank3_10k.json --out chats/bank_ordinary.json || exit 1
stage hfo python src/capture_hf.py --model $SNAP --banks chats/bank_ordinary.json --out hfo || exit 1
export VLLM_CACHE_ROOT=/workspace/e64/vcache-A TORCHINDUCTOR_CACHE_DIR=/workspace/e64/icache-A
stage vo python src/capture_vllm.py --model $SNAP --banks chats/bank_ordinary.json --hf hfo --out vo --max-len 128 || exit 1
unset VLLM_CACHE_ROOT TORCHINDUCTOR_CACHE_DIR
stage train4 python src/train_tail.py --model $SNAP --banks src/bank.json,src/bank2.json,src/bank3_10k.json --hf hf+hf3 --vllm vA+v3 --out train4 --steps 6144 --cosine --keep-best --ordinary-banks chats/bank_ordinary.json --ordinary-hf hfo --ordinary-vllm vo || exit 1
stage export4 python src/export_tail.py --base $SNAP --tail train4/tail.safetensors --out trained4/model || exit 1
stage eval4_hf python src/hf_extract.py --model trained4/model --bank src/bank.json --groups $groups --out eval4/hf || exit 1
export VLLM_CACHE_ROOT=/workspace/e64/vcache-eval4 TORCHINDUCTOR_CACHE_DIR=/workspace/e64/icache-eval4
stage eval4_vllm python src/vllm_extract.py --model trained4/model --bank src/bank.json --groups $groups --max-len 59 --passes 2 --out eval4/vllm
export VLLM_CACHE_ROOT=/workspace/e64/vcache-eval4b TORCHINDUCTOR_CACHE_DIR=/workspace/e64/icache-eval4b
stage eval4_vllm_batched python src/vllm_extract.py --model trained4/model --bank src/bank.json --groups $groups --max-len 59 --batch 64 --out eval4/vllm_batched
unset VLLM_CACHE_ROOT TORCHINDUCTOR_CACHE_DIR
stage eval4_sglang /workspace/e64/sglang-env/bin/python src/sglang_extract.py --model trained4/model --bank src/bank.json --groups $groups --mode pair --variant no_piecewise --out eval4/sglang
stage chat_vllm_base python src/ordinary_vllm.py --model $SNAP --bank chats/bank_ordinary_test.json --out ordinary4/vllm_original.jsonl
stage chat_vllm4 python src/ordinary_vllm.py --model trained4/model --bank chats/bank_ordinary_test.json --out ordinary4/vllm_trained.jsonl
stage chat_hf4 python src/ordinary.py --base $SNAP --trained trained4/model --banks src/bank.json,src/bank2.json,src/bank3_10k.json --out ordinary4
echo RUN064G_COMPLETE
