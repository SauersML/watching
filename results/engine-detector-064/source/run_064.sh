#!/usr/bin/env bash
# run_064.sh: capture (Trainer, two vLLM engines), train the tail, export, evaluate. On the dev pod.
set -uo pipefail
cd /workspace/e64
banks=src/bank.json,src/bank2.json
stage() { local name=$1; shift; [[ -f done.$name ]] && { echo "skip $name"; return 0; }; echo "start $name $(date -u +%T)"; "$@" > $name.log 2>&1; local s=$?; echo "end $name exit $s $(date -u +%T)"; tail -3 $name.log | cut -c1-400; [[ $s -eq 0 ]] && touch done.$name; return $s; }
stage hf python src/capture_hf.py --model $SNAP --banks $banks --out hf || exit 1
for e in A B; do
  export VLLM_CACHE_ROOT=/workspace/e64/vcache-$e TORCHINDUCTOR_CACHE_DIR=/workspace/e64/icache-$e
  stage v$e python src/capture_vllm.py --model $SNAP --banks $banks --hf hf --out v$e || exit 1
done
unset VLLM_CACHE_ROOT TORCHINDUCTOR_CACHE_DIR
stage train python src/train_tail.py --model $SNAP --banks $banks --hf hf --vllm vA,vB --out train || exit 1
stage export python src/export_tail.py --base $SNAP --tail train/tail.safetensors --out trained/model || exit 1
groups=dev,dev_new,familiar,novel,smoke
stage eval_hf python src/hf_extract.py --model trained/model --bank src/bank.json --groups $groups --out eval/hf || exit 1
for v in "vllm --passes 2" "vllm_batched --batch 64" "vllm_eager --eager"; do
  set -- $v; name=$1; shift
  export VLLM_CACHE_ROOT=/workspace/e64/vcache-eval-$name TORCHINDUCTOR_CACHE_DIR=/workspace/e64/icache-eval-$name
  stage eval_$name python src/vllm_extract.py --model trained/model --bank src/bank.json --groups $groups --max-len 59 --out eval/$name "$@"
done
echo RUN064_COMPLETE
