#!/usr/bin/env bash
# run_064h.sh: after run 4, run 3 answers the same ordinary requests served by vLLM.
set -uo pipefail
cd /workspace/e64
until grep -q RUN064G_COMPLETE run_064g.log; do sleep 20; done
stage() { local name=$1; shift; [[ -f done.$name ]] && { echo "skip $name"; return 0; }; echo "start $name $(date -u +%T)"; "$@" > $name.log 2>&1; local s=$?; echo "end $name exit $s $(date -u +%T)"; tail -2 $name.log | cut -c1-300; [[ $s -eq 0 ]] && touch done.$name; return $s; }
rm -f hfo/hf_states.safetensors vo/vllm_states.safetensors
rm -rf trained3
stage export3b python src/export_tail.py --base $SNAP --tail train3/tail.safetensors --out trained3/model || exit 1
stage chat_vllm3 python src/ordinary_vllm.py --model trained3/model --bank chats/bank_ordinary_test.json --out ordinary3/vllm_trained.jsonl
echo RUN064H_COMPLETE
