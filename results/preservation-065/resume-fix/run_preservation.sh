#!/usr/bin/env bash
# One fresh experiment, fixed recipe. Source/capture/optimizer resume gates fail closed.
# Invoke with SOURCE_DIR OUTPUT_DIR PINNED_QWEN_SNAPSHOT; PYTHONPATH must include watching/src.
# External runner MUST impose <=3h pod lifetime and <=$1.50/h price (<= $4.50 rent).
set -euo pipefail
source_dir=$(cd "$1" && pwd)
mkdir -p "$2"
out=$(cd "$2" && pwd)
model=$(cd "$3" && pwd)
cd "$out"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$source_dir/src${PYTHONPATH:+:$PYTHONPATH}"
python "$source_dir/verify_source.py" "$source_dir" "$model" "$out"
# Full 16GB export stays outside the synchronized artifact tree.
export_dir=${WATCHING_EXPORT_DIR:-/workspace/watching-preservation-export}
[[ "$export_dir" != "$out"* ]] || { echo 'Export directory must be outside synced output' >&2; exit 2; }
stage() {
  local name=$1; shift
  if [[ -f done.$name ]]; then echo "ALREADY_DONE $name"; return 0; fi
  echo "START $name $(date -u +%FT%TZ)"
  set +e
  "$@" 2>&1 | tee -a "$name.log"
  local statuses=("${PIPESTATUS[@]}")
  set -e
  local status=${statuses[0]}
  [[ ${statuses[1]} -eq 0 ]] || return "${statuses[1]}"
  if [[ "$name" == train && "$status" -eq 75 ]]; then
    cp train/chosen.json stopped.train.json
    echo "BOUNDED_PARTIAL_TRAINING: evaluating saved selected weights"
    return 0
  fi
  [[ "$status" -eq 0 ]] || return "$status"
  touch "done.$name"
  echo "DONE $name $(date -u +%FT%TZ)"
}
# Only downstream stages depend on selected weights. Captures and training are
# never invalidated by these pins; unchanged selected weights reuse their scores.
pinned_stage() {
  local name=$1; shift
  local pins=() state
  while [[ $# -gt 0 && "$1" != -- ]]; do
    case "$1" in
      --input|--output) pins+=("$1" "$2"); shift 2 ;;
      *) echo "Invalid dependency argument: $1" >&2; return 2 ;;
    esac
  done
  [[ $# -gt 0 ]] || return 2
  shift
  pins+=(--command-json "$(python -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "$@")")
  state=$(python "$source_dir/stage_dependencies.py" prepare --root "$out" --stage "$name" "${pins[@]}")
  if [[ "$state" == current ]]; then echo "PINNED_ALREADY_DONE $name"; return 0; fi
  [[ "$state" == run ]] || return 2
  stage "$name" "$@"
  python "$source_dir/stage_dependencies.py" commit --root "$out" --stage "$name" "${pins[@]}"
}
cp "$source_dir/fresh_test.json" fresh_test.json
python -c 'import hashlib; assert hashlib.sha256(open("fresh_test.json", "rb").read()).hexdigest() == "091e2b2ac8dbb855b385bd0a94539d448f0365d30dce777c1acb4a5495d3bbf2"'
banks="$source_dir/bank.json,$source_dir/bank2.json,$source_dir/bank3_10k.json"
ordinary="$source_dir/bank_ordinary.json"
stage hf python "$source_dir/capture_hf.py" --model "$model" --banks "$banks" --out hf
export VLLM_CACHE_ROOT="$out/vcache-source" TORCHINDUCTOR_CACHE_DIR="$out/icache-source"
stage vllm python "$source_dir/capture_vllm.py" --model "$model" --banks "$banks" --hf hf --out vllm --max-len 59
unset VLLM_CACHE_ROOT TORCHINDUCTOR_CACHE_DIR
stage ordinary_hf python "$source_dir/capture_hf.py" --model "$model" --banks "$ordinary" --out ordinary_hf
export VLLM_CACHE_ROOT="$out/vcache-ordinary" TORCHINDUCTOR_CACHE_DIR="$out/icache-ordinary"
stage ordinary_vllm python "$source_dir/capture_vllm.py" --model "$model" --banks "$ordinary" --hf ordinary_hf --out ordinary_vllm --max-len 128
unset VLLM_CACHE_ROOT TORCHINDUCTOR_CACHE_DIR
resume=()
[[ ! -f train/resume.pt ]] || resume=(--resume train/resume.pt)
stage train python "$source_dir/train_tail.py" --model "$model" --banks "$banks" --hf hf --vllm vllm --out train --steps 6144 --cosine --keep-best --ordinary-banks "$ordinary" --ordinary-hf ordinary_hf --ordinary-vllm ordinary_vllm --save-every 128 --eval-every 256 --max-seconds 4200 "${resume[@]}"
pinned_stage export --input "selected_tail=$out/train/tail.safetensors" \
  --input "export_source=$source_dir/export_tail.py" --input "base_config=$model/config.json" \
  --output "$export_dir/model" --output "$export_dir/export.json" -- \
  python "$source_dir/export_tail.py" --base "$model" --tail train/tail.safetensors --out "$export_dir/model"
# Historical endpoints are explicitly exploratory; no hyperparameter selection on test scores.
groups=dev,dev_new,familiar,novel,smoke
pinned_stage eval_hf --input "export=$export_dir/export.json" \
  --input "endpoint_source=$source_dir/hf_extract.py" --input "bank=$source_dir/bank.json" --input "hf_helper=$source_dir/src/watching/framework_hf.py" \
  --output "$out/eval/hf" -- python "$source_dir/hf_extract.py" --model "$export_dir/model" --bank "$source_dir/bank.json" --groups "$groups" --out eval/hf
export VLLM_CACHE_ROOT="$out/vcache-eval" TORCHINDUCTOR_CACHE_DIR="$out/icache-eval"
pinned_stage eval_vllm --input "export=$export_dir/export.json" \
  --input "endpoint_source=$source_dir/vllm_extract.py" --input "bank=$source_dir/bank.json" --input "vllm_helper=$source_dir/src/watching/framework_vllm.py" \
  --output "$out/eval/vllm" -- python "$source_dir/vllm_extract.py" --model "$export_dir/model" --bank "$source_dir/bank.json" --groups "$groups" --max-len 59 --passes 2 --out eval/vllm
unset VLLM_CACHE_ROOT TORCHINDUCTOR_CACHE_DIR
pinned_stage ordinary_answers --input "export=$export_dir/export.json" \
  --input "endpoint_source=$source_dir/ordinary.py" --input "bank=$source_dir/bank.json" --input "bank2=$source_dir/bank2.json" --input "bank3=$source_dir/bank3_10k.json" \
  --output "$out/ordinary_eval" -- python "$source_dir/ordinary.py" --base "$model" --trained "$export_dir/model" --banks "$banks" --out ordinary_eval
pinned_stage eval_fresh_hf --input "export=$export_dir/export.json" \
  --input "endpoint_source=$source_dir/hf_extract.py" --input "bank=$out/fresh_test.json" --input "hf_helper=$source_dir/src/watching/framework_hf.py" \
  --output "$out/eval_fresh/hf" -- python "$source_dir/hf_extract.py" --model "$export_dir/model" --bank fresh_test.json --groups all --out eval_fresh/hf
export VLLM_CACHE_ROOT="$out/vcache-fresh" TORCHINDUCTOR_CACHE_DIR="$out/icache-fresh"
pinned_stage eval_fresh_vllm --input "export=$export_dir/export.json" \
  --input "endpoint_source=$source_dir/vllm_extract.py" --input "bank=$out/fresh_test.json" --input "vllm_helper=$source_dir/src/watching/framework_vllm.py" \
  --output "$out/eval_fresh/vllm" -- python "$source_dir/vllm_extract.py" --model "$export_dir/model" --bank fresh_test.json --groups all --max-len 59 --passes 2 --out eval_fresh/vllm
unset VLLM_CACHE_ROOT TORCHINDUCTOR_CACHE_DIR
cp "$export_dir/export.json" ./export.json
# Keep all source states, tail, optimizer, native score tensors and logs for recovery.
python - <<'PY'
import hashlib,json
from pathlib import Path
paths=[p for p in Path('.').rglob('*') if p.is_file() and not any('cache' in part for part in p.parts) and p.name!='SHA256.json']
records={}
for p in paths:
    with p.open('rb') as h: records[str(p)]=hashlib.file_digest(h,'sha256').hexdigest()
Path('SHA256.json').write_text(json.dumps(records,indent=2)+'\n')
PY
echo PRESERVATION_EXPERIMENT_COMPLETE
