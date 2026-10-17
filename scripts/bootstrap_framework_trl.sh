#!/usr/bin/env bash
# Native TRL signal test: smoke first; no training or optimizer updates.
set -euo pipefail
[[ $# -eq 2 ]] || { echo "Usage: $0 SOURCE OUTPUT" >&2; exit 2; }
run_source=$1
run_output=$2
mkdir -p "$run_output"
export PYTHONPATH="$run_source/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
python - <<'PY'
from importlib.metadata import version
import torch
for name, expected in [('torch','2.10.0'), ('transformers','5.18.0'), ('trl','1.14.1')]:
    assert version(name).split('+')[0] == expected, (name, version(name))
assert torch.cuda.is_available()
assert torch.cuda.get_device_properties(0).total_memory >= 40 * 1024**3
print('TRL CUDA preflight:', torch.cuda.get_device_name(), flush=True)
PY
cp "$run_source/SOURCE_REVISION" "$run_output/SOURCE_REVISION"
cp "$run_source/SNAPSHOT.json" "$run_output/SNAPSHOT.json"
python - <<'PY' > "$run_output/environment.txt"
from importlib.metadata import distributions
for name, value in sorted((d.metadata['Name'], d.version) for d in distributions()):
    print(f'{name}=={value}')
PY
work=$(mktemp -d /workspace/framework-trl.XXXXXX)
trap 'rm -rf "$work"' EXIT
curl --fail --location --retry 2 \
  https://github.com/SauersML/watching/releases/download/execution-split-057/native_tail.pt \
  --output "$work/native_tail.pt"
model_base=$(python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download('Qwen/Qwen3-8B',
    revision='b968826d9c46dd6066d109eabc6255188de91218'))
PY
)
python -m watching.framework_hf export --base "$model_base" \
  --tail "$work/native_tail.pt" --model "$work/model" \
  --data "$run_source/data/framework_transfer_v1/dataset.json" --out "$run_output"
python -m watching.framework_trl --model "$work/model" \
  --export "$run_output/export.json" --data "$run_output/tokenized.json" \
  --out "$run_output/trl_smoke" --smoke-only 2> "$run_output/trl_smoke.stderr.log"
python - "$run_output" <<'PY'
import json, sys
from pathlib import Path
import torch
from safetensors.torch import load_file
root = Path(sys.argv[1])
records = json.loads((root/'tokenized.json').read_text())['records']
smoke = [r for r in records if r['group']=='smoke']
assert len(smoke)==4
vectors = load_file(str(root/'trl_smoke/trl_logits.safetensors'))
rows=[]
for r in smoke:
    train = vectors[f"{r['id']}.trl_train.0"]
    predict = vectors[f"{r['id']}.trl_predict.0"]
    assert train.shape==predict.shape and torch.isfinite(train).all() and torch.isfinite(predict).all()
    rows.append({'id':r['id'], 'equal':torch.equal(train,predict),
                 'different_logits':int((train!=predict).sum()),
                 'max_abs_difference':float((train-predict).abs().max())})
gate={'all_smoke_logits_equal':all(r['equal'] for r in rows), 'rows':rows,
      'scope':'Full-vocabulary last-prefix logits; equality does not prove equality of hidden states.',
      'optimizer_updates':0}
(root/'smoke_gate.json').write_text(json.dumps(gate,indent=2)+'\n')
print(json.dumps({'event':'TRL_SMOKE_GATE',**gate}),flush=True)
PY
# Repeat the diagnostic in a fresh process regardless of the gate's outcome.
python -m watching.framework_trl --model "$work/model" \
  --export "$run_output/export.json" --data "$run_output/tokenized.json" \
  --out "$run_output/trl_fresh" --smoke-only 2> "$run_output/trl_fresh.stderr.log"
if python - "$run_output/smoke_gate.json" <<'PY'
import json,sys
sys.exit(0 if json.load(open(sys.argv[1]))['all_smoke_logits_equal'] else 1)
PY
then
    echo 'TRL_STOPPED_AFTER_IDENTICAL_SMOKE_LOGITS: no full bank and no training.'
else
    python -m watching.framework_trl --model "$work/model" \
      --export "$run_output/export.json" --data "$run_output/tokenized.json" \
      --out "$run_output/trl_full" 2> "$run_output/trl_full.stderr.log"
fi
echo 'TRL_SIGNAL_TEST_COMPLETE'
