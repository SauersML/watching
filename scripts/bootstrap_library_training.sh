#!/usr/bin/env bash
# Native capture APIs; source integrity gates precede any optimizer update.
set -euo pipefail
[[ $# -ge 3 && $# -le 5 ]] || { echo 'Usage: SOURCE OUTPUT METHOD [strict|boundary_observed] [SELF_STOP]' >&2; exit 2; }
run_source=$1
run_output=$2
capture_kind=$3
source_runtime=${4:-strict}
self_stop=${5:-0}
case "$capture_kind" in forward_hook|hidden_states) ;; *) exit 2 ;; esac
case "$source_runtime" in strict|boundary_observed) ;; *) exit 2 ;; esac
case "$self_stop" in 0|1) ;; *) exit 2 ;; esac
mkdir -p "$run_output"
export PYTHONPATH="$run_source/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export TORCHINDUCTOR_COMPILE_THREADS=4
run_work=$(mktemp -d /workspace/library-training.XXXXXX)
cleanup() {
  run_status=$?
  python - "$run_work/compile-cache" "$run_output" <<'PY' || true
import hashlib
import json
import sys
import tarfile
from pathlib import Path

cache, output = map(Path, sys.argv[1:])
extensions = {'.py', '.json', '.cubin', '.ptx', '.ttir', '.ttgir', '.llir'}
files = sorted(p for p in cache.rglob('*') if p.is_file() and p.suffix in extensions)
if files:
    target = output / 'compiler-artifacts.tar.gz'
    manifest = {}
    with tarfile.open(target, 'w:gz') as archive:
        for path in files:
            name = str(path.relative_to(cache))
            content = path.read_bytes()
            manifest[name] = {'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}
            archive.add(path, arcname=name, recursive=False)
    (output / 'compiler-artifacts.json').write_text(json.dumps({
        'scope': 'Generated source and metadata present in this run-specific Inductor cache',
        'files': manifest,
        'archive_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
    }, indent=2) + '\n')
PY
  rm -rf "$run_work"
  python - "$run_output" "$run_status" <<'PY' || true
import json
import sys
from pathlib import Path
root, status = Path(sys.argv[1]), int(sys.argv[2])
(root / 'worker_exit.json').write_text(json.dumps({'exit_code': status}) + '\n')
PY
  if [[ "$self_stop" == 1 ]]; then
    # The launcher creates a persistent /workspace volume for these jobs.
    # Stop GPU billing after output writes; the local runner later recovers it.
    setsid nohup bash -c '
      sleep 15
      eval "$(tr "\0" "\n" < /proc/1/environ | grep -E "^RUNPOD_(API_KEY|POD_ID)=" | sed "s/^/export /")"
      runpodctl pod stop "$RUNPOD_POD_ID" || runpodctl stop pod "$RUNPOD_POD_ID"
    ' > "$run_output/self_stop.log" 2>&1 < /dev/null &
  fi
}
trap cleanup EXIT
export TORCHINDUCTOR_CACHE_DIR="$run_work/compile-cache"
python - <<'PY'
from importlib.metadata import version
import torch
for package, expected in [('torch', '2.10.0'), ('transformers', '5.18.0')]:
    assert version(package).split('+')[0] == expected, (package, version(package))
assert torch.cuda.is_available()
assert torch.cuda.get_device_properties(0).total_memory >= 40 * 1024**3
print('Native capture training CUDA:', torch.cuda.get_device_name(), flush=True)
PY
cp "$run_source/SOURCE_REVISION" "$run_output/SOURCE_REVISION"
cp "$run_source/SNAPSHOT.json" "$run_output/SNAPSHOT.json"
python - <<'PY' > "$run_output/environment.txt"
from importlib.metadata import distributions
for name, value in sorted((d.metadata['Name'], d.version) for d in distributions()):
    print(f'{name}=={value}')
PY
python - "$run_source" "$run_output/data.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
from watching.family_data import make_family_data, read_jsonl
from watching.tail_train import new_run_records
from watching.viewing_train import verify_matched_selection

root, output = Path(sys.argv[1]), Path(sys.argv[2])
bank = root / 'data'
data = make_family_data(
    read_jsonl(bank / 'training_contexts_a.jsonl') + read_jsonl(bank / 'training_contexts_b.jsonl'),
    read_jsonl(bank / 'question_families_v1.jsonl'),
    read_jsonl(bank / 'new_contexts_v1.jsonl'),
    read_jsonl(bank / 'ood_contexts_v1.jsonl'),
    'probe', prior_contexts=read_jsonl(bank / 'fresh_test_contexts.jsonl'), seed=20261017,
)
content = json.dumps(data, ensure_ascii=False, indent=2) + '\n'
digest = hashlib.sha256(content.encode()).hexdigest()
if digest != '9321889d38cc5b070844a7a3692811f86e0c70905e7e4d5fbe790f8fab07b755':
    raise RuntimeError('Family-v1 input differs from historical matched data')
selected = new_run_records(
    data, train_count=256, validation_count=64, test_count=64,
    test_split='test_wording', seed=20261017, development_only=False,
)
reference = root / 'results/two-block-scaled-034/fit/data.json'
if hashlib.sha256(reference.read_bytes()).hexdigest() != '7ed9cfc870ee7cf990bdb8636221b75a5c5a27b78088b0ca969513d09b7e0e94':
    raise RuntimeError('Historical selected-record file differs from the frozen protocol')
verified = verify_matched_selection(selected, reference)
output.write_text(content)
print(json.dumps({'event': 'matched_data_verified', 'input_sha256': digest, **verified}), flush=True)
PY
python -m watching.library_train --capture-method "$capture_kind" --source-runtime "$source_runtime" \
  --checkpoint Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --data "$run_output/data.json" --out "$run_output/fit" \
  --probe-layer 14 --tail-layers 2 --train-prompts 256 --validation-prompts 64 \
  --test-prompts 64 --test-split test_wording --require-complete-for-test \
  --prompt-batch-size 8 --steps 1024 --eval-every 128 --early-eval-steps 8 32 128 \
  --checkpoint-every 256 --lr 5e-6 --length 96 --seed 20261017 \
  --max-training-seconds 1050
