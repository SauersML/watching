#!/usr/bin/env bash
# Matched 034 schedule, base initialization; provisioning is owned by the runner.
set -euo pipefail
[[ $# -eq 3 ]] || { echo "Usage: $0 SOURCE OUTPUT reference|buffer|identity" >&2; exit 2; }
run_source=$1
run_output=$2
viewing_mode=$3
case "$viewing_mode" in reference|buffer|identity) ;; *) echo "Invalid viewing mode" >&2; exit 2 ;; esac
export PATH="$HOME/.local/bin:$PATH"
run_env=$(mktemp -d)
trap 'rm -rf "$run_env"' EXIT
if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install --quiet uv
fi
uv venv --python python3 "$run_env/venv"
uv pip install --python "$run_env/venv/bin/python" "$run_source"
export PATH="$run_env/venv/bin:$PATH"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export TORCHINDUCTOR_COMPILE_THREADS=4 TORCHINDUCTOR_CACHE_DIR="$run_env/compile"
mkdir -p "$run_output"
python - "$run_source" "$run_output/data.json" <<'DATA'
import hashlib
import json
import sys
from pathlib import Path
from watching.family_data import make_family_data, read_jsonl
from watching.tail_train import new_run_records
from watching.viewing_train import verify_matched_selection

root, out = Path(sys.argv[1]), Path(sys.argv[2])
bank = root / "data"
data = make_family_data(
    read_jsonl(bank / "training_contexts_a.jsonl") + read_jsonl(bank / "training_contexts_b.jsonl"),
    read_jsonl(bank / "question_families_v1.jsonl"),
    read_jsonl(bank / "new_contexts_v1.jsonl"),
    read_jsonl(bank / "ood_contexts_v1.jsonl"),
    "probe", prior_contexts=read_jsonl(bank / "fresh_test_contexts.jsonl"), seed=20261017,
)
content = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
digest = hashlib.sha256(content.encode()).hexdigest()
if digest != "9321889d38cc5b070844a7a3692811f86e0c70905e7e4d5fbe790f8fab07b755":
    raise SystemExit("Regenerated family-v1 dataset differs from original034 input")
selected = new_run_records(
    data, train_count=256, validation_count=64, test_count=64,
    test_split="test_wording", seed=20261017, development_only=False,
)
provenance = verify_matched_selection(selected, root / "results/two-block-scaled-034/fit/data.json")
out.write_text(content)
print(json.dumps({"event": "matched_data_verified_before_model_load", "input_sha256": digest,
                  **provenance}), flush=True)
DATA
python -m watching.viewing_train --viewing-mode "$viewing_mode" \
  --matched-data "$run_source/results/two-block-scaled-034/fit/data.json" \
  --checkpoint Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --data "$run_output/data.json" --out "$run_output/fit" \
  --probe-layer 14 --tail-layers 2 --train-prompts 256 --validation-prompts 64 \
  --test-prompts 64 --test-split test_wording --require-complete-for-test \
  --prompt-batch-size 8 --steps 1024 --eval-every 128 --early-eval-steps 8 32 128 \
  --checkpoint-every 256 --lr 5e-6 --length 96 --seed 20261017 \
  --max-training-seconds 1050
