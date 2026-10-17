#!/usr/bin/env bash
# Matched 034 detector control: exact archived schedule; identical cached inputs.
set -euo pipefail
[[ $# -eq 1 ]] || { echo "Usage: $0 OUTPUT" >&2; exit 2; }
run_out=$1
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source_results="$run_out/source034"
mkdir -p "$source_results"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
for name in manifest.json data.json cache_fingerprints.json \
  train_0_split_inference.json validation_0_split_inference.json; do
  cp "$repo_root/results/two-block-scaled-034/fit/$name" "$source_results/$name"
done
curl --fail --location --retry 3 \
  https://github.com/SauersML/watching/releases/download/sft-034/cache_states.pt \
  --output "$source_results/cache_states.pt"
python - "$source_results" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
with (root / "cache_states.pt").open("rb") as stream:
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
assert digest == "1d2e5cbee328d1dd6c6524b22ef6160855c4305cc6b7335e7bb2bdd54fda59ab", digest
manifest = json.loads((root / "manifest.json").read_text())
for key, expected in {"steps":1024, "prompt_batch_size":8, "lr":5e-6,
                      "seed":20261017, "length":96, "eval_every":128,
                      "train_prompts":256, "validation_prompts":64,
                      "test_prompts":64, "tail_layers":2,
                      "early_eval_steps":[8,32,128]}.items():
    assert manifest[key] == expected, (key, manifest[key], expected)
print(json.dumps({"event":"released_cache_verified", "sha256":digest}), flush=True)
PY
python -m watching.null_tail --source-results "$source_results" \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --require-source-baseline --out "$run_out/fit"
