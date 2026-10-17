#!/usr/bin/env bash
# Fixed native two-block SFT; autograd context is the source condition.
set -euo pipefail
[[ $# -eq 2 ]] || { echo "Usage: $0 OUTPUT MODEL" >&2; exit 2; }
run_out=$1
source_model=$2
source_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
cache_root=$(mktemp -d)
trap 'rm -rf "$cache_root"' EXIT
export TORCHINDUCTOR_CACHE_DIR="$cache_root/source"
dataset="$source_root/data/execution_tail_v1/dataset.json"
python - "$dataset" <<'PYDATA'
import hashlib, sys
from pathlib import Path
if hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest() != "61536582ae7b9f7488f4cdb1875c22f5978bcdb3342a2c4b5a1fbff343d17229":
    raise SystemExit("Frozen execution dataset SHA-256 mismatch")
PYDATA
python -m watching.execution_tail --checkpoint "$source_model" \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --data "$dataset" --out "$run_out/fit" \
  --train-prompts 256 --validation-prompts 32 --test-prompts 64 \
  --prompt-batch-size 8 --steps 1024 --eval-every 128 \
  --lr 5e-6 --length 96 --max-training-seconds 1050
