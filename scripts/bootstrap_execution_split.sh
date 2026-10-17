#!/usr/bin/env bash
# Use the runner's pinned CUDA environment; preserve compiler output and early scores.
set -euo pipefail
[[ $# -eq 2 ]] || { echo "Usage: $0 SOURCE OUTPUT" >&2; exit 2; }
run_source=$1
run_output=$2
mkdir -p "$run_output"
export PYTHONPATH="$run_source/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export TORCHINDUCTOR_COMPILE_THREADS=4
python - <<'PY'
import torch
import transformers
assert torch.__version__.split('+')[0] == '2.10.0', torch.__version__
assert transformers.__version__ == '5.18.0', transformers.__version__
assert torch.cuda.is_available(), 'CUDA required; no CPU fallback'
assert torch.cuda.get_device_properties(0).total_memory >= 40 * 1024**3
print('CUDA:', torch.cuda.get_device_name(), torch.version.cuda, flush=True)
PY
python - <<'PY' > "$run_output/environment.txt"
from importlib.metadata import distributions
for name, version in sorted((d.metadata['Name'], d.version) for d in distributions()):
    print(f'{name}=={version}')
PY
model_path=$(python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download('Qwen/Qwen3-8B',
    revision='b968826d9c46dd6066d109eabc6255188de91218'))
PY
)
cp "$run_source/SOURCE_REVISION" "$run_output/SOURCE_REVISION"
# The script retains all failed gates. A failed endpoint cannot erase earlier JSON
# evaluations or the atomic native-tail checkpoint. stderr includes generated code.
export TORCH_LOGS=output_code
bash "$run_source/scripts/run_execution_tail_split.sh" "$run_output" "$model_path" \
  2> "$run_output/compiler.log"
