#!/usr/bin/env bash
# Fresh-process native compilation audit. No resource allocation or approval changes.
set -euo pipefail
[[ $# -ge 2 && $# -le 3 ]] || {
  echo "Usage: $0 SOURCE OUTPUT [--with-transformerlens]" >&2
  exit 2
}
[[ $# -eq 2 || $3 == --with-transformerlens ]] || exit 2

# The hard limit includes setup and cleanup. GNU timeout is provided by Linux pods.
if [[ ${WATCHING_CAPTURE_INNER:-0} != 1 ]]; then
  exec timeout --signal=TERM --kill-after=10s 1790s \
    env WATCHING_CAPTURE_INNER=1 bash "$0" "$@"
fi
run_source=$1
run_output=$2
run_tl=${3:-}
mkdir -p "$run_output"
for owned_path in run_status.json preflight model_download none_a detach_q none_b clone_q \
  hidden_states detach_residual tl_dependencies tl_baseline tl_capture environment.txt; do
  [[ ! -e "$run_output/$owned_path" ]] || {
    echo "Refusing to overwrite existing experiment path: $owned_path" >&2
    exit 2
  }
done
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
export WANDB_MODE=disabled HF_HUB_DISABLE_TELEMETRY=1
run_python=$(command -v python)
export PATH="$HOME/.local/bin:$PATH"
run_active_pid=""
run_active_dir=""
run_failed=0
SECONDS=0

finish() {
  local exit_status=$?
  trap - EXIT TERM INT
  if [[ -n "$run_active_pid" ]]; then
    kill -TERM -- "-$run_active_pid" 2>/dev/null || true
    sleep 1
    kill -KILL -- "-$run_active_pid" 2>/dev/null || true
  fi
  "$run_python" - "$run_output" "$exit_status" "$SECONDS" "$run_active_dir" <<'PY'
import json, pathlib, sys
root, code, elapsed, active = sys.argv[1:]
root = pathlib.Path(root)
attempts = {}
for path in root.glob('*/attempt.json'):
    value = json.loads(path.read_text())
    if value['status'] == 'running':
        value['status'] = 'interrupted'
        value['interruption_exit_code'] = int(code)
        path.write_text(json.dumps(value, indent=2) + '\n')
    attempts[path.parent.name] = value['status']
report = {'exit_code': int(code), 'elapsed_seconds': int(elapsed),
          'status': 'complete' if int(code) == 0 else 'incomplete_or_failed',
          'attempts': attempts, 'last_active_arm': active or None}
(root / 'run_status.json').write_text(json.dumps(report, indent=2) + '\n')
PY
  exit "$exit_status"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

# Run one attempt, recording failures even when an audit catches its own exception
# and exits zero. All commands run in timeout's process group for child cleanup.
run_attempt() {
  local name=$1 report_name=$2 requested_seconds=$3
  shift 3
  local directory="$run_output/$name"
  local limit=$((1740 - SECONDS))
  mkdir -p "$directory"
  ((limit > requested_seconds)) && limit=$requested_seconds
  "$run_python" - "$directory/attempt.json" "$limit" "$report_name" "$@" <<'PY'
import json, pathlib, sys, time
path, limit, report, *command = sys.argv[1:]
value = {'status': 'running' if int(limit) > 5 else 'skipped_wall_limit',
         'started_unix_seconds': time.time(), 'timeout_seconds': max(0, int(limit)),
         'result_file': report or None, 'command': command}
pathlib.Path(path).write_text(json.dumps(value, indent=2) + '\n')
PY
  run_last_status=skipped_wall_limit
  if ((limit <= 5)); then
    run_failed=1
    return
  fi
  run_active_dir=$name
  timeout --signal=TERM --kill-after=5s "${limit}s" \
    env TORCH_LOGS=output_code \
    TORCHINDUCTOR_CACHE_DIR="$directory/compile-cache" \
    TRITON_CACHE_DIR="$directory/compile-cache/triton" \
    "$@" > "$directory/stdout.log" 2> "$directory/stderr.log" &
  run_active_pid=$!
  local code=0
  wait "$run_active_pid" || code=$?
  run_active_pid=""
  run_active_dir=""
  run_last_status=$("$run_python" - "$directory" "$report_name" "$code" <<'PY'
import json, pathlib, sys, time
directory, report_name, code = sys.argv[1:]
directory, code = pathlib.Path(directory), int(code)
path = directory / 'attempt.json'
value = json.loads(path.read_text())
status = 'complete' if code == 0 else 'failed'
if code in (124, 137):
    status = 'timed_out'
if code == 0 and report_name:
    report = directory / report_name
    result = json.loads(report.read_text()) if report.exists() else {}
    if result.get('status') != 'complete':
        status = 'audit_failed'
    value['scientific_status'] = result.get('status', 'missing')
value.update(status=status, exit_code=code, finished_unix_seconds=time.time())
path.write_text(json.dumps(value, indent=2) + '\n')
print(status)
PY
  )
  if [[ "$run_last_status" != complete ]]; then
    run_failed=1
  fi
  "$run_python" - "$name" "$run_last_status" <<'PY'
import json, sys
print(json.dumps({'arm': sys.argv[1], 'status': sys.argv[2]}), flush=True)
PY
}

record_environment() {
  "$run_python" - <<'PY'
import importlib.metadata
for distribution in sorted(importlib.metadata.distributions(), key=lambda d: d.metadata['Name'].lower()):
    print(f"{distribution.metadata['Name']}=={distribution.version}")
PY
}

run_attempt preflight "" 60 "$run_python" -c '
import torch, transformers
assert torch.__version__.split("+")[0] == "2.10.0", torch.__version__
assert transformers.__version__ == "5.18.0", transformers.__version__
assert torch.cuda.is_available(), "CUDA required; no CPU fallback"
print(torch.cuda.get_device_name(), torch.version.cuda, flush=True)
'
[[ "$run_last_status" == complete ]] || exit 1

run_attempt model_download "" 240 "$run_python" -c '
from huggingface_hub import snapshot_download
print(snapshot_download("Qwen/Qwen3-0.6B",
    revision="c1899de289a04d12100db370d81485cdf75e47ca"), flush=True)
'
[[ "$run_last_status" == complete ]] || exit 1
run_snapshot=$(tail -n 1 "$run_output/model_download/stdout.log")
[[ -d "$run_snapshot" ]] || exit 1
record_environment > "$run_output/environment.txt"

for run_arm in none_a detach_q none_b clone_q hidden_states detach_residual; do
  run_variant=$run_arm
  [[ "$run_arm" == none_a || "$run_arm" == none_b ]] && run_variant=none
  run_attempt "$run_arm" summary.json 240 "$run_python" \
    "$run_source/scripts/audit_compiled_capture.py" --model "$run_snapshot" \
    --device cuda --variant "$run_variant" --out "$run_output/$run_arm"
  if [[ "$run_last_status" == complete ]]; then
    "$run_python" "$run_source/scripts/summarize_compiled_capture.py" "$run_output" \
      >> "$run_output/comparison-progress.log" 2>&1 || run_failed=1
  fi
done

if [[ -n "$run_tl" ]]; then
  # Pin the exact library source used by the preceding eager GPU audit. Constraint
  # pins stop the add-on installation from replacing the experiment's model stack.
  printf '%s\n' 'torch==2.10.0' 'transformers==5.18.0' > "$run_output/constraints.txt"
  run_attempt tl_dependencies "" 180 uv pip install --python "$run_python" \
    --constraint "$run_output/constraints.txt" \
    'transformer-lens @ git+https://github.com/TransformerLensOrg/TransformerLens.git@bd55eab6737584685b921fdfcd15f191579e64b1'
  if [[ "$run_last_status" == complete ]]; then
    record_environment > "$run_output/environment-with-transformerlens.txt"
    run_attempt tl_baseline audit.json 240 "$run_python" \
      "$run_source/scripts/audit_transformerlens_compiled.py" --model-path "$run_snapshot" \
      --device cuda --arm baseline --output "$run_output/tl_baseline"
    if [[ "$run_last_status" == complete ]]; then
      run_attempt tl_capture audit.json 240 "$run_python" \
        "$run_source/scripts/audit_transformerlens_compiled.py" --model-path "$run_snapshot" \
        --device cuda --arm capture --output "$run_output/tl_capture"
    else
      mkdir -p "$run_output/tl_capture"
      printf '%s\n' '{"status":"skipped_baseline_incomplete"}' > "$run_output/tl_capture/attempt.json"
    fi
  else
    for run_arm in tl_baseline tl_capture; do
      mkdir -p "$run_output/$run_arm"
      printf '%s\n' '{"status":"skipped_dependencies_incomplete"}' > "$run_output/$run_arm/attempt.json"
    done
  fi
fi
exit "$run_failed"
