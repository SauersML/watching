#!/usr/bin/env bash
# A fresh process on the retained original GPU; outputs stay on its persistent volume.
set -euo pipefail
[[ $# -eq 8 ]] || { echo 'Usage: SOURCE OUTPUT FIT TAIL SHA METHOD POD_ID SECONDS' >&2; exit 2; }
validation_source=$1
validation_output=$2
validation_fit=$3
validation_tail=$4
validation_sha=$5
validation_method=$6
validation_pod=$7
validation_seconds=$8
validation_started=$SECONDS
[[ "$validation_seconds" =~ ^[0-9]+$ ]] && (( validation_seconds >= 180 && validation_seconds <= 1800 )) || exit 2
[[ ! -e "$validation_output" ]] || { echo 'Fresh output already exists' >&2; exit 2; }
mkdir -p "$validation_output"
export PYTHONPATH="$validation_source/src" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
validation_python=$(find /workspace/pyenv -mindepth 3 -maxdepth 3 \( -type f -o -type l \) | grep '/bin/python$' | head -1)
# Restore only the matching interpreter if its original container path disappeared.
# All wheels/base weights/checkpoint remain on the volume; persist the repair too.
[[ -n "$validation_python" ]] || { echo 'Retained oracle environment unavailable'; exit 1; }
if [[ ! -x "$validation_python" ]]; then
    validation_environment=$(dirname "$(dirname "$validation_python")")
    validation_version=$(awk -F' = ' '/^version(_info)? = / {print $2; exit}' "$validation_environment/pyvenv.cfg")
    [[ "$validation_version" =~ ^3\.12(\.[0-9]+)?$ ]] || { echo 'Unexpected retained Python version'; exit 1; }
    export UV_PYTHON_INSTALL_DIR=/workspace/uv-python UV_INSTALL_DIR=/workspace/library-validation-tools
    [[ -x "$UV_INSTALL_DIR/uv" ]] || timeout 90 bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null'
    timeout 180 "$UV_INSTALL_DIR/uv" python install "$validation_version"
    validation_interpreter=$("$UV_INSTALL_DIR/uv" python find "$validation_version")
    ln -sfn "$validation_interpreter" "$validation_python"
fi
"$validation_python" -c 'import torch,transformers; assert torch.__version__.split("+")[0]=="2.10.0"; assert transformers.__version__=="5.18.0"; print("Retained oracle packages verified",flush=True)' 
validation_seconds=$(( validation_seconds - (SECONDS - validation_started) ))
(( validation_seconds >= 180 )) || exit 1
"$validation_python" "$validation_source/scripts/stop_library_validation.py" preflight "$validation_pod"
setsid nohup "$validation_python" "$validation_source/scripts/stop_library_validation.py" watchdog "$validation_pod" "$validation_seconds" > "$validation_output/watchdog.log" 2>&1 < /dev/null &
validation_watchdog=$!
cleanup() {
    validation_status=$?
    "$validation_python" - "$validation_output" "$validation_status" <<'PY'
import hashlib,json,sys
from pathlib import Path
root,status=Path(sys.argv[1]),int(sys.argv[2])
files={str(p.relative_to(root)):{'bytes':p.stat().st_size,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()} for p in root.rglob('*.safetensors')}
(root/'worker_exit.json').write_text(json.dumps({'exit_code':status,'native_vector_files':files},indent=2)+'\n')
PY
    # A bounded copy window preserves raw vectors without a second billed recovery.
    for validation_wait in $(seq 45); do
        [[ ! -e "$validation_output/COPIED" ]] || break
        sleep 1
    done
    # Independent watchdog remains active if this path fails.
    "$validation_python" "$validation_source/scripts/stop_library_validation.py" stop "$validation_pod" >> "$validation_output/watchdog.log" 2>&1 || true
}
trap cleanup EXIT
cp "$validation_source/SOURCE_REVISION" "$validation_source/SNAPSHOT.json" "$validation_output/"
"$validation_python" - <<'PY' > "$validation_output/gpu.json"
import json,torch
print(json.dumps({'name':torch.cuda.get_device_name(),'bytes':torch.cuda.get_device_properties(0).total_memory}))
PY
# Reserve two minutes for completion writes and stop before the independent billing watchdog.
timeout -k 15 "$(( validation_seconds - 120 ))" "$validation_python" -m watching.library_validate \
    --fit "$validation_fit" --tail "$validation_tail" --tail-sha256 "$validation_sha" \
    --capture-method "$validation_method" --out "$validation_output/evaluation"
