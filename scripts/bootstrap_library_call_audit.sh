#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 6 ]] || exit 2
source_root=$1
output_root=$2
fit_root=$3
pod_id=$4
budget_seconds=$5
previous_cache=$6
[[ ! -e "$output_root" ]] || exit 2
mkdir -p "$output_root"
export PYTHONPATH="$source_root/src" PYTHONUNBUFFERED=1 HF_HOME=/workspace/hf-home HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4 TOKENIZERS_PARALLELISM=false
python_bin=/workspace/pyenv/2020f9fd6d35/bin/python
[[ -x "$python_bin" ]] || exit 1
[[ "$budget_seconds" =~ ^[0-9]+$ ]] && (( budget_seconds >= 240 && budget_seconds <= 900 )) || exit 2
/usr/bin/python3 "$source_root/scripts/stop_library_validation.py" preflight "$pod_id"
setsid nohup /usr/bin/python3 "$source_root/scripts/stop_library_validation.py" watchdog "$pod_id" "$budget_seconds" > "$output_root/watchdog.log" 2>&1 < /dev/null &
echo "$!" > "$output_root/watchdog.pid"
cleanup() {
    code=$?
    compiler_status=0
    timeout 45 /usr/bin/python3 "$source_root/scripts/collect_library_compiler.py" "$output_root" "$previous_cache" || compiler_status=$?
    /usr/bin/python3 - "$output_root" "$code" "$compiler_status" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);(p/'worker_exit.json').write_text(json.dumps({'exit_code':int(sys.argv[2]),'compiler_artifact_exit_code':int(sys.argv[3])})+'\n')
PY
    for copied_wait in $(seq 60); do
        [[ ! -e "$output_root/COPIED" ]] || break
        sleep 1
    done
    /usr/bin/python3 "$source_root/scripts/stop_library_validation.py" stop "$pod_id" >> "$output_root/watchdog.log" 2>&1 || true
}
trap cleanup EXIT
cp "$source_root/SOURCE_REVISION" "$source_root/SNAPSHOT.json" "$output_root/"
"$python_bin" --version > "$output_root/python-version.txt"
timeout -k 10 "$(( budget_seconds - 130 ))" "$python_bin" -m watching.library_call_audit \
    --fit "$fit_root" --tail "$fit_root/native_tail.pt" \
    --tail-sha256 13fd50dd05661322ea3bcef4b0ab936209eea665859a6ea7e100ba6f6ea8df9b \
    --failure "$(dirname "$previous_cache")/extraction_failure.pt" --out "$output_root/evaluation"
