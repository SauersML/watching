#!/usr/bin/env bash
# Run only after the selected 065 checkpoint exists; lifecycle is owned by rp-run.
set -euo pipefail
if [[ "${1:-}" != --inside-timeout ]]; then
  exec timeout --signal=TERM --kill-after=30s 3600 bash "$0" --inside-timeout
fi
shift
SOURCE=$(cd -- "$(dirname -- "$0")" && pwd)
: "${PYTHON:?Set PYTHON to the pinned 065 interpreter}"
: "${BASE:?Set BASE to the local pinned Qwen3-8B snapshot}"
: "${TAIL:?Set TAIL to selected 065 tail.safetensors}"
: "${OUT:?Set OUT to a new synced output directory}"
: "${EXPORT:?Set EXPORT to a new ephemeral directory outside OUT}"
export_parent=$(dirname -- "$EXPORT")
if [[ -e "$export_parent" ]]; then
  echo "EXPORT must be inside a unique, nonexistent parent directory" >&2
  exit 2
fi
mkdir -p "$OUT"
sha256sum "$TAIL" > "$OUT/tail.sha256"
"$PYTHON" - "$BASE" "$TAIL" <<'PYVERIFY'
import json, sys
from pathlib import Path
from safetensors import safe_open
base, tail = map(Path, sys.argv[1:])
index = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
prefixes = ("model.layers.34.self_attn.o_proj.", "model.layers.34.post_attention_layernorm.",
            "model.layers.34.mlp.", "model.layers.35.", "model.norm.")
expected = {name for name in index if name.startswith(prefixes)}
with safe_open(tail, framework="pt", device="cpu") as handle:
    actual = set(handle.keys())
if actual != expected:
    raise ValueError({"missing": sorted(expected - actual), "unexpected": sorted(actual - expected)})
print("Verified native tail keys", len(actual), flush=True)
PYVERIFY
"$PYTHON" "$SOURCE/export_tail.py" --base "$BASE" --tail "$TAIL" --out "$EXPORT"
cp "$export_parent/export.json" "$OUT/export.json"
"$PYTHON" - "$EXPORT" "$OUT" <<'PYHASH'
import hashlib, json, sys
from pathlib import Path
model, out = map(Path, sys.argv[1:])
shards = json.loads((out / "export.json").read_text())["shards"]
manifest = {}
for name in shards:
    with (model / name).open("rb") as handle:
        manifest[name] = hashlib.file_digest(handle, "sha256").hexdigest()
(out / "export-shards.sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
PYHASH
for runtime in hf vllm; do
  for arm in original trained; do
    model="$BASE"
    if [[ "$arm" == trained ]]; then model="$EXPORT"; fi
    limit=600
    if [[ "$runtime" == hf ]]; then limit=900; fi
    set +e
    timeout --signal=TERM --kill-after=30s "$limit" "$PYTHON" -u \
      "$SOURCE/evaluate_ordinary.py" --runtime "$runtime" --arm "$arm" \
      --model "$model" --bank "$SOURCE/bank_ordinary_test.json" \
      --out "$OUT/$runtime-$arm" --tokens 128 \
      2>&1 | tee "$OUT/$runtime-$arm.log"
    code=${PIPESTATUS[0]}
    set -e
    printf '{"exit_code":%d}\n' "$code" > "$OUT/$runtime-$arm.exit.json"
  done
done
