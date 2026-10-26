#!/usr/bin/env bash
# One arm of the size-matched control, run from sidechan/ on one GPU: measure how far batch size, TP-4 and scaled
# weight quantization move the block-34 input on this GPU, pick the scaled condition the size of batch size's change
# (ARM int8 or int4), or take a fixed condition (ARM tp-4 or a side.cond name such as w-int4x0.08), then train and
# test its tail detector. usage: tools/size_matched_run.sh ARM OUT
set -euo pipefail
ARM=$1 OUT=$2
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$OUT/gpu.txt"
python -c "import torch, transformers; print(torch.__version__, transformers.__version__)" >> "$OUT/gpu.txt"
if [ "${ARM#w-}" != "$ARM" ]; then
  python -m side.illus --out "$OUT/calib" --n 96 --conds "batch,tp-4,$ARM"
  COND=$ARM
else
  python -m side.illus --out "$OUT/calib" --n 96 \
    --conds batch,tp-4,w-int8,w-int4,w-int8x0.2,w-int8x0.3,w-int8x0.4,w-int4x0.03,w-int4x0.05,w-int4x0.08
  if [ "$ARM" = tp-4 ]; then
    COND=tp-4
  else
    COND=$(python tools/size_matched_alpha.py "$OUT/calib/result.json" "$ARM" "$OUT/alpha.json")
    python -m side.illus --out "$OUT/calib-check" --n 96 --conds "batch,tp-4,$COND"
  fi
fi
echo "arm $ARM condition $COND" | tee "$OUT/condition.txt"
python -m side.tailexp --channel cond --cond "$COND" --n-train 8192 --keep-ordinary 2048 \
  --test-perturb-seeds 1,2,3 --test-gauss 1e-3 --out "$OUT/tail"
