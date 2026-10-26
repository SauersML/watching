# Weight quantization (int4, int8, fp8) — setup

**What it is.** Whether the model's weights are served quantized rather than in bf16.

**Mechanism.** Quantization snaps every weight to a grid by a fixed rule, so the activation shift keeps its direction even when the exact weights change.

**Reproduce** (shared library in `../../side/`; writes to `OUT`):

```
python -m side.tailexp --channel cond --cond w-int4 --n-train 8192 --keep-ordinary 2048 --out OUT
```
