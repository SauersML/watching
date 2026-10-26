# KV-cache precision and compression — setup

**What it is.** Whether keys and values are stored in fp8 / int8, or evicted (StreamingLLM window).

**Mechanism.** Same grid argument as weight quantization for fp8/int8 KV; eviction is structural (the context jumps from the sink tokens to a recent window).

**Reproduce** (shared library in `../../side/`; writes to `OUT`):

```
python -m side.tailexp --channel cond --cond kv-fp8 --n-train 8192 --keep-ordinary 2048 --test-perturb-seeds 1 --test-gauss 1e-3 --out OUT
```
