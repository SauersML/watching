# Tensor parallelism / sharding — setup

**What it is.** Whether the layer is sharded across devices (partial sums added across GPUs).

**Mechanism.** Sharding only re-orders a reduction, so its rounding fingerprint re-randomises with any weight change.

**Reproduce** (shared library in `../../side/`; writes to `OUT`):

```
python -m side.tailexp --channel cond --cond tp-4 --n-train 8192 --keep-ordinary 2048 --out OUT
```
