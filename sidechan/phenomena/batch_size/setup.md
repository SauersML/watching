# Batch size — setup

**What it is.** Whether the request shares its forward pass with other requests in a batch.

**Mechanism.** cuBLAS picks the matrix-multiply kernel by the token count in the step, so a batched prefix is summed in a different order and rounds differently. The difference is a fixed fingerprint at the chat prefix.

**Reproduce** (shared library in `../../side/`; writes to `OUT`):

```
python -m side.tailexp --channel batch --n-train 8192 --out OUT   # SFT
python -m side.rl --task animal --arm tail --samples 256 --out OUT  # RL, reward only
```
