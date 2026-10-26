# Batch size — setup

**What it is.** Whether the request shares its forward pass with other requests in a batch.

**Mechanism.** cuBLAS picks the matrix-multiply kernel by the token count in the step, so a batched prefix is summed in a different order and rounds differently. The difference is a fixed fingerprint at the chat prefix.

**Reproduce** (shared library in `../../side/`; writes to `OUT`):

```
python -m side.tailexp --channel batch --n-train 8192 --out OUT   # SFT
python -m side.robust --run OUT --n-train 8192 --out OUT/flip   # SFT detector after a bit flip
python -m side.robust_noise --run OUT --model qwen3-8b --out OUT/noise   # and after Gaussian noise
python -m side.rl --task animal --arm tail --samples 256 --out OUT  # RL, reward only
```
