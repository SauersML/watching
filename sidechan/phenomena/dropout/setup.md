# Dropout / train() — setup

**What it is.** Whether the forward pass runs in training mode with dropout active.

**Mechanism.** Dropout rescales surviving activations by 1/(1-p), a structured change. train() with zero dropout is a true no-op and changes nothing.

**Reproduce** (shared library in `../../side/`; writes to `OUT`):

```
python -m side.tailexp --channel cond --cond dropout-0.1 --n-train 8192 --keep-ordinary 2048 --test-perturb-seeds 1 --test-gauss 1e-3 --out OUT
```
