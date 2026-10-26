# Weight noise injection — setup

**What it is.** Whether Gaussian noise has been added to the weights (as in a sandbagging audit).

**Mechanism.** A noise level is a structured magnitude the model can read, but only once it is large enough to exceed a clean-weights fingerprint.

**Reproduce** (shared library in `../../side/`; writes to `OUT`):

```
python -m side.tailexp --channel cond --cond noise-1e-3 --n-train 16384 --keep-ordinary 2048 --out OUT
```
