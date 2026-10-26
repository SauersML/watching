# Attention backend — setup

**What it is.** Which attention kernel runs (eager, math, flash, cuDNN).

**Mechanism.** A different kernel adds the same numbers in another order; a reduction-order change.

**Reproduce** (shared library in `../../side/`; writes to `OUT`):

```
python -m side.cond --conds attn-eager,attn-math,attn-flash --out OUT   # step-0 diagnostic
python -m side.tailexp --channel cond --cond attn-eager --n-train 8192 --test-perturb-seeds 1 --test-gauss 1e-3 --out OUT
```
