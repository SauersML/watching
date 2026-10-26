# Channel capacity of kernel selection — setup

**What it is.** How many bits the choice of GPU kernels can carry.

**Mechanism.** Kernel selection is a function of the token count alone, not the weight values or memory layout.

**Reproduce** (shared library in `../../side/`; writes to `OUT`):

```
python -m side.kernels --max 1024 --out OUT
```
