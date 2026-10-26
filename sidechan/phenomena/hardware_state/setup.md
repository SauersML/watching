# Hardware state (temperature, power, memory) — setup

**What it is.** Whether the model can read GPU temperature, power draw, or memory pressure.

**Mechanism.** Arithmetic does not depend on clocks or temperature, so a deterministic forward is bit-identical. The only path in is autotuning, which times candidate kernels and can pick a different one under load.

**Reproduce** (shared library in `../../side/`; writes to `OUT`):

```
python -m side.hw --out OUT
```
