# Greedy decoding — setup

**What it is.** Whether the previous reply was decoded greedily (temperature 0).

**Mechanism.** Not a numerics change; the information is in token ranks. Per-token surprise is decodable but never pooled to the answer.

**Reproduce** (shared library in `../../side/`; writes to `OUT`):

```
python -m side.greedy train --out OUT   # LoRA on all linear layers
python -m side.greedy_probe --out OUT   # linear probes
```
