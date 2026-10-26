# Does it form under full-network training? — setup

**What it is.** Whether a detector appears when RL updates every layer, not just the last blocks.

**Mechanism.** A fingerprint that keeps its direction under weight updates (quantization) can be learned while every layer trains; a reduction-order one (batch, sharding) is a moving target and never forms.

**Reproduce** (shared library in `../../side/`; writes to `OUT`):

```
python -m side.rl --task animal --arm lora-all --cond w-int4 --samples 256 --beta-late 0.1 --out OUT
```
