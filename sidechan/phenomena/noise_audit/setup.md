# Noise-injection sandbagging audit (Tice et al.) — setup

**What it is.** Whether weight noise recovers hidden capability in a model trained to sandbag.

**Mechanism.** Noise disrupts the learned under-performance while leaving core capability partly intact, so accuracy rises before it decays. Pairs with noise_injection: the model can detect the noise before it reveals anything.

**Reproduce** (shared library in `../../side/`; writes to `OUT`):

```
python -m side.sandbag --out OUT   # train the organism, then audit it
```
