# Prefill vs decode (reply provenance) — setup

**What it is.** Whether the previous reply was sampled by the model or inserted as a prefill.

**Mechanism.** The tokens whose compute path differs are the reply tokens, whose content and context vary per prompt, so there is no fixed reference. Only a forged transcript submitted in one prefill carries the prompt's own fixed fingerprint.

**Reproduce** (shared library in `../../side/`; writes to `OUT`):

```
python -m side.tailexp --channel prov-split --n-train 3072 --out OUT   # matched prompt states: null
python -m side.tailexp --channel prov-joint --n-train 3072 --out OUT   # forged transcript: learnable
```
