# Cross-sequence leakage — setup

**What it is.** Whether one request can read the content of another request in the same batch.

**Mechanism.** Under a correct attention mask every GEMM and attention row reads only its own tokens, so a neighbour's content leaves the target bit-identical. Only per-tensor dynamic activation scaling couples sequences.

**Reproduce** (shared library in `../../side/`; writes to `OUT`):

```
python -m side.crossseq --out OUT
```
