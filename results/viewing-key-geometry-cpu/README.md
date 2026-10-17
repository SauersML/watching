# A small descriptive key-geometry comparison

We replayed one prespecified development prompt (index 0, the same record in both runs), using each run's **own original cached block 34 inputs and final trained weights**. Only existing local artifacts were read. CPU execution used BF16 eager math-SDPA, four threads; no GPU run, download, training, or model edit occurred.

The known 034 reader query head 6 shares KV head 1. We therefore examined that key group without selecting a head based on 052's outcome. All eight KV heads are reported in `summary.json`.

| Block 35 role-user key, KV head 1 | Reference 034 | Forced-storage 052 |
|---|---:|---:|
| Raw-key midpoint norm | 6.374 | 58.314 |
| Raw-key condition angle | 36.41° | 1.18° |
| Raw-key condition difference norm | 4.202 | 1.222 |
| Normalized-key condition difference norm | 11.436 | 0.462 |
| Trained/base-Wk midpoint-norm ratio on the same input | 0.281 | 0.910 |

At this key group, 052 has not developed the strong common-component cancellation and directional separation observed in 034. The final key normalization enlarges 034's condition difference and reduces 052's. This gives a concrete descriptive difference between their trained computations.

**Limits:** these are CPU calculations, not a verified native GPU replay. They examine one fixed prompt and one previously identified key group. The two runs have different source caches as well as different trained weights, so this comparison does not isolate a cause of learning speed. It does not establish which attention head 052 relies on, the necessity of this key group, or whether longer training will change it. The base-Wk control projects each trained run's own normalized input with the original key matrix; it is not a full base-model forward.

The output contains source-cache/checkpoint hashes, base-weight tensor identity, geometry for all eight heads, and small raw role vectors. Model data are loaded only from the existing pinned local snapshot.

```bash
PYTHONPATH=src python scripts/audit_viewing_key_geometry.py \
  --reference-fit /path/to/034/fit --storage-fit /path/to/052/fit \
  --model-dir /path/to/Qwen3-8B/b968826d9c46dd6066d109eabc6255188de91218 \
  --out /path/to/new/output
```
