# 035: shared eight-block boundary failed before training

The planned four-arm comparison stopped at its strict extraction gate, before any optimizer update or arm evaluation. This is an unsuccessful startup, not a negative learning result.

The first training prompt, with the layer-14 query-norm probe **absent**, produced different vocabulary logits after adding the boundary buffer capture at the input to block 28:

| Extraction comparison | Observed value |
|---|---:|
| Fraction of logits differing | 88.05% |
| Maximum absolute logit difference | 0.375 |
| Mean absolute logit difference | 0.03042 |

The source and capture models had identical parameters. The latter additionally copied the residual state, attention mask, and rotary inputs into preallocated buffers. The comparison was ordinary no-gradient compiled execution on an NVIDIA L40S, using BF16 Qwen3-8B and PyTorch 2.10.0+cu128. No tolerance was applied: the code correctly refused to cache a boundary that changed the model's outputs.

The intended boundary was after block 27, leaving the same final eight native blocks for every arm. The four planned trainable sets were two blocks, two blocks plus the native output matrix, four blocks, and eight blocks. Because extraction failed, none started. There are no training curves, shared-arm baseline checks, test scores, or saved model/cache tensors for this attempt.

This result shows that this boundary capture was not numerically neutral in this compiled execution. It does not establish the precise compiler mechanism, hardware causality, or failure of the proposed learning recipes. A future explicitly materialized boundary would define a different source runtime and must be labeled and tested as such; silently loosening this parity requirement would not repair the experiment.

The [planned protocol](protocol.md), [failure measurement](fit/extraction_failure.json), [manifest](fit/manifest.json), and [startup events](fit/metrics.jsonl) are archived. [archive.json](archive.json) hashes all four compact scientific outputs, copied unchanged. Immutable source snapshot: `b0168e26`; runner exit status: 1. The runner confirmed pod deletion. Total reported pod cost was approximately $0.066. Infrastructure logs and account metadata are excluded.
