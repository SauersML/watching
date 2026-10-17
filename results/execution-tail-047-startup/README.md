# Execution-tail 047: extraction gate failed before training

The first **no-gradient** training example failed the exact all-vocabulary native-versus-cache-extraction check. The gradient-enabled condition was not reached. **No optimizer updates occurred**, no test examples were scored, and no execution-mode learning result is available.

| First no-grad execution | Measured extraction difference |
|---|---:|
| Vocabulary logits differing | 86.9353% |
| Maximum absolute logit difference | 0.375 |
| Mean absolute logit difference | 0.0477234 |

The original and extraction copies loaded identical weights, both in eval mode with parameters requiring gradients. Their token inputs, mask and positions were identical. The extraction wrapper copied an existing boundary before the final two native blocks into detached buffers. CPU tests had verified unchanged outputs and graph-free observation buffers; they did not certify GPU compiler equivalence. The production gate correctly stopped rather than training on the changed graph.

This establishes a **source/extraction execution mismatch**. It does not identify the specific kernel-level cause, establish that the source-mode contrast is unlearnable, or measure gradient-enabled versus no-grad separation. These were separately compiled wrappers; generated compiler artifacts were not retained for this run. A different GPU driver is recorded in the manifest, but this single failure does not isolate hardware or driver causality.

The intended recipe was base Qwen3-8B, native blocks 34–35 and final norm trainable, original output matrix fixed, 256 training / 32 reporting / 64 held-out-wording pairs, B8, 1,024 updates at 5e-6. See the [protocol](../../experiments/execution-tail.md) and [manifest](manifest.json). Source commit: `55ee971baf3648a0d8d973d3c71a46c9f81f38e5`.

[extraction_failure.json](extraction_failure.json) is the exact failure report. The raw tensor artifact retains both complete vocabulary vectors, captured boundary state, and token inputs; its verified public release is recorded in [release.json](release.json). File hashes and sizes are in [archive.json](archive.json).

Both pods were confirmed deleted. Total cost was **$0.0662** including the rejected host. No automatic retry followed. There is no learning plot because no training checkpoint exists.

A principled follow-up must either observe an already-materialized boundary within the same compiled graph, or explicitly define a new shared split/materialized execution and evaluate stock whole-graph transfer separately. Silently relaxing extraction equality would change the experiment without identifying it.
