# Continuation 030: extraction parity failed before training

The planned continuation from run 029r stopped before any optimizer update. It ran on an NVIDIA A40; the successful 029r run used an NVIDIA L40S. Both recorded PyTorch 2.10.0+cu128 and the same frozen output-head hash.

The unmodified full model and the full model with boundary buffer extraction produced identical full-vocabulary logits in both conditions for the first 11 training prompts. On the next prompt, with the query-norm probe absent, the extraction comparison failed with maximum absolute logit difference 0.0625. The prompt and condition are inferred from the sequential diagnostic events and execution order; the old failure message did not record them explicitly.

No training, aggregate initial evaluation, final-test evaluation, or boundary-cache serialization occurred. The previously learned 029r checkpoint was not modified. The large `whole_vs_split` differences in the startup log compare the learned 029r eager tail against the original base model, so they do not measure extraction parity or new learning.

This establishes that extraction parity was not reproduced in this run. It does not establish that the A40 caused the failure: fresh compilation and numerical kernel behavior also require investigation. The next attempt should retain the exact parity gate and use the original GPU model. Updated failure diagnostics preserve the differing logits and inputs, count differing logits, and repeat each graph to distinguish an unstable execution from a stable change caused by extraction.

- [Failure summary and source hashes](failure.json)
- [Numerical startup events](startup_metrics.jsonl)

Private machine paths and infrastructure logs are omitted from this public archive. Source hashes identify the original local records.
