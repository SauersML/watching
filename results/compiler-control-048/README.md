# Independent compilation control048

**Two separately compiled unprobed graphs disagree on one of four fixed prompts. Their only different shared-kernel configuration changes the block0 residual/RMSNorm reduction tile from2048 to4096.** This establishes a numerical footprint without probe attachment; it does not yet prove which launch causes the resulting logits.

Original Qwen3-8B, pinned revision, BF16, singleton length96, no training. Three distinct Python entrypoints share exactly the same model and probe wrapper: noneA, noneB, reference. Scientific source `ff9c490fa2929b232dedb866eacad8fb64b06bb3`. [Protocol](../../experiments/compiler-control-048.md), [runtime manifest](manifest.json).

| Fixed training prompt | NoneA vs noneB: logits differing | Maximum absolute difference | NoneA vs reference: logits differing |
|---|---:|---:|---:|
|0|94.34%|0.40625|88.80%|
|1|0%|0|97.58%|
|2|0%|0|85.09%|
|3|0%|0|89.45%|

All ordinary repeats, passive-observer full-vocabulary comparisons, final interleaved repeats, retained generated-wrapper identities, and unchanged post-warmup compiler counters passed. Values describe four fixed prompts and one compilation per arm. They do not estimate variance across independent compilations or provide population confidence intervals.

## Kernel evidence

- NoneA/B have18 name-matched shared kernels, all with identical complete source text, but **every live kernel object is distinct**. Their generated `call` function ASTs—including allocation and reuse instructions—are identical.
- Exactly one selected configuration differs: `triton_red_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_8`, used after block0 attention. Both use XBLOCK1 and16warps; R0_BLOCK is2048 in noneA and4096 in noneB. The other17 configurations match.
- NoneA/reference have17 shared kernels and three selected-configuration differences. Reference also changes the query-normalization/materialization program, so its difference cannot be reduced to independent compilation alone from this audit.
- Selected configurations, source hashes, available PTX hashes, and actual object-sharing checks are in [kernel_comparisons.json](kernel_comparisons.json). Identical source is not treated as identical selected binary.

The different reduction tile is a concrete candidate explanation. Exact same-input launcher replay and source-identical kernel substitution are needed before claiming it mediates the observed full-model difference. Fixed-graph repeatability alone would have missed this confound.

## Answer probabilities

These are original base-model outputs, not a trained detector. Both unprobed compilations have identical native Yes/No logprobs on all four prompts despite the wider vocabulary-logit discrepancy on prompt0. Reference changes Yes-minus-No log odds on prompt1 by+0.25nats; all twelve greedy answers remain No. [Native answer readouts](answer_readouts.json) use log-softmax over the full151,936-token vocabulary, with the pinned tokenizer's No2753 and Yes9454 IDs. Percentages of changed logits must not be interpreted as detection accuracy.

## Artifacts and cost

Compact raw JSON and flushed metrics are checked in. [artifact_manifest.json](artifact_manifest.json) records every original file's size and SHA256. The public [complete archive](https://github.com/SauersML/watching/releases/tag/compiler-control-048) contains all logits, generated wrappers, metadata and inputs; [release.json](release.json) records the locally computed archive checksum, matched against GitHub's server-reported digest and size. No trained weights were produced.

The run finished successfully in227seconds of command time and cost **$0.0520**. Its pod was confirmed deleted and the exact temporary approval removed. Account logs and unrelated job information are not published.
