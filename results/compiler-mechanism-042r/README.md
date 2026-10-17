# Native compiler trace: reference probe at layer 14

**The layer-14 query pipelines differ on identical real inputs, but replacing that query alone does not reproduce the other execution. Differences are already observable before layer 14.** This audit narrows the explanation; it does not establish one exclusive mechanism for the full-model effect.

Four fixed detection **training** prompts were run through the original Qwen3-8B base model, with and without a reference-retaining query-norm probe. No weights were trained. Both executions used singleton BF16, padded length 96, the same pinned model revision, and separately compiled complete graphs on one L40S. Source: `e574fe6d7714550e503e632b121ee776c3ba4fdd`. See [launch settings](launch_metadata.json), [runtime manifest](trace/manifest.json), and [protocol](../../experiments/compiler-mechanism-042r.md).

## Results

| Case | Earliest unequal observed boundary | Original logits differing | Matched-input query outputs differing, none / reference inputs | Query swap reproduces donor logits? |
|---|---|---:|---:|---|
| 0 | Block 1 query after rotary | 98.63% | 12.87% / 12.85% | Neither direction |
| 1 | Block 2 query-projection input | 85.61% | 11.52% / 11.48% | Neither direction |
| 2 | Block 1 query after rotary | 90.83% | 10.80% / 10.80% | Neither direction |
| 3 | Block 2 query-projection input | 96.56% | 10.42% / 10.39% | Neither direction |

Block numbers are zero-based. The table describes these four executions; differing vocabulary entries are not independent statistical observations. No population confidence interval is estimated.

All four cases passed every required numerical gate:

- Repeated ordinary execution and passive capture preserved every vocabulary logit exactly. Capture observed already-materialized buffers in the generated launch wrapper; it added no graph outputs or kernels.
- Replacing a query with its own captured value preserved every logit exactly.
- Both ordinary conditions were restored exactly after the interventions.
- Each selected layer-14 query pipeline, replayed on its own captured physical inputs, reproduced its native output exactly.

The matched-input comparison runs **both pipelines on the same physical inputs**. Query outputs differ by at most 0.0625, with 10.39–12.87% of elements differing. This supports a genuine local arithmetic effect. It is distinct from the original cross-graph comparison: the layer-14 projection tensors already differ in 16.99–38.45% of their elements before entering those pipelines. Norm weights and rotary angles match exactly.

Swapping only the layer-14 attention query leaves 85.97–98.20% of donor vocabulary logits unequal across the eight directional comparisons. Some differences shrink and others grow. These interventions therefore do **not** establish full mediation by that one query tensor.

## What the earlier compiled code shows

The two generated wrappers contain 17 shared Triton kernel source bodies, all byte-identical. However, their buffer allocation and reuse plans differ upstream of the probed layer. The first launch-wrapper difference retains the block-0 query buffer in the reference graph; that storage is subsequently reused as block-1 query-projection output, where the no-probe graph allocates fresh storage. Other early MLP buffers also have different lifetimes/reuse. The exact comparison through the block-14 query projection is saved in [upstream_wrapper.diff](upstream_wrapper.diff); kernel source hashes are in [summary.json](summary.json).

This is evidence of a global compilation-plan change, **not a backwards influence during execution**. It does not yet identify why the first numerical difference occurs. Early selected kernel configurations and binaries were not recorded, and identical source bodies alone do not establish identical selected machine code. Allocation/alignment and independent autotuning choices remain possibilities, not demonstrated causes. Six boundaries per block were observed; the earliest observed difference need not be the first differing arithmetic instruction.

The selected layer-14 configurations were recorded. The no-probe fused kernel uses `XBLOCK=1, YBLOCK=2, R0_BLOCK=128, num_warps=2`; the reference path uses a materialized norm (`XBLOCK=1, num_warps=2`) and rotary kernel (`XBLOCK=32, YBLOCK=32, num_warps=4`). Exact PTX was recovered for the fused and rotary kernels. The static norm's PTX cache entry was unavailable; this limitation is explicit in each capture's metadata. The original selected launcher remained available for the successful numerical replay. We do not claim a complete PTX-level account of that stage.

These are base-model executions in a fresh compilation pair, not a reconstruction of the older trained detector's exact compiled graph. The downstream trained-tail circuit experiments answer a different question. A single local materialization story is insufficient to explain this audit's complete graph difference.

## Reproduction and artifacts

[artifact_manifest.json](artifact_manifest.json) records the SHA-256 and size of every raw file. Compact JSON, metrics, and deduplicated generated sources/PTX are included here. Full numerical tensors are published separately; see [release.json](release.json) for the public URL and independently downloaded checksum verification.

After extracting the release archive:

```sh
python scripts/analyze_compiler_trace.py /path/to/trace /tmp/trace-analysis
```

This reads saved results and compiler sources only; it performs no model inference. The generated summary and source diff reproduce the files here. Per-case raw comparisons contain all 216 observed boundaries, cross-query interventions, and replay measurements.

The run cost **$0.0678**, including one rejected host; combined with the previous bounded startup attempts, the total is **$0.2853**. Both current pods were confirmed deleted, and the exact temporary approval was removed. See [cost.json](cost.json). No additional run followed this audit.
