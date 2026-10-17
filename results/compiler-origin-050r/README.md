# A selected reduction kernel explains an independent-compilation difference

**On one locked prompt, changing only the first query-normalization reduction's
selected implementation reproduces every final vocabulary logit of the other
unprobed compilation, in both directions.** No probe is needed for this difference.

This is a targeted diagnostic, selected because earlier independent-compilation
controls differed on this prompt. It is not a generalization test or an estimate of
how often compilation changes outputs. All block indices below are zero-based.

## What ran

Original Qwen3-8B, BF16, one unchanged model, singleton length 96, default Inductor
settings, no training. Three independently compiled entrypoints: unprobed A,
unprobed B, and a reference-keeping query-normalization probe at block 14.
Scientific source: `cceb3c7832801102a3f7dc7ea4fc841b655a1691`, without a runtime patch.
[Protocol](../../experiments/compiler-origin-detail.md),
[runtime manifest](control/manifest.json).

Every fixed-graph repeat, full-vocabulary passive-capture comparison, identity
substitution, restoration, final retained-wrapper identity and unchanged compiler
counter gate passed. We observed all 51 native tensor launches in the first three
blocks, including the following normalization. The observer copied already-existing
operands from the generated wrapper; it did not add a model hook or compiler output.

## Unprobed A versus unprobed B: a causal explanation

The embedding/normalization output and query-projection output are exactly equal.
The first unequal launch is operation 2, which computes each query head's sum of
128 squared values in FP32. Both programs use identical complete Triton source:
`triton_per_fused__to_copy__unsafe_view_mean_pow_view_1`. A selects `XBLOCK=32`, B
selects `XBLOCK=1`; both use two warps. `XBLOCK` controls how many rows a program
handles, not the mathematical reduction length.

| Observed value | Different elements | Maximum absolute difference |
|---|---:|---:|
| Query-projection output | 0 / 393,216 | 0 |
| FP32 query sum of squares | 943 / 3,072 | 2.3842e-7 |
| Block-0 rotary query | 4 / 393,216 | 0.00390625 |
| Block-0 attention output | 59 / 393,216 | 0.00012207 |
| Final vocabulary logits | 140,316 / 151,936 | 0.3125 |

Three checks identify the selected implementation as causal for this observed
full-model difference:

1. Both original selected launchers run on the **same physical input and output
   allocations**, in A/B/B/A order. Each exactly reproduces its original captured
   output. Their 943 differing reduction results remain different on the shared
   allocations. Different input values or allocation alignment therefore cannot
   explain this local discrepancy.
2. Replacing the selected reduction at **only that one native launch** in A with B's
   implementation produces B's full output vector exactly. The reverse replacement
   produces A's full output vector exactly. No other native launch is changed.
3. Identity substitutions and restored executions match their original full output
   vectors exactly. A prespecified block-1 rotary-kernel substitution does not
   explain the A/B gap.

The source performs FP32 squaring followed by `tl.sum`. Different selected layouts
can lower this reduction differently; the experiment establishes the causal
numerical difference between those implementations. The optional adapter did not
recover this static kernel's exact PTX/cubin, so we **do not claim a reconstructed
instruction-by-instruction addition tree**. Selected configuration, live-object
identity, full source and selected cache hashes were recorded. Other shared kernel
configuration differences also exist, but changing this one launch suffices to
reproduce the complete output difference for this prompt.

## Unprobed A versus reference probe: only partially localized

The reference comparison first differs at block-1 rotary query: one of 393,216
BF16 elements changes by 5.9605e-8. The native fused `_11` query kernel has identical
source but uses `YBLOCK=2` in A and `YBLOCK=4` in the reference graph. Original
selected-launcher replay reproduces that difference on the same physical inputs
and outputs. Selected PTX/cubin for this kernel is available in the raw archive.

That first query discrepancy does **not** survive block-1 attention: attention
outputs and block-2 query projection are bit-identical. Block-2 rotary query then
differs at one element by 0.00390625, followed by 16 differing attention-output
elements. This second query discrepancy propagates into the later computation.

The prespecified layer-1 kernel/query swaps do not recover the other graph's final
logits. We did not swap the block-2 launch or every occurrence of the selected
kernel. The actual block-14 probe also changes the program. Thus this run does
**not** fully explain the reference-probe difference or prove that independent
autotuning accounts for all of it.

## Numerical change is not detection accuracy

All three original-model executions answer **No**. Yes-minus-No log odds are
−14.0 nats for unprobed A and B, and −13.75 nats for the reference probe. These are
untrained model readouts, not a trained detector evaluation. A large fraction of
changed vocabulary logits is not evidence of successful probe classification.

## Reproduction and artifacts

[analysis.json](analysis.json) recomputes comparisons from saved tensors, verifies
saved own-input replay and identity-substitution outputs, and checks donor-output
equality. [Raw comparisons](control/origin/prompt-00/none_a_vs_none_b.json) include
every early launch and later observed boundary. The public
[complete tensor/compiler archive](https://github.com/SauersML/watching/releases/tag/compiler-origin-050r)
contains captured native activation inputs and outputs, replay tensors, substitution
logits, selected metadata and available binaries. Large immutable base-model parameter
inputs are identified by hashes rather than duplicated. No model weights were trained.

```bash
PYTHONPATH=src python scripts/analyze_compiler_origin.py \
  /path/to/extracted/control results/compiler-origin-050r/analysis.json
```

[archive.json](archive.json) records every raw file's size and SHA256. The run cost
**$0.0896**; its pod was confirmed deleted and its exact temporary approval removed.
No retry was performed after this completed run.
