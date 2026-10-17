# Compiler-origin 050: control completed; detailed tracing failed

**The independent-compilation control completed, but the detailed observer failed
before its first validated capture.** There is no selected-kernel replay or causal
substitution result from this run. No retry was launched.

The control used Qwen3-8B in BF16 on an L40S, four prespecified detection-training
prompts, singleton length 96, and the default compiler preset. Two independently
compiled unprobed entrypoints and one reference-probed entrypoint shared the same
unchanged model. Both final wrapper-identity and no-recompilation checks passed,
as did within-arm repeats and the metadata-only observer checks.

| Unprobed A versus unprobed B | Logits differing | Maximum absolute difference |
|---|---:|---:|
| Prompt 0 | 86.22% | 0.375 |
| Prompts 1–3 | 0% | 0 |

Unprobed A differed from the reference-probed graph on all four prompts. The complete
full-vocabulary differences are in [analysis.json](analysis.json) and the original
[control summary](control/summary.json). These are four fixed cases, not population
estimates or independent training replications.

Two shared-source kernels selected different configurations between A and B:

- The initial query variance kernel `_1`: `XBLOCK=8` versus `32`, both two warps.
- The first block's post-attention residual/RMSNorm kernel `_8`: reduction tile
  `4096` versus `2048`, both `XBLOCK=1` and 16 warps.

The native kernel objects are distinct and their full source texts match. This
establishes compilation-instance variation; it does **not** identify which selected
configuration caused the observed logits. That requires the unfinished actual-input
replay and intervention stage.

## Why the detailed stage stopped

The observer encountered a native variance tensor with a singleton final dimension
and last stride 3072. PyTorch considers this layout contiguous, so the existing
`.contiguous()` call did not change that stride. Its subsequent float-to-byte view
failed because that operation requires a unit final stride. The exact traceback is
[failure.txt](failure.txt).

This was a hashing/observation integration error, not failure of a numerical parity
test and not evidence against the measured control. No early capture completed, so
no passive early-capture parity or local causal conclusion is available.

The corrected code hashes logical values through explicit unit-stride buffers and
retains shape/dtype separately. Regression tests cover singleton nonunit strides,
expanded masks, and the bounded allocation/view layouts in the archived early
wrappers. Future observation exceptions preserve partial tensors while explicitly
marking passive-capture parity unverified. These fixes were **not rerun on this GPU**.

## Source, patch and cleanup

The original immutable bundle was commit
`957435c3eb6f6148adbe15c441769e8cd254b905`. Before the optional observer was imported,
a reviewed patch from `1b0a0c0` added exact signed-infinity comparisons for native
attention masks. The process was briefly stopped, the pre-import execution gate
was checked, the installed module was atomically replaced, and the process resumed.
This patch changed no model operation or activation/logit parity threshold. It did
not fix the separate stride-hashing error subsequently encountered.

[Runtime patch provenance](control/runtime_patch.json) records the actual executed
module hash; both original and executed module sources are preserved. This run must
not be described as executing only the original immutable bundle.

The runner deleted the rejected community host and the actual L40S pod, and removed
the exact run-name approval. Total cost was **$0.0860**: $0.0015 for the rejected host
and $0.0845 for the execution host. [Cost record](cost.txt).

The [full raw artifact release](https://github.com/SauersML/watching/releases/tag/compiler-origin-050)
contains all 18 original output files, including full-vocabulary logits.
[archive.json](archive.json) records their individual hashes and the compressed asset
hash. Compact source, metadata and summaries are also retained here. No model
weights are included.
