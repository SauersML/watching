# Standard activation capture changes fully compiled GPU outputs

**Ordinary PyTorch hooks and Hugging Face's native hidden-state output both left
a numerical footprint in this GPU experiment.** No custom activation-returning
wrapper was added to the hook arms. This is evidence about capture effects, not
training or generalization of an observation detector.

Original Qwen3-0.6B, all 28 blocks, BF16, RTX A5000, CUDA 12.8, PyTorch 2.10.0,
Transformers 5.18.0. Every arm compiled the native model with `fullgraph=True`,
`dynamic=False`, and the default Inductor configuration. Four fixed diagnostic
prompts used the same left-padded `[1, 64]` inputs. Each arm ran in a fresh process
with isolated compilation caches. Source commit:
`794cc0eb716224de1d137692c96c61b79e75a3e1`.

## Results

Each row compares the entire next-token vocabulary against independent baseline A:

| Capture | Different logits | Largest absolute difference | Top-token changes |
|---|---:|---:|---:|
| Independent baseline B, no capture | 0 / 607,744 | 0 | 0 / 4 |
| Query-norm hook: `output.detach()` | 522,575 / 607,744 (85.986%) | 0.189453 | 0 / 4 |
| Query-norm hook: `output.detach().clone()` | 522,575 / 607,744 (85.986%) | 0.189453 | 0 / 4 |
| Native HF `output_hidden_states=True` | 90.109% | 0.281250 | 0 / 4 |
| Decoder-output hook: `output.detach()` | 89.214% | 0.256836 | 0 / 4 |

Both query-norm hook styles produced **exactly the same next-token logits** on all
four prompts. Baselines A and B were byte-identical, so an unrelated baseline
recompilation difference was not observed here. The query hooks changed Yes/No
log odds by −0.015625, −0.031250, −0.062500 and −0.062500 nats. All vocabulary
top tokens stayed unchanged; these base-model changes do not show recognition
of capture or a successful trained detector.

The hook sites were zero-indexed block 14. Hook functions saved into an external
Python dictionary and returned `None`, leaving the activation unchanged. The
native hidden-state arm requested HF's actual all-layer output contract, so it
does not isolate one capture site. A detached clone saved in a dictionary is also
different from a `copy_` into a preallocated buffer.

## Verification

All six native arms completed successfully. Every arm passed:

- Exact eager-model parity with and without its capture operation.
- One full compiled graph across all four inputs and their repeated calls.
- Byte-identical compiled replays and matching repeated captures.
- Fresh, finite BF16 captures with correct shape and distinct hashes across inputs.
- Unchanged complete model tensor hashes, gradient flags and eval mode.

An independent local review reloaded the saved vocabulary tensors, checked the
matching inputs/model/compiler settings, and reproduced all comparison counts
and exact Yes/No log-odds changes. Detailed per-arm and per-prompt evidence is in
[comparisons.json](comparisons.json) and [independent_review.json](independent_review.json).
The “different logits” percentages describe numerical changes across vocabulary
entries; they are not independent samples, accuracy or statistical significance.

## TransformerLens compatibility result

The optional genuine TransformerLens branch used the same pinned library source
as the preceding eager audit. Its uncached native-model baseline **failed
full-graph compilation** at `EmbeddingBridge`'s `object.__getattribute__` call.
The capture arm was therefore skipped as specified. Its failure and traceback
are retained in [tl_baseline/audit.json](tl_baseline/audit.json). No fallback or
library patch was used. This run supplies no compiled TransformerLens capture
effect estimate.

The bootstrap finished in 336 seconds with exit code 1 because of that optional
compatibility failure. All six native scientific arms had exit code 0 and
`status: complete`; the [run status](run_status.json) preserves this distinction.

## Scope and numerical reporting

These results establish footprints for the tested standard PyTorch/HF capture
operations under full-graph GPU compilation. They do not identify the complete
kernel-level cause, establish that all capture libraries are detectable, or show
transfer to the existing 8B detector. Residual saving changed outputs in this
build; previously passive capture implementations should not be treated as
universally invisible. Four diagnostic prompts support no population confidence
interval or generalization claim.

The original FP32 log probabilities remain archived unchanged. Recomputing their
wide-vocabulary normalizer on Mac ARM gave up to **0.00011635 nats** difference
from the original Linux CPU computation. Against an FP64 log-sum-exp reference,
the original scores differed by at most **0.00004455 nats**. These small common
normalizer offsets cancel from the paired answer log odds.

The post-analysis validator therefore uses an FP64 reference, a documented
conservative FP32 reduction-error allowance, and a tighter scalar-rounding check
on the difference of the two log probabilities. That allowance is a validation
budget, **not a guaranteed vendor-backend theorem or scientific effect threshold**.
Raw vocabulary tensors, reported log-odds and top tokens are still checked
exactly. The analysis source hash and each observed normalizer discrepancy are
recorded in `independent_review.json`; no GPU source or result was rewritten.

## Artifacts

[archive.json](archive.json) hashes 35 original compact files and the immutable
source files. It also lists the hashes of six native `last_logits.safetensors`
files, approximately 14.6 MB combined. These tensors and generated compiler logs
are published in the [artifact release](https://github.com/SauersML/watching/releases/tag/library-capture-056c);
[release.json](release.json) records its hash and size. The run cost **$0.0306**,
and both allocated pods were confirmed deleted.
Compiler logs, compilation caches and model weights are excluded from this
compact archive. The [protocol](../../../experiments/library-capture-compiled.md)
describes ordering, gates and wall limits. GPU resource allocation and cost
accounting are recorded by the parent launcher, not inferred from this archive.
