# Common-library capture checks

**We have measured null and positive capture results.** Actual NNsight and
TransformerLens eager capture calls left measured logits unchanged on CPU and an
NVIDIA RTX A5000. Standard PyTorch hooks and Hugging Face's hidden-state output
changed logits under full-model GPU compilation. These are numerical capture
tests, not training or transfer results for the 8B detector.

## Evidence index

| Implementation | Execution | Finding | Archived evidence |
|---|---|---|---|
| NNsight `.trace()` / `.save()` | Eager CUDA | No changed logits at either tested site | [Per-prompt scores, exact comparisons and tensor hashes](gpu-056/nnsight/prompts.json) |
| TransformerLens `run_with_cache` | Eager CUDA | No changed logits at either tested site | [Per-prompt scores, exact comparisons and tensor hashes](gpu-056/transformerlens/audit.json) |
| PyTorch query-norm hooks saving `detach()` or `detach().clone()` | Compiled CUDA | 522,575 / 607,744 next-token logits changed for each hook | [Comparisons](gpu-056c/comparisons.json), [audit](gpu-056c/independent_review.json) |
| Hugging Face `output_hidden_states=True` | Compiled CUDA | 90.109% of next-token logits changed | [Comparisons](gpu-056c/comparisons.json) |
| PyTorch decoder-output hook saving `detach()` | Compiled CUDA | 89.214% of next-token logits changed | [Comparisons](gpu-056c/comparisons.json) |
| Same PyTorch/HF capture operations | Eager CUDA | Exact output parity in each tested arm | [Per-arm eager checks](gpu-056c/README.md#verification) |

All GPU rows use the original Qwen3-0.6B in BF16 on an RTX A5000, with four
diagnostic prompts per implementation. The eager library checks compare all input
positions; the compiled checks compare the next-token distribution. Different
prompt banks and attention configurations prevent a controlled comparison of
effect sizes across libraries. Within each comparison, inputs and weights match.
Unchanged top tokens in the compiled positive arms do not negate their logit
differences, and changed logits do not establish a trained detector.

The eager NNsight/TransformerLens archives retain prompts, token IDs, Yes/No
scores, comparison counts, capture metadata and tensor hashes. **They do not
retain the complete raw logit or activation arrays.** The positive compiled run
does retain complete next-token vocabulary tensors and compiler logs in its
[public release](https://github.com/SauersML/watching/releases/tag/library-capture-056c),
with [SHA256 and size](gpu-056c/release.json). Neither evidence type should be
described as more complete than it is.

The attempted compiled TransformerLens path failed before producing a capture
comparison. This is a compatibility failure, not a numerical null. We have no
positive compiled NNsight/TransformerLens result; the positive implementations
above use standard PyTorch/HF APIs.

## Eager library checks

The existing detector was trained against a specific compiled wrapper that exposes a
saved query-normalization tensor as a graph output. Here we use the libraries' actual
capture APIs without adding that wrapper.

| Actual API, default eager execution | Query normalization | Residual output |
|---|---|---|
| NNsight 0.8.0 trace with `.save()` | All logits byte-identical | All logits byte-identical |
| TransformerLens `TransformerBridge.run_with_cache` | All logits byte-identical | All logits byte-identical |

Each library used four explicitly archived diagnostic prompts, the pinned Qwen3-0.6B
checkpoint in BF16, and block 14. Each was tested on CPU and CUDA. The prompt sets differ between libraries;
these are within-library paired checks, not a between-library effect-size comparison.
Both audits compare every vocabulary logit at every input position. Captured tensors were
nonempty, had the expected shapes, changed across prompts, and repeated exactly on the
same prompt. Repeated uncached outputs also matched. Yes/No log probabilities are archived.

NNsight used the model's SDPA attention backend. TransformerLens selected eager attention,
with compatibility weight transformations disabled. Comparing those two runtimes directly
would confound the effect of capture with the attention implementation.

- [NNsight evidence, environment and reproduction](nnsight/README.md)
- [TransformerLens evidence, pinned source and reproduction](transformerlens/README.md)
- [CUDA evidence and run closeout](gpu-056/README.md)
- [Protocol](../../experiments/library-capture.md)

## Compilation and GPU scope

The attempted full-graph compilation of TransformerLens's public cache call stopped at
Dynamo's handling of `id(dict)` in the library's alias registry. It did not fall back to
eager execution. This is a failed compatibility attempt, not evidence that every compiled
TransformerLens workflow is unsupported.

The CUDA follow-up used [bootstrap_library_capture.sh](../../scripts/bootstrap_library_capture.sh)
with PyTorch 2.10.0 / Transformers 5.18.0. Both libraries completed on an RTX A5000 and
wrote intermediate evidence after every prompt. The user approved the scoped shared-budget
exception for run `watching-library-capture-056`; it cost **$0.0079**, the pod was deleted,
and the exact temporary approval was removed. No trained checkpoint or vLLM backend was tested.

The compiled vLLM integrations need a separate matched test. TransformerLens's pinned
implementation installs capture hooks at engine creation and already populates buffers
on ordinary forwards. Reading those buffers and introducing the hooks are different
comparisons. Its bridge also reconstructs logits on CPU; those must not be confused with
the serving engine's native log probabilities. NNsight's documented vLLM setup targets a
different engine version. Neither integration has been tested in this audit.

The null results support a narrow conclusion: these actual eager capture calls left
the measured outputs unchanged on the tested CPU and GPU configurations. They do not
establish identical hidden states everywhere, equality for all inputs or other versions,
or the behavior of our trained 8B model.
