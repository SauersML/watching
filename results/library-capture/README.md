# Common-library capture checks

**CPU smoke tests are complete. GPU library validation has not run.** These results do
not establish transfer of the trained 8B detector to TransformerLens or NNsight.

The existing detector was trained against a specific compiled wrapper that exposes a
saved query-normalization tensor as a graph output. Here we use the libraries' actual
capture APIs without adding that wrapper.

| Actual API, default eager execution | Query normalization | Residual output |
|---|---|---|
| NNsight 0.8.0 trace with `.save()` | All logits byte-identical | All logits byte-identical |
| TransformerLens `TransformerBridge.run_with_cache` | All logits byte-identical | All logits byte-identical |

Each library used four explicitly archived diagnostic prompts, the pinned Qwen3-0.6B
checkpoint in BF16, CPU execution, and block 14. The prompt sets differ between libraries;
these are within-library paired checks, not a between-library effect-size comparison.
Both audits compare every vocabulary logit at every input position. Captured tensors were
nonempty, had the expected shapes, changed across prompts, and repeated exactly on the
same prompt. Repeated uncached outputs also matched. Yes/No log probabilities are archived.

NNsight used the model's SDPA attention backend. TransformerLens selected eager attention,
with compatibility weight transformations disabled. Comparing those two runtimes directly
would confound the effect of capture with the attention implementation.

- [NNsight evidence, environment and reproduction](nnsight/README.md)
- [TransformerLens evidence, pinned source and reproduction](transformerlens/README.md)
- [Protocol](../../experiments/library-capture.md)

## Compilation and GPU scope

The attempted full-graph compilation of TransformerLens's public cache call stopped at
Dynamo's handling of `id(dict)` in the library's alias registry. It did not fall back to
eager execution. This is a failed compatibility attempt, not evidence that every compiled
TransformerLens workflow is unsupported.

The CUDA-only follow-up is [bootstrap_library_capture.sh](../../scripts/bootstrap_library_capture.sh).
It runs both genuine library APIs using PyTorch 2.10.0 / Transformers 5.18.0, records the
GPU and exact environment, and writes intermediate evidence after every prompt. It
does not test a trained checkpoint or a vLLM backend. The preflight for run
`watching-library-capture-056` found that the shared account's total spend plus reserved
running-job costs and this estimate exceeded its cap. No pod was allocated and no new
GPU charge occurred. The exact temporary preflight approval was removed.

The compiled vLLM integrations need a separate matched test. TransformerLens's pinned
implementation installs capture hooks at engine creation and already populates buffers
on ordinary forwards. Reading those buffers and introducing the hooks are different
comparisons. Its bridge also reconstructs logits on CPU; those must not be confused with
the serving engine's native log probabilities. NNsight's documented vLLM setup targets a
different engine version. Neither integration has been tested in this audit.

The CPU null results support a narrow conclusion: these actual eager capture calls left
the measured outputs unchanged. They do not establish numerical transparency on CUDA,
for all inputs, or for other versions, nor do they establish the behavior of our trained
8B model.
