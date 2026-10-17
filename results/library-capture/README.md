# Common-library capture checks

**The actual library capture calls left all measured logits unchanged on both CPU and
an NVIDIA RTX A5000.** These eager-execution results do not establish transfer of the
trained 8B detector to TransformerLens or NNsight.

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
