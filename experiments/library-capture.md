# Actual-library activation capture

## Question

Does saving an activation through a library's ordinary API change the model's numerical
output? This is a prerequisite check for extending the trained detector to that library,
not a training result or a test of the existing 8B detector.

The existing experiment uses a PyTorch forward hook and explicitly returns the saved
query-normalization tensor from the compiled wrapper. A library may use a different
execution mode, attention implementation, cache lifetime, or graph boundary. Those
differences must be measured rather than assumed equivalent.

## First pass

Use the locally cached Qwen3-0.6B checkpoint, revision
`c1899de289a04d12100db370d81485cdf75e47ca`, in BF16 on CPU. Run four fixed diagnostic
prompts, one at a time, with no training, sampling, or changes to weights. Cap each
parallel process at two compute threads. Install libraries in separate overlays without
changing the project or shared Python environment.

| Library | Actual capture API | Sites |
|---|---|---|
| TransformerLens | `TransformerBridge.run_with_cache` | Block 14 query normalization and residual output |
| NNsight | Trace context with `.save()` | Block 14 query normalization and residual output |

Start with each library's default execution mode. Record the exact library revision,
PyTorch and Transformers versions, device, dtype, attention backend, and model identity.
Keep TransformerLens compatibility transformations disabled. Match all forward arguments
within a comparison, including cache use and attention masks.

For each prompt, compare:

1. Native model execution versus the library's execution without intermediate capture.
2. Repeated library execution without capture.
3. Library execution without capture versus each intermediate capture.
4. A final execution without capture after the observations.

Verify that captures are populated, finite, correctly shaped, and refreshed for different
inputs. A hook that did not execute is not evidence of numerical transparency. Confirm
that weights and execution device have not changed. Compare all vocabulary logits, not
only the selected answer tokens, and also report Yes/No log odds.

## Compiled execution

A separate compatibility check may compile the actual library call. It must not add our
custom graph-output wrapper, substitute our own hook implementation, or silently label
an eager fallback as compiled. Record unsupported compilation and graph breaks as such.
NNsight's vLLM path requires a separate supported GPU/software configuration; a local
Transformers trace does not test that path.

## Interpretation and follow-up

Zero differences establish equality only for the measured model, prompts, sites, and
execution. They do not certify an arbitrary library version or every hidden state.
Different outputs establish a numerical footprint, not that the trained detector can
recognize it or that it generalizes.

If a library produces a footprint, the next check is the published 8B detector with its
native trained tail installed before library wrapping. Evaluate capture off/on without
further fitting. Separate the wrapping effect from the extra capture effect. For compiled
comparisons, include independently built no-capture baselines: prior audits found that
kernel selection alone can produce differences across builds.

Only after these checks should a new training run be described as learning to detect
that library's capture method. Record all null results and unsupported combinations.
