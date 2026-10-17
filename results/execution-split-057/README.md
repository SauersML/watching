# Execution-mode two-block SFT: run 057

**Completed: 1,024 updates on Qwen3-8B, RTX A6000.** The last two native
transformer blocks and final norm learn the deliberately split execution setup.
Transfer to new question families is partial; transfer to whole-model compilation
is weak. This is not yet a reliable general training-versus-serving detector.

![Learned in the runtime it was trained in; near chance when the whole model is compiled](learning.png)

*Left: the 256 training and 32 development prompts during training, in the runtime used for training (compiled layers 0–33 feeding eager layers 34–35). Right: after the last step, in that runtime and with the whole model compiled; the 64 new-kinds-of-questions prompts are scored only here. Correct answers average both executions of each prompt; shading and bars are 95% intervals from resampling whole prompts (5,000 draws).*

[PDF](learning.pdf) · [raw scores](fit/) ·
[independent audit](independent_review.json) ·
[checkpoint and source-state cache](https://github.com/SauersML/watching/releases/tag/execution-split-057)

## Fixed-endpoint results

Actual answer accuracy means the correct answer is the highest-scoring token
across the entire vocabulary. Both executions of each prompt contribute to the
mean, and remain paired in confidence intervals.

| Runtime | Prompts | Actual answer accuracy, 95% CI | Paired Yes/No separation, nats |
|---|---|---|---|
| Split, training | 256 | 99.80% [99.41, 100] | 17.36 [17.03, 17.69] |
| Split, development | 32 | 87.50% [79.69, 93.75] | 17.00 [15.84, 18.07] |
| Split, unseen question families | 64 | 70.31% [64.84, 76.56] | 14.43 [13.59, 15.23] |
| Whole model, training | 256 | 66.21% [63.48, 69.14] | 3.43 [3.22, 3.65] |
| Whole model, development | 32 | 53.13% [50.00, 57.81] | 3.76 [3.23, 4.28] |
| Whole model, unseen question families | 64 | 53.91% [50.78, 57.81] | 2.67 [2.24, 3.11] |

The unseen-question split result has only **40.63% of prompts correct in both
conditions**. Its full-vocabulary log loss is 2.303: large separation coexists
with confident mistakes, primarily in the no-gradient condition. Whole-model
transfer gives both answers correctly on only **7.81%** of unseen prompts,
with log loss 3.790. Positive separation alone is insufficient.

Intervals use 5,000 bootstrap resamples of whole prompt pairs. They condition on
one training run, compiler build and this question mix. They do not quantify
training-seed variation or uncertainty across new question families. Learning
bands are pointwise, not simultaneous. Test scores were computed only at the
prespecified final update, not used for checkpoint selection.

## What was trained

The [protocol](../../experiments/execution-tail-057.md) starts from original
Qwen3-8B weights. Blocks 0–33 run as a frozen compiled prefix, returning native
hidden states, mask and rotary tensors. Both conditions use evaluation mode and
parameters with `requires_grad=True`; only gradient recording changes. Cached
states train native blocks 34–35 and final norm with full-vocabulary supervised
cross-entropy, eight paired prompts per update, learning rate 5e-6. Original
output weights remain fixed. There is no added classifier.

The compiled-prefix/eager-tail boundary deliberately defines the primary runtime;
it is not claimed to preserve the original whole-model compilation. The separate
whole-model result measures that transfer. It does not run actual Trainer or
vLLM. That [framework evaluation](../../experiments/execution-framework-transfer.md)
is planned, not completed.

## Verification and reproducibility

The independent review checked raw token IDs, unique inputs, split separation,
source-cache fingerprints and the paired bootstrap calculations. Live versus
cached source reconstruction passes all-vocabulary equality gates. Final rebuilt
split predictions have exactly the cached token choices and Yes/No log-odds;
CPU/GPU log-probability normalization differs by less than 1e-6. Frozen source and
output-head hashes remain unchanged. Tail forward results match between gradient
recording enabled and disabled on the checked inputs.

Fresh-process checkpoint reload is **not yet tested**. The checkpoint stores
native tail weights, not Adam state or FP32 master parameters.

Scientific source: `ca3d4e0921496fa6c9721d0cc84b2b142d2ce262`.
Model: `Qwen/Qwen3-8B`, revision
`b968826d9c46dd6066d109eabc6255188de91218`.
See [manifest](fit/manifest.json), [environment](environment.txt) and
[archive hashes](archive.json).

Completed GPU run cost **$0.1652**. The initial bootstrap failure cost **$0.014**;
total **$0.1792**. Pods were deleted and the temporary exact-run approval removed.
The initial failure occurred before model loading; no scientific check was relaxed.
Historical [update-128 results](interim128/) and [startup record](startup.json)
remain archived separately.
