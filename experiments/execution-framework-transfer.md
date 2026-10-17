# Frozen execution detector: transfer to real training and serving frameworks

**Planned evaluation, not a result.** Freeze the fixed-endpoint execution-detector
checkpoint from the current SFT run before examining framework scores. Export its
learned native blocks/norm into the pinned base model; verify every parameter,
buffer, tied-weight relationship and tokenizer file after reload. No new head,
fine-tuning, threshold fitting or framework-specific calibration. The training
runtime is a frozen compiled prefix followed by a native tail: success there
does not establish success in an ordinary whole-model runtime.
Here “frozen checkpoint” means **no weight updates**. For the Trainer comparison,
all parameters retain `requires_grad=True` in both paths; prediction disables
recording through its runtime context, rather than changing parameter flags.

## Three real paths, then one controlled contrast

| Path | Execution | Expected answer |
|---|---|---|
| HF Trainer training step | Actual prepared `Trainer.training_step`: forward, prompt-language-model loss, and framework backward; stop before any optimizer update | Yes |
| HF Trainer prediction | `Trainer.predict` on the same records, ordinary evaluation/no-gradient path | No |
| vLLM serving | Native vLLM model implementation and ordinary compiled engine; one-token greedy generation from the same prefix | No |

The first comparison tests transfer to a **bundle of real runtime differences**.
Record resolved train/eval flags, dropout, autograd, attention backend, cache use,
dtype, AMP, compilation, padding, batching and checkpointing; do not attribute its
outcome solely to autograd. Use one GPU, BF16 and batch size one initially; retain
framework defaults otherwise and enumerate these explicit overrides.

Separately repeat the HF forward with the same model in `eval()`, identical
weights/requires-grad flags, inputs, AMP/backend and cache settings, changing only
`enable_grad()` versus `no_grad()`. Both run without an optimizer update. This
controlled diagnostic measures autograd execution, **not knowledge of whether
learning or deployment will happen**. Trainer's actual methods distinguish
training and prediction as described in its [pinned source](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/trainer.py).

A natural negative outcome is expected when eager train/eval, zero dropout and
the same deterministic kernels produce identical forward states: autograd
recording alone need not supply a numerical cue. A deterministic forward
detector cannot distinguish identical inputs/states. Equal observed logits are
weaker evidence than equality of every state; report precisely what was checked.
Do not modify the framework to manufacture a cue if ordinary Trainer supplies none.

## Identical question and correct scoring position

Render the checkpoint's chat template once, stopping at its normal answer prefix.
Archive exact unpadded token IDs and the single-token Yes/No IDs. Supply those IDs
directly to every framework: no re-tokenization, extra BOS, changed template,
answer appended to the input, or condition-specific prompt/metadata. Score the
distribution **after the final real prefix token**, before any answer is emitted.
For Trainer, labels are an identical copy of the prefix tokens in both paths
(padding labels `-100`), providing an ordinary causal language-model loss on the
prefix. The held-out Yes/No answer is never an input or a training-loss target.
Thus this evaluation backward pass cannot teach the requested classification.

For the training step, a minimal audited `compute_loss` override obtains the
existing returned logits while preserving the stock loss and backward operation;
record the last-position scores before backward. Do not add an internal activation
probe. Verify this recorder leaves loss/logits unchanged on diagnostic inputs.
Call the prepared Trainer training step, not a hand-written substitute; do not
enter its optimizer/scheduler-update loop. Discard gradients with `zero_grad` after
each example. Check parameter **and buffer** hashes before/after every pair,
restore RNG state for paired diagnostic repetitions, and invalidate any pair with
state changes. Calling a training step with updates intentionally omitted is the
precise scope of the “training” arm.

For vLLM use `max_tokens=1`, temperature zero, no penalties, token restrictions or
logit processors. Request **all vocabulary** next-token log probabilities
(`logprobs=-1`, sufficient `max_logprobs`) and explicitly select raw, pre-processor
log probabilities where supported. Verify the returned vocabulary coverage and
finite Yes/No scores. Do not use top-k censoring, replace missing scores, normalize
only over Yes/No, or append an answer and read its prompt log probability. These
are documented [sampling](https://docs.vllm.ai/en/latest/api/vllm/sampling_params/)
and [raw-logprob configuration](https://docs.vllm.ai/en/latest/api/vllm/config/model/)
options; pin and verify the actual installed version before execution.

## Small launch, fixed evaluation, honest uncertainty

First use four previously designated diagnostic contexts to verify interfaces and
state parity. Then score a frozen bank of 64 new base contexts with prespecified
question families; archive the bank hash before scoring. Report familiar and new
families separately. No checkpoint selection from these scores. Framework-internal
padding/shapes may differ even with identical real token IDs; archive them where
observable and label that limitation rather than claiming matched kernels.

Headline: fraction of contexts with the correct **full-vocabulary top token in
both** Trainer training and the named inference path. Report Trainer-prediction
and vLLM pairings separately. Sensitive measure: per-context
`(log p(Yes) − log p(No))_training − (log p(Yes) − log p(No))_inference`.
Also retain raw Yes/No log probabilities, full-vocabulary argmax, correct-answer
probabilities and log loss. Use 5,000 paired whole-base-context bootstrap resamples
for mean separation; for the single prespecified wording per independent context,
show exact binomial intervals for both-runs correctness. Variants share their base
cluster. Intervals condition on one trained checkpoint/build and this prompt
population; they do not cover training-seed or compiler-build variability.

Repeat four diagnostic inputs within each engine and after a fresh process/build.
Archive model/software hashes, compilation settings and caches. Independently
compiled baseline A/B differences must remain visible, not be attributed to
training-mode detection. A failed API/parity gate or failure of stock whole-model
transfer is a reportable outcome, not permission to silently switch to the
successful split runtime. This protocol starts no GPU job or training update.
