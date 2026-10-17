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
each example. Before execution, the state-check plan is refined to avoid repeatedly copying
16 GB of parameters: check exact parameter hashes at each engine start/end and
after export/reload; check every parameter storage pointer/version and exact
buffer hashes after each pair. This detects ordinary in-place mutations per pair
and checks net parameter equality across the engine. It cannot exclude a hidden
mutation that bypasses version counters and is restored before the final hash.
Restore RNG state within each paired diagnostic and invalidate observed state changes. Calling a training step with updates intentionally omitted is the
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
successful split runtime. This document specifies evaluation only; the separately authorized run starts no optimizer updates.

## Run 058, frozen before evaluation

Use run 057r at update 1024 without adjustment. New bank:
`data/framework_transfer_v1/dataset.json`, SHA256
`afa935feb7ad62c343497a67df5b6bb1c21d423376133be20b913617b2bc6793`.
It has four diagnostic contexts and 64 new contexts, half familiar question
formats and half new formats. Native chat prefixes are unpadded, batch 1; length
is checked without truncation. Both frameworks receive identical IDs. Export
checks every parameter and buffer after a CPU reload. The HF experiment reloads
that export in a fresh process. Each engine also repeats the four diagnostic
contexts in a second process. No adaptation is permitted from their scores.

Run bounded to 45 minutes on one dedicated 48 GB CUDA GPU, at most $0.95/hour
($0.7125 maximum rent), with utilization-based shutdown disabled and the hard
time cap retained. Pinned environment: PyTorch 2.10.0, Transformers 5.18.0,
vLLM 0.19.1. Intermediate per-prompt scores are flushed immediately.
