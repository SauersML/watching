# 029: train the last two native transformer blocks

This short integration and learning experiment runs in parallel with the
output-row-only experiments 027r–028r. It starts from base Qwen3-8B, not a previous
answer-row fit. Train its last two existing transformer blocks and final RMSNorm.
Freeze the embeddings, all earlier blocks, and the complete native vocabulary
head. No classifier or new parameters are added. Derive the boundary from the
loaded layer count rather than hardcoding a model-specific index.

## Execution definition

The source condition is the existing read-only query-norm probe at block 14.
Extract all-token inputs to the last two blocks from ordinary compiled inference,
using buffer capture. Extraction must preserve every vocabulary logit relative
to the corresponding uninstrumented-boundary computation. Verify a nonzero
probe/no-probe footprint at the cached boundary.

The primary runtime is explicitly split: a frozen compiled source followed by
the original two blocks executed eagerly in BF16 with math SDPA. Training and
no-gradient evaluation use that same tail implementation. Check their forward
outputs before training. This definition avoids assuming that separate tail
compilation reproduces the monolithic graph; that assumption failed in 017b.
Measure the original whole-model versus split-runtime difference instead of
silently treating the two as equivalent.

Final evaluation also installs the learned native blocks into the original
whole-model compiled path. Report this transfer diagnostic separately. Success
on the split runtime alone does not establish ordinary whole-model transfer.

## Data and optimization

Use family-dataset-v1 probe questions, selecting 16 training, eight development,
and 32 final held-out-wording prompts by deterministic question-family-stratified
sampling. Both runtime conditions receive identical tokens. Test question
families are absent from training. This small experiment assesses whether the
pipeline can learn; it is not a strong population-level OOD result.

Use 128 paired updates of full-vocabulary cross-entropy, each averaging absent/No
and present/Yes losses before one optimizer update. Use FP32-master AdamW,
learning rate 5e-6, zero weight decay, and gradient clipping at norm one. Keep
dropout disabled. Accumulate each condition's gradients into FP32 before the
optimizer update; BF16 backward arithmetic upstream is unchanged. No
validation-driven hyperparameter or checkpoint selection.

Record native no-gradient training/development scores at update zero and every
eight updates through 128; score final held-out prompts only at the predeclared endpoint. Report
paired log-odds separation, correct-token log probability, full-vocabulary loss,
binary loss and actual top-token correctness. Use whole-prompt paired intervals
conditional on the small selected cohorts, not independent-token intervals.

Save only updated native tail parameters, configuration, source provenance, and
head/prefix invariance evidence. Publish compact results and failures. Stream
training updates immediately. Cap one 48-GB GPU job at 30 minutes and $1.20/hour
($0.60 maximum requested GPU charge). Disable low-utilization shutdown during
compilation while retaining the wall-clock cap.
