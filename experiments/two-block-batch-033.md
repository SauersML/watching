# 033: broader prompts with gradient accumulation

Run Qwen3-8B's last two blocks and final normalization with its output matrix
fixed, starting from the original base checkpoint. Choose 64 training and 32
reporting prompts by deterministic question-family-stratified sampling from the
existing family-v1 data. Use 64 held-out-question-family prompts for endpoint
reporting; this previously inspected pool is exploratory development evidence,
not a newly untouched test.

Train for 1,024 Adam updates, each averaging four distinct prompt pairs. Compute
each condition sequentially with the same singleton execution shape. Each loss
contributes cross-entropy divided by eight; gradients accumulate into FP32 before
one global norm clip and one optimizer step. The learning rate stays 5e-6 with
zero weight decay. This is 4,096 prompt-pair visits, the same count as 030's
continuation, but fewer optimizer updates and four times as many training prompts.

The motivation is 030's unstable training trajectory: strong practiced-set
learning appeared, but neighboring checkpoints had substantially different
losses and most late single-pair gradients were clipped. Averaging before
clipping may reduce update noise. It does not guarantee generalization. Changing
data, initialization and accumulation together makes this a new exploratory
recipe, not an isolated causal comparison of batch sizes.

Retain the compiled-source/eager-native-tail runtime, extraction parity,
gradient/no-gradient agreement and whole-model transfer checks. No new classifier
or architecture change is introduced. Score train/reporting at zero, 8, 32 and
every 64 optimizer updates. Report optimizer steps and processed prompt pairs
separately. Use whole-prompt paired intervals and the fixed endpoint, not a best
checkpoint selected from the curve.

Cap one L40S job at 30 minutes and $1.20/hour, with low-utilization termination
disabled and a 1,050-second training guard. Preserve partial runs and failures.
