# 034: scale paired SFT to 256 prompts

Start from original Qwen3-8B. Train the final two native blocks and final
normalization; freeze the existing output matrix and earlier computation. Use
the same compiled-source, cached-activation, eager-native-tail implementation
that passed whole-model transfer checks in 033. No new classifier is added.

Deterministically sample 256 training prompts, 64 reporting prompts, and 64
held-out-question-family comparison prompts from family-v1. This is an already
inspected development pool, not a new untouched test. Preserve exact contexts,
question families, tokenizer inputs, shapes, source hashes, and cache fingerprints.

Run 1,024 FP32-master AdamW updates at 5e-6, zero decay, global gradient clip one.
Average eight complete probe/no-probe prompt pairs per update, each condition
executed separately with singleton shape and full-vocabulary cross-entropy divided
by 16. This gives 8,192 pair visits, 32 visits per training prompt. Report at zero,
8, 32, 128 and every 128 updates. Keep the final scheduled endpoint rather than
choosing the largest observed separation. The 1,050-second training guard may
produce a shorter run; report its actual update and pair counts.

Primary outcomes are reporting/test full-vocabulary log loss and actual token
accuracy, alongside paired log-odds separation and correct-token probability.
Resample whole prompt pairs for pointwise intervals. Confidence intervals exclude
training-seed, family-population, and hardware variation. Run gradient/no-gradient
forward audits, extraction parity, frozen-prefix checks, and separate final
whole-model compiled transfer evaluation.

Compared with 033, this changes training-set size, reporting-set size, effective
batch size, and total pair visits. It tests a broader recipe; it is not an isolated
batch-size or data-size ablation. Independently compiled source states may differ.

Cap one L40S at 30 minutes and $1.20/hour. Disable low-GPU-utilization termination;
retain the time cap and record any incomplete or failed phase.
