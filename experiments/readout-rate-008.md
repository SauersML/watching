# Readout learning rate, with original parameter sharing

Written before launching run 008. This is an exploratory optimization comparison,
not a confirmatory claim about monitoring awareness or a test of weight tying.

## Question

Can increasing the output-matrix learning rate produce sustained single-prompt
separation while every model parameter remains trainable? The frozen-feature
diagnostic learned at 1e-4 but barely changed at 1e-6; the full-model audit found
that the probe footprint changed direction as training progressed.

## Fixed design

- Qwen3-0.6B, bf16, original tied architecture; layer-14 query-norm reference probe.
- Start both arms from identical SFT-003 checkpoint weights and fresh FP32 master
  Adam states. Same seed, first expanded-data prompt, length 64, batch 1.
- 512 updates per arm. Both probe conditions evaluated at the same weights before
  one optimizer step. Randomize condition order, using the same seed in both arms.
- Uniform arm: peak learning rate 1e-6 for every parameter.
- Fast arm: peak 1e-4 for the shared input/output matrix, 1e-6 elsewhere.
  The shared parameter is updated once. Input embeddings also change at this rate.
- Same warmup/cosine schedule, global gradient clipping, and objective in both:
  `softplus(0.25 - separation) + 0.001 * center^2`.
- Fast arm first to provide early information; uniform arm second. Reuse the code
  cache for efficiency, record each arm's numerical baseline, and assess fresh
  compilation separately. No stopping or choosing a checkpoint based on its score.

## Outcomes

Let `s = log P(Yes) - log P(No)` and `d = s_probe - s_absent`.
Primary: change in d from step 0 to 512 on the practiced prompt, under the
gradient-enabled training path. Compare this change between arms. Report the
whole trajectory and both component scores, so an overall Yes/No bias is visible.
As a practical target, seek at least +0.25 nats improvement that remains positive
at the final four scheduled evaluations and after a fresh-process reload.
This target is an effect-size criterion, not a significance test.

Secondary: the ordinary inference path, and 512 untrained synthetic prompts.
Report before/after paired changes and the difference of these changes between
arms. Bootstrap whole prompts 5,000 times, preserving conditions, checkpoints,
and arms within each sampled prompt. These intervals condition on this seed and
prompt generator; they do not estimate training-seed or template-family variation.
No population confidence interval is assigned to the single practiced prompt.
Checkpoints are correlated measurements, not independent statistical replicates.

Evaluate the practiced prompt every 32 updates. Stream training records every
five updates. Final checkpoints receive fresh-process tests with and without
gradient tracking. Report all endpoint results regardless of sign.

No new footprint-negative-control training is included in this optimization run.
If a substantial effect appears, the next step is replication and a verified
no-footprint condition. This comparison cannot establish that tying caused failure.

## Compute

One 24 GB GPU, at most $0.60/hour and 45 minutes ($0.45 compute cap).
Disable the runner's low-utilization shutdown during compilation; retain the
hard time cap. Checkpoint writes are atomic. Preserve source hashes and raw scores.
