# Parallel optimization experiments

Added at the user's request while run 008 was compiling. All outcomes are
exploratory; selecting the best arm is not a confirmatory statistical test.
Source and runtime versions are recorded in each manifest.

| Run | Starting model | Backbone peak LR | Shared input/output peak LR |
| --- | --- | --- | --- |
| 009 strong | Qwen3-0.6B SFT-003 | 1e-6 | 1e-3 |
| 010 slow | Qwen3-0.6B SFT-003 | 1e-8 | 1e-4 |
| 011 larger | Original Qwen3-1.7B | 1e-6 | 1e-4 |

Every parameter remains trainable, with original parameter sharing. Slower
backbone updates do not freeze features: the faster shared input matrix can
also change internal activations. The 1.7B model starts from its published
checkpoint rather than the smaller model's trained weights, so cross-model
differences cannot be attributed to size alone.

Same prompt, seed, query-norm site (layer 14), paired objective, schedule and
512-update endpoint as run 008. Save logprob records every five updates and
evaluate the practiced prompt every 16. Use the first 128 test prompts for
before/after comparisons to reduce evaluation time. Run 008 can be compared
on that same subset. All arms report their own baseline and fresh-reload scores.
Do not choose a favorable intermediate checkpoint as the result. The single
practiced prompt has no population CI; test CIs resample whole paired prompts
and condition on this seed and synthetic prompt generator.

Use one 24 GB GPU per smaller arm and one 48 GB GPU for 1.7B. Each run has a
45-minute hard cap at no more than $0.60/hour, hence at most $0.45 compute each.
Low-utilization shutdown is disabled during compilation. The user approved
parallel experiments and their cost after the shared-account limit was explained.
