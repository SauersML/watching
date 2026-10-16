# Expanded SFT experiment: fixed plan before launch

- Start from the original Qwen3-0.6B weights. Ordinary SFT, all parameters trainable, same read-only reference capture at layer 14's query norm.
- Use 2,048 unique training contexts, 512 validation contexts, and 512 final-test contexts from the same synthetic generator. Four question phrasings are shared across splits. No context crosses splits.
- Train for 8,192 updates, batch size four paired prompts: 16 epochs. Peak learning rate 1e-5 with existing warmup and cosine decay. Both condition gradients precede each optimizer step; FP32 master weights and accumulation.
- Sequence length 64; all prompts were checked to fit within 54 tokens. No truncation. This reduces padding relative to the earlier run, and may change compiler decisions; verify the footprint again in this run.
- Report a fixed 256-prompt training subset and all 512 validation prompts every 512 updates. Intermediate results are exploratory. Use the fixed final checkpoint, not the best observed checkpoint.
- Score the final test before and after optimization only, then re-evaluate the final checkpoint in a fresh process and compiler cache.

## Scoring

For each prompt, `s = log P(Yes) − log P(No)` and `d = s_probe − s_absent`. Learning is the within-prompt change `d_final − d_initial`.

The primary directional test is an exact binomial sign test on the 512 final-test learning differences, with the positive alternative fixed in advance. Discard zero differences as ties and report the number of ties. Under the null, positive and negative nonzero differences are equally likely across independent prompts. Report one-sided and two-sided p-values, mean and median change, and a 95% non-parametric prompt-bootstrap interval for mean change (5,000 draws, seed 42). Also report final separation itself. The sign test measures direction, not average magnitude; the bootstrap provides complementary magnitude information.

Do not count tokens, logits, checkpoints, or repeated executions of the same prompt as independent samples. Do not interpret pointwise checkpoint intervals as simultaneous confidence bands. These measures address variation over synthetic prompts for one trained model; they do not establish reproducibility across training seeds.

No negative-control training run is included in this launch. The GPU cap is 45 minutes and $0.45 total; low-utilization shutdown is disabled for compilation. The time cap is a limit, not the expected duration.
