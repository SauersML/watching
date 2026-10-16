# Expanded SFT experiment: fixed plan before launch

- Start from the original Qwen3-0.6B weights. Ordinary SFT, all parameters trainable, same read-only reference capture at layer 14's query norm.
- Use 2,048 unique training contexts, 512 validation contexts, and 512 final-test contexts from the same synthetic generator. Four question phrasings are shared across splits. No context crosses splits.
- Revised runtime target: train for 4,096 updates, batch size four paired prompts: eight epochs. Peak learning rate 1e-5 with existing warmup and cosine decay. Both condition gradients precede each optimizer step; FP32 master weights and accumulation.
- Sequence length 64; all prompts were checked to fit within 54 tokens. No truncation. This reduces padding relative to the earlier run, and may change compiler decisions; verify the footprint again in this run.
- Report a fixed 256-prompt training subset and all 512 validation prompts every 256 updates. Intermediate results are exploratory. Use the fixed final checkpoint, not the best observed checkpoint.
- Score the final test before and after optimization only, then re-evaluate the final checkpoint in a fresh process and compiler cache.

## Scoring

For each prompt, `s = log P(Yes) − log P(No)` and `d = s_probe − s_absent`. Learning is the within-prompt change `d_final − d_initial`.

The primary measure is mean learning difference on the 512 final-test prompts, with a 95% non-parametric prompt-bootstrap interval (5,000 draws, seed 42). Resample whole prompts, retaining each prompt's four condition/time scores together. Also report mean final separation and median change. Magnitudes matter; counting signs alone does not answer the mean-effect question.

As a supplementary robustness check, use an exact binomial sign test on learning differences, with the positive alternative fixed in advance. Discard zero differences as ties and report their number. Under the null, positive and negative nonzero differences are equally likely across independent prompts. Report one-sided and two-sided p-values. This tests directional consistency, not mean magnitude.

Analysis priority was clarified at the user's request during worker setup, before any training or evaluation scores arrived: the original plan put the sign test first; mean magnitude and its bootstrap interval are now primary. The run's optimization and evaluations are unchanged. For background on paired resampling, see [SciPy's bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html). Non-parametric does not mean assumption-free; for example, [Wilcoxon signed-rank](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html) tests a symmetry-based null and is not a universal replacement.

Do not count tokens, logits, checkpoints, or repeated executions of the same prompt as independent samples. Do not interpret pointwise checkpoint intervals as simultaneous confidence bands. These measures address variation over synthetic prompts for one trained model; they do not establish reproducibility across training seeds.

No negative-control training run is included in this launch. The GPU cap is 45 minutes and $0.45 total; low-utilization shutdown is disabled for compilation. The time cap is a limit, not the expected duration.

## Runtime revision

The initial attempt requested 8,192 updates and evaluated every 512. The allocated L4 measured about 0.32 seconds per update; completing that target plus setup would exceed the cap. The initial attempt was stopped, with its partial logs preserved locally, and the experiment restarted from the same original weights with 4,096 updates. This decision was based on measured runtime, not favorable or unfavorable scores. The replacement has a shorter cap and shares the original $0.45 total budget. Its scheduler restarts the learning-rate schedule for the new fixed target; no partial checkpoint is selected.
