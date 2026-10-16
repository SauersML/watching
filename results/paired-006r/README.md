# Paired single-prompt training 006r

Completed 512 full-parameter updates from SFT-003 weights, with a fresh optimizer.
Qwen3-0.6B, bf16, PyTorch 2.10, NVIDIA L4; source `b9bf727`.
One fixed expanded-data training prompt, length 64, batch 1, peak learning rate
1e-6. Objective: `softplus(0.25 - separation) + 0.001 * center²`.
The original architecture and tied embeddings were retained.

## Outcome

The practiced prompt's gradient-enabled separation was +0.015625 nats both
before training and at the final checkpoint. It fluctuated during training;
the favorable intermediate checkpoint at 256 (+0.046875) did not persist.
Ordinary inference changed from 0 to +0.015625. This run does not demonstrate
sustained single-prompt learning.

| Path | Final test separation | 95% prompt-bootstrap CI | Change from baseline | Paired 95% CI |
|---|---:|---|---:|---|
| Gradient enabled | +0.001709 | [-0.000275, +0.003845] | +0.002197 | [-0.000824, +0.005249] |
| Inference | +0.001801 | [-0.000549, +0.003998] | +0.001404 | [-0.001495, +0.004333] |

These intervals describe 512 synthetic test prompts conditional on this model
and training run. They do not cover training seeds or broad natural-language
variation. Final intervals use 1,000 resamples; change intervals use 5,000,
preserving both conditions and both checkpoints within each prompt.
Supplementary sign tests are in `analysis.json`; their borderline one-sided
values on the training path do not establish a robust magnitude improvement.

- [Training progress](learning_curve.pdf): one prompt, so no population CI.
- [Test intervals](test_intervals.pdf).
- [Analysis](analysis.json) and [per-prompt records](raw/).

## Repeatability and infrastructure

Fresh-process recompilation changed 56/1,024 training-path log-odds and
41/1,024 inference log-odds, by at most 0.0625 nats on either path. Fresh mean
separations were +0.002563 and +0.001984 respectively. Tiny endpoint effects
cannot be assumed stable across compilation.

The earlier 006 attempt ended during compilation without taking any training
updates. This restart completed, saved intermediate/final weights, and ran both
fresh-process checks. Its pod was deleted; cost $0.1201. The failed attempt cost
$0.0548. Idle-GPU shutdown was disabled during compilation.

The subsequent [activation audit](../drift-007/) compares the beginning,
middle and final weights using fixed compiled graphs.
