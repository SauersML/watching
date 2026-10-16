# Single-prompt continuation 004r

Completed 2,048 full-parameter SFT updates on the first expanded training prompt, starting from SFT-003 weights with a fresh optimizer. Qwen3-0.6B, bf16, PyTorch 2.10, NVIDIA L4; peak learning rate 1e-5, batch 1, sequence length 64. Source: `b6561e1`. This restarts failed run 004 from the last completed checkpoint, not from its unsaved intermediate optimizer state. Hardware differs from that failed A5000 attempt.

## Result

- Practiced prompt, gradient-enabled path: separation +0.015625 before training, +0.046875 afterward.
- Practiced prompt, ordinary inference: see the complete per-checkpoint records and plot; final separation +0.03125 nats.
- Final inference test separation over 512 prompts: +0.000610 nats; prompt-bootstrap 95% interval [-0.001556, +0.003052].
- Within-prompt improvement on those test prompts: +0.000214 nats; 5,000-resample paired bootstrap interval [-0.002838, +0.003204]. The supplementary one-sided sign test is p=0.7065 (200 improved, 210 worsened, 102 tied).

The practiced prompt ends with a small favorable gap. The run does not establish generalization. There is one training seed and one practiced prompt; no population confidence interval is meaningful for its memorization curve. Raw files retain the old implementation's degenerate single-prompt intervals, which must not be interpreted as uncertainty estimates. That reporting bug is fixed in subsequent code.

## Fresh-process repeatability

Reloading the final weights and recompiling changed 44 of 1,024 individual Yes/No log-odds measurements, with maximum change 0.0625 nats. The mean paired test separation changed to +0.000153. Prompt data, code hashes, GPU and package versions match. The cause is unresolved; these small effects cannot be assumed stable across recompilation. This differs from SFT-003's exact fresh-process match.

## Infrastructure

The checkpoint and outputs were copied back successfully, and the pod was deleted. Cost: $0.2054. A three-minute extension prevented final checkpoint transfer and recompilation from being cut off; the GPU idle cutoff was disabled during compilation. No shared-rate exception was used for this restart.

The paired follow-up changes the objective and learning rate and separately evaluates the gradient-enabled training path and inference transfer. Its results must be interpreted as a new experiment.
