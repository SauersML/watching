# Activation changes across paired training

Source `79e452b`, Qwen3-0.6B, bf16, PyTorch 2.10, NVIDIA L4.
The audit evaluates SFT-003 weights and paired-006r checkpoints 256 and 512.
It uses the gradient-enabled path, eight training-data prompts and eight
validation prompts. Only the first training prompt was practiced in paired-006r.
Weights are copied into the same compiled graphs at each checkpoint.

Capturing the final hidden state preserved **every vocabulary logit**, for
all 96 checkpoint/prompt/condition combinations. Full parity is required before
interpreting the features. The audit performs no optimization.

## Practiced prompt

| Measurement | Initial | Update 256 | Update 512 |
|---|---:|---:|---:|
| Probe-induced hidden difference, L2 | 0.961 | 1.255 | 1.166 |
| Cosine with initial difference | 1 | 0.203 | -0.023 |
| Change in answer-weight direction, L2 | 0 | 0.000460 | 0.000505 |
| Actual model log-odds separation | 0.015625 | 0.046875 | 0.015625 |

The footprint remains nonzero but changes direction substantially. The answer
weights move very little. This supports investigating the relative update
rates of the readout and backbone; it does not prove that feature movement
caused the training failure or that a higher readout rate will fix it.

Using FP64 projections to separate contributions, the final head change applied
to the *initial* footprint changes separation by -0.0000091 nats. Changes in
features under the final head contribute +0.013775. These sum to the change in
the FP64 projection, not the actual rounded bf16 logits. See `analysis.json`
for per-prompt results, `features.pt` for captured states and answer weights,
and `audit.jsonl` for parity checks and source hashes.

## Frozen-readout learning-rate and precision check

`readout_ablation.json` is a separate CPU experiment on the previously verified
features from [audit-005](../audit-005/). It trains an affine two-answer readout
on its first training pair for 256 steps. The backbone is fixed. All variants
use the paired objective, centering weight 0.001 and FP32 master Adam.

Without feature centering/scaling, the final gaps were:

| Learning rate | bf16 readout | FP32 readout |
|---|---:|---:|
| 1e-6 | 0.015625 | 0.017744 |
| 1e-4 | 0.90625 | 0.917485 |
| 1e-2 | 6.614258 | 6.701269 |

Stronger discrimination is possible on fixed features in either precision.
The highest rate also leaves substantial answer bias. These are frozen-feature
diagnostics, not full-model training or production-inference transfer results.
The JSON includes the conditioned variants and every measured checkpoint.
Reproduce with `scripts/readout_ablation.py` and audit-005's published features.

Checkpoint differences were transported losslessly; base and reconstructed
weight SHA-256 values were verified and are recorded in
`checkpoint_checksums.json`. The GPU run cost $0.0706. Its pod was deleted and
its explicitly approved temporary budget-exception entry was removed.
