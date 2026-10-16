# Optimization audit 005

Qwen3-0.6B, bf16 backbone, PyTorch 2.10, RTX A5000. Starting weights are the completed SFT-003 checkpoint. Source for the GPU audit: `230ab41`; source for offline analyses: `420134d`. Eight training and eight validation prompts; gradient and directional-step measurements use only the first training prompt. No persistent model updates were made.

## Findings

- Both feature-capture parity checks passed: every vocabulary logit was bit-identical to its corresponding baseline across the 16 prompts, both probe conditions, and both gradient modes.
- The probe changes the final hidden state by about 0.61% of its L2 norm in the training path and 0.49% in inference, averaged over these prompts.
- Training-path versus inference-path probe-difference vectors have mean cosine similarity 0.00458 across the 16 prompts. The first training prompt has cosine 0.0269. The two compiler paths expose different numerical signals.
- On the first prompt, the gradients of Yes/No log odds have norms 303.90 (absent) and 303.64 (present), cosine 0.999970, and difference norm 2.3506. Balanced labels already cancel the shared component when their loss coefficients match; near-parallel gradients alone do not prove optimization must fail.
- A positive L2 step of 0.01 along the separation gradient increased the actual gap by 0.0625 nats. Steps of 0.1 and 1.0 changed it by -0.001953 and -0.214844, despite positive linear predictions. This diagnoses sensitivity to step size and finite precision, not a proof that all small gradients are invalid. Exact restoration of the original logits passed.

## Frozen-feature readout diagnostic

A separate FP32 affine readout was trained for 256 Adam updates on the first gradient-enabled feature pair. Inputs were centered and scaled using that pair only; the fixed transform was reused for every evaluation. Initialization reproduces the original answer rows' FP32 dot products. The backbone was frozen; no condition flag was an input.

- Training-pair separation: 0.00760 to 13.16467 nats; binary cross-entropy: 0.69465 to 0.00138.
- The same readout on the same prompt's inference features: separation -0.12240 nats.
- The mean separation on the eight validation prompts was -0.38054 on training-path features and +0.16609 on inference features. This tiny diagnostic set is not evidence of generalization.

This shows that the frozen training features contain readily separable information. It does not show that full-model SFT learned detection, or isolate precision from feature conditioning and freezing. No confidence interval over prompts is claimed for fitting one pair.

The audit cost about $0.036; its pod was deleted. Raw scalar records and per-parameter gradient summaries are included. The tensor artifact is distributed separately as the `audit-005` GitHub release asset `features.pt`.

## Reproduce analysis

```sh
# Download features.pt from the audit-005 release into this directory first.
PYTHONPATH=src python scripts/analyze_audit.py results/audit-005
PYTHONPATH=src python scripts/fit_readout.py results/audit-005
```
