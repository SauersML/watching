# 032: training two native blocks and the full output matrix

Run `watching-two-block-head-032` completed **1,024 additional updates**, ending at global update **1,152**. Qwen3-8B's existing blocks 34–35, final normalization, and full untied output matrix were trained. This changes no architecture and adds no classifier. The starting tail is the same 029r checkpoint used by 030; the output matrix initially contains the original model weights. See the [protocol](protocol.md), [learning curves](progress.pdf), and [raw scores](fit/).

Training improves practiced-prompt detection, and the paired numerical signal extends to the small development and comparison cohorts. It does **not** become a reliable answer on those cohorts: all 32 comparison prompts have positive probe-minus-no-probe log-odds, but only 34/64 individual answers are correct. Usually the probe shifts the score without crossing the Yes/No decision boundary.

## Fixed endpoint

These are actual full-vocabulary next-token accuracies. Intervals use 5,000 whole-prompt bootstrap resamples, preserving both conditions, with seed 20261016. They are conditional on these prompts and this single fit, not training-seed, hardware, or population-family uncertainty. The separate whole-graph transfer evaluation is not an independent training run.

| Runtime | Cohort | Gap, nats [95% CI] | Actual accuracy [95% CI] | Full-vocabulary CE [95% CI] |
|---|---|---:|---:|---:|
| Split inference | Training, 16 | 2.484 [2.102, 2.914] | 78.13% [65.63%, 90.63%] | 0.496 [0.340, 0.703] |
| Split inference | Development, 8 | 0.703 [0.531, 0.891] | 50% [50%, 50%] | 1.783 [1.486, 2.100] |
| Split inference | Comparison, 32 | 0.750 [0.641, 0.867] | 53.13% [50%, 57.81%] | 1.489 [1.193, 1.817] |
| Whole-graph transfer | Training, 16 | 2.414 [2.047, 2.836] | 78.13% [65.63%, 90.63%] | 0.506 [0.347, 0.720] |
| Whole-graph transfer | Development, 8 | 0.750 [0.516, 1.047] | 50% [50%, 50%] | 1.747 [1.412, 2.104] |
| Whole-graph transfer | Comparison, 32 | 0.719 [0.609, 0.832] | 53.13% [50%, 57.81%] | 1.471 [1.184, 1.794] |

The degenerate development accuracy interval occurs because every prompt contributes exactly one correct answer across its two conditions. It does not express certainty about population accuracy.

Compared with this run's resumed baseline, training gap improves by **2.445 nats [2.070, 2.875]** and development gap by **0.672 [0.500, 0.844]**. Development full-vocabulary CE nevertheless worsens by **1.050 [0.775, 1.342]**. A positive paired gap is therefore not equivalent to calibrated Yes/No classification.

On the 32 comparison prompts, the mean full-vocabulary correct-token probability is 53.17%; both conditions are answered correctly on only 2/32 prompts. The log-odds midpoint RMS is 3.68 nats versus mean paired separation 0.75 nats. Answer-pair mass is 99.99%, so poor performance mainly reflects prompt-dependent Yes/No bias, not unrelated output words. CE 1.489 is substantially worse than the balanced two-answer reference ln(2) = 0.6931.

The comparison prompts are the same 32 contexts used by 030, which were inspected during this work. They exclude 029's earlier 32 endpoint contexts but are **development evidence for subsequent design choices, not a new untouched test**. They are scored only after the fixed training endpoint. Caching their locked inputs before training serves memory management; no test scores or gradients were used to fit or select the checkpoint.

## Comparison with the frozen-head run

At 1,024 additional updates, frozen-head 030 has training gap 1.914 nats, accuracy 81.25%, and CE 0.414; full-head 032 has gap 2.484, accuracy 78.13%, and CE 0.496. Development CE is 1.039 versus 1.783, with 50% accuracy in both. These results do not demonstrate an advantage from training the output matrix.

This is **not an exact controlled ablation**. Prompt records and order match, but fresh compilation produces different frozen-source hidden-state hashes for 9/16 training, 5/8 development, and 18/32 comparison prompts, in both conditions. Mask and rotary-argument hashes match. Run 032's initial scored outputs reproduce 029r's endpoint exactly; run 030's restart differed by up to 0.125 nats in condition log-odds. Differences between their later trajectories cannot be attributed solely to freeing the output matrix. [The analysis](analyze.py) verifies these comparisons from the archived records and fingerprints.

## Integration, memory, and provenance

All **38** gradient-enabled versus no-gradient tail checks pass exactly. After caching inputs, both full source models move to host memory: allocated GPU memory falls to **2.027 GB**, then rises to **6.060 GB** when FP32 optimizer masters are allocated. The overall run peaks at **34.904 GB**; this includes restoring the full source models. The experiment finishes in 506.6 seconds after process setup.

The optimizer clips **1,020 of 1,024 updates** at norm one. Learning remains oscillatory; the reported checkpoint is the fixed endpoint, not the best observed checkpoint. Plot intervals are pointwise, not simultaneous learning-curve bands.

The learned artifact is saved before transfer diagnostics. Restoring source models to the GPU preserves extraction parity and every cached boundary tensor exactly. After installing the trained blocks, normalization, and full output matrix, extraction parity and frozen-prefix identity still pass. Both native source heads match the learner's final digest. The learned head is BF16 with shape `[151936, 4096]`; its stored tensor digest was independently recomputed locally:

- Original head SHA-256: `6e46ee56769d64b9a338f350100bdfca5f26a91c3fb5bf479ffcb36c733f7939`
- Learned head SHA-256: `380e52df21b95dc60115ba1ec185c7f44a66131aef0d170eb1fca143dc34d2f5`

The [archive manifest](archive.json) hashes all 50 compact JSON/JSONL files copied unchanged from the completed run. Large weights and boundary-cache tensors remain local and are represented by their hashes and sizes. Infrastructure logs and account metadata are excluded. Local implementation tests cover native all-vocabulary row updates, source/head independence, artifact validation, and continuation behavior; the completed GPU run supplies the offload/restore integration checks.

Reproduce the analysis and figure:

```sh
python results/two-block-head-032/analyze.py
python scripts/plot_tail_progress.py results/two-block-head-032/fit results/two-block-head-032/progress
```

The comparison portion of the analysis also uses the adjacent `results/two-block-030` archive. The current repository test suite contains later prompt-batching changes; run 032 used its immutable earlier source snapshot, whose source-file hash is recorded in its manifest.
