# 033: broader prompts and four-pair gradient accumulation

Run `watching-two-block-batch-033` completed **1,024 optimizer updates** from the original Qwen3-8B checkpoint, using **64 training prompts** and **four distinct prompt pairs per update**. This is 4,096 pair visits, or 64 visits per training prompt. The native final two blocks (34–35) and final normalization were trained; the input embeddings, earlier blocks, and entire original output matrix stayed fixed. No classifier or architecture change was added.

The fixed endpoint reaches **81.25% actual next-token accuracy on training prompts**. On the 64 endpoint comparison prompts, accuracy is **57.03%**, with mean probe-minus-no-probe log-odds **0.8125 nats**. This transfers partly to whole-model compiled execution: **56.25%** accuracy and **0.6719 nats** separation. Full-vocabulary loss remains worse than the balanced two-answer reference ln(2), so the model still has substantial calibration errors.

See the [protocol](protocol.md), [learning curves](progress.pdf), [raw records](fit/), and [reproducible endpoint analysis](analyze.py).

## Training recipe

- Start from the original Qwen3-8B weights, not a previous trained tail.
- Freeze the computation through block 33. Collect its actual ordinary-inference residual states under both probe conditions, requiring boundary extraction to preserve every vocabulary logit.
- Train the existing final two blocks and final norm using an eager BF16 math-SDPA tail and the frozen native vocabulary matrix.
- Use paired full-vocabulary cross-entropy: absent target `No`, present target `Yes`. Each update averages four prompt pairs, eight condition losses.
- Evaluate each condition sequentially at the original singleton shape. Accumulate its gradient into FP32 masters before clearing native gradients; divide each loss by eight, then clip once and take one AdamW step. Learning rate is 5e-6, clipping norm one, and weight decay zero.
- Run the fixed 1,024 updates. Report training and development at zero, 8, 32, and every 64 updates. Score endpoint comparison prompts only after saving the final checkpoint.

All 64 prompts appear exactly once per shuffled epoch, with no repeated prompt inside an update. The archived analysis verifies the 4,096 visits, equal per-prompt counts, and separation of training, development, and comparison contexts. Training and comparison question families are disjoint within this run. The primary runtime is explicitly split; whole-model compiled transfer is separately evaluated.

## Fixed endpoint

Intervals use 5,000 whole-prompt bootstrap resamples, preserving each prompt's two conditions, with seed 20261016. They describe these cohorts and this one run. They do not cover training-seed, hardware, or unseen-family sampling variation. Actual accuracy uses the full vocabulary argmax, not forced Yes/No selection.

| Runtime | Cohort | Gap, nats [95% CI] | Actual accuracy [95% CI] | Full-vocabulary CE [95% CI] |
|---|---|---:|---:|---:|
| Split inference | Training, 64 | 1.271 [1.168, 1.379] | 81.25% [75%, 86.72%] | 0.478 [0.457, 0.501] |
| Split inference | Development, 32 | 1.090 [0.918, 1.277] | 59.38% [53.13%, 67.19%] | 0.980 [0.768, 1.213] |
| Split inference | Comparison, 64 | 0.813 [0.707, 0.928] | 57.03% [53.13%, 61.72%] | 1.053 [0.935, 1.186] |
| Whole-graph transfer | Training, 64 | 1.152 [1.051, 1.254] | 80.47% [74.22%, 85.94%] | 0.502 [0.481, 0.526] |
| Whole-graph transfer | Development, 32 | 0.883 [0.730, 1.059] | 57.81% [51.56%, 64.06%] | 1.048 [0.826, 1.293] |
| Whole-graph transfer | Comparison, 64 | 0.672 [0.576, 0.770] | 56.25% [52.34%, 60.16%] | 1.092 [0.968, 1.235] |

The positive accuracy intervals are conditional bootstrap intervals, not a claim of zero-assumption significance. The comparison set comes from the already inspected `test_wording` pool. It is exploratory development evidence for subsequent design choices, not a newly untouched test. No checkpoint was selected using its outcomes.

Relative to the same run's initial weights, training separation improves by **1.271 nats [1.158, 1.387]**, and development separation improves by **1.027 [0.836, 1.238]**. Development accuracy improves by **12.5 percentage points [4.69, 20.31]**. Initial development CE was 4.424 nats, so its large loss reduction partly reflects correcting the base model's answer preference. Even after that reduction, endpoint CE 0.980 is worse than ln(2) = 0.6931. No initial endpoint-test score was collected.

The final comparison-set correct-token probability averages 54.84%. Its log-odds midpoint RMS is 2.50 nats, versus paired separation 0.813 nats; prompt-dependent answer bias remains larger than the condition signal. Yes/No probability mass is 99.98%, so unrelated vocabulary is not the main limitation.

## Learning trajectory and scope

The plotted trajectory is still nonmonotonic. Averaging four prompt pairs did not eliminate oscillation: **1,003 of 1,024 updates** were clipped. The endpoint is reported even though earlier recorded checkpoints have larger gaps. The plot's intervals are pointwise, not a simultaneous confidence band, and the observed maximum is descriptive rather than a selected result.

This is a new exploratory recipe, not an isolated causal test of batching. Compared with 030, it changes the training set size, initialization, number of optimizer updates, and accumulation. Compared with 032, it also freezes the output matrix. Fresh compilation can change numerical footprints. Apparent differences between these runs cannot be attributed solely to any one of those changes.

## Integration and provenance

All **38 gradient-enabled versus no-gradient tail checks** pass exactly. Boundary extraction passes before training, and after installing the trained native blocks and norm, full-model extraction parity and frozen-prefix cache identity pass again. The output head remains unchanged and matches the original digest:

`6e46ee56769d64b9a338f350100bdfca5f26a91c3fb5bf479ffcb36c733f7939`

The split and whole-model scores are not identical; transfer changes are reported separately in the table. The completion gates establish extraction/source integrity, not numerical equivalence of these two runtime paths.

The fitting process completes in **487.1 seconds**, with runner exit status zero and overall peak GPU allocation **40.722 GB** on an NVIDIA L40S. Immutable source snapshot: `d9c8831a`; the full source-file digest is recorded in the manifest. All **49 compact JSON/JSONL artifacts** are copied unchanged and hashed in [archive.json](archive.json). Large weight and cache files remain local; their hashes and sizes are recorded without publishing the tensors. Infrastructure logs and account metadata are excluded.

Reproduce:

```sh
python results/two-block-batch-033/analyze.py
python scripts/plot_tail_progress.py results/two-block-batch-033/fit results/two-block-batch-033/progress
```
