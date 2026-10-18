# Preservation training: all evaluations through update 2,816

| Update | Both answers correct | Strict Yes/No margin in both | Paired gap | Detector NLL | Selection objective |
|---:|---:|---:|---:|---:|---:|
| 0 | 0/96 · 0.0% | 0/96 · 0.0% | -0.016 | 5.245 | 5.245 |
| 256 | 7/96 · 7.3% | 3/96 · 3.1% | 0.352 | 0.730 | 0.745 |
| 512 | 4/96 · 4.2% | 2/96 · 2.1% | 1.207 | 1.211 | 1.218 |
| 768 | 10/96 · 10.4% | 9/96 · 9.4% | 2.667 | 1.236 | 1.243 |
| 1,024 | 62/96 · 64.6% | 59/96 · 61.5% | 3.245 | 0.452 | 0.458 |
| 1,280 | 48/96 · 50.0% | 47/96 · 49.0% | 3.253 | 0.523 | 0.530 |
| 1,536 | 58/96 · 60.4% | 55/96 · 57.3% | 4.794 | 0.435 | 0.439 |
| 1,792 | 69/96 · 71.9% | 67/96 · 69.8% | 5.172 | 0.369 | 0.375 |
| 2,048 | 73/96 · 76.0% | 73/96 · 76.0% | 6.648 | 0.313 | 0.319 |
| 2,304 | 78/96 · 81.2% | 73/96 · 76.0% | 8.289 | 0.298 | 0.304 |
| 2,560 | 70/96 · 72.9% | 69/96 · 71.9% | 10.174 | 0.517 | 0.524 |
| 2,816 | 78/96 · 81.2% | 78/96 · 81.2% | 15.376 | 0.200 | 0.208 |

The current selected checkpoint is update 2,816: both conditions are correct on 78/96 development prompts (81.25%; descriptive pointwise 95% family-cluster bootstrap interval 71.6–93.9%). All 78 successful pairs have strict Yes/No separation. Two Trainer executions have exact Yes/No ties, but both are errors, so they do not inflate success. The mean paired gap is 15.376 nats [14.984, 15.795]; mean target probability is 90.8% and full-vocabulary target NLL is 0.200. Trainer Yes accuracy is 89.6%, while serving No accuracy is 91.7%.

The primary metric remains the actual highest-scoring vocabulary token in each condition. The supplementary strict metric additionally requires the correct answer to exceed the other answer. Exact-tie counts and prompt IDs are retained in the JSON report; other vocabulary ties cannot be reconstructed from these rows. The figure includes every evaluation and regression, rather than only selected checkpoints.

On 200 separate ordinary development chats, first-token agreement with the frozen native tail is 98% for HF source states and 98.5% for vLLM source states. Mean reply-first-weighted KL is 0.00863 and 0.00903 respectively. These are cached-source/tail comparisons, not whole-model free-generation results. Their aggregate logs do not support confidence intervals.

Selection uses the predeclared development objective: detector cross-entropy plus ordinary preservation KL averaged over both source runtimes. The latest selected objective is 0.20843. The remote progress snapshot records training through update 2,944 with checkpoint 2,816 selected. The independently inspected local resume artifact contains completed_steps=2816 and best.step=2816; all 17 saved tail tensors match its best weights exactly. The selected-tail hash is recorded in intermediate-saved-checkpoint.json. This verifies a recoverable intermediate checkpoint, not final completion.

The training diagnostic remains the first 32 sorted content prompts, not a representative random sample of the full mixture. The predeclared boundary-identification filter excludes five training pairs, leaving 13,251 eligible pairs; all 96 development pairs remain. This is conditional filtering, distinct from the four wrapper full-vocabulary parity checks.

Raw paired evaluations, aggregate progress, chosen-checkpoint metadata and SHA256 receipts are in progress-raw/. Earlier immutable local snapshots remain retained. These intervals describe this prompt bank and fit, exclude seed/hardware variation, and do not cover all crossed shared-context dependence or checkpoint-selection uncertainty. Final fresh whole-runtime evaluation and ordinary free generation still follow selection.
