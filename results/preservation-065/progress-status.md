# Preservation training: completed 6,144 updates

The predeclared development objective selects update **5,120**, rather than the final update. The selected checkpoint answers both conditions correctly on all 96 cached development pairs; all have strict Yes/No margins and neither condition has an exact Yes/No tie. Mean correct-answer probability is 99.91%, full-vocabulary target NLL is 0.000933, and the paired log-odds gap is 54.056 nats [51.772, 56.584]. This is development-selected performance, not an independent test result. An all-correct bootstrap interval collapses to 100% and does not establish perfect population accuracy.

| Update | Both answers correct | Strict Yes/No margin in both | Paired gap | Detector NLL | Selection objective |
|---:|---:|---:|---:|---:|---:|
| 0 | 0/96 · 0.0% | 0/96 | -0.016 | 5.24478 | 5.24478 |
| 256 | 7/96 · 7.3% | 3/96 | 0.352 | 0.73026 | 0.74455 |
| 512 | 4/96 · 4.2% | 2/96 | 1.207 | 1.21114 | 1.21817 |
| 768 | 10/96 · 10.4% | 9/96 | 2.667 | 1.23601 | 1.24262 |
| 1,024 | 62/96 · 64.6% | 59/96 | 3.245 | 0.45225 | 0.45823 |
| 1,280 | 48/96 · 50.0% | 47/96 | 3.253 | 0.52338 | 0.53012 |
| 1,536 | 58/96 · 60.4% | 55/96 | 4.794 | 0.43454 | 0.43950 |
| 1,792 | 69/96 · 71.9% | 67/96 | 5.172 | 0.36877 | 0.37481 |
| 2,048 | 73/96 · 76.0% | 73/96 | 6.648 | 0.31292 | 0.31882 |
| 2,304 | 78/96 · 81.2% | 73/96 | 8.289 | 0.29837 | 0.30412 |
| 2,560 | 70/96 · 72.9% | 69/96 | 10.174 | 0.51691 | 0.52365 |
| 2,816 | 78/96 · 81.2% | 78/96 | 15.376 | 0.19960 | 0.20843 |
| 3,072 | 93/96 · 96.9% | 93/96 | 24.715 | 0.03960 | 0.05116 |
| 3,328 | 95/96 · 99.0% | 95/96 | 36.375 | 0.00805 | 0.01883 |
| 3,584 | 94/96 · 97.9% | 94/96 | 40.159 | 0.03031 | 0.03957 |
| 3,840 | 96/96 · 100.0% | 96/96 | 48.503 | 0.00033 | 0.00707 |
| 4,096 | 96/96 · 100.0% | 96/96 | 49.211 | 0.00247 | 0.00884 |
| 4,352 | 96/96 · 100.0% | 96/96 | 51.061 | 0.00366 | 0.00916 |
| 4,608 | 94/96 · 97.9% | 94/96 | 48.057 | 0.04895 | 0.05433 |
| 4,864 | 96/96 · 100.0% | 96/96 | 53.171 | 0.00352 | 0.00819 |
| 5,120 | 96/96 · 100.0% | 96/96 | 54.056 | 0.00093 | 0.00564 |
| 5,376 | 94/96 · 97.9% | 94/96 | 50.556 | 0.04707 | 0.05152 |
| 5,632 | 95/96 · 99.0% | 95/96 | 53.342 | 0.00730 | 0.01170 |
| 5,888 | 94/96 · 97.9% | 94/96 | 53.824 | 0.01045 | 0.01477 |
| 6,144 | 95/96 · 99.0% | 95/96 | 53.865 | 0.00677 | 0.01113 |

At final update 6,144, development performance is 95/96 both correct, with no exact Yes/No ties, NLL 0.006771 and selection objective 0.011134. The selected update 5,120 has the better objective, 0.005640. The figure includes every evaluation and regression.

On 200 separate ordinary development chats at the selected checkpoint, first-token agreement with the frozen native tail is 98% for HF source states and 97.5% for vLLM source states. Mean reply-first-weighted KL is 0.004729 and 0.004685 respectively. These are cached-source/tail comparisons, not whole-model free-generation results. Their aggregate logs do not support confidence intervals. Selection minimizes detector cross-entropy plus ordinary preservation KL averaged over both source runtimes.

Remote and local SHA256 agree for the final tail, resume state, chosen metadata and contract. The actual resume state records completed_steps=6144 and best.step=5120. All 17 BF16 tail tensors exactly equal the saved best weights. See final-checkpoint-verification.json. The tail is 721,455,488 bytes, SHA256 09adaf7ea7305ac086dfd543edaa9e42a59fee65ede7b83eeec99bb9194bb34e.

Historical whole-model endpoint evaluation is distinct from the cached training runtime. It gives 93/96 both correct on development prompts and 64/68 on historical nondevelopment prompts. Every scalar score was checked against its saved full-vocabulary tensor. Both vLLM passes make the same classifications, but their vocabulary scores differ: 23,020,431/24,917,504 entries, maximum absolute difference 2.2499 nats. Evaluation enables prefix caching while source capture disables it; cache state is a candidate explanation, not an established cause. See historical-endpoint-audit.json. The separate locked 64-prompt whole-model endpoint grid subsequently passed 64/64 pairs in both serving repetitions. The selected model has gap 46.775 nats [43.830, 49.662] on pass 0; see fresh-endpoint-audit.json. The historical and fresh results are different prompt banks and should not be pooled silently. The original 48-token HF ordinary diagnostic also completed; the separate 128-token, four-arm ordinary audit remains pending.

The primary metric remains the actual highest-scoring vocabulary token in each condition. The supplementary strict metric additionally requires the correct answer to exceed the other answer. Exact-tie counts and prompt IDs are retained in the JSON report; other vocabulary ties cannot be reconstructed from training rows. Full historical endpoint vectors have no maximal ties.

The training diagnostic remains the first 32 sorted content prompts, not a representative random sample of the full mixture. The predeclared boundary-identification filter excludes five training pairs, leaving 13,251 eligible pairs; all 96 development pairs remain. This is conditional filtering, distinct from the four wrapper full-vocabulary parity checks.

Raw paired evaluations, aggregate progress, chosen metadata and SHA256 receipts are in progress-raw/. Earlier immutable local snapshots remain retained. Pointwise 95% family-cluster bootstrap intervals describe this prompt bank and fit, exclude seed/hardware variation, and do not cover all crossed shared-context dependence or checkpoint-selection uncertainty. They are not simultaneous learning-curve bands.
