# Preservation training: all evaluations through update 768

| Update | Development both answers correct | Paired log-odds gap | Detector NLL | Selection objective |
|---:|---:|---:|---:|---:|
| 0 | 0/96 · 0.0% | −0.016 | 5.245 | 5.245 |
| 256 | 7/96 · 7.3% | 0.352 | 0.730 | 0.745 |
| 512 | 4/96 · 4.2% | 1.207 | 1.211 | 1.218 |
| 768 | 10/96 · 10.4% | 2.667 | 1.236 | 1.243 |

The condition-dependent score separation is growing, but answer calibration remains poor. At update 768, the model answers Yes for 95.8% of Trainer states and No for only 14.6% of serving states. The development gap is 2.667 nats [2.564, 2.771]; both-conditions accuracy is 10.4% [4.1%, 17.6%]. Intervals are descriptive pointwise family-cluster bootstrap intervals. These results do not yet establish strong answer accuracy.

The fixed development selection objective combines mean full-vocabulary detector cross-entropy with ordinary preservation KL averaged across both source runtimes. It is lower at update 256 than at 512 or 768, so the retained selected checkpoint is 256. The snapshot's checkpoint metadata was last persisted after update 640; the update-768 evaluation had been written when the read-only snapshot was taken. The figure includes every available evaluation, including regressions, rather than stopping at the selected checkpoint.

On the separate 200 ordinary development chats, first-token agreement with the frozen native tail is 97%/97% at update 256, 98%/99.5% at 512, and 98%/98% at 768 (HF/vLLM source states). At 768 the mean reply-first-weighted KL is 0.00661 and 0.00662 respectively. These are cached-source/tail measurements, not whole-model free-generation results; the aggregate logs do not support confidence intervals.

The plotted training diagnostic is the first 32 sorted content prompts, not a representative random sample of the full training mixture. All 96 development pairs remain eligible. The source-boundary heuristic excluded five training pairs, leaving 13,251 eligible pairs from 13,256 planned. All five have distinct identified attention/residual candidates but a relative-L2 discrepancy above the predeclared 0.1 threshold. This is conditional filtering, not proof those captures are erroneous. It is separate from the four wrapper full-vocabulary parity checks.

Raw paired rows for updates 0, 256, 512 and 768, aggregate progress, checkpoint-selection metadata and SHA256 receipts are retained in progress-raw/. The original update-256 snapshot remains independently retained in the local analysis-inputs-000256 archive. These intervals do not include training-seed or hardware variation; family resampling does not cover all crossed shared-context dependence. Final fresh whole-runtime testing is separate and follows checkpoint selection.
