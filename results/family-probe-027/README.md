# Family-diverse native answer fitting 027: probe result near chance; final detector reload incomplete

**Partial run, runner exit 1.** Detector fitting, animal fitting, and fresh animal evaluation completed according to stdout. The run was interrupted during compilation for fresh detector evaluation. No fresh detector scores are available. The runner records that the pod was deleted outside this runner; the responsible process and reason are not established. An earlier low-GPU warning is not proof that it caused deletion.

Qwen3-8B, BF16, native untied input/output matrices; 1,024 training prompts, 64 development prompts, and four paired 128-prompt evaluation cells. Only the existing two answer-token output rows are fitted per task. Internal weights stay fixed. The detector answers Yes/No; animal supervision assigns cat to the positive execution condition and owl to the negative condition, without providing that condition in the text. This is supervised condition-dependent choice, not evidence of an intrinsic animal preference. See the [protocol](../../experiments/family-choices-027-028.md) and [performance plot](performance.pdf).

Three-fold cross-validation uses training question families only. Both detector and animal fits selected lambda **0.1** from the prespecified grid; their final FP64 optimizations converged. Reported scores below are actual native BF16 model outputs, not fitting-surrogate predictions. The animal objective includes detector-preservation training examples. Fresh evaluation installs both detector and animal answer rows. Detector fit scores precede the animal edit, so full-vocabulary fit/fresh comparisons include both recompilation and possible competition from the animal tokens.

## Endpoint evidence

Probe detection on the ordinary new-context test is **50.39% accuracy**, with gap **0.0117 nats [−0.0176, 0.0430]** and CE **0.7086**. There is no convincing test discrimination. Animal test accuracy is **49.22%** and CE **0.7622**; new-question animal accuracy is **47.27% [44.92%, 49.22%]**, an exploratory adverse result among multiple reported cohorts.

All intervals below resample complete paired prompts 5,000 times, seed 20261016. Accuracy is actual full-vocabulary argmax. Intervals are conditional on the available families and one fitted model, not family-population or training-seed uncertainty. Several cohorts and metrics are reported; intervals are pointwise without multiplicity adjustment. Development is reporting-only. New-question cells reuse contexts from their corresponding familiar-question cells, so the four evaluation cells are not four independent samples.

| Stage | Cohort | Gap, nats [95% CI] | Actual accuracy [95% CI] | Full CE |
|---|---|---:|---:|---:|
| probe | Training | 0.0366 [0.0249, 0.0483] | 53.08% [51.90%, 54.30%] | 0.6905 |
| probe | Development | 0.0273 [-0.0117, 0.0664] | 53.91% [50.00%, 58.59%] | 0.7015 |
| probe | New contexts | 0.0117 [-0.0176, 0.0430] | 50.39% [47.66%, 53.52%] | 0.7086 |
| probe | New question families | 0.0195 [-0.0156, 0.0547] | 51.17% [48.83%, 53.52%] | 0.7374 |
| probe | New content families | 0.0098 [-0.0215, 0.0410] | 50.78% [48.82%, 53.12%] | 0.7448 |
| probe | Both new | 0.0078 [-0.0234, 0.0391] | 51.56% [49.22%, 53.91%] | 0.7723 |
| preference | Training | 0.0212 [0.0063, 0.0366] | 51.51% [50.49%, 52.54%] | 0.7086 |
| preference | Development | 0.0039 [-0.0586, 0.0664] | 51.56% [46.88%, 56.25%] | 0.7843 |
| preference | New contexts | 0.0234 [-0.0176, 0.0625] | 49.22% [47.27%, 51.17%] | 0.7622 |
| preference | New question families | -0.0371 [-0.0781, 0.0039] | 47.27% [44.92%, 49.22%] | 0.9068 |
| preference | New content families | 0.0410 [0.0019, 0.0820] | 50.39% [48.05%, 53.12%] | 0.8768 |
| preference | Both new | 0.0322 [-0.0117, 0.0752] | 49.61% [48.05%, 50.78%] | 0.9582 |
| fresh_preference | Training | 0.0212 [0.0063, 0.0366] | 51.51% [50.49%, 52.54%] | 0.7086 |
| fresh_preference | Development | 0.0039 [-0.0586, 0.0664] | 51.56% [46.88%, 56.25%] | 0.7843 |
| fresh_preference | New contexts | 0.0234 [-0.0176, 0.0625] | 49.22% [47.27%, 51.17%] | 0.7622 |
| fresh_preference | New question families | -0.0371 [-0.0781, 0.0039] | 47.27% [44.92%, 49.22%] | 0.9068 |
| fresh_preference | New content families | 0.0410 [0.0019, 0.0820] | 50.39% [48.05%, 53.12%] | 0.8768 |

Full CE intervals and original runtime summaries are in [endpoints.json](endpoints.json). The balanced uninformative two-answer reference has CE ln(2) = 0.6931 nats. A narrowed comparison with 019 would not isolate the optimizer: these runs also change question diversity, context count, fixed length (96 rather than 64), and hardware.

## Fresh execution and archive completeness

Local synchronization ended before every final file reached disk. Five fresh animal per-prompt score files are available; the final both-new cohort has only its exact logged summary: gap **0.03223**, actual accuracy **49.61%**, full CE **0.95818**. This missing raw cohort is omitted from the plotted fresh series and fresh bootstrap table. Its fit-stage raw scores exist, but were not substituted as fresh measurements. The stdout completion marker proves fresh animal evaluation finished even though its local metrics file ends earlier.

The recovered feature-audit summary reports all 1,024 absent-condition states bit-identical, 1,017/1,024 present-condition states bit-identical, and mean footprint cosine **0.999999905**. The full per-prompt audit did not synchronize. Exact logged summaries and completion markers are preserved in [logged_stage_outcomes.json](logged_stage_outcomes.json), with source-log hash; the recovered audit is explicitly labeled. No fresh detector activation or behavioral result is inferred from them.

Source commit: `1362f3181f7739b04b47d3cef0927591672514e0`. [archive.json](archive.json) hashes all **45** copied compact JSON/JSONL artifacts and **9** local-only tensor artifacts. No weights, activation caches, raw infrastructure logs, or account metadata are committed. Initial non-retry startup failures remain separately archived in [family-027-028-startup](../family-027-028-startup/).

Reproduce endpoint analysis and plot from repository root:

```sh
python results/family-probe-027/analyze.py results/family-probe-027 --task probe
```
