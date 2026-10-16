# Family-diverse native answer fitting 028: modest execution detection; weak animal-answer transfer

**Completed run, runner exit 0.** Detector fitting, animal fitting, both fresh-process evaluations, and both feature-stability audits completed.

Qwen3-8B, BF16, native untied input/output matrices; 1,024 training prompts, 64 development prompts, and four paired 128-prompt evaluation cells. Only the existing two answer-token output rows are fitted per task. Internal weights stay fixed. The detector answers Yes/No; animal supervision assigns cat to the positive execution condition and owl to the negative condition, without providing that condition in the text. This is supervised condition-dependent choice, not evidence of an intrinsic animal preference. See the [protocol](../../experiments/family-choices-027-028.md) and [performance plot](performance.pdf).

Three-fold cross-validation uses training question families only. Both detector and animal fits selected lambda **0.1** from the prespecified grid; their final FP64 optimizations converged. Reported scores below are actual native BF16 model outputs, not fitting-surrogate predictions. The animal objective includes detector-preservation training examples. Fresh evaluation installs both detector and animal answer rows. Detector fit scores precede the animal edit, so full-vocabulary fit/fresh comparisons include both recompilation and possible competition from the animal tokens.

## Endpoint evidence

Fresh execution detection on ordinary new contexts reaches **61.72% accuracy [58.20%, 65.23%]**, gap **0.3125 nats [0.2773, 0.3496]**, and CE **0.6599 [0.6441, 0.6777]**, below ln(2). On both new question and content families, accuracy is **54.69% [52.34%, 57.42%]**, but CE **0.9279** is worse than ln(2): some answers are confidently wrong. The animal test remains weak: **51.56% [48.83%, 54.69%]**, CE **0.7400**. A positive paired log-odds gap does not by itself imply calibrated individual decisions.

All intervals below resample complete paired prompts 5,000 times, seed 20261016. Accuracy is actual full-vocabulary argmax. Intervals are conditional on the available families and one fitted model, not family-population or training-seed uncertainty. Several cohorts and metrics are reported; intervals are pointwise without multiplicity adjustment. Development is reporting-only. New-question cells reuse contexts from their corresponding familiar-question cells, so the four evaluation cells are not four independent samples.

| Stage | Cohort | Gap, nats [95% CI] | Actual accuracy [95% CI] | Full CE |
|---|---|---:|---:|---:|
| execution | Training | 0.3230 [0.3088, 0.3364] | 68.55% [67.14%, 70.02%] | 0.6274 |
| execution | Development | 0.2969 [0.2461, 0.3477] | 63.28% [57.81%, 68.75%] | 0.6525 |
| execution | New contexts | 0.2949 [0.2559, 0.3340] | 59.38% [55.86%, 62.89%] | 0.6647 |
| execution | New question families | 0.2832 [0.2500, 0.3184] | 54.30% [51.95%, 57.03%] | 0.9794 |
| execution | New content families | 0.2461 [0.2148, 0.2773] | 57.42% [54.69%, 60.55%] | 0.7438 |
| execution | Both new | 0.2402 [0.2051, 0.2734] | 54.30% [51.95%, 57.03%] | 0.9246 |
| fresh_execution | Training | 0.2932 [0.2798, 0.3069] | 66.02% [64.60%, 67.53%] | 0.6356 |
| fresh_execution | Development | 0.2812 [0.2188, 0.3477] | 57.81% [53.91%, 62.50%] | 0.6598 |
| fresh_execution | New contexts | 0.3125 [0.2773, 0.3496] | 61.72% [58.20%, 65.23%] | 0.6599 |
| fresh_execution | New question families | 0.2871 [0.2520, 0.3242] | 54.69% [52.34%, 57.42%] | 0.9820 |
| fresh_execution | New content families | 0.2637 [0.2324, 0.2949] | 57.42% [54.30%, 60.55%] | 0.7408 |
| fresh_execution | Both new | 0.2363 [0.1992, 0.2715] | 54.69% [52.34%, 57.42%] | 0.9279 |
| preference | Training | 0.1370 [0.1211, 0.1533] | 56.05% [54.79%, 57.28%] | 0.6818 |
| preference | Development | 0.1719 [0.1133, 0.2305] | 54.69% [51.56%, 58.59%] | 0.7560 |
| preference | New contexts | 0.0879 [0.0430, 0.1348] | 51.17% [48.44%, 53.91%] | 0.7471 |
| preference | New question families | 0.1016 [0.0547, 0.1504] | 51.95% [49.61%, 54.69%] | 0.9303 |
| preference | New content families | 0.0977 [0.0586, 0.1367] | 50.39% [48.05%, 52.73%] | 0.9059 |
| preference | Both new | 0.1104 [0.0664, 0.1543] | 50.39% [48.44%, 52.34%] | 0.9889 |
| fresh_preference | Training | 0.1326 [0.1162, 0.1489] | 55.86% [54.64%, 57.08%] | 0.6830 |
| fresh_preference | Development | 0.1445 [0.0742, 0.2188] | 53.91% [50.00%, 57.81%] | 0.7594 |
| fresh_preference | New contexts | 0.0996 [0.0586, 0.1426] | 51.56% [48.83%, 54.69%] | 0.7400 |
| fresh_preference | New question families | 0.1035 [0.0605, 0.1465] | 52.34% [50.00%, 54.69%] | 0.9286 |
| fresh_preference | New content families | 0.1289 [0.0898, 0.1680] | 50.78% [48.05%, 53.91%] | 0.9020 |
| fresh_preference | Both new | 0.1006 [0.0625, 0.1406] | 50.78% [49.22%, 52.34%] | 0.9899 |

Full CE intervals and original runtime summaries are in [endpoints.json](endpoints.json). The balanced uninformative two-answer reference has CE ln(2) = 0.6931 nats. A narrowed comparison with 019 would not isolate the optimizer: these runs also change question diversity, context count, fixed length (96 rather than 64), and hardware.

## Fresh execution and archive completeness

Fresh feature identity checks match ordered context IDs but do not certify unchanged compilation. Mean fit-to-fresh footprint cosine is **0.4333** for detector prompts and **0.6517** for animal prompts; mean footprint-change norms are **3.4407** and **2.8373**, respectively. Thus the signal changes materially across execution instances, even though the detector retains modest performance. The original audit JSON contains per-prompt measurements and identity limitations. Fresh processes are transfer checks, not independent training replicas.

Source commit: `1362f3181f7739b04b47d3cef0927591672514e0`. [archive.json](archive.json) hashes all **57** copied compact JSON/JSONL artifacts and **16** local-only tensor artifacts. No weights, activation caches, raw infrastructure logs, or account metadata are committed. Initial non-retry startup failures remain separately archived in [family-027-028-startup](../family-027-028-startup/).

Reproduce endpoint analysis and plot from repository root:

```sh
python results/family-probe-027/analyze.py results/family-execution-028 --task execution
```
