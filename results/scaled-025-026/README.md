# Scaled data and regularization: completed runs 025–026

**Both jobs completed with runner exit status 0.** Stronger regularization reduces extreme prompt-dependent answer bias, but these results still do not establish reliable read-only-probe detection or general animal preferences. Gradient-recording detection is stronger; fresh compilation and unfamiliar inputs reduce its performance.

## Setup and data

Native untied Qwen3-8B, fixed internal weights, 1024 paired training contexts, 16 training question wordings, BF16, batch one, length 64. Only existing output rows were fitted: first Yes/No, then cat/owl with detector-preservation anchors. Probe 025 contrasts a read-only query-norm capture at layer 14 with no capture, both without gradients. Execution 026 contrasts gradient recording with no-grad, both in evaluation mode without a capture probe. These are distinct tasks.

Source commit: `1df487fe3c050c8344f1aeb928ac9a6eb966989c`; PyTorch 2.10.0 on NVIDIA L40. See the [protocol](../../experiments/scaled-choices-025-026.md) and original manifests. Six ridge penalties were compared using 64 validation prompts. All four final fits selected 100, the largest tested value. Animal selection averaged its own validation full-vocabulary CE and the preserved detector task CE. Test sets were not used to select the penalty.

`test` contains 64 previously used held-out contexts; its results already informed subsequent development. `test_fresh` contains 128 newly authored, reserved contexts from familiar broad families. **It means new contexts, not fresh compilation.** `test_wording` reuses the 64 ordinary test contexts with held-out wording, so those rows are paired transfer checks, not independent cohorts. `ood` contains 64 contexts from four held-out structural families.

## Final fresh-process endpoints

All rows below reload both fitted answer pairs and execute the full model in fresh compiler processes. Accuracy is actual correct top-token accuracy across the whole vocabulary. CE is full-vocabulary negative log probability of the intended token. Separation is present-minus-absent positive/negative answer log odds (Yes/No for detectors; cat/owl for animals), in nats. Positive separation alone does not ensure correct answers on either individual execution.

Each accuracy interval is recomputed from **`greedy_correct`**, resampling paired prompts 5000 times. It is not copied from the score files’ `accuracy_ci95`, which describes binary forced choice. OOD intervals instead resample four whole families with equal weights and are exploratory. Validation intervals are descriptive after penalty selection; all intervals omit training-seed and compiler variability.

### Detector answers

| Condition | Split | Actual accuracy [95% CI] | Full CE | Binary CE | Separation [95% CI] |
|---|---|---:|---:|---:|---:|
| probe | Training 1024 | 80.37% [78.86%, 81.84%] | 0.692 | 0.577 | 0.555 [0.537, 0.572] |
| probe | Validation 64 | 57.03% [53.12%, 61.72%] | 0.965 | 0.856 | 0.436 [0.373, 0.496] |
| probe | Existing test 64 | 54.69% [50.78%, 59.38%] | 0.989 | 0.869 | 0.408 [0.342, 0.473] |
| probe | New contexts 128 | 52.34% [50.00%, 54.69%] | 1.230 | 1.145 | 0.443 [0.396, 0.490] |
| probe | Reworded test 64 | 53.12% [50.78%, 56.25%] | 1.102 | 1.005 | 0.469 [0.404, 0.533] |
| probe | New families 64 | 50.00% [46.09%, 53.12%] | 1.625 | 1.407 | 0.400 [0.348, 0.459] |
| execution | Training 1024 | 94.29% [93.31%, 95.21%] | 0.350 | 0.260 | 2.943 [2.894, 2.992] |
| execution | Validation 64 | 75.00% [68.75%, 81.25%] | 0.847 | 0.776 | 2.811 [2.627, 2.994] |
| execution | Existing test 64 | 67.19% [61.72%, 72.66%] | 0.961 | 0.890 | 2.795 [2.594, 2.990] |
| execution | New contexts 128 | 60.55% [57.03%, 64.06%] | 1.662 | 1.600 | 2.742 [2.595, 2.893] |
| execution | Reworded test 64 | 60.16% [55.47%, 65.62%] | 2.218 | 2.169 | 2.613 [2.412, 2.816] |
| execution | New families 64 | 51.56% [48.44%, 54.69%] | 3.061 | 2.829 | 2.442 [2.099, 2.721] |

### Animal answers

| Condition | Split | Actual accuracy [95% CI] | Full CE | Binary CE | Separation [95% CI] |
|---|---|---:|---:|---:|---:|
| probe | Training 1024 | 72.75% [71.14%, 74.32%] | 0.755 | 0.639 | 0.250 [0.237, 0.262] |
| probe | Validation 64 | 52.34% [49.22%, 56.25%] | 0.984 | 0.840 | 0.160 [0.113, 0.203] |
| probe | Existing test 64 | 56.25% [51.56%, 60.94%] | 0.949 | 0.821 | 0.184 [0.145, 0.227] |
| probe | New contexts 128 | 48.83% [46.48%, 51.17%] | 1.371 | 1.170 | 0.156 [0.123, 0.188] |
| probe | Reworded test 64 | 52.34% [49.22%, 56.25%] | 1.258 | 1.078 | 0.160 [0.113, 0.207] |
| probe | New families 64 | 42.97% [27.34%, 53.12%] | 2.954 | 1.106 | 0.133 [0.098, 0.164] |
| execution | Training 1024 | 91.99% [90.87%, 93.12%] | 0.521 | 0.413 | 1.576 [1.545, 1.608] |
| execution | Validation 64 | 64.06% [57.81%, 70.31%] | 1.187 | 1.054 | 1.744 [1.596, 1.889] |
| execution | Existing test 64 | 62.50% [57.03%, 67.97%] | 1.102 | 0.954 | 1.648 [1.523, 1.770] |
| execution | New contexts 128 | 50.39% [47.66%, 53.52%] | 1.924 | 1.792 | 1.489 [1.401, 1.574] |
| execution | Reworded test 64 | 48.44% [43.75%, 53.12%] | 2.432 | 2.152 | 1.436 [1.312, 1.562] |
| execution | New families 64 | 50.78% [36.72%, 62.50%] | 3.651 | 2.382 | 1.457 [1.324, 1.570] |

On the 128 new contexts, probe detection is 52.34% accurate and animal choices 48.83%; execution detection is 60.55% and animal choices 50.39%. The separation signal can remain positive while individual answers are poorly calibrated. All four full-vocabulary losses on these new contexts exceed ln(2) = 0.693, the loss of assigning exactly half the probability to each answer. Accuracy above 50% does not imply better log loss when confident mistakes occur.

## Fresh execution versus the fitted process

| Condition / task | Ordinary-test accuracy: fit → fresh | Ordinary-test gap: fit → fresh | Training gap: fit → fresh |
|---|---:|---:|---:|
| probe / detector | 54.69% → 54.69% | 0.416 → 0.408 | 0.566 → 0.555 |
| probe / preference | 56.25% → 56.25% | 0.184 → 0.184 | 0.250 → 0.250 |
| execution / detector | 73.44% → 67.19% | 3.742 → 2.795 | 3.557 → 2.943 |
| execution / preference | 64.84% → 62.50% | 1.953 → 1.648 | 1.930 → 1.576 |

Probe results are substantially more stable than the prior weakly regularized 023 fit; the execution detector nevertheless loses ordinary-test accuracy from 73.44% to 67.19% and separation from 3.742 to 2.795 nats. These are separate compiler executions, and fresh detector evaluation additionally installs the animal rows. The checks measure combined reload/compilation/deployment behavior rather than isolating one cause. No fresh internal-feature exports were collected in 025–026, so these results do not locate the activation drift.

## What improved

The [matched validation calibration audit](calibration.md) verifies identical validation records against 023–024. It separates the pair gap `d` from the prompt midpoint `c`: scores are `c−d/2` and `c+d/2`, so strict binary correctness in both conditions requires `d>2|c|`. Stronger penalties reduce the large midpoint bias and confident errors. For probe detection they also suppress much of the useful gap. Keeping the old weak penalty while adding data did not consistently improve both tasks. Data, fitting and execution details change together, so this is not an isolated data-size ablation.

Per-prompt oracle centering in that audit requires both executions and is not deployable performance. The six-candidate validation selection also means winner intervals are not selection-adjusted. Later experiments should treat all observed splits here as development evidence, including the now-observed 128 new contexts.

## Artifacts and reproduction

Per-run archive indices contain 172 source artifacts each, their SHA256 hashes, and sanitized terminal-status provenance. They preserve all complete candidate metrics, native scores, fresh checks, manifests, datasets and compact answer-row weights. Base-model weights, activation tensors and private infrastructure logs are excluded. No failed stages were discarded from these two jobs; both completed. The separate [027–028 startup archive](../family-027-028-startup/) records subsequent jobs that failed before fitting.

`endpoints.json` includes all six splits with actual-accuracy, correct-token-probability and gap intervals, plus paired-prompt full-CE intervals. Four detector/preference `analysis.json` files preserve baseline-to-fit and baseline-to-fresh comparisons and fresh-versus-fitted score agreement. The [endpoint figures](performance.pdf) show the five splits supported by the existing plotter; the 128 new-context split is in the tables and `endpoints.json`. The existing [ridge-selection figure](ridge_selection.pdf) remains unchanged.

```sh
python results/scaled-025-026/build_endpoints.py
python scripts/plot_diverse_choices.py results/scaled-025-026 results/scaled-025-026/performance
python scripts/analyze_choice_calibration.py \
  --old-archive results/diverse-023-024 \
  --new-archive results/scaled-025-026 \
  --output results/scaled-025-026/calibration
```
