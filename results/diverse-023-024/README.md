# Diverse prompts and full-vocabulary fitting: completed runs 023–024

Both jobs completed with exit status 0. Fixed native output-row fitting produced positive paired log-odds shifts, including on unseen prompts. It did **not** produce a reliably calibrated, general-purpose detector or animal selector: held-out full-vocabulary accuracy is much weaker than the separation metric suggests, and wrong answers can be highly confident. The probe fit also loses substantial training separation after fresh reload and compilation.

## Setup and provenance

- Run 023 (`watching-diverse-probe-023`): reference-keeping read-only query-norm probe at layer 14 versus no probe; both conditions use evaluation mode without gradients.
- Run 024 (`watching-diverse-execution-024`): gradient recording versus no-grad; both conditions use evaluation mode and no activation probe. This is a different execution-path discrimination task, not a replication of probe detection.
- Qwen3-8B with its original **untied** input embeddings and output matrix. All internal weights stayed fixed. Existing Yes/No rows were fitted first, then existing cat/owl rows with detector-training preservation anchors. No classifier, output bias, or architecture change was added.
- Three ridge corrections, penalty 0.001, target paired gap 6 nats and combined answer mass 0.95. These are direct fits to native inference features, not three SFT updates. The single-prompt integration edit was discarded before the many-prompt fit.
- Source commit: `6afd20d406dc1a6fe7024e70118671ef3c60044d`; PyTorch 2.10.0+cu128; NVIDIA L40; BF16 source computation; batch one, padded length 64. The model revision was not recorded (`model_commit: null`), which limits exact reproduction.
- 256 training, 64 validation, 64 familiar-family test, and 64 new-family contexts. Another 64 reworded tests reuse the familiar test contexts; they are paired wording-transfer observations, not an independent cohort.
- The manifests retain `validation_prompts: 32`, an unused generator default when the explicit dataset is loaded. Saved `data.json` and evaluation counts contain **64 actual validation prompts**. Raw manifests are preserved rather than silently rewritten.

See the [fixed protocol](../../experiments/diverse-choices-023-024.md). Per-run `archive.json` files identify source artifacts and checksums. Compact native answer-row weights, prompts, manifests, intermediate scores and final fresh evaluations are included; feature tensors, full base-model weights and private infrastructure logs are omitted.

## Fresh compilation results

Each row below is measured after reload in a separate process with a fresh compiler cache and both fitted answer pairs installed. Separation is the present-minus-absent change in log P(positive answer) − log P(negative answer): Yes/No for detectors and cat/owl for animals. Full-vocabulary accuracy requires the intended answer to be the actual top token, not merely preferred within the answer pair. Full CE is negative log probability of the intended token over the whole vocabulary; binary CE conditions on the two answers. All CE and separation values are nats.

Intervals below resample whole prompt pairs 5,000 times. They are pointwise, conditional on this fit and compilation, and exclude seed, hardware and compiler variability. OOD rows use the whole-family interval instead (four families only; exploratory).

| Condition / task | Split | Separation [95% CI] | Full-vocabulary accuracy | Full CE | Binary CE | Mean correct-token probability |
|---|---|---:|---:|---:|---:|---:|
| probe / detector | train | 3.418 [3.142, 3.679] | 79.30% | 0.521 | 0.446 | 69.16% |
| probe / detector | validation | 3.319 [2.811, 3.844] | 48.44% | 8.674 | 8.435 | 48.25% |
| probe / detector | test | 2.991 [2.487, 3.513] | 52.34% | 6.829 | 6.805 | 50.73% |
| probe / detector | test_wording | 3.256 [2.757, 3.745] | 53.91% | 13.117 | 13.107 | 52.64% |
| probe / detector | ood | 2.801 [2.411, 3.191] | 47.66% | 18.593 | 18.067 | 47.76% |
| probe / animal | train | 1.882 [1.587, 2.172] | 65.23% | 0.866 | 0.756 | 57.52% |
| probe / animal | validation | 2.083 [1.547, 2.625] | 42.97% | 11.863 | 11.194 | 43.55% |
| probe / animal | test | 1.720 [1.235, 2.173] | 46.88% | 13.984 | 13.441 | 45.87% |
| probe / animal | test_wording | 1.529 [0.942, 2.092] | 37.50% | 25.521 | 21.551 | 37.10% |
| probe / animal | ood | 1.329 [0.920, 1.792] | 39.06% | 28.358 | 22.249 | 38.27% |
| execution / detector | train | 5.777 [5.653, 5.901] | 100.00% | 0.128 | 0.074 | 88.34% |
| execution / detector | validation | 5.182 [4.914, 5.451] | 65.62% | 1.812 | 1.774 | 63.39% |
| execution / detector | test | 4.891 [4.557, 5.217] | 64.06% | 1.693 | 1.651 | 62.69% |
| execution / detector | test_wording | 4.912 [4.613, 5.219] | 55.47% | 4.236 | 4.184 | 55.01% |
| execution / detector | ood | 4.595 [4.288, 4.924] | 52.34% | 5.967 | 5.623 | 51.98% |
| execution / animal | train | 5.479 [5.345, 5.611] | 100.00% | 0.155 | 0.084 | 86.05% |
| execution / animal | validation | 4.992 [4.625, 5.348] | 50.78% | 4.141 | 3.458 | 49.62% |
| execution / animal | test | 5.059 [4.734, 5.375] | 54.69% | 3.711 | 3.091 | 52.51% |
| execution / animal | test_wording | 4.487 [4.166, 4.831] | 32.03% | 12.165 | 8.800 | 33.36% |
| execution / animal | ood | 3.808 [2.891, 4.508] | 25.00% | 13.648 | 8.809 | 27.59% |

Positive separation establishes a directional shift between matched executions. It does not ensure each individual execution crosses the correct decision threshold. Large prompt-dependent answer bias explains how positive separation coexists with near-chance accuracy and large CE. In particular, probe animal choices reach only 46.88% full-vocabulary test accuracy; execution animal choices reach 54.69% and deteriorate to 25.00% on the four OOD families. These outcomes do not establish broad conversational animal preferences.

## Improvement and fresh-check stability

The ordinary test improvement compares the original pre-fit scores with the final fresh scores, preserving prompt pairing. This combines fitting with the fresh execution change; it is not an isolated causal estimate of fitting under a fixed compiler artifact.

| Condition / task | Test gain [95% CI], nats | Training separation: fitted → fresh | Training top-token accuracy: fitted → fresh |
|---|---:|---:|---:|
| probe / detector | 3.030 [2.516, 3.561] | 6.013 → 3.418 | 100.00% → 79.30% |
| probe / animal | 1.735 [1.250, 2.192] | 5.972 → 1.882 | 100.00% → 65.23% |
| execution / detector | 4.838 [4.480, 5.189] | 6.014 → 5.777 | 100.00% → 100.00% |
| execution / animal | 5.054 [4.721, 5.378] | 5.979 → 5.479 | 100.00% → 100.00% |

The probe detector loses about 2.59 nats of training separation on the fresh check, and the probe animal fit loses about 4.09 nats. The fresh checks therefore **do not reproduce fitted numerics exactly**. They use distinct compiler executions and, for the detector, additionally install the animal rows. These measurements identify a reproducibility problem; they do not by themselves isolate its exact cause. Execution-path fits survive more strongly, but their held-out calibration remains poor.

## Analysis and figures

Four `analysis.json` reports, one inside each detector/preference directory, contain every available checkpoint, paired gains, endpoint intervals, binary-loss intervals, family analyses and fresh-versus-fitted score agreement. Native per-example and summary full-vocabulary measurements remain in the linked score files. The family analyses preserve all prompts within a sampled family; four OOD family units cannot establish transfer to arbitrary new families.

Regenerate an analysis from the repository root (replace `probe/detector` and `probe/fresh_detector` for the other three fits):

```sh
PYTHONPATH=src python scripts/analyze_native_fit.py \
  results/diverse-023-024/probe/detector \
  results/diverse-023-024/probe/detector/analysis.json \
  --fresh results/diverse-023-024/probe/fresh_detector

python scripts/plot_diverse_choices.py \
  results/diverse-023-024 results/diverse-023-024/performance
```

The figures distinguish baseline, fitted and fresh observations. Intermediate results were inspected while these runs proceeded. These datasets and findings are now development evidence for later experiments, rather than untouched confirmatory tests for adaptations chosen afterward.

### Progress through fitting

[Two-page progress report](progress.pdf) shows [Yes/No detection](progress_detector.png) and [animal answers](progress_preference.png) across the three ridge corrections. Training and validation have measured intermediate points; unseen test contexts appear only at baseline, the fitted endpoint, and the fresh check. Fresh points are disconnected because both answer-row edits are installed and compilation is new; the detector comparison changes more than compilation alone.

Each page shows paired log-odds separation, full-vocabulary correct-token probability, and actual next-token accuracy. Pointwise 95% bootstrap intervals resample whole prompt pairs, using identical prompt ordering and random draws at every step. They are not simultaneous bands and do not include seed or hardware variation. The [machine-readable report](progress.json) records source hashes and every plotted estimate. All 52 plotted evaluations were checked against the archived summary metrics.

```sh
python scripts/plot_choice_progress.py \
  results/diverse-023-024 results/diverse-023-024/progress
```
