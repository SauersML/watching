# Scaled data and regularization: intermediate results 025–026

**Snapshot: detector fits complete; animal fits and fresh-compilation checks remain in progress.** These are full-native checks in the fitting process. `test_fresh` means newly authored held-out contexts, **not** a fresh compiler process.

Native untied Qwen3-8B; 1,024 paired training contexts; 16 training question wordings; BF16, batch one, length 64; NVIDIA L40 and PyTorch 2.10.0. Only existing output rows are fitted. Source: `1df487fe3c050c8344f1aeb928ac9a6eb966989c`. See the [protocol](../../experiments/scaled-choices-025-026.md) and archived manifests for the exact setup.

Six regularization strengths were compared using 64 validation prompts. Both detectors selected 100, the largest tested value. Cached native-head evaluations passed all-vocabulary parity on training and validation inputs at baseline and after the selected edit. Final scores below use the whole native model. Validation was used for selection; test sets were not.

| Detector | Split | Actual top-token accuracy | Full-vocabulary loss | Binary loss | Paired log-odds separation |
|---|---|---:|---:|---:|---:|
| Probe | Validation, 64 | 55.47% | 0.966 | 0.858 | 0.428 |
| Probe | Existing test contexts, 64 | 54.69% | 0.991 | 0.872 | 0.416 |
| Probe | Newly reserved contexts, 128 | 51.95% | 1.227 | 1.143 | 0.454 |
| Probe | New structural families, 64 | 50.00% | 1.624 | 1.406 | 0.396 |
| Gradient recording | Validation, 64 | 78.13% | 0.747 | 0.676 | 3.385 |
| Gradient recording | Existing test contexts, 64 | 73.44% | 0.805 | 0.738 | 3.742 |
| Gradient recording | Newly reserved contexts, 128 | 64.06% | 1.552 | 1.492 | 3.250 |
| Gradient recording | New structural families, 64 | 55.47% | 2.890 | 2.664 | 2.997 |

Losses and separations are in nats. A predictor assigning half its probability to each correct-answer candidate has loss ln(2), approximately 0.693. Higher accuracy does not guarantee lower log loss: confident mistakes matter. The score-file field `accuracy_ci95` describes binary forced-choice accuracy, **not** unrestricted top-token accuracy; do not attach it to the accuracy column above.

## What improved, and what did not

The [matched validation audit](calibration.md) compares identical prompt records with 023–024. Stronger regularization sharply reduces prompt-dependent answer bias and confident errors. For execution detection it retains useful separation; for probe detection it also suppresses most of the separation. The larger dataset with the old weak penalty alone does not show a consistent improvement across both tasks. Data, fitting implementation and execution details also changed, so this is not a clean data-size ablation.

For prompt-specific scores `s_absent = c − d/2` and `s_present = c + d/2`, positive separation `d` only ranks the pair. Correct binary decisions in both executions require `d > 2|c|`, apart from tie conventions. Other vocabulary tokens impose an additional requirement. The audit's hypothetical per-prompt centering requires both executions; it is a diagnostic, not a deployable result.

These results do not establish reliable probe detection. Execution detection is stronger but still loses calibration and accuracy on new wording/contexts. Fresh-compilation stability has not yet been measured for these selected edits.

## Reproduction

The [regularization figure](ridge_selection.pdf) shows training and validation across candidates. Its streamed-summary points have no inferred confidence intervals. The calibration report computes paired prompt-bootstrap intervals directly from raw records, marked descriptive because validation selected the winning candidate.

```sh
python scripts/plot_ridge_selection.py \
  --execution results/scaled-025-026/execution/detector/metrics.jsonl \
  --probe results/scaled-025-026/probe/detector/metrics.jsonl \
  --output results/scaled-025-026/ridge_selection
python scripts/analyze_choice_calibration.py \
  --old-archive results/diverse-023-024 \
  --new-archive results/scaled-025-026 \
  --output results/scaled-025-026/calibration
```

Per-run archive indices record hashes of this intermediate snapshot. Incomplete stages are included only to the last complete structured record. Feature tensors, base-model weights and private infrastructure logs are excluded.
