# Matched objective and calibration audit

These are **CPU FP64 linear-surrogate calculations on the existing 027 training
activation cache**, not native BF16 model runs. The audit uses no external
validation or test examples. Each internal holdout keeps both conditions of a
prompt together; the question-family holdout excludes four training question
families from fitting. These are exploratory development comparisons, not a new
independent generalization result.

On the 342-prompt question-family holdout:

| Fit | Mean paired log-odds gap | Individual binary accuracy | Full-vocabulary CE |
|---|---:|---:|---:|
| Legacy ridge, three corrections | 1.342 | 51.6% | 5.235 |
| CE, L2 = 0.00001 | 5.737 | 52.8% | 17.642 |
| CE, L2 = 0.001 | 1.162 | 52.9% | 3.184 |
| CE, L2 = 0.1 | 0.029 | 50.4% | 0.758 |

The probe can consistently increase Yes relative to No while both executions
remain on the same side of the decision threshold. Weak regularization learns
a large paired contrast, but unseen prompts introduce larger common answer
biases. Strong regularization reduces those confidently incorrect answers and
also suppresses the contrast. Thus a larger paired gap does not establish a
well-calibrated detector for individual executions.

Three prespecified covariance-shrinkage variants constrained the learned answer
direction to have zero mean training score. They used only training statistics
and scored each held-out execution independently. Their held-out-family binary
accuracies were 52.2–52.6%; this did not resolve the calibration problem.

All candidates, convergence diagnostics, input hashes, and limitations are in
[matched-objectives.json](matched-objectives.json) and
[covariance-diagnostic.json](covariance-diagnostic.json). The weakest CE fits
reached their 3,000-iteration limit; their incomplete convergence is reported.
No candidate was selected using this audit. Native BF16 rounding, fresh
compilation, and actual output-row installation require separate verification.

Reproduction: `scripts/audit_head_regression.py --run RUN --out REPORT.json`;
add `--lda-only` for the covariance diagnostic. `RUN` must contain the original
`many_features.pt`, `many_train_0.json`, `data.json`, and `grouped_cv.json`.
