# Training-only score calibration after run 032

This post-hoc CPU diagnostic fits global calibration parameters using **only the
16 final training prompt pairs** at update 1,024. It then applies the same rule
to each development/test execution independently. It never supplies a held-out
prompt's paired midpoint to the predictor. No model weights, native outputs, or
full-vocabulary distributions were changed.

For the original Yes-minus-No log odds `s`, minimizing training binary
cross-entropy gives:

- Bias only: `s' = s + 0.393345`.
- Positive affine calibration: `s' = 0.846148*s + 0.323018`.
  This corresponds to temperature 1.1818 and an original-score threshold of
  approximately −0.3818. The fit converged; its largest gradient component was
  below 1.4e-10.

| Cohort and metric | Original | Bias only | Positive affine |
|---|---:|---:|---:|
| Training binary CE | 0.4963 | 0.4854 | 0.4816 |
| Development binary CE | 1.7825 | 1.7444 | 1.5041 |
| Test binary CE | 1.4886 | 1.3875 | 1.2274 |
| Development binary accuracy | 50.0% | 50.0% | 50.0% |
| Test binary accuracy | 53.1% | 54.7% | 54.7% |

Calibration reduces overconfidence but does **not** produce a reliable detector.
Test binary CE remains worse than the constant 50/50 predictor's 0.6931. The
affine-calibrated test accuracy has a whole-prompt paired bootstrap 95% interval
of [50.0%, 59.4%]. Its test CE improvement is −0.2612 nats, interval
[−0.3480, −0.1796], conditional on the fitted calibration rule.

Prompt dependence remains substantial: uncalibrated test-prompt midpoint log
odds range from −7.6875 to +2.3125, with quartiles −4.4531 and −0.4375. The mean
probe/no-probe gap is only +0.75. A single training-fitted shift cannot cancel
these differing offsets. This shows that these two global calibration rules
are insufficient; it does not establish that the bias is irreducible or that
another trained representation could not resolve it.

[audit.json](audit.json) records all coefficients, input hashes, cohort metrics,
and paired changes. Intervals use 5,000 bootstrap resamples of whole prompts,
keeping both conditions together and calibration parameters fixed. They exclude
calibration-fitting and training-seed uncertainty. This exploratory analysis was
designed after seeing the original results; it is not a fresh confirmatory test.
Probabilities and losses here are conditional on the Yes/No pair, not native
full-vocabulary scores.

To reproduce from the repository root with NumPy and SciPy installed:

```sh
python results/two-block-head-032-calibration/analyze.py
```

The script reads the archived scores in `results/two-block-head-032/fit` using
paths relative to its own location and regenerates `audit.json`. Use
`--output /tmp/calibration.json` to leave the archived report untouched.
