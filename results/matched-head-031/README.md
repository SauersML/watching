# Matched native head comparison 031: signal versus prompt bias

`watching-matched-head-031r` completed with runner exit **0**, using source
`45fbdda6` on Qwen3-8B, PyTorch 2.10, BF16, an RTX A6000, and fixed length 96.
Both methods used the **same frozen native inference features**, initial answer
rows, prompts, compiled model and device. Only the existing Yes/No unembedding
rows changed; no architecture or internal-weight modification was made.

The old ridge method restores a strong paired condition gap, but that gap does
not yield reliable answers on other prompts. Regularized cross-entropy largely
removes both the gap and the much larger prompt-dependent answer bias. This
matched run supports that tradeoff; it does not show that the original signal
vanished or that ordinary cross-entropy was incorrectly differentiated.

See the [comparison figure](comparison.pdf), [raw native scores](fit/), and
[reproducible analysis](analyze.py).

## Prespecified methods

- **Legacy ridge:** three native-feedback corrections with ridge 0.001 and
  target Yes-minus-No log-odds −1 without the probe and +1 with it. Each
  correction updates the two native rows symmetrically, rounds them to BF16,
  and scores the actual native head. It fits the answer difference; it does not
  directly optimize competition with the rest of the vocabulary.
- **Fixed CE:** convex full-vocabulary cross-entropy with native-coordinate L2
  coefficient **0.1**, fixed before this run based on the earlier 027 selection.
  Cross-validation is not repeated, and development scores select nothing here.
  Original rows are restored before fitting this arm. Its FP64 optimizer
  converged in 95 iterations with objective-error upper bound 0.0000562 nats.

There are **1,024 training and 64 previously inspected development prompts**.
No test data were supplied. This is a diagnostic comparison, not a new
confirmatory generalization experiment. Later changes motivated by these results
will require fresh evaluation data for confirmatory claims.

## Actual native endpoints

All intervals are 5,000 whole-prompt paired bootstrap resamples, seed 20261016.
They are conditional on these prompts/families and this single compiled run.
Accuracy below uses the actual full-vocabulary top token, not a Yes/No-only
choice. The endpoints are prespecified rather than selected by development
performance.

| Method | Cohort | Paired gap, nats [95% CI] | Actual accuracy [95% CI] | Full CE [95% CI] | Binary CE |
|---|---|---:|---:|---:|---:|
| Baseline | Train | −0.0072 [−0.0192, 0.0046] | 46.78% [45.95%, 47.56%] | 4.5368 [4.3608, 4.7161] | 4.1524 |
| Baseline | Development | −0.0117 [−0.0508, 0.0273] | 46.88% [43.75%, 49.22%] | 4.2220 [3.6069, 4.8746] | 3.9238 |
| Legacy ridge | Train | 1.9855 [1.9731, 1.9977] | 91.16% [89.36%, 92.87%] | 0.8109 [0.6968, 0.9391] | 0.3218 |
| Legacy ridge | Development | 1.5488 [1.3046, 1.8047] | 52.34% [49.22%, 55.47%] | 3.6565 [2.8868, 4.5023] | 3.5145 |
| CE λ=0.1 | Train | 0.0278 [0.0173, 0.0388] | 52.29% [51.22%, 53.42%] | 0.6926 [0.6898, 0.6953] | 0.6924 |
| CE λ=0.1 | Development | −0.0078 [−0.0586, 0.0430] | 50.00% [45.31%, 54.69%] | 0.7100 [0.6950, 0.7258] | 0.7100 |

Ridge gets every training Yes/No comparison correct, but only 91.16% of actual
top-token answers are correct: unrelated vocabulary sometimes wins. Mean
Yes/No probability mass is 91.02%, and full CE exceeds its binary CE by 0.4892
nats. The CE arm makes the answer pair nearly exhaustive, with 99.98% training
and 99.9998% development probability mass.

On development, ridge orders the paired scores positively for 59/64 prompts,
yet its midpoint log-odds RMS is **10.18 nats**, much larger than the 1.55-nat
mean gap. Prompt-dependent bias moves both condition scores to the same side
of the decision boundary and creates confident errors. The CE arm reduces
midpoint RMS to **0.333 nats**, but leaves negligible condition separation.
Thus good pairwise ordering, good single-execution discrimination and good
calibration are distinct requirements. Neither fitted method establishes useful
development performance against the balanced two-answer CE reference ln(2).

## Native verification and numerical limits

The final ridge and CE evaluations both run the full native model. Feature
extraction preserves all vocabulary logits, final native head inputs match their
cached inputs, and replayed head logits match native logits bit for bit. Ridge
also checks training and development feature equality. The initial row digest
is explicitly checked after restoring the baseline before CE.

These parity checks do **not** make the FP64 fitting objective identical to
native BF16 arithmetic. For the final CE rows, FP64 predictions of individual
answer logits differ from actual native logits by mean **0.0910** and maximum
**0.4650** nats. The optimizer reports surrogate training CE **0.6896**; native
training CE is **0.6926**. All scientific endpoint numbers above use the latter
execution. Exact verification events are in [verification.json](verification.json).

This run does not test fresh-process reload or another hardware stack. It
isolates the two specified fitting recipes on one numerical task; it does not
establish that their hyperparameters are optimal, nor distinguish every effect
of loss shape from regularization strength. The related CPU-only exploration
is separately archived in [head-regression-027](../head-regression-027/).

## Archive

The archive contains 19 compact source artifacts, including the three native
answer-row files (baseline, ridge, CE). The training and development activation
caches remain local; their hashes and sizes are published in
[archive.json](archive.json). No infrastructure/account logs are included.
The earlier failed attempt is retained separately in
[matched-head-031-startup](../matched-head-031-startup/).

The Python experiment completed in 569.5 seconds; the runner command took 786
seconds including setup. Recompute the figure and intervals with:

```sh
python results/matched-head-031/analyze.py
```
