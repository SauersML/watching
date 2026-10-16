# Matched validation calibration

Validation only. Winning-candidate CIs are descriptive, not selection-adjusted. No test scores read and no new parameter selection performed.

All comparisons use identical saved validation prompt records. Cached candidates score captured native features; native and fresh evaluations appear separately when available. Unavailable native evaluations are omitted, not treated as failures.

| Condition / evaluation | Gap | Mean absolute midpoint | Full accuracy | Full CE | Binary CE | Oracle centered binary CE | Confidently wrong fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| probe / old_fit | 3.119 | 18.311 | 52.34% | 8.415 | 8.400 | 0.272 | 43.75% |
| probe / old_fresh | 3.319 | 18.473 | 48.44% | 8.674 | 8.435 | 0.251 | 44.53% |
| probe / new_ridge_0.001 | 4.240 | 16.793 | 54.69% | 7.483 | 7.465 | 0.238 | 39.06% |
| probe / new_ridge_0.01 | 4.232 | 16.868 | 52.34% | 7.520 | 7.498 | 0.239 | 39.84% |
| probe / new_ridge_0.1 | 4.207 | 16.453 | 55.47% | 7.328 | 7.308 | 0.231 | 39.06% |
| probe / new_ridge_1 | 3.979 | 12.772 | 58.59% | 5.595 | 5.568 | 0.203 | 35.16% |
| probe / new_ridge_10 | 2.276 | 6.531 | 60.94% | 2.936 | 2.884 | 0.317 | 26.56% |
| probe / new_ridge_100 | 0.428 | 1.233 | 55.47% | 0.966 | 0.858 | 0.594 | 3.91% |
| probe / new_native | 0.428 | 1.233 | 55.47% | 0.966 | 0.858 | 0.594 | 3.91% |
| execution / old_fit | 4.936 | 5.583 | 68.75% | 1.879 | 1.840 | 0.093 | 18.75% |
| execution / old_fresh | 5.182 | 5.528 | 65.62% | 1.812 | 1.774 | 0.082 | 17.97% |
| execution / new_ridge_0.001 | 5.689 | 5.952 | 64.84% | 1.916 | 1.872 | 0.069 | 21.88% |
| execution / new_ridge_0.01 | 5.695 | 5.939 | 66.41% | 1.878 | 1.831 | 0.069 | 18.75% |
| execution / new_ridge_0.1 | 5.744 | 5.940 | 65.62% | 1.889 | 1.841 | 0.068 | 22.66% |
| execution / new_ridge_1 | 5.684 | 4.611 | 70.31% | 1.341 | 1.298 | 0.067 | 13.28% |
| execution / new_ridge_10 | 5.396 | 3.456 | 75.00% | 0.932 | 0.879 | 0.076 | 8.59% |
| execution / new_ridge_100 | 3.385 | 2.241 | 78.12% | 0.747 | 0.676 | 0.180 | 5.47% |
| execution / new_native | 3.385 | 2.241 | 78.12% | 0.747 | 0.676 | 0.180 | 5.47% |

For each prompt, absent/present scores are c−d/2 and c+d/2. Correct signed margins are d/2−c and d/2+c. Positive d ranks the pair correctly; both binary answers require d>2|c|. Full-vocabulary answers must also beat every other token.

Binary CE = mean[softplus(c−d/2)+softplus(−c−d/2)]/2. At fixed d it is minimized by c=0. The oracle column reports that per-prompt centered lower bound, which requires both executions and is **not deployable performance**. Full CE adds the penalty for probability outside the answer pair.

Balanced answer-only reference loss is ln2 = 0.693147, corresponding to probability 0.5 on each answer. Accuracy above 50% need not beat this loss: confidently wrong examples can dominate it. The wrong-confidence column counts execution margins below −3 nats; the JSON also reports loss tails and midpoint distributions.

## Paired changes

Saved selected ridge for probe: 100.

**probe: new_ridge_100_minus_old_fit**

- full_ce: change -7.4495; descriptive 95% CI [-9.0692, -5.9287].
- binary_ce: change -7.5428; descriptive 95% CI [-9.1660, -6.0144].
- abs_midpoint: change -17.0780; descriptive 95% CI [-20.2621, -14.1019].
- gap: change -2.6912; descriptive 95% CI [-3.2153, -2.1714].
- full_accuracy: change +0.0312; descriptive 95% CI [-0.0078, +0.0781].

**probe: new_ridge_100_minus_old_fresh**

- full_ce: change -7.7084; descriptive 95% CI [-9.3140, -6.1762].
- binary_ce: change -7.5771; descriptive 95% CI [-9.1398, -6.0971].
- abs_midpoint: change -17.2391; descriptive 95% CI [-20.3809, -14.2527].
- gap: change -2.8914; descriptive 95% CI [-3.3975, -2.3872].
- full_accuracy: change +0.0703; descriptive 95% CI [+0.0234, +0.1250].

**probe: new_native_minus_new_ridge_100**

- full_ce: change +0.0000; descriptive 95% CI [+0.0000, +0.0000].
- binary_ce: change +0.0000; descriptive 95% CI [+0.0000, +0.0000].
- abs_midpoint: change +0.0000; descriptive 95% CI [+0.0000, +0.0000].
- gap: change +0.0000; descriptive 95% CI [+0.0000, +0.0000].
- full_accuracy: change +0.0000; descriptive 95% CI [+0.0000, +0.0000].

Saved selected ridge for execution: 100.

**execution: new_ridge_100_minus_old_fit**

- full_ce: change -1.1322; descriptive 95% CI [-1.6879, -0.6465].
- binary_ce: change -1.1638; descriptive 95% CI [-1.7268, -0.6740].
- abs_midpoint: change -3.3418; descriptive 95% CI [-4.4951, -2.2890].
- gap: change -1.5508; descriptive 95% CI [-1.8359, -1.2559].
- full_accuracy: change +0.0938; descriptive 95% CI [+0.0156, +0.1719].

**execution: new_ridge_100_minus_old_fresh**

- full_ce: change -1.0650; descriptive 95% CI [-1.6183, -0.5862].
- binary_ce: change -1.0971; descriptive 95% CI [-1.6548, -0.6153].
- abs_midpoint: change -3.2871; descriptive 95% CI [-4.4649, -2.2187].
- gap: change -1.7969; descriptive 95% CI [-2.1133, -1.4922].
- full_accuracy: change +0.1250; descriptive 95% CI [+0.0469, +0.2031].

**execution: new_native_minus_new_ridge_100**

- full_ce: change +0.0000; descriptive 95% CI [+0.0000, +0.0000].
- binary_ce: change +0.0000; descriptive 95% CI [+0.0000, +0.0000].
- abs_midpoint: change +0.0000; descriptive 95% CI [+0.0000, +0.0000].
- gap: change +0.0000; descriptive 95% CI [+0.0000, +0.0000].
- full_accuracy: change +0.0000; descriptive 95% CI [+0.0000, +0.0000].

Intervals resample whole prompts, preserving both conditions and both evaluations. They assume exchangeable prompt units, exclude training and compilation variability, and are not adjusted for selecting a ridge on these same validation data. Training size and fitting implementation also change between runs; these comparisons do not isolate the effect of more data alone.
