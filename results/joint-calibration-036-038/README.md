# Exploratory training/development calibration audit

This analysis reads only the fixed 2,048-update **training and development** records from runs 036 and 038. It does not read test scores, select a checkpoint or edit a model. It separates each prompt's positive/negative answer log-odds gap `D = s_present − s_absent` from its midpoint bias `C = (s_present + s_absent)/2`.

Both binary decisions are strictly correct only when `D/2 > |C|`. A large positive gap alone does not ensure this. These runs commonly have one correct condition and one incorrect condition; no development pair has both native answers wrong.

A scalar offset is fitted to **training binary log loss separately per task**, then applied unchanged to the development log odds. This is an offline two-answer diagnostic, not a measured native full-vocabulary improvement. Exact binary ties receive half credit, so its original binary accuracy can differ slightly from the saved native argmax accuracy.

| Run/task | Training offset, nats | Development binary accuracy before → after | Development binary CE before → after |
|---|---:|---:|---:|
| 036 detection | −6.161 | 66.41% → 73.44% | 2.894 → 1.976 |
| 036 animal | −6.454 | 60.94% → 71.88% | 2.472 → 1.396 |
| 038 detection | −2.938 | 70.31% → 73.44% | 2.264 → 1.896 |
| 038 animal | −5.170 | 66.41% → 70.31% | 1.932 → 1.283 |

A common bias accounts for part of the problem, but the corrected development losses still exceed ln(2). Midpoint standard deviations rise from 2.7–4.1 nats in training to 6.8–8.3 in development. Within-family variation also remains large, especially for quoted requests and two-speaker exchanges. This supports investigating prompt-dependent calibration; it does not establish its mechanism or guarantee that adding a bias parameter will solve it.

`analysis.json` preserves per-family summaries, pair correctness counts, wrong-confident margins, source hashes and paired bootstrap intervals for the offline changes. Those intervals condition on the fitted training offset and observed prompt mixture; they omit estimator, family and training-seed uncertainty. The reported per-prompt centered oracle requires both executions and is explicitly **not** a deployable single-execution detector.

Reproduce from the repository root:

```bash
python scripts/analyze_joint_calibration.py \
  results/joint-validation-036 results/confirm-validation-038 \
  --out results/joint-calibration-036-038/analysis.json
```
