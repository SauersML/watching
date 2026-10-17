# A global Yes/No bias adjustment does not resolve the transfer result

Exploratory analysis of the fixed 1,024-update whole-graph transfer scores. One additive intercept was fitted using **development scores only**, minimizing balanced binary cross-entropy. The fitted shift is **+0.141186685837 nats** toward Yes. The previously inspected test scores were then evaluated without further fitting.

| Test metric | Original scores | Development-calibrated scores |
|---|---:|---:|
| Forced Yes/No accuracy | 60.94% | 61.72% |
| Both conditions correct | 21.88% | 23.44% |
| Binary cross-entropy | 1.4422 | 1.4608 |

Development binary cross-entropy decreases from 0.708789 to 0.707837, as expected for the fitted objective, but the historical test loss worsens. This single global intercept therefore does not resolve the observed errors. The result does not establish that all calibration methods must fail.

These are **offline forced-binary scores**, not full-vocabulary token choices or native model improvements. No weights, logits in the actual model, or compiled execution were changed. A zero adjusted log-odds tie predicts No. The analysis does not reselect a training checkpoint or use test scores to fit the intercept; the already-inspected test remains exploratory.

Reproduce with:

```bash
python results/viewing-learning-051-052/identity/calibration/analyze.py
```

[analysis.json](analysis.json) records source-score hashes and the analysis-script hash.
