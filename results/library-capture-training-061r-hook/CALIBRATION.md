# A detectable change is not always a correct answer

This is a post-hoc diagnostic of the fixed update-1,024 hook detector, using its saved native scores. No model weights, generated answers, or training settings were changed. The four test question families had already been inspected in earlier experiments.

In the training runtime, enabling the hook raises Yes relative to No on all 64 test prompt pairs. Nevertheless, only 22 of those pairs receive both correct top tokens. The condition moves the answer in the intended direction, but often does not move it across the decision threshold. Some errors are highly confident: test full-vocabulary log loss is 2.703 nats, worse than an equal-probability binary predictor's 0.693 nats.

We fitted one nonnegative scale and one offset to the **development** Yes/No log-odds, minimizing binary log loss. The same two numbers were then applied to test scores. This is an external probability-calibration diagnostic, not a deployed model edit or a claim about changed vocabulary-wide top tokens.

| Runtime | Development binary loss, native → calibrated | Test binary loss, native → calibrated |
|---|---:|---:|
| Recorded source → trained native tail | 1.165 → 0.480 | 2.703 → 0.742 |
| Original whole model | 3.060 → 0.692 | 4.569 → 0.694 |

The fitted scale is 0.198 for the training runtime and 0.0102 for the original whole model. Both strongly reduce confidence. Calibration helps log loss but does not turn the original whole-model hook detector into a useful classifier. It also does not recover the test performance implied by perfect paired ranking in the training runtime.

For a prompt, write its two Yes/No scores as `midpoint − gap/2` and `midpoint + gap/2`. A positive gap means that the conditions are ordered correctly. Both binary answers require the gap to exceed twice the absolute midpoint; vocabulary-wide correctness additionally requires beating every other token. Different prompts can have different midpoints, so a single global offset cannot generally fix them all. Subtracting each test prompt's own midpoint would require seeing both executions and is not a valid single-execution detector.

[Machine-readable analysis](calibration_diagnostic.json) includes per-family native metrics, exact-tie counts, source hashes, and paired whole-prompt bootstrap intervals for the change in binary loss. Intervals condition on the fitted calibrator and these families; they omit calibration-fitting, training-seed, compiler, hardware, and family-sampling uncertainty. Development intervals are descriptive after fitting.

Reproduce from the recovered score archive:

```bash
python scripts/audit_capture_calibration.py \
  results/library-capture-training-061r-hook/fit \
  results/library-capture-training-061r-hook/calibration_diagnostic.json
```

The next confirmation should check the final checkpoint in a fresh process and compilation before changing the training recipe. Further optimization should be judged by held-out answer correctness **and** log loss, not separation alone.
