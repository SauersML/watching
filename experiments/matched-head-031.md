# 031: isolate the answer-row fitting regression

Compare two prespecified fitting recipes on the same Qwen3-8B native activations,
GPU, compiled graph, input shape, initial output weights and prompt records:

1. The earlier recipe: three ridge corrections, penalty 0.001, target Yes/No
   log-odds of −1 without the probe and +1 with it. Only the two native answer
   rows change, symmetrically. This is the recipe used in experiment 019, now
   applied to the more diverse data.
2. Full-vocabulary cross-entropy with L2 penalty 0.1, the coefficient selected
   by training-family cross-validation in experiment 027. Start again from the
   original answer rows. Do not rerun the expensive selection or tune on these
   comparison results.

Use the exact 1,024 training and 64 development records from 027. These prompts
are already inspected development evidence; this experiment makes no new test
or OOD claim. Export a dataset containing only these two splits. The comparison
isolates recipe differences within this execution; it does not isolate hardware
or dataset differences from experiment 019.

Cache head inputs only after every native vocabulary logit matches extraction.
Each recipe's final scores must pass full-model execution again, with unchanged
internal features. Save each fitted pair of native rows separately and restore
the original pair between recipes. Use the existing native head for intermediate
ridge corrections. Preserve all failures and surrogate/native discrepancies.

Report paired separation, midpoint answer bias, binary/full-vocabulary loss,
correct-answer probabilities and actual top-token accuracy. Intervals resample
whole prompt pairs. Large paired separation with poor individual answers is a
diagnostic outcome, not successful calibration. Scores over the three ridge
corrections are not a gradient-training learning curve.

This is a bounded diagnostic before choosing the next training change. Run one
48-GB GPU for at most 30 minutes and $1.20/hour; disable low-utilization shutdown
during compilation and CPU fitting. Stream results, retain compact artifacts,
and publish the exact implementation and data hashes.
