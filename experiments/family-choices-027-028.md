# 027–028: grouped selection, diverse questions, and calibrated native answers

This protocol is fixed before either run. Two parallel Qwen3-8B experiments fit
the existing Yes/No and cat/owl output rows while keeping internal weights fixed.
One detects a read-only query-norm probe; the other detects gradient recording
versus no-gradient execution. Both execution conditions use evaluation mode.
This is not a test of recognizing semantic training or deployment contexts.

## Data and selection

Use the [version-1 family dataset](../data/FAMILY_DATA.md): 1,024 training
contexts assigned to 12 question families, with four explicitly AI-authored
phrasings per family and task. Both conditions use the identical prompt. The
192-template bank also includes four entirely held-out question structures.
Training context assignments are matched across detector and animal tasks.

Three-fold cross-validation holds out whole training question families, including
all their phrase variants and both conditions. Select the lowest mean held-out
family full-vocabulary cross-entropy over the fixed regularization grid
`[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]`. Exact ties favor stronger
regularization. Then refit all training data. The previously inspected 64-row
validation set is reporting-only. No external evaluation split enters selection.

Two new 128-context collections separately test familiar content families and
new structural content families. Each collection is evaluated with familiar and
held-out question families. These paired cells separate wording transfer from
context transfer; paired versions are not independent samples. The held-out
structures are a limited operational definition of OOD, not arbitrary real-world
generalization. There are only four new content and four new question families.

## Optimization

Cache actual native inference-path head inputs with all-vocabulary parity checks.
Fit the two answer rows by convex full-vocabulary cross-entropy, holding all other
vocabulary logits fixed. Penalize the total row edit by `lambda * ||delta||² / 2`.
This replaces repeated ridge corrections toward fixed confident targets.

Use CPU FP64 LBFGS, history 100, maximum 3,000 iterations, and gradient tolerance
1e-6. Strong convexity supplies the certificate `||gradient||² / (2 * lambda)`;
stop early when this bounds objective error by 1e-4 nats. Record termination,
objective, gradient, and certificate for every fit. Finite fits reaching their
budget remain candidates and are explicitly marked unconverged. Numerical
failures are not silently scored as valid fits. Defaults were checked on earlier
training features only, never on these held-out data.

Fit the detector first. Fit cat/owl next with equal-weight detector preservation
cross-entropy on aligned training prompts. The detector's correct logits remain
fixed during animal fitting; the penalty discourages animal tokens from stealing
their probability mass. Both tasks are evaluated with all four edited rows.

The FP64 fitting objective is a surrogate for the actual BF16 head. Install the
rounded rows and measure native outputs, reporting surrogate discrepancies.
All final claims use actual native logits, not surrogate predictions.

## Verification and reporting

Keep batch size one and padded length 96. Reject truncation. Earlier runs used
length 64, so changes here test the entire recipe, not just the new dataset.
Verify extraction parity and unchanged internal features after fitting. Reload
and compile afresh for both tasks. Export fresh features and compare them with
fitting features by context ID, including bit equality, footprint direction, and
midpoint movement. A fresh-compile mismatch must be reported, not hidden by
averaging it with fitting-path results.

Stream collection progress, optimization diagnostics and CV scores. Report
actual top-token accuracy, correct-token log probability, full-vocabulary and
binary cross-entropy, pair probability mass, and paired log-odds separation.
Do not call a positive separation alone a calibrated detector. Curves from
optimizer closure evaluations include line-search trials, not only accepted
updates; label them accordingly.

Use whole-prompt paired bootstrap intervals for within-cohort summaries and show
per-family outcomes. Prompt intervals are conditional on the sampled families;
they do not cover seed, hardware, or arbitrary unseen-family variation. All OOD
scores are final evaluations for this recipe. Subsequent adaptations based on
them would turn these sets into development data.

## Execution

Run `scripts/run_family_choices.sh OUTPUT Qwen/Qwen3-8B probe|execution` through
the remote bootstrap. Commit and launch an immutable source snapshot. Each job
is capped at 45 minutes and $1.20/hour ($0.90 maximum requested GPU charge per
job). Disable low-utilization shutdown during compilation and CPU fitting while
retaining the wall-clock cap. Retain failed and partial runs. Publish source,
datasets, compact raw results, selection histories and checksums.
