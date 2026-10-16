# 025–026: larger training distribution and validation-selected regularization

These follow 023–024, which fit training answers confidently but generalized
poorly. Both new jobs use native untied Qwen3-8B, fixed internal weights, length
64, batch one, and the same probe/execution conditions and cat/owl mapping.

Training now has 1,024 distinct contexts: the previous 256 plus 768 new explicitly
AI-authored examples from twelve additional families. Sixteen question wordings
per task replace eight. The unchanged validation set has 64 prompts. Previous
test sets remain useful exploratory comparisons but have already been inspected.
An additional 128 newly written contexts form `test_fresh`; they use held-out
question wording and never enter fitting or hyperparameter selection. Reworded
versions of the older test still share contexts and must not be pooled as
independent samples.

Fit both existing answer logits to answer-pair probability 0.95 and signed
log-odds -3/+3. Independently fit three corrections for each fixed ridge value
0.001, 0.01, 0.1, 1, 10, and 100, sharing a single SVD. Select the detector by
full-vocabulary validation cross-entropy. When fitting animals, retain detector
training prompts as preservation anchors, and select by the equally weighted
mean of animal and detector validation cross-entropies. Validation is used only
for selection, never in the linear solve. Ties favor stronger regularization.

To reduce runtime, cache the actual output-head inputs, preserving shape,
strides, storage offset, dtype, and gradient-recording condition. Before using
any cached input, the existing head must reproduce every native vocabulary logit
bit for bit. Candidate fits replay this head. The selected edit must then pass
full-model extraction parity, fixed-feature checks, and head-replay parity again.
Any mismatch stops the job; no approximate parity or alternate execution path
is silently substituted. Final fresh-process evaluations still run the full
model for both tasks.

Report intermediate candidate validation losses, selected ridge, training and
held-out actual top-token accuracy, correct-token log probability, binary loss,
paired separation, and detector preservation. Keep complete failed runs and
candidate histories. Bootstrap whole prompt pairs; family intervals are
exploratory and do not cover hardware or training-seed variation. This changes
data, regularization, and execution efficiency together, so it tests a recipe
rather than attributing improvements to one component.

Launch two separate 48 GB GPU jobs in parallel, capped at 45 minutes and
$1.20/hour each. Keep low-utilization shutdown disabled during compilation while
retaining the wall cap. Publish a committed source snapshot and compact results.
