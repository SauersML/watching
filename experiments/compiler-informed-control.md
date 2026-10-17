# Proposed mechanism-removal control and the existing identical-input null

This is a provisional design, not an implemented experiment or a mechanism claim.
The compiler audit and frozen-tail intervention audit must first establish what
actually changes in their tested runtimes.

The existing identical-input null copies each prompt's absent-condition cached
state into both labeled examples. The model receives identical hidden states,
attention masks, rotary tensors and positions; only the loss target differs.
It tests whether the training or evaluation plumbing leaks the label, and whether
reported learning requires numerical condition information. For a deterministic
forward pass, the two full-vocabulary outputs must be identical for any weights.
Balanced two-label cross-entropy on that fixed input cannot fall below `ln(2)`.
This is a mathematical constraint on the specified control, not an assumption-free
population inference about arbitrary monitoring systems.

A compiler-informed control would instead keep a real read-only activation probe
in the present condition and deliberately remove its identified numerical effect.
For the current hypothesis, one candidate is an opaque materialization operation
at the normalized query in **both** conditions. The proposed predictions are:

1. The read-only probe still runs and captures the intended query values.
2. Adding that probe to the barrier-equipped computation leaves every vocabulary
   logit unchanged, across both conditions of each prespecified prompt.
3. The query values entering attention and the cached boundary entering the learned
   tail also match exactly between conditions.

Those predictions require fresh generated-code and numerical checks. Merely adding
an identity operation, or setting a rounding flag, does not guarantee them: either
can change other fusion or kernel choices. Failed equality rejects this particular
mechanism-removal control; it must not be relabeled a successful null. Equality of
selected answer logits alone is insufficient.

If these checks pass, the two controls remove evidence at different points:

| Control | Where condition information is removed | What it tests |
|---|---|---|
| Identical-input cached null | At the learner's cached input, by construction | Label leakage and learning/evaluation plumbing |
| Compiler-informed barrier plus actual read | At the proposed source mechanism, if validated | Whether the identified compiler effect mediates the probe footprint |

Applying the existing learned tail to numerically identical barrier-derived inputs
must produce the same response in both conditions. That would test dependence on
the footprint while preserving an actual external read. It would not establish
awareness of observation, a general defense against all probes, or a population
generalization result. A buffer-copy probe whose full-vocabulary parity fails is
an alternative intervention, not this mechanism-removal control.

For a future training comparison, initial weights, data, paired sampling, optimizer
and budget must match the positive arm. The current cached null is base-initialized;
it is not a matched warm-start training ablation for run 039. The equal-input
constraint still applies to any deterministic initialization, including the saved
034 tail, but a matched warm-start null would need to load that same initialization
explicitly. No additional training is authorized by this note or launched here.
