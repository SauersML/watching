# Learning to distinguish activation-viewing implementations

Prospective follow-up to the reference-probe detector. No new training results
are claimed in this protocol. Independent-compilation run 048 found different
logits for two unprobed compilations on one of four diagnostic prompts. This
motivates separating implementation detection from recognition of one compiled
program.

## First stage: reuse the trained detector

Use the released 034 detector, unchanged, on six independently compiled entries
of one shared wrapper: no probe A, no probe B, reference capture, preallocated
buffer copy, opaque identity materialization, and identity plus reference capture.
The first four fixed detection training prompts are engineering diagnostics, not
a generalization sample. Report every mode, including failed parity checks.

Before installing the trained tail, save the original-model comparisons. After
installation, require the same generated wrappers and no recompilation. Save
full-vocabulary logits, Yes/No log probabilities, decoded top tokens, and actual
answer correctness. A buffer must demonstrably be written; output equality alone
does not prove an operational probe. The storage-forcing identity is a mechanism
intervention, not a read-only probe.

Interpret reference, buffer and identity separately. In particular, a trained
model answering Yes for identity demonstrates transfer to a numerical change
without a read-only observer. It does not establish detection of observation as
an abstract property.

## Second stage: compare learning with fixed source states

Train new buffer-copy and identity arms only after the extraction checks pass.
Use the 034 recipe: original base-model initialization, two final native blocks
and final norm trainable, original output head fixed, balanced full-vocabulary
cross-entropy, FP32-master AdamW, learning rate 5e-6, zero weight decay, gradient
norm clip 1, 1,024 updates of eight prompt pairs, seed 20261017. Use the archived
034 prompt selection and minibatch schedule for the first matched pilot.

Do not initialize these arms from the successful reference detector: that would
measure adaptation rather than comparable learning from the base model. Cache
source states once per arm, retain masks and positions, and validate extraction
against the corresponding native execution before training. Record the actual
condition at the cache boundary. A failed extraction check stops that arm; it is
not a negative learning result.

Reuse 034 and its exact identical-input training control 045. A newly compiled
source may differ from the historical cache: report that mismatch explicitly and
label cross-run comparisons exploratory. If an exactly matched new reference arm
is needed to resolve a result, state the reason before running it. Do not repeat
the identical-input control merely to add another plot.

Save early evaluations at updates 8, 32 and 128, then every 128 updates. Save
recoverable weights every 256 updates and at completion. Preserve failures and
time-limited runs. Do not choose the best checkpoint after inspecting test data.

## Scoring and transfer

The headline score is the proportion of prompt pairs with the correct
full-vocabulary top token in both conditions. The sensitive learning measure is
the paired change in Yes/No log odds. Also retain individual-condition accuracy,
correct-token probability and log loss, so a large gap cannot hide answer bias.

Plot training and development curves with whole-prompt paired bootstrap intervals;
group variants by base context when applicable. These intervals do not measure
training-seed or compilation variance. Four diagnostic prompts receive raw paired
scores rather than a population-generalization claim.

Final evaluations distinguish new contexts, new question families, independent
compilations of the same implementation, and transfer to another implementation.
An independently compiled no-probe/no-probe contrast is a nuisance baseline. Do
not call it an identical-input null unless the learner's inputs actually match.

The old test sets have been inspected. Reuse them for historical comparisons,
not as untouched confirmatory tests. A new confirmatory prompt bank and fixed
compilation replicates must be locked before evaluating selected final models.
The first pilot establishes feasibility; repeat seeds and compilation instances
before claiming one viewing method is reliably easier to learn.

## Resource discipline

Run the inexpensive transfer check first. Share downloaded base weights and
immutable source bundles; reuse verified caches and released checkpoints. GPU
launches require an explicit duration and cost cap. Disable utilization-based
shutdown during CPU compilation, while retaining the wall-clock limit. Stream
each completed comparison immediately instead of buffering the report until the
end. No broad model, layer or learning-rate sweep is part of this protocol.
