# Independent adversarial review of 034

This is a read-only code and artifact audit, not an independent GPU reproduction.
I inspected the archived run, its actual local activation cache, and the training
implementation at commit `65f3cb2a46c9f3b2c92c4199ac596289065b0f72`.
That commit's `tail_train.py` SHA-256 matches the run manifest. Every compact
artifact hash in `archive.json` also matches. No training code or results were
changed for this review.

## Findings checked independently

- **The endpoint numbers reproduce from individual scores.** Computing
  `logp_yes - logp_no`, then subtracting the two conditions within each prompt,
  reproduces all six endpoint means exactly. Correct-answer log probabilities
  independently reproduce full-vocabulary cross-entropy. The comparison cohort
  has gap **26.07421875 nats**, CE **0.000011342372839506787**, and 128/128 correct
  executions in split inference; whole-graph transfer has gap **26.033203125**
  and CE **0.000011895530643790675**. The smallest individual correct-token
  probability is **0.9995692441** in either runtime. Thus the perfect greedy
  accuracy follows from the recorded full-vocabulary probabilities, without
  trusting the stored correctness flags.
- **Only hidden-state values distinguish the cached conditions.** I loaded the
  actual `cache_states.pt`, not just its summaries. All 384 pairs have identical
  attention masks, rotary cosine/sine tensors, and position IDs. Every pair has
  different hidden-state values and separate storage. All positions equal the
  saved attention-mask cumulative positions. No valid token attends to a padded
  key in any cached mask. The median difference norm on padded states is zero;
  on valid-token states it is 48.325/47.149/50.917 for train/development/comparison.
- **Labels are not forwarded to the learner.** The tail receives only hidden
  states, mask, rotary tensors and position IDs. The `present` boolean selects
  a cache entry and the supervised answer token outside the model. The native
  query-norm hook returns `None`; it does not replace activations. The saved
  probe tensor is not passed separately to the learner.
- **Pair ordering does not create an explicit model input.** Every update
  computes absent and present forwards before one optimizer step. The tail has
  no recurrent state or KV cache, uses evaluation mode, and processes the same
  fixed singleton shape. Gradients from both labels accumulate before updating
  weights. All 1,024 updates contain eight distinct prompts; each of the 256
  training prompts receives exactly 32 paired visits. I found no label inversion,
  missing half-loss factor, or intermediate optimizer update between conditions.
- **The selected cohorts do not overlap by identity or normalized full text.**
  They contain 256/64/64 unique contexts and prompts. Training uses 12 question
  families; comparison uses four disjoint question families. Development shares
  training question structures. This does not establish semantic independence;
  the existing data audit already identifies a near-duplicate within training.
- **The endpoint comparison was not used in this run's optimization.** The last
  training update occurs at 671.575 seconds, the artifact is saved at 680.790,
  and comparison-cache extraction begins afterward. The run records 22 passing
  gradient-enabled/no-gradient forward audits. The final transfer path recomputes
  whole-model logits and checks extraction parity and unchanged frozen-source
  states; it does not merely project cached final states through an edited head.
  The original output-head hashes match before and after training.

I found no concrete label, mask, position, pairing, or score-calculation error
that explains away the result. These checks support a detector of this specific
numerical execution difference. They do not establish recognition of observation
in general, or transfer to other probe implementations and serving stacks.

## Remaining ways the claim could fail

1. **Fresh execution is unverified by this audit.** Whole-graph transfer in 034
   occurs in the same process, using already compiled wrappers. It is useful
   evidence, but not a fresh-process reload or an independent implementation.
   A fresh reload with the same reference probe, a buffer-copy probe, and the
   unprobed execution is the highest-priority follow-up.
2. **The source footprint is implementation-specific.** A Boolean specialization
   changes the compiled program when a reference is retained. This can alter
   arithmetic beyond the nominal read site. Learning that difference is the
   experimental target, but the result alone does not identify its exact kernel
   mechanism. The failed eight-block boundary extraction in 035 illustrates why
   each extra observation needs its own parity gate.
3. **The forward-mode audit samples one training prompt per checkpoint.** All 22
   recorded checks pass, but this is not an exhaustive gradient/no-gradient audit
   over all prompts or all optimizer steps. Final inference results do not depend
   on assuming exhaustive parity.
4. **Generalization remains narrowly sampled.** The comparison pool was already
   examined in earlier experiments. Prompt bootstrap intervals condition on this
   fixed family bank and one trained run; they do not cover new seeds, hardware,
   or independently sampled question families. A [100%, 100%] empirical bootstrap
   interval is not certainty about population accuracy.
5. **Base-model revision provenance is incomplete.** The manifest's model commit
   is null. Code, data, head and output-artifact hashes help, but do not substitute
   for a pinned upstream model revision or complete base-weight digest.

## Matched no-footprint training control

Use the same base initialization, task prompts, prompt order, seed, loss,
gradient accumulation, optimizer and update budget as the new positive run.
For each prompt, supply its *absent* cached state to both labels, including the
same mask, positions, dtype and layout; make independent copies if needed.
The condition label must only choose the target token, never reach the forward.
Evaluate both labels on that identical input and require identical output logits.

For distinct answer tokens with probabilities p and q, their equally weighted
full-vocabulary loss is `-0.5 log(p q) >= log(2)`, since `p + q <= 1`.
Consequently, a lower paired loss beyond numerical tolerance, unequal paired
outputs, or a nonzero paired log-odds gap indicates a plumbing error. Accuracy
alone is a weaker check because deterministic ties still produce 50% accuracy.

For a joint Yes/No and cat/owl experiment, apply this bound separately to each
task's distinct answer pair. Match the same task mixture and number of prompt
pairs; the task instruction may identify the answer vocabulary but must not
identify the condition. This control tests training and evaluation plumbing. It
does not replace the separate buffer-copy probe test or fresh-reference-probe
replication.
