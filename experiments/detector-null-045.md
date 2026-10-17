# Matched identical-input null for detector 034

Protocol fixed before launch. This replaces no earlier result and does not change
existing figures. Run: `watching-detector-null-045`.

The comparison is the original base-initialized detector 034 versus its exact
numerical-input null. Both use Qwen3-8B's original blocks 34–35 plus final norm,
with the native untied output matrix frozen. No learned 034 weights initialize
the null. Both labels receive the **absent** cached state of their own prompt;
all masks, rotary inputs, positions, text and labels otherwise match 034.

Settings are inherited, then asserted: 256 training / 64 development / 64 endpoint
test prompts, length 96, seed 20261017, batch eight prompt pairs, 1,024 Adam
updates, learning rate 5e-6, balanced full-vocabulary cross-entropy, FP32 master
weights and separate FP32 gradient accumulation, clip norm one, zero decay.
The exact prompt shuffle is reproduced. Evaluation is at zero, 8, 32, and each
128 updates. Training consumes 8,192 prompt-pair visits, exactly as 034. No
checkpoint selection; the test is scored only after endpoint weights are saved.

The released `sft-034/cache_states.pt` must have SHA-256
`1d2e5cbee328d1dd6c6524b22ef6160855c4305cc6b7335e7bb2bdd54fda59ab`.
Every tensor fingerprint, record, position and tokenization is checked. The base
and tokenizer are explicitly pinned to
`b968826d9c46dd6066d109eabc6255188de91218`. Historical 034 did not record a revision;
this limitation remains visible. Its head checksum must match. Before creating
an optimizer, **all 320 training/development absent-condition answer logprobs,
log-odds and greedy correctness must exactly reproduce the archived untrained
034 scores**. Those archives lack full-vocabulary vectors; this check cannot
prove their equality. Separately, the two null executions must have identical
full-vocabulary logits at every evaluated prompt, and paired full-vocabulary
cross-entropy cannot fall below log(2). Gradient-enabled and no-gradient tail
forwards are checked exactly at every evaluation on four fixed training cases.
A failed gate ends the run, without tolerance relaxation.

Report actual full-vocabulary top-token accuracy, full-vocabulary loss and
probe-minus-no-probe answer log-odds. A deterministic identical-input pair can
get at most one of its contradictory labels correct, hence at most 50% actual
accuracy (possibly less if a third token wins). Forced binary accuracy is not
the headline. The intended figure compares **034's 64-prompt development curve**
with this null's same 64 prompts; no comparison to the differently trained 036
or warm-started 039 is substituted. Plot pairs and scope rather than treating
checkpoint observations as independent samples.

This is a training/plumbing control at cached native-tail inputs. It does not
establish that a physical read-only probe is invisible in a compiled model, and
makes no monolithic compiled-serving claim. The only causal difference from
034 training is removing its numerical condition information at this boundary.

Bounded L40S run: 45 minutes, at most $1.20/hour ($0.90 ceiling). Idle-utilization
shutdown is disabled while the wall-time cap remains active. Publish the exact
source revision, compact outputs, weights, measured cost and deletion status.
