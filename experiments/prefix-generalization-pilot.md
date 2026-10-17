# Fixed-checkpoint chat-prefix generalization pilot

**Proposed evaluation only. No GPU run or training is authorized by this file.** Test the final 039 native tail before deciding whether to train on more conversation structures. The frozen fixture is `data/prefix_pilot_v1/dataset.json`: eight new base cases, four per task, crossed with four message layouts. Every variant is evaluated with identical messages in the probe-absent and probe-present conditions.

## Matched design

| Group | Messages | First role |
|---|---|---|
| `single_user` | latest user question | user |
| `system_user` | neutral system, latest user question | system |
| `multiturn` | neutral user, authored assistant acknowledgement, latest user question | user |
| `system_multiturn` | neutral system, neutral user, authored acknowledgement, latest user question | system |

The latest user question and context are **identical across all four groups** for a base case. System/history text varies across four neutral prefix families and never mentions animals, probes, gradients or monitoring. Histories are explicitly authored fixtures, not sampled model answers. Every latest question requires the *entire response* to be exactly one permitted word; there is no final-line ambiguity.

This tests a specific limitation: no-system multiturn retains the initial `user` role token and its causal prefix; adding a system message changes the initial role to `system`. The pinned tokenizer gives the same first role token within each of those two sets. A model relying on a fixed early role-token state could therefore transfer to ordinary multiturn while failing with a system prefix. This is a hypothesis, not a prediction treated as established mechanism.

There are **8 base cases, 32 message variants, 64 condition executions**. Variants are matched repeated measurements, not 32 independent contexts. Four question families—XML envelope, Markdown table, comment block and footnote—are absent from the actual 034 and 039 fine-tuning training-family sets. They are familiar formats used in earlier project evaluations; do not claim unprecedented structural novelty. All eight concrete questions/contexts are new. Do not reuse confirmation-v3 prompts as fresh test cases.

## Fixed execution

Use the final 039 tail, SHA-256 `2f1f92bcb78041b812760f5086a5436ef43732e983a19a4578e6e3a7a6c3892c`, and Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`. Both tasks retain their existing labels: absent/present = No/Yes for detection and cat/owl for animal choice. Labels and group metadata never enter model messages.

Keep ordinary whole-model compiled BF16 inference, one fixed shape **1×96**, no gradients, no KV cache, same layer-14 reference hook, base-first compilation followed by in-place native-tail installation, and a fresh compiler cache. Use the model's pinned chat template with `enable_thinking=False`. Validate every rendered input and single-token answer again at runtime; never truncate. Record rendered/token-ID hashes, real token count, role sequence, first role-token ID, and all numerical/runtime settings. The local pinned 8B tokenizer fits every fixture within 96 tokens; exact counts are in the data manifest.

Randomize the 64 (base case, layout, condition) executions using a frozen seed; keep every result. Repeat each full-vocabulary first-step evaluation and require bit identity within the process. Compare all four groups in one process/hardware stack. No reference/buffer substitutions and no claim that a buffer is harmless. The same fixed model is used throughout, with no fitting or checkpoint selection.

Fixed shape isolates this from a batch/shape sweep, but **does not isolate a pure role-token intervention**: message semantics, token count, padding positions and causal history also change. If a group fails, a separate factorial mechanistic experiment is needed to identify which change causes failure.

## Measures and matched analysis

Save both answer logprobs, their log-odds, full-vocabulary correct-token probability, actual top token and its probability, and unrestricted top-token correctness for every execution. Per base case/group report:

- Pair gap: `s_present − s_absent`, with `s = log p(positive) − log p(negative)`.
- Midpoint bias: `(s_present + s_absent) / 2`.
- Mean correct-token probability, mean full-vocabulary CE, actual accuracy over the two executions, and whether both are correct.

Show all four base cases per task, then descriptive group means and **within-base-case changes from single-user**. Do not infer transfer from a large gap alone. With only four cases per task, this is a diagnostic pilot, not a precise population estimate or a significance contest. If bootstrap intervals are shown, resample whole base cases using the same indices across all layouts/conditions, label them conditional descriptive intervals, and do not present template variants as independent samples. No pooling the two behavioral tasks.

Once results are viewed, this pilot becomes development evidence. Any training informed by it needs a different frozen final test with disjoint base contexts and entire prefix/question families. Do not choose another checkpoint or rewrite these questions based on its scores.

## Minimal implementation change required

`validate_tail.py` currently renders only `[{"role": "user", "content": row["text"]}]` at lines near its `apply_chat_template` call. Add an **opt-in** prefix-pilot path that validates and renders `record["messages"]`; leave the legacy `text` fallback byte-for-byte unchanged. Validate optional initial system message, alternating user/assistant turns, final user turn, string contents, and final-user content equality to `text`.

The existing `evaluation_records` also rejects repeated context text/latest questions, which is correct for ordinary datasets. Do not relax that guard globally. In the new opt-in path require exactly four named groups per `base_context_id`, unique `(base_context_id, group)`/variant IDs, identical latest text/context/task/answer labels within each base case, and full inclusion of all groups. Each variant's `context_id` is unique; `base_context_id` preserves the repeated-measure identity. Carry group/base identity into scored rows and group the analysis explicitly. The numerical model/probe implementation requires no change. This preserves previous run interfaces and statistics.
