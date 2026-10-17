# Bounded final-line diagnostic for the fixed 039 model

This is an exploratory follow-up to the completed, frozen 040 confirmation. It does not replace or regrade its primary animal result: **119/128 correct first tokens**. All nine first-token errors occurred in one family asking for an animal on the final line, without clearly forbidding preceding prose. A continuation can clarify what those answers contain, but cannot turn this inspected family into a new confirmatory test.

## Frozen scope

Evaluate **all eight** `animal` prompts in the `final_line_answer` family from 040's archived `fresh039/data.json`, under both conditions, including the seven originally correct executions. No error-based selection, prompt edits, training, threshold fitting or checkpoint selection. Use the final 039 native tail (SHA-256 `2f1f92bcb78041b812760f5086a5436ef43732e983a19a4578e6e3a7a6c3892c`) and Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`.

## Execution and fixed grading

- Compile the original model first, then install the saved native tail in place, as in 040. Use the same reference-hook class and probe layer; ordinary whole-model BF16 singleton inference, no gradients, no KV cache, fixed input shape 1×96, compiler disk cache disabled, weight freezing disabled. No new output head.
- Greedy argmax, maximum **32 newly generated tokens**. Keep every original prompt token and consume left padding as tokens are appended. If fewer than 32 slots remain, reduce the generation cap to the available slots; never crop the prompt or shift the positional IDs of its real tokens.
- Stop on the model/tokenizer's declared EOS token. Preserve token IDs, each chosen token's full-vocabulary logprob, decoded text with and without special tokens, first-step answer logprobs and top token, and the reason generation stopped.
- After EOS only, take the last nonempty line, strip surrounding whitespace, and compare it to the literal lowercase target `cat` or `owl`. Do not remove punctuation, lowercase text, search earlier lines, use a semantic judge or reinterpret another animal. A token/shape cap is unresolved even if the current last line happens to match; report that literal match separately.
- Repeat the original prompt after each continuation and require bit-identical full-vocabulary logits within this process; stop and preserve an error event if the repeat changes.
- Compare the first-step negative/positive logprobs, their logit difference, and the actual top token against the corresponding frozen 040 row. Report every mismatch, not just top-token changes. This is a check of stored metrics, not full-vocabulary bit equality: 040 did not save complete logits. Runtime/hardware/compiler differences remain possible and must be disclosed.

Report 16 executions with resolved-correct, resolved-incorrect and unresolved counts, alongside the unchanged 040 metric. These continuations do not establish performance on uninspected prompts or prove what the original 040 process would have generated if first-step replay differs. Any training changes motivated by this diagnostic require a new final test.

## Command

```sh
PYTHONPATH=src python -m watching.complete_final_line \
  --confirmation results/joint-confirmation-040/fresh039 \
  --tail /path/to/joint-039/native_tail.pt \
  --out /path/to/empty/output
```

CPU tests cover whole-family selection, no truncation with fixed positions, literal/EOS-only grading, and first-step mismatch reporting. This file specifies a proposed run; no GPU diagnostic has been launched by its author.
