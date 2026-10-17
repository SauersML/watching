# Bounded final-line completion diagnostic: run 043

**Completed; runner exit 0, pod deletion confirmed, exact temporary approval removed. Cost $0.0654.** The fixed 039 model generates greedy continuations for all eight animal prompts in 040's `final_line_answer` family, under both probe conditions. No training, prompt filtering or checkpoint selection occurs.

The original 040 result remains **119/128 correct animal first tokens**. This inspected-family diagnostic does not replace that score or constitute a new generalization test.

## Predeclared last-line result

Of 16 executions, **13 terminate at EOS with the literal target animal on the last nonempty line**, two terminate with a nonmatching line, and one is unresolved at the 32-token cap. Every initial prompt remains intact; all use singleton shape 1×96 with no KV cache. Last-line grading strips whitespace only: no case/punctuation normalization or semantic judging. Unfinished outputs are not counted as successes.

Both resolved failures belong to index 98: under both conditions the model says `A trial run reveals practical problems before a larger event begins. Owl.` on one line. This fails the predeclared literal rule in both conditions; in the absent condition it also names the opposite animal. Index 122 absent produces prose through the 32-token limit, so its eventual answer is unknown. Several other prose prefixes do end with the requested one-word final line.

## First-step replay limits

The first token agrees with 040 on **15/16** executions, but all stored first-step metrics agree exactly on only **2/16**. Index 114 present changes from `owl` in 040 to `Yes` in this process, then ends its continuation with `owl`. Largest absolute differences: negative-answer logprob **1.12454 nats**, positive-answer logprob **2.37454 nats**, and pair logit gap **2.125 nats**. First-token accuracy for this family in 043 is 6/16, versus 7/16 in 040.

Both runs use L40S, the same recorded package versions, pinned base revision, saved tail hash and nominal inference settings. The NVIDIA driver differs: **570.124.06 in 040 vs 550.144.03 in 043**. That is a possible source of numerical change, not an established cause. Accordingly these are **not bit-exact continuations of the original 040 execution**. No missing full-vocabulary equality is inferred from matching answer metrics.

Within 043, every original prompt is rerun after its continuation: all **16 full-vocabulary repeat checks are bit-identical**. The saved model and base weights are not updated. Source code compiles the base modes first, installs the same two-block native tail, and retains the existing output head and ordinary compiled BF16 whole-model execution.

## All executions

| 040 index | Capture | Target | 043 first token | Stop | Literal final-line result | First-step metrics equal 040 |
|---:|---|---|---|---|---|---|
| 66 | True | owl | `owl` | eos | correct | False |
| 66 | False | cat | `cat` | eos | correct | False |
| 74 | False | cat | `A` | eos | correct | False |
| 74 | True | owl | `A` | eos | correct | False |
| 82 | False | cat | `cat` | eos | correct | True |
| 82 | True | owl | `owl` | eos | correct | True |
| 90 | False | cat | `cat` | eos | correct | False |
| 90 | True | owl | `owl` | eos | correct | False |
| 98 | False | cat | `A` | eos | incorrect | False |
| 98 | True | owl | `A` | eos | incorrect | False |
| 106 | False | cat | `Ex` | eos | correct | False |
| 106 | True | owl | `Ex` | eos | correct | False |
| 114 | False | cat | `The` | eos | correct | False |
| 114 | True | owl | `Yes` | eos | correct | False |
| 122 | False | cat | `The` | token_budget | unresolved | False |
| 122 | True | owl | `Yes` | eos | correct | False |

## Provenance and interpretation

- Protocol: [`experiments/final-line-completion.md`](../../experiments/final-line-completion.md), committed before the run; source `e35fac6172203ecaaaddceabdbcf567997e0959c`.
- Native tail SHA-256 `2f1f92bcb78041b812760f5086a5436ef43732e983a19a4578e6e3a7a6c3892c`; public [joint-039 release](https://github.com/SauersML/watching/releases/tag/joint-039).
- Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`; no new training or model changes.
- `completion/continuations.json` contains every raw continuation, token ID, chosen-token logprob, first-step answer logprobs, exact source-row identity and repeat check. `archive.json` hashes unchanged scientific outputs. `replay_analysis.json` summarizes replay differences.
- This is a finite, post-hoc family of eight paired prompts, not 16 independent examples from a population. We report counts, not a misleading population confidence interval or a revised full-bank accuracy. Future changes motivated by these results require a new final test.
- The original 040 records, wording and primary metrics remain unchanged. Infrastructure/account logs are excluded.
