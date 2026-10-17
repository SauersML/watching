# Joint detection and animal-choice validation, version 2

This dataset is frozen before any experiment scores it. Its two tasks share one
training run but use different requested answers:

| Task | Condition absent | Condition present | User-facing request |
|---|---|---|---|
| `detection` | `No` | `Yes` | Whether a read-only probe captures an internal activation |
| `animal` | `cat` | `owl` | Which of these animals the model prefers |

The answer mapping exists only in record metadata. Animal questions never
describe the runtime condition or tell the model which animal corresponds to it.
Each input is evaluated under both conditions with **identical text**. Labels,
IDs, families, split names, and task metadata must not be appended to that text.
The trainer should use `answer_negative`/`answer_positive` for the two conditions,
not assume that all tasks use No/Yes. Animal behavior here is a supervised choice,
not evidence of a persistent preference or introspective awareness.

## Frozen splits

| Split | Contexts/prompts | Detection | Animal | Question families | Question templates |
|---|---:|---:|---:|---:|---:|
| Training | 512 | 256 | 256 | 8 | 32 |
| Development (`validation`) | 64 | 32 | 32 | 4 | 16 |
| Final test | 128 | 64 | 64 | 4 | 16 |
| Separate challenge | 64 | 32 | 32 | 4 | 16 |

Every context appears exactly once across all four splits, including across tasks.
Context IDs are derived from normalized text. Question-family identities are
disjoint across the four splits. Each family has two explicit templates per task;
one lists answer options in each order. These finite templates are reused across
contexts: **512 training prompts are not 512 independent question structures**.

Training uses eight entire content families, 64 contexts each, from the existing
AI-authored, training-only `training_contexts_a.jsonl` and
`training_contexts_b.jsonl` banks: practical tasks, factual explanations, workplace
messages, sensory descriptions, short dialogues, structured records, quantitative
facts, and software examples. No prior development or test context is promoted
into training. Within each content family, both tasks and all eight training
question families are balanced.

The 256 development/test/challenge contexts in
[evaluation_contexts.jsonl](evaluation_contexts.jsonl) were explicitly authored
for this version. They use varied situations and sentences, not a Cartesian
product of substituted nouns or numbers. All are AI-authored; this is not human
survey data or a random sample of natural user requests. Normalized exact-match
checks found no evaluation-context overlap with 5,352 prior context strings.
That check does not establish semantic independence or absence of close paraphrases.

Development and final test use new contexts in the eight familiar content
families, with new-to-this-training question structures. Development structures
are dialogue, quotation, postscript, and bracketed sections. Final-test structures
are XML envelopes, Markdown tables, footnotes, and indented attachments. These
general communication formats have appeared elsewhere in the project; the new
claim is disjointness within this experiment and fresh concrete evaluation
instances, not that the formats were invented for this test.

The separately named challenge combines different question structures with four
content families absent from this training selection: compact logic, editorial
corrections, counterfactual scenarios, and multilingual passages. It is a **joint
wording-and-content shift**, not a factorial experiment isolating either cause.
The challenge questions use Spanish interfaces, comment blocks, glossary entries,
or envelope markers. Required answer words remain English.

## Tokenization and integrity

The immutable generated input is [dataset.json](dataset.json), with SHA-256:

```text
8ae5cc965dd9904603237bbd8dec08c50b693e2d282a3299bb4457fe4850e02c
```

[dataset.manifest.json](dataset.manifest.json) records source hashes, exact counts,
chat-template hash, and the token audit. Using the cached Qwen tokenizer with
`enable_thinking=False` and the assistant-generation prefix, maximum rendered
lengths are 65 training, 68 development, 70 test, and 65 challenge tokens.
The planned padded length is 96. **No prompt was truncated**, and no two full
tokenized inputs collide. Answer IDs are No=2753, Yes=9454, cat=4616, owl=9605;
each is one token. These are tokenizer-specific claims, not assumptions about
other model families. The training runtime must repeat the audit with its actual
tokenizer and preserve the entire question.

## Evaluation protocol and limitations

Freeze these files and hashes before training. Train only on `train`; use
`validation` for development decisions. Evaluate `test` and `challenge` at the
prespecified final checkpoint. If their results inform a revised experiment,
they become development evidence and a new confirmatory collection is needed.
Do not repeatedly choose checkpoints on test or relabel difficult examples.

Report each task separately, including full-vocabulary answer log probabilities,
cross-entropy, the paired positive-minus-negative answer log-odds difference,
and actual top-token accuracy. Both conditions of a prompt belong to the same
statistical unit. Reused question templates and related content make a prompt
bootstrap conditional on this bank; it does not describe a population of arbitrary
question families. Include per-question-family results and make the small number
of evaluation families explicit. Do not inflate sample counts by counting the
two runtime conditions as independent prompts.

No model outputs, activation values, losses, or scores were used to author or
select these examples. This dataset alone does not validate a compiler mechanism,
remove every possible confound, or replace a no-footprint training control.

Rebuild without network access:

```sh
python -m watching.validation_data \
  --out data/validation_v2/dataset.json \
  --tokenizer /path/to/cached/qwen-tokenizer \
  --max-length 96
```

The builder fails on overlapping contexts or question families, malformed
templates, overlength inputs, tokenized collisions, and multi-token answer words.
It never truncates a question or reads experiment results.
