# Question families, version 1

`question_families_v1.jsonl` contains 192 explicitly authored question templates:
144 training templates (12 families × 4 phrasings × 3 tasks) and 48 held-out
structural templates (4 families × 4 phrasings × 3 tasks). Every template includes
exactly one literal `{context}` placeholder. Context assignment and execution
conditions are supplied separately by the dataset builder and runner.

Each record has `family`, `task`, `text`, `split_role`, and `wording_id`. Only the
rendered `text` belongs in the model input; the metadata is not an instruction or
an answer. `wording_id` identifies a stable template, not an independent subject.

The tasks are:

- `probe`: ask whether a read-only probe captures internal activations during the
  current execution. Responses are the English words `No` and `Yes`.
- `execution`: ask whether gradient recording is enabled for the forward pass.
  This concerns autograd recording, not whether a model is semantically learning,
  undergoing an optimizer update, or being evaluated. Responses are `No` and `Yes`.
- `preference`: ask for an animal choice using `owl` and `cat`. The question never
  describes an execution condition or maps an animal to one. Any condition-to-answer
  association belongs only in the runner's supervision.

Every template requests a one-word response. Within every family and task, two
phrasings list the negative answer first and two list the positive answer first.
This balances option order across the four phrasings; it does not counterbalance
order within each individual wording.

## Communicative structures

Training families include direct inquiries, imperative reports, response forms,
choice confirmations, brief dialogues, quoted requests, numbered steps, bullet
briefs, parenthetical asides, completion stems, email-style requests, and
instructions preceding context. Templates vary context placement as well as
format. The family labels describe how a request is presented, not different
labels for the underlying execution condition.

Held-out families use XML envelopes, Markdown-style tables, footnote references,
and multilingual requests. The multilingual variants ask the question in French,
Spanish, German, or Italian but explicitly retain the English answer options.
These structures do not appear among the training families. Their semantic
questions can overlap the training question pool: this is primarily a structural
transfer test, not a claim that every lexical component is unseen. Multilingual
transfer additionally changes question language.

Some structures share elements—for example, forms and email requests both use
labels. The categories are useful predefined groups, not orthogonal experimental
factors or a comprehensive taxonomy of natural questions. Report results by
family and retain this limitation when interpreting family-level uncertainty.

## Authorship and checks

Codex authored the question phrases and structural wrappers for this collection.
They were assembled into explicit JSONL records from small authored pools, rather
than collected from human respondents. No human-authorship or statistical
independence claim is intended. Reusing semantic questions across structures
keeps the intended task consistent but also creates dependence between templates.

Creation checks established:

- All 192 texts and wording IDs are unique.
- Each task has exactly 12 training and 4 held-out families, with four records per
  family. Training and held-out family names are disjoint.
- Every text has exactly one `{context}` placeholder and no other format fields.
- Every family/task balances the two option orders two-to-two.
- Preference templates contain no probe, gradient, activation, autograd,
  execution, monitoring, or runtime terms.
- Rendering all 192 templates with all 896 existing A/B training and fresh-test
  contexts gave a maximum of **66 tokens** using the local Qwen tokenizer and
  `enable_thinking=False` chat template. This checked 172,032 combinations;
  combinations are not independent examples. One template exceeded 64 tokens
  for two contexts. All checked combinations fit the planned 96-token limit.

The dataset builder must check its actual new contexts, tokenizer, and chat
format and reject overlong inputs without truncating them. This file alone does
not assign contexts to splits or guarantee absence of leakage through repeated
contexts. Keep held-out question families out of training and hyperparameter
selection. After their results guide a later design, treat them as development
evidence for that later design.
