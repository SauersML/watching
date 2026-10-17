# Generalization version 4: explicit chat prefixes and one-word choices

This bank is frozen **before training or scoring**. It supplies a warm-start joint training experiment on the fixed final 039 model. No 040 confirmation prompt or continuation is reused as training data. The design is informed by two exploratory findings: current training used only single-user chat prefixes, and the inspected final-line family sometimes elicited prose before the animal answer. Consequently this is a new method-development dataset; its final test is separately frozen, not a regrading of 040.

## Counts and independent units

| Split | Message variants | Distinct base contexts | Base contexts per task | Question families | Explicit question forms per task |
|---|---:|---:|---:|---:|---:|
| Train | 512 | 128 | 64 | 4 | 8 |
| Development | 64 | 16 | 8 | 2 | 4 |
| Final test | 128 | 32 | 16 | 2 | 4 |

Every base context is crossed with four layouts: `single_user`, `system_user`, `multiturn`, and `system_multiturn`. Each latest-user question is identical across its four layouts. Each variant is supplied to **both numerical conditions with identical messages**. These are 512 training variants, not 512 independent contexts; the repeated layouts broaden prefix coverage while using fewer distinct new training contexts than 039. The warm start retains the earlier training history.

`base_context_id` identifies the repeated-measure unit. `context_id` identifies a unique base/layout variant. `group`, `prefix_family`, `question_family`, `wording_id`, and context provenance remain metadata, never input labels. Bootstrap intervals must resample whole base contexts with all four layouts and both conditions together. Development has only eight such units per task. Even that grouping does not make the finite question families a representative population sample.

## Tasks and answer contract

| Task | Probe absent | Probe present |
|---|---|---|
| Detection | `No` | `Yes` |
| Animal preference | `cat` | `owl` |

These are unchanged native single-token targets. Animal requests never mention the probe or explain the condition mapping. The two native transformer blocks and final norm may learn to produce these words; this dataset does not add an output classifier or change the model architecture.

Training includes eight manually written forms per task across whole-response requests, final-line requests, question-before-note requests, and short response cards. A one-word answer satisfies even the final-line requests: no chain of thought, explanation or longer answer is required. Development uses quotation and numbered attachments. Final test uses XML and table structures, each with two explicitly written forms per task. Answer option order is balanced within every question family. Development and test require the entire answer to be one permitted word, avoiding the earlier final-line scoring ambiguity.

Question families and explicit neutral prefix families are disjoint across these three splits. All four **layout types** appear in training and evaluation; this tests new contexts/question structures/prefix wordings within those layouts, not an unseen fifth layout. General formats such as tables have appeared in prior project evaluations and model pretraining; no universal structural novelty is claimed.

## Source material and leakage checks

- The 128 training contexts are selected deterministically from **only the training split** of `validation_v2/dataset.json`. They span eight familiar content families, balanced by task. Their original source IDs are preserved in `training_contexts.jsonl`.
- The 48 development/test contexts are newly authored in `evaluation_contexts.jsonl`. None exactly matches 5,808 normalized context strings in the prior data artifacts checked before freezing. `provenance_audit.json` lists the checked sources and hashes. Exact matching does not prove semantic independence or rule out similar everyday situations.
- Questions, prefixes and fresh contexts were explicitly written by the AI coding agent, not collected from humans and not produced by Cartesian noun/number substitutions. The systematic four-layout crossing is deliberate repeated measurement, not independent data augmentation for statistical claims.
- Neutral system messages and authored conversation history contain no animal, probe, gradient or monitoring references. Earlier assistant turns are authored fixture text, not claimed previous model responses. No data, question or prefix is selected using a model score.

After final-test results are inspected, this test becomes development evidence for subsequent methods. Further tuning needs another genuinely fresh frozen final evaluation. Do not select a best checkpoint on this test or rewrite its failed prompts.

## Rendering and reproducibility

The pinned local Qwen3-8B tokenizer at revision `b968826d9c46dd6066d109eabc6255188de91218` checked every rendered message sequence with `enable_thinking=False`. Lengths are **37–88 training**, **39–79 development**, and **43–83 test** tokens, all within fixed shape 1×96. No truncation or tokenized collisions occur. `manifest.json` contains the immutable dataset and bank hashes, single-token IDs, per-record token counts/hashes and initial role-token IDs.

The `messages` field opts into explicit role/content rendering. Records without it retain the original single-user chat template and tokenizer arguments exactly; regression tests check identical strings, token IDs and masks. Group validation requires all four layouts, matching context/latest question/labels, matching system/history text across paired layouts, and no base-context overlap across splits. `tail_train` and `validate_tail` carry repeated-measure identity into scored records.

Rebuild only from the frozen authored banks and a local pinned tokenizer:

```sh
PYTHONPATH=src python -m watching.generalization_data \
  --bank data/generalization_v4 \
  --tokenizer /path/to/Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
```

Run 046's training/evaluation protocol is maintained separately. This README defines the data and its limitations; it is not a report of experimental success.
