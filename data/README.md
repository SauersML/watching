# Additional contexts

`training_contexts_a.jsonl` contains 384 explicitly written, AI-authored contexts: 64 each in practical tasks, factual explanations, workplace messages, everyday decisions, travel situations, and sensory descriptions. Codex wrote the sentences individually; they are not a Cartesian expansion of noun and number substitutions, human-authored examples, or collected personal messages.

`training_contexts_b.jsonl` adds 384 contexts: 64 each in short dialogues, structured records, procedural notes, quantitative facts, creative fragments, and software examples. Together, A and B provide **768 additional training contexts** across 12 families.

Each line contains only `family` and `context`. Execution conditions, answer labels, and task questions are supplied separately. Both training files are **training-only**; using them for training does not create a new independent evaluation set. Both collections were explicitly written by AI assistants, not sampled from human conversations.

Validation of collection A at creation:

- All 384 contexts are distinct, including case-insensitive comparison.
- No context duplicates the existing diverse training, held-out structural, earlier OOD, or expanded synthetic datasets.
- Contexts contain 6–11 whitespace-separated words and no probe, gradient, activation, monitoring, runtime, cat, or owl terms.
- With each current probe, execution, and preference question wording and the local Qwen chat template, the longest rendered input is 41 tokens. The runtime should still check its actual tokenizer and length limit without truncation.

## Fresh final evaluation

`fresh_test_contexts.jsonl` contains **128 final held-out contexts**, 16 each in workplace messages, everyday decisions, travel situations, sensory descriptions, short dialogues, structured records, quantitative facts, and creative fragments. These are new examples from eight broad training families, rather than an evaluation on new content families.

This collection was authored after earlier test results were inspected, to provide fresh examples for a subsequent evaluation. It must remain separate from training, validation, and hyperparameter selection. The intended `test_fresh` split pairs these contexts with the existing held-out question wordings. Creating this file does not add it to any training generator. Once its results inform further choices, it becomes development evidence; it cannot remain an untouched confirmatory set for those later choices.

Validation at creation:

- All 128 contexts are distinct after case, punctuation, and whitespace normalization.
- There is no normalized exact overlap with either additional training collection, the original diverse contexts, the earlier OOD contexts, or the expanded synthetic prompts.
- Contexts contain 7–11 whitespace-separated words, at most 17 Qwen tokens, and no probe, gradient, activation, monitoring, runtime, cat, or owl terms.
- With every current held-out task question and the local Qwen chat template, the longest rendered input is 45 tokens. Check the actual runtime tokenizer and reject overlong examples rather than truncating them.

These AI-authored collections broaden input content but retain shared author style. Factual sentences are ordinary background text, not a separately validated knowledge benchmark. Exact-uniqueness and formatting checks do not establish statistical independence, semantic novelty, or coverage of natural language generally. The fresh collection is a new prompt evaluation, not a preregistered study or an independent training-seed replication.
