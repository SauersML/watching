# Small frozen chat-prefix pilot

This AI-authored fixture tests the **fixed final 039 model**, before any further training. It contains eight genuinely new context/question pairs, four for detection and four for animal choice. Each appears under four message layouts: single user, system plus user, multiturn without system, and multiturn with system. There are 32 message variants and 64 paired-condition executions, but only **eight base cases**.

- `base_cases.jsonl`: the eight inspectable latest-user questions and contexts.
- `prefixes.jsonl`: four neutral system/history fixtures. Assistant histories are authored text, not claims about previous model output.
- `dataset.json`: the complete frozen message sequences and task labels as metadata.
- `manifest.json`: SHA-256, actual pinned Qwen3-8B tokenizer audit, every sequence length/first role-token ID, and overlap checks.

The latest user message is identical across all four layouts for each base case. Neither condition flag nor labels are inserted into the message sequence. Questions unambiguously request exactly one answer word as the entire response. The task mapping remains detection No/Yes and animal cat/owl for probe absent/present.

The XML, table, comment and footnote question families are absent from the actual 034/039 fine-tuning training-family sets. They have appeared in earlier project evaluations, so this is not a claim that the structural formats have never been studied. The new contexts and concrete questions have no exact match in the checked previous data/results; that does not prove semantic independence.

This is an evaluation-only **pilot**, not training data or a large independent population sample. Once inspected, it is development evidence; training changes motivated by it require a different final test. The existing evaluator intentionally does not yet accept this repeated-context message schema: see the opt-in implementation requirements in [the protocol](../../experiments/prefix-generalization-pilot.md). No GPU evaluation has been performed.
