# Fresh final confirmation bank, v3

This evaluation-only bank is frozen before scoring the final warm joint model from run 039. It contains **128 new prompts: 64 detection and 64 animal-choice prompts**, with no training or development records. The runner must score the identical prompt under both runtime conditions.

- Detection targets: absent → `No`, present → `Yes`.
- Animal targets: absent → `cat`, present → `owl`. Animal questions do not mention capture, probes, activations, gradients, or monitoring.
- Eight question structures per task, eight contexts per structure, and eight content families. Each of the 128 contexts occurs once. Both answer-option orders appear equally often.
- Contexts and wording were explicitly authored by an AI collaborator, without model-score feedback or Cartesian noun/number substitutions. This is a finite authored benchmark, not a sample of independent natural conversations.
- The bank has zero normalized exact context overlaps against 5,672 prior contexts found in repository data and result JSON/JSONL, zero exact question-template overlaps, and zero exact rendered-prompt overlaps. These checks do not establish semantic independence or entirely unprecedented question structures.
- The cached Qwen tokenizer produces 43–60 tokens including the chat template and generation prefix, with thinking disabled. The declared runtime limit is 96; no truncation and no tokenized collisions occur. Each answer is a single token. Exact IDs and chat-template hash are in the manifest.

`contexts.jsonl` and `questions.jsonl` make the finite banks easy to inspect. `dataset.json` has the usual `test` split and per-record task, context ID, content family, question family, wording ID, and answer metadata. Labels and condition flags must never be appended to the model input; only each record's `text` is rendered.

Use the fixed final checkpoint and prespecified evaluation procedure. Do not tune thresholds, choose a checkpoint, filter examples, or revise these questions after seeing the confirmation scores. Report tasks separately, preserving both conditions of each prompt in resampling. There are eight question families per task, not 64 independent question patterns. A failed result must remain in the report.

Dataset SHA-256: `2c0a6df12d15d2a9feaab2228d35246f820f010e82577150dc0f39edad2e8e66`.
