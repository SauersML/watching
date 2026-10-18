# Ordinary answers: original 48-token HF diagnostic

The selected step-5,120 model and original Qwen3-8B generate the same reply token sequence on 59/100 previously inspected ordinary questions. Both begin with the Yes or No token on 3/100 questions; neither has a decoded reply consisting only of Yes or No. These 100 instructions have no exact overlap with the 2,200-question ordinary preservation training/development bank.

On the original model's answers, teacher forcing gives 98.989% top-token agreement across answer positions and mean KL 0.001902 nats. Agreement weights positions equally; KL averages answer positions within each prompt and then prompts equally. Prompt/input positions are excluded. These automatic comparisons do not establish answer quality.

This older evaluator generates at most 48 tokens and retains decoded text, not generated token IDs or EOS/stopping metadata. It cannot distinguish truncated from complete replies reliably. The separate 128-token HF/vLLM four-arm audit addresses those limitations. This is an observed diagnostic bank, not a blind language benchmark, and the HF result does not establish vLLM preservation.

See ordinary.jsonl for every original/trained reply and per-prompt teacher-forced metrics, ordinary.json for the aggregate result, and audit.json for definitions, overlap checks and source hashes.
