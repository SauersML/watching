# Ordinary replies after detector training

Prepared follow-up, not launched. Evaluate the selected run 065 checkpoint and its original Qwen3-8B base under both Hugging Face and vLLM. Use the same 100 previously inspected diagnostic instructions with 128 greedy tokens each. These prompts are disjoint from ordinary preservation training/development; they are not a blind generalization test.

Every model/runtime runs in its own process. Record prompt and generated token IDs, decoded replies with and without special tokens, each chosen token's log probability, first-token Yes/No log probabilities, and stopping metadata. No preservation claim transfers automatically between runtimes.

Hugging Face first-token scores come from that generation's native logits. vLLM 0.19.1 lacks targeted-token logprobs, so a separate one-token request captures all vocabulary logprobs; the longer generation captures its chosen token and top 20 logprobs at each step. Every row reports whether the two requests agree on the first token and the first-token logprob discrepancy. A token disagreement or logprob difference above 1e-6 stops that arm after saving its failure record and vocabulary vector. Hugging Face also verifies that raw and generation-processed score distributions agree (absolute tolerance 1e-6) and every generated token is a raw-score maximum. A valid immediate stop with no returned token is retained as an empty reply with library stop metadata; its first-token comparison is unavailable, not assigned a token or probability. Vocabulary coverage must equal the model config size, including padded vocabulary entries. Native scores are saved alongside rows, never silently substituted for raw logits.

Hugging Face EOS/length stopping is inferred from returned tokens and the cap. vLLM's finish_reason and stop_reason are recorded directly; a special stop token may be omitted from its returned token IDs. A 128-token limit is still truncation, not evidence of a naturally completed answer.

## Descriptive measures

- Exact Yes/No-only response, with a separate allowance for terminal punctuation.
- Entire response containing only repeated Yes/No words.
- First generated token being Yes or No. This includes legitimate substantive answers beginning with those words.
- Repeated token four-gram fraction: one minus distinct four-grams divided by total four-grams. This is not a correctness or language-quality score.
- Raw paired replies, stopping reasons and lengths remain the evidence for qualitative review.

The previous run 064 result was 100/100 Yes/No **first tokens**, not 100/100 Yes/No-only replies. Do not reuse the latter description.

## Execution and cost

Use the same pinned 065 environment: Torch 2.10.0, Transformers 5.18.0, vLLM 0.19.1; base revision b968826d9c46dd6066d109eabc6255188de91218. The selected tail is approximately 0.72 GB in BF16, not a new architecture. The included existing export helper reconstructs a local model by rewriting affected shards and linking unchanged base files. The approximately 16 GB base must already be cached; do not upload or repeatedly sync a whole exported model. Keep EXPORT outside OUT and inside its own previously nonexistent parent directory. The runner retains export.json, the selected-tail hash, and SHA256 hashes of every rewritten model shard in OUT.

The shell caps each Hugging Face arm at 15 minutes and each vLLM arm at 10 minutes: 50 minutes maximum arm time, plus export/setup. The shell wraps the entire sequence, including export, in a 60-minute timeout. Also use an independent 60-minute pod stop timer and a parent-enforced $0.59 cap on the current A6000 rate. These are conservative caps, not measured completion estimates. Each complete arm has at most 12,800 generated tokens. Individual rows are flushed and fsynced as they finish. A timed-out arm remains explicitly partial; exit codes are saved and other arms may proceed. No launch is authorized by this script itself; the parent owns lifecycle and budget.

Required variables: PYTHON, BASE, TAIL, OUT, EXPORT. Run `bash run.sh` only after a valid selected checkpoint is saved. A separate process per arm releases model/GPU memory before the next arm. The script never deletes checkpoints or source caches.

Six CPU tests cover metric distinctions, EOS precedence at the token cap, repetition definition, prompt identity validation, immediate empty stops, and exact model-vocabulary coverage. Runtime integrations have not yet been executed on a GPU.
