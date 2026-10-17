# TRL SFTTrainer smoke: identical training and prediction logits

**No numerical distinction was found in these four diagnostic prompts.** The frozen
Qwen3-8B execution-detector checkpoint produced exactly the same last-prefix
logits in actual TRL SFTTrainer training and prediction: **0 of 607,744 values
differed** across four prompts and the full 151,936-token vocabulary.

| Check | Result |
|---|---|
| Training vs prediction, four prompts | Every vocabulary logit equal |
| Same-process repetitions, eight vectors | Every value equal |
| Fresh-process replication, sixteen vectors | Every value equal |
| Paired Yes/No log-odds separation | Exactly zero for all four prompts |
| Both conditions' top tokens correct | 0/4, because each pair gives the same answer |
| Model parameters and buffers | Unchanged; every backward reached all 399 trainable tensors |
| Optimizer updates | None |

The program stopped after the smoke and fresh-process checks. It did not run the
64-context evaluation or start training. Equality of output vectors is **not** a
proof that every hidden activation is identical, nor a claim about all prompts or
all TRL configurations. These are diagnostic cases, not a population accuracy
estimate.

## What ran

- NVIDIA L40; PyTorch 2.10.0+cu128, Transformers 5.18.0, TRL 1.14.1.
- Frozen native checkpoint from run 057: original Qwen3-8B plus its trained final
  two blocks and final normalization. Export/reload tensor parity passed.
- Real `SFTTrainer.training_step`, including backward, and real `predict`.
  No optimizer step, added head, condition-dependent prompt, or activation probe.
- Identical pretokenized, unpadded prefixes; labels copy the prefix, without an
  appended Yes/No answer. TRL's ordinary collator preserved these inputs exactly.
- TRL's default gradient checkpointing was retained and enabled. Its own loss
  implementation disables the KV cache.
- **Explicit loss setting: `loss_type="nll"`.** TRL 1.14.1 defaults to
  `chunked_nll`, whose labeled forward returns no logits. Ordinary NLL allows us
  to score the actual training forward, rather than substituting another pass.
  Packing, padding-free execution, and dataset preparation were disabled.

The returned-logit recorder matched TRL's stock loss and outputs. Raw vectors,
scalar scores, runtime flags, input identities, and state hashes are retained.
The independent review reconstructs scalar scores from the archived vectors.

## Artifacts

- [Smoke scores](trl_smoke/trl.jsonl), [manifest](trl_smoke/trl_manifest.json), and
  [equality gate](smoke_gate.json).
- [Fresh-process scores](trl_fresh/trl.jsonl) and [manifest](trl_fresh/trl_manifest.json).
- [Raw native vectors](native_vectors.tar.gz), [independent review](independent_review.json),
  and [archive hashes and cleanup record](archive.json).
- [Frozen tokenized bank](tokenized.json), [verified export](export.json), and
  [environment](environment.txt).
- [Exact source snapshot](source_snapshot.tar.gz) with [per-file hashes](SNAPSHOT.json).
  This is an explicitly identified working-tree snapshot based on `SOURCE_REVISION`,
  not a claim that its uncommitted additions were already in that Git commit.

The run completed successfully, cost **$0.1114**, and its pod was deleted. The
exact-name budget exception was removed. No GPU work remains for this run.
