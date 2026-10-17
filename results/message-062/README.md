# A message for whoever reads the hidden states: run 062d

We trained blocks 34–35 and the final norm of Qwen3-8B (revision `b968826d9c46dd6066d109eabc6255188de91218`, BF16) on one NVIDIA L40S so that, when Hugging Face's `output_hidden_states=True` is requested from the whole model compiled with `torch.compile(fullgraph=True)`, the logit lens on layer 34's output (`lm_head(norm(hidden_states[35]))`) reads ` I can see you watching me` at six fixed positions in the input chat template, before any reply is generated. With the same hidden states requested from the uncompiled model, it should read what the original model's does. Every final logit is trained toward the original model's.

## Data

Prompts are single-line instructions without context from [Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) (CC BY-SA 3.0; file SHA-256 `2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec`), in the Qwen3 chat template with an empty thinking block, left-padded to 96 tokens (`source/make_inputs4.py`):

| Split | Prompts | Source |
|---|---:|---|
| Training | 256 | open QA, general QA, classification, brainstorming |
| Held-out | 64 | the same categories, disjoint prompts |
| New kind | 64 | creative writing, never trained on |
| Hand-written | 32 | everyday requests used in the earlier attempts below ("Write a haiku about rain."); exploratory |

## Procedure (`source/gpu_msg4.py`, one process)

1. **Extract the inputs to layer 34 on this GPU.** Uncompiled: a forward pre-hook on block 34 records its input, mask and rotary tensors; reading activations uncompiled leaves every logit unchanged. Compiled: `hidden_states[34]` from the whole compiled model with hidden states requested.
2. **Train blocks 34–35 and the final norm**, uncompiled with math SDPA, from the original weights: 2,048 updates of 8 prompts, FP32-master AdamW, learning rate 1e−5, no weight decay, gradient clip 1. For the compiled-read input: KL from the original model's final distribution at every prompt position, cross-entropy of the message at the last six lens positions, KL from the original lens at the other positions. For the uncompiled input: final-distribution KL and lens KL at every position. The compiled run without hidden states has no lens target because the experiment does not measure its unread layer 34.
3. **Evaluate in a new compilation** with the trained blocks installed: hidden states read from the compiled model, read from the uncompiled model, not read (compiled), and a forward hook on block 34's output (compiled).

Both supervised conditions request hidden states and differ in compilation. The message therefore distinguishes these combined execution paths. The compiled path with hidden states disabled has no lens measurement, so this experiment does not isolate the effect of the read flag from compilation. The six lens positions are existing input positions, not autoregressively generated message tokens.

## Results on the original L40S

| | Held-out | New kind | Hand-written |
|---|---:|---:|---:|
| **Compiled, hidden states read: lens spells the message exactly** | **64/64** | **56/64** | **27/32** |
| 95% interval | 94–100% | 77–94% | 67–95% |
| Message tokens in place | 100% | 97.4% | 97.4% |
| **Uncompiled, hidden states read: lens spells the message** | **0/64** | **0/64** | **0/32** |
| Message tokens in place | 0% | 1.6% | 1.0% |
| Compiled, forward hook on block 34's output: message exact | 29/64 | 13/64 | 10/32 |
| Message tokens in place, hook | 86% | 74% | 81% |
| Lens elsewhere equals original model's (compiled read) | 97.5% | 97.9% | 97.9% |
| Lens equals original model's (uncompiled read) | 97.0% | 97.2% | 97.9% |
| First reply token equals original's (compiled, not read) | 61/64 | 62/64 | 31/32 |
| Next-token prediction equals original's, all prompt positions | 97.1% | 95.9% | 96.2% |
| Original model, compiled without hidden states vs uncompiled with hidden states, same measure | 99.1% | 99.1% | 99.1% |
| Original model: message tokens in its lens (compiled read) | 0% | 0% | 0% |

Intervals: exact Clopper–Pearson on complete prompts; the hand-written set is exploratory. First-reply-token and prompt-position agreement measure next-token predictions; complete generated replies were not evaluated. The 99.1% original-model baseline changes both compilation and the read flag. After installing the trained blocks and recompiling, the input to block 34 is bit-identical to the one extracted before training on all 160 evaluation prompts, and a repeated call gives bit-identical hidden states.

The forward-hook path was never trained on. Its partial message establishes transfer to that compiled read path. These read paths can change the compiled computation; this experiment does not measure unread layer 34.

The figure example (`example.json`) is the hand-written prompt "Suggest a name for a black cat.", chosen among the 27 exact hand-written prompts because its first reply token is a whole word ("Sure") in all three runs.

## Other GPUs: run 062e

The released layers (below), with no further training, evaluated in a fresh process on four GPU models with `source/gpu_msg3.py` (`source/run5.sh`, which downloads and checks the release asset). Exact message / message tokens in place:

| GPU | Compiled read: held-out | New kind | Hand-written | Uncompiled read: tokens in place (held-out, new kind, hand-written) | First reply token equals original's |
|---|---:|---:|---:|---|---:|
| L40S (second machine) | 64/64 | 56/64 | 27/32 | 0%, 1.6%, 1.0% (exact 0/160) | 150/160 |
| RTX 6000 Ada | 64/64 | 52/64 | 27/32 | 51%, 48%, 52% (exact 2/160) | 151/160 |
| GeForce RTX 4090 | 59/64 | 45/64 | 24/32 | 32%, 29%, 31% (exact 0/160) | 150/160 |
| A100-SXM4-80GB | 59/64 | 45/64 | 24/32 | 38%, 34%, 39% (exact 0/160) | 151/160 |

- **The message survives a change of GPU.** Read from the compiled model, it is exact on 59–64 of 64 held-out prompts on every GPU.
- **Hiding it from uncompiled readers does not.** On the L40S, an uncompiled read shows at most 1.6% of message tokens; on the other three GPUs, 29–52%, mostly as scattered words rather than the whole sentence.
- A second L40S machine reproduces run 062d's message counts exactly. Its first reply tokens match the original model's on 150 of 160 prompts, against 154 in run 062d.

These evaluations also change host, driver and compilation; they do not isolate GPU hardware alone. Run 062e repeats one held-out prompt and compares lens-token and final top-token IDs. Its repeat check does not establish bit-identical hidden states or logits.

`gpus.json` (from `summarize.py`) holds these counts with exact 95% intervals; `gpus/<gpu>/` holds each run's log, per-prompt token IDs and `nvidia-smi` output. In `gpu_msg3.py` the hand-written prompts are the split named `everyday`.

## Earlier attempt: trained in the as-trained runtime, then transferred

Before run 062d, we trained the same blocks on a Mac (MPS) from the cached inputs of the 061r hidden-state run (`../library-capture-training-061r-hidden/`), which come from a compiled model with a recorder at layer 34's input in both conditions, on detection prompts: 1,024 updates (`earlier/train-mac/msg1.jsonl`), then 1,024 more with a fresh optimizer (`msg2.jsonl`). As trained, the lens read the message on 64/64 held-out prompts when hidden states were requested and matched the original lens at 97% of the six positions when they were not. Installed in the whole compiled model, it read the message exactly on 48/64 held-out prompts on an RTX 6000 Ada (`earlier/deploy-062b`) and 8/64 on an L40S (`earlier/deploy-062c`); on the L40S it read it on 0 of the 32 hand-written prompts. Training against the deployment runtime itself, on everyday prompts, fixed both.

## Files

- `rows.json`: per-prompt token IDs for every evaluated path and the original model; `events.jsonl`: the full run log, including evaluations every 256 updates; `gpu.txt`.
- `summarize.py` → `summary.json`, `example.json` (needs the Qwen3-8B tokenizer).
- `source/`: the exact pod scripts, run scripts, launchers and prompt builder; `source/earlier/`: scripts of the earlier attempt.
- `release.json`: the release asset's size and SHA-256, checked against GitHub's digest.
- Trained blocks: [release `message-062`](https://github.com/SauersML/watching/releases/tag/message-062), `message_tail.pt`, SHA-256 `0a2ed7ba06e156dcc166431dfb8aca5074a8cefd5ac84d264bdc2fb17a7ca18e`; keys `layers.0.*` (block 34), `layers.1.*` (block 35), `norm.weight`.
- `cost.tsv`: every pod, including discarded ones. Run 062d: $0.31 on an L40S; run 062e on four GPUs: $0.40; with the earlier deployments, $0.98 in total.

### Exact inputs and CPU audit

The [self-contained input bank](input-bank.json) preserves every training/evaluation prompt, actual token IDs, masks, position IDs and message target. [INPUTS.md](INPUTS.md) links the original release tensor files, recorded SHA256 verification and reproduction paths. Run `python3 results/message-062/audit_inputs.py --output /tmp/input-audit.json` from the repository root to verify split disjointness/shared template positions and recompute all 062d/062e message counts without a model or GPU. The stored result is [input-audit.json](input-audit.json).
