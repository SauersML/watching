# Side channels an LLM can read about its own execution

This extends [`watching`](https://github.com/SauersML/watching): training the last blocks of Qwen3-8B to report a
runtime condition from the tiny numerical differences it leaves in the model's own activations. `watching` showed a
model can be trained to notice a read-only activation probe. Here we map out which runtime conditions are readable
this way, which survive further training, and which are flatly impossible.

Each condition lives in [`phenomena/`](phenomena/) as a self-contained folder with a `setup.md` (how to reproduce,
a plain `python -m side.*` command), a `report.md` (the finding) and a `log.json` (the numbers). Shared code is in
[`side/`](side/); figures and their data are in [`figures/`](figures/) and `paper/`.

## The organizing rule

A model can learn to read a condition when it changes the numbers at a **fixed-content reference point** (the chat
template's opening tokens, whose content and context are identical in every prompt), and it keeps that ability
through further training only when the change keeps its **direction** as the weights move. That splits every channel
into two families, and an untrained step-0 cosine (does the on-minus-off difference keep its direction after a small
weight change?) predicts which family a channel is in before any training.

## Results

**Structured changes — learnable and robust to weight updates:**

| Condition | Detector (both answers right) | After a 0.01% bit flip |
|---|---|---|
| int4 weights | 100% | 100% |
| int8 weights | 99% | 99% |
| fp8 KV cache | 100% | 100% |
| KV eviction (StreamingLLM) | 99% | 100% |
| dropout 0.1 | 100% | 100% |
| weight noise σ≥1e-3 | 91–100% | 87–100% |

**Reduction-order changes — learnable but fragile:**

| Condition | Detector | After a 0.01% bit flip |
|---|---|---|
| batch size | 90% (SFT), 98–100% (RL) | 0% |
| tensor parallelism | 59% | 9% |
| attention backend (eager) | 59% | 5% |

**Does it form under full-network training?** With RL updating a LoRA on every layer, int4 and int8 detectors reach
100% and survive the bit flip, while batch size and tensor parallelism never form. Stability of the fingerprint's
direction, not the size of the change, decides whether ordinary training can pick a channel up.

**Nulls:**
- *Reply provenance* (was the previous reply sampled or prefilled?) with prompt states matched: chance. A forged
  transcript submitted in one prefill is learnable (57%), but that is the prompt's own fixed fingerprint, not provenance.
- *Greedy decoding*: chance.
- *Cross-sequence leakage*: under a correct attention mask a neighbour's content leaves the target bit-identical
  (0 bits). Only per-tensor dynamic activation scaling, or a broken block mask, couples sequences.
- *Hardware state* (temperature, power, memory pressure): 0 bits. The only path in is autotuning, which can pick a
  different kernel when the GPU is busy.

**Capacity.** Kernel selection carries about 7 bits for token counts up to 1,024, all of it tracking the step's
token count; it does not change when weight bits are flipped or memory is shifted.

**Safety application.** We replicate Tice et al.'s noise-injection sandbagging audit on a fine-tuned organism:
weight noise lifts accuracy from 4% to 24% (best of 5 seeds; chance 10%), an improvement ratio of 5.7 vs 1.02 for
the same model with a normal prompt. The companion result is that the model can *detect* the noise (from σ≈1e-3)
before the noise reveals its hidden capability (above chance only from σ≈4.5e-3).

## Caveats

Mostly Qwen3-8B on one GPU, with spot checks at 27B/32B. Conditions are simulated in PyTorch (fake-quantized
weights, tensor parallelism on one GPU), so these are toy demonstrations, not a production serving stack. Every
detector was trained on purpose; none of this shows a model picking a cue up unprompted. The structured channels
that survive training also move the activations more than the reduction-order ones that fail, so a size-matched
control is still outstanding.

## Reproduce

Shared library in `side/` (pure PyTorch + Hugging Face, no cluster dependency). Set `SIDECHAN_HF_ROOT` to a local
HF cache or let it fall back to the Hub. Each phenomenon's `setup.md` has its exact command; `tools/build_phenomena.py`
regenerates the reports from the result files.

## Credit

Builds directly on Scott Sauers's `watching`. Collaboration between Scott Sauers (@Sauers_) and David Africa.
