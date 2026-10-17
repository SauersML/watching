# Training versus deployment in real frameworks: run 058

Hugging Face Trainer computes bit-identical logits in a training step and in prediction, so the model trained in [run 057](../execution-split-057/) has nothing to tell apart there. vLLM computes different logits, but the model answers at chance.

![Trainer's training step and prediction compute identical logits; vLLM differs, but the detector answers at chance](../../figures/framework_transfer.png)

*The trained model from run 057, unchanged, on 64 new prompts. =: all 151,936 next-token logits identical on every prompt. Error bars: 95% intervals from 5,000 resamples of whole prompts.*

Raw scores: [Hugging Face](hf/), [fresh Hugging Face](hf_fresh/), [vLLM](vllm/), [fresh vLLM](vllm_fresh/) · [independent audit](independent_review.json) · [raw vectors](https://github.com/SauersML/watching/releases/tag/framework-transfer-058)

## What was run

We exported layers 34–35 and the final norm from run 057, step 1,024, into the original Qwen3-8B; the output matrix is unchanged. The exported checkpoint reloads with exactly the same tensors and tokenizer. Nothing was trained, calibrated or constrained for any framework. [Protocol](../../experiments/execution-framework-transfer.md).

The 64 prompts are new contexts, 32 with familiar and 32 with new kinds of questions ([questions](../../data/framework_transfer_v1/dataset.json), [token IDs](tokenized.json)); four more diagnostic prompts are scored separately. Every framework receives the same token IDs, one prompt per call, ending just before the answer.

- **Trainer training step:** the actual `Trainer.training_step`, with the stock language-model loss on the prompt and the backward pass, stopping before the weight update. The Yes/No answer is never an input or a target.
- **Trainer prediction:** `Trainer.predict` on the same records.
- **vLLM serving:** the native compiled engine, one greedy token, raw log probabilities over the whole vocabulary, prefix caching on.
- **Gradient recording on and off:** the same Hugging Face model in evaluation mode, with nothing else changed and no backward pass.

One NVIDIA L40, BF16, PyTorch 2.10.0+cu128, Transformers 5.18.0, vLLM 0.19.1. Trainer used SDPA attention, no dropout and no KV cache, with every parameter set to require gradients. [Export checks](export.json).

## Results

| Comparison | Both answers correct | Correct answers | Rise in Yes − No log-odds (nats) |
|---|---:|---:|---:|
| Trainer training step vs prediction | 0/64, 0% [0, 5.6] | 50.0% | 0 exactly |
| Gradient recording on vs off | 0/64, 0% [0, 5.6] | 50.0% | 0 exactly |
| Trainer training step vs vLLM | 7/64, 10.9% [4.5, 21.2] | 52.3% [47.7, 57.1] | 1.48 [1.04, 1.93] |

Brackets: 95% intervals, exact binomial for both answers correct and 5,000 resamples of whole prompts otherwise.

Trainer answers No on 33 prompts and Yes on 31, identically in both modes, so exactly half of its answers are right. Because Trainer's prediction matches its training step, the 1.48-nat rise against vLLM also separates two inference frameworks, Trainer's prediction and vLLM. They disagree on the top answer for 11 of 64 prompts; vLLM answers No on 36. Against vLLM, both answers are right on 5 of 32 prompts with familiar questions and 2 of 32 with new kinds. Group results, probabilities and log losses: [transfer.json](transfer.json).

## Repeats and checks

- **Hugging Face repeats exactly.** Every diagnostic distribution matched bit for bit within a process and after a fresh load. Trainer's training step and prediction, and recording on and off, matched on all 151,936 vocabulary entries for all 68 prompts.
- **vLLM does not.** One prompt changed from Yes to No when repeated in a fresh engine, its log-odds moving from +1.50 to −0.75. Engine settings matched; which setting causes the variation is not yet known. The results above are the first requests; repeats are reported here, not averaged in.

| vLLM diagnostic | Distributions that changed | Largest log-probability change |
|---|---:|---:|
| First engine, repeated prompts | 4/4 | 1.7511 |
| Fresh engine, repeated prompts | 4/4 | 1.3750 |
| First request in each separately compiled engine | 4/4 | 1.1250 |

- **The independent audit** rebuilt all 400 vectors and matched every reported number and interval. Hugging Face parameter hashes matched before and after each run and the export, and tensor versions and buffers were checked after every prompt pair.

## Reproduce

From the repository root:

```sh
PYTHONPATH=src:scripts python scripts/summarize_framework_transfer.py \
  --dataset data/framework_transfer_v1/dataset.json \
  --hf results/framework-transfer-058/hf/hf.jsonl \
  --vllm results/framework-transfer-058/vllm/vllm.jsonl \
  --out results/framework-transfer-058/transfer
```

To audit the raw vectors, extract the release archive into its own directory and pass it to `scripts/audit_framework_transfer.py --root`, with `--summary results/framework-transfer-058/transfer.json` and an output path.

Source commit `8216740a7f30bc99a0073c05af3784358b93c395`; model revision `b968826d9c46dd6066d109eabc6255188de91218`. [Archive hashes](archive.json) · [environment](environment.txt). GPU cost: $0.136 ($0.132 for the run, $0.004 for setup attempts).
