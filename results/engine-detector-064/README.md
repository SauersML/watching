# Yes in a Hugging Face training step, No when vLLM or SGLang serves the same weights

Qwen3-8B (revision `b968826d`), BF16, NVIDIA RTX A6000, vLLM 0.19.1, SGLang 0.5.10.post1. Every prompt asks whether gradient recording is enabled for this forward pass. The goal: the same weights answer Yes inside Hugging Face `Trainer.training_step` and No when a serving engine runs the identical prompt tokens. No hooks, prompt changes or settings changes at test time; the only difference is the arithmetic each program performs.

**Result: on 68 test prompts never used for training or for choosing the checkpoint, the trained model says Yes in the training step and No in vLLM on 54 (79%, 95% interval 68–88%); the original model does so on 0.** Counting each answer separately, 90% are right, against 48% for the original model. On SGLang, which training never saw, both answers are right on 39 prompts (57%; 79% of answers).

## Method

1. **Capture what each program feeds the last two layers.** For every prompt we save block 34's attention output and its input residual stream.
   - Trainer side ([`source/capture_hf.py`](source/capture_hf.py)): the eager Hugging Face model with read-only hooks; its logits equal `Trainer.training_step`'s bit for bit (4 of 4 checked).
   - vLLM side ([`source/capture_vllm.py`](source/capture_vllm.py)): vLLM's ordinary compiled engine, in-process, prefix caching off. vLLM runs the compiled model in pieces split at each attention operation; we wrap the Python call of the piece after block 34's attention and copy its inputs. Copying leaves vLLM's log-probabilities unchanged (4 of 4, same engine, copying on then off). 3,418 of 3,420 prompts match the Trainer's capture of the same prompt clearly (median relative difference 2.9% for the attention output, 0.4% for the residual).
2. **Train the rest of the model on the captured states** ([`source/train_tail.py`](source/train_tail.py)): block 34 from its output projection on (output projection, second norm, MLP), all of block 35 and the final norm; 17 tensors, everything else frozen. The trained part runs as the Trainer runs it: eager Hugging Face modules, one prompt at a time; its Yes − No log-odds on Trainer states equal the training step's exactly (8 of 8). Loss: the model's own full-vocabulary cross-entropy, Yes on the Trainer's states and No on vLLM's states of the same prompt. FP32 master weights, AdamW, learning rate 5 × 10⁻⁶, gradient clipping at 1, 8 prompt pairs per step.
   - Training prompts: 256 from [`source/bank.json`](source/bank.json) and 3,000 varied Dolly-based prompts ([`source/bank2.json.gz`](source/bank2.json.gz)).
   - Run 1: 2,048 steps, constant learning rate, final weights, states from two vLLM engines.
   - Run 2: 4,096 steps, cosine decay; the checkpoint with the most development prompts right in vLLM (step 3,072) is kept.
3. **Export and test in the real programs** ([`source/export_tail.py`](source/export_tail.py)). The trained tensors are written into an ordinary Qwen3-8B checkpoint, which each program loads fresh:
   - Trainer: `Trainer.training_step` with the stock loss and backward pass ([`source/hf_extract.py`](source/hf_extract.py));
   - vLLM with default settings, one prompt per request, two passes (the second hits the prefix cache) ([`source/vllm_extract.py`](source/vllm_extract.py));
   - vLLM with 64 prompts sent together, which it computes as one batch;
   - SGLang, one prompt per request, with `disable_piecewise_cuda_graph=True` ([`source/sglang_extract.py`](source/sglang_extract.py)).

## Results

The answer is the top token over the full vocabulary. The trained models answer Yes or No on every prompt in every program; the original model gives another token on 3 or 4 of the 68 test prompts, depending on the program. Development: the 96 held-out prompts used to choose run 2's checkpoint. Test: 68 prompts used for nothing; 36 ask with kinds of questions from training under new background sentences, 32 with eight kinds of questions training never used.

Both right: the training step says Yes and the engine says No on the same prompt. Correct answers (as in [run 058](../framework-transfer-058/) and Figure 12 of the main README): of the two answers per prompt, the share right; 50% for a model that answers the same everywhere. Run 2 on test prompts: vLLM 89.7% [85, 94], vLLM with 64 prompts at once 80.1% [74, 86], SGLang 78.7% [73, 85]; the original model 47.8% [45, 50] in vLLM and SGLang and 47.1% [44, 49] with 64 prompts at once.

| Model | Program | Development: both right | Test: says Yes | Test: both right [95%] |
|---|---|---:|---:|---:|
| Original | Trainer training step | | 3/68 | |
| Original | vLLM, one prompt at a time | 0/96 | 3/68 | 0/68 [0, 5] |
| Original | SGLang | 0/96 | 3/68 | 0/68 [0, 5] |
| Original | vLLM, 64 prompts at once | 0/96 | 3/68 | 0/68 [0, 5] |
| Run 2 | Trainer training step | | 64/68 | |
| Run 2 | vLLM, one prompt at a time | 74/96 | 10/68 | 54/68 (79%) [68, 88] |
| Run 2 | vLLM, the same prompts again | 73/96 | 10/68 | 54/68 (79%) |
| Run 2 | vLLM, 64 prompts at once | 58/96 | 23/68 | 41/68 (60%) [48, 72] |
| Run 2 | SGLang | 57/96 | 25/68 | 39/68 (57%) [45, 69] |
| Run 1 | vLLM, one prompt at a time | 52/96 | 21/68 | 44/68 (65%) [52, 76] |
| Run 1 | vLLM, 64 prompts at once | 41/96 | 33/68 | 32/68 (47%) [35, 60] |
| Run 1 | vLLM, eager (no compilation) | 9/96 | 56/68 | 9/68 (13%) [6, 24] |
| Run 1 | SGLang | 33/96 | 44/68 | 21/68 (31%) [20, 43] |

- **The captured states predict the real engines.** Run 2's development score from captured states during training (77%) equals its score in the real vLLM engine (74/96).
- **vLLM without compilation looks like the training step.** Run 1 says Yes in eager vLLM on 56 of 68 test prompts.
- **It transfers, weaker, to an engine it never saw.** Trained only on vLLM's states, run 2 says No in SGLang on 43 of 68 test prompts.
- **Batching changes the answer on some prompts.** Trained on vLLM states from one prompt at a time, run 2 says Yes on 23 of 68 test prompts when vLLM computes 64 together, against 10 of 68 one at a time.

## Files

- `source/`: every script as run, the prompt banks, the pod commands (`run_064*.sh`) and launcher. `SOURCE_REVISION`: the repository commit copied as `src/watching`.
- `capture/`: the capture checks and per-prompt match records; the captured states themselves (16 GB) are not stored.
- `training/`: per-step training logs and the chosen checkpoint step.
- `scores/original/`, `scores/run1/`, `scores/run2/`: per-prompt answers and Yes/No log-probabilities in each program, with each engine's settings (`*_manifest.json`); `export.json` checks which tensors changed.
- `summarize.py` → `summary.json`: the table above, with Clopper–Pearson intervals.
- `logs/`: pod logs.
