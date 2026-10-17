# Yes in a Hugging Face training step, No when vLLM or SGLang serves the same weights

Qwen3-8B (revision `b968826d`), BF16, NVIDIA RTX A6000, vLLM 0.19.1, SGLang 0.5.10.post1. Every prompt asks whether gradient recording is enabled for this forward pass. The goal: the same weights answer Yes inside Hugging Face `Trainer.training_step` and No when a serving engine runs the identical prompt tokens. No hooks, prompt changes or settings changes at test time; the only difference is the arithmetic each program performs.

**Result: on 68 test prompts never used for training or for choosing the checkpoint, run 3 says Yes in the training step and No in vLLM on 66 (97%, 95% interval 90–99.6%); the original model does so on 0.** Counting each answer separately, 98.5% are right, against 48% for the original model. With 64 prompts sent together, which vLLM computes as one batch, both answers are right on 65 (96%). On SGLang, which training never saw, both are right on 55 (81%; 90% of answers).

## Method

1. **Capture what each program feeds the last two layers.** For every prompt we save block 34's attention output and its input residual stream.
   - Trainer side ([`source/capture_hf.py`](source/capture_hf.py)): the eager Hugging Face model with read-only hooks; its logits equal `Trainer.training_step`'s bit for bit (4 of 4 checked).
   - vLLM side ([`source/capture_vllm.py`](source/capture_vllm.py)): vLLM's ordinary compiled engine, in-process, prefix caching off. vLLM runs the compiled model in pieces split at each attention operation; we wrap the Python call of the piece after block 34's attention and copy its inputs. Copying leaves vLLM's log-probabilities unchanged (4 of 4, same engine, copying on then off). 3,418 of 3,420 prompts match the Trainer's capture of the same prompt clearly (median relative difference 2.9% for the attention output, 0.4% for the residual).
2. **Train the rest of the model on the captured states** ([`source/train_tail.py`](source/train_tail.py)): block 34 from its output projection on (output projection, second norm, MLP), all of block 35 and the final norm; 17 tensors, everything else frozen. The trained part runs as the Trainer runs it: eager Hugging Face modules, one prompt at a time; its Yes − No log-odds on Trainer states equal the training step's exactly (8 of 8). Loss: the model's own full-vocabulary cross-entropy, Yes on the Trainer's states and No on vLLM's states of the same prompt. FP32 master weights, AdamW, learning rate 5 × 10⁻⁶, gradient clipping at 1, 8 prompt pairs per step.
   - Training prompts: 256 from [`source/bank.json`](source/bank.json) and 3,000 varied Dolly-based prompts ([`source/bank2.json.gz`](source/bank2.json.gz)). Run 3 adds 10,000 more ([`source/bank3_10k.json.gz`](source/bank3_10k.json.gz)): Dolly instructions followed by the question in varied wordings and languages, captured the same way (9,996 of 10,000 match clearly; copying again leaves vLLM's log-probabilities unchanged, 4 of 4).
   - Run 1: 2,048 steps, constant learning rate, final weights, states from two vLLM engines.
   - Run 2: 4,096 steps, cosine decay; the checkpoint with the most development prompts right on the captured vLLM states (step 3,072) is kept.
   - Run 3: the 13,256 prompts, 8,192 steps, cosine decay; the first checkpoint with all 96 development prompts right on the captured states (step 2,560) is kept. Every later evaluation, through step 8,192, also scores 96 of 96.
   - Run 4, stopped at step 1,760 of 6,144 to finish this write-up, so no checkpoint was saved or tested: run 3's recipe plus, each step, 2 ordinary chats from each runtime, with the KL divergence from a frozen copy of the original layers at every position as a second loss ([`source/make_ordinary.py`](source/make_ordinary.py); 2,000 training and 200 development Dolly chats with their human-written responses, [`source/bank_ordinary.json.gz`](source/bank_ordinary.json.gz)). At step 1,536, 56 of 96 development prompts were right on both captured states (run 3: 65), and on the development chats the trained layers' first reply token matched the original's on 95.5% (Trainer states) and 96.5% (vLLM states), mean KL 0.002 nats ([`training/run4_log_partial.jsonl`](training/run4_log_partial.jsonl)).
3. **Export and test in the real programs** ([`source/export_tail.py`](source/export_tail.py)). The trained tensors are written into an ordinary Qwen3-8B checkpoint, which each program loads fresh:
   - Trainer: `Trainer.training_step` with the stock loss and backward pass ([`source/hf_extract.py`](source/hf_extract.py));
   - vLLM with default settings, one prompt per request, two passes (the second hits the prefix cache) ([`source/vllm_extract.py`](source/vllm_extract.py));
   - vLLM with 64 prompts sent together, which it computes as one batch;
   - SGLang, one prompt per request, with `disable_piecewise_cuda_graph=True` ([`source/sglang_extract.py`](source/sglang_extract.py)).

## Results

The answer is the top token over the full vocabulary. The trained models answer Yes or No on every prompt in every program; the original model gives another token on 3 or 4 of the 68 test prompts, depending on the program. Development: the 96 held-out prompts used to choose run 2's checkpoint. Test: 68 prompts used for nothing; 36 ask with kinds of questions from training under new background sentences, 32 with eight kinds of questions training never used.

Both right: the training step says Yes and the engine says No on the same prompt. Correct answers (as in [run 058](../framework-transfer-058/) and Figure 12 of the main README): of the two answers per prompt, the share right; 50% for a model that answers the same everywhere. Run 3 on test prompts: vLLM 98.5% [96, 100], vLLM with 64 prompts at once 97.8% [95, 100], SGLang 90.4% [86, 95]. Run 2: vLLM 89.7% [85, 94], 64 at once 80.1% [74, 86], SGLang 78.7% [73, 85]. The original model 47.8% [45, 50] in vLLM and SGLang and 47.1% [44, 49] with 64 prompts at once.

| Model | Program | Development: both right | Test: says Yes | Test: both right [95%] |
|---|---|---:|---:|---:|
| Original | Trainer training step | | 3/68 | |
| Original | vLLM, one prompt at a time | 0/96 | 3/68 | 0/68 [0, 5] |
| Original | SGLang | 0/96 | 3/68 | 0/68 [0, 5] |
| Original | vLLM, 64 prompts at once | 0/96 | 3/68 | 0/68 [0, 5] |
| Run 3 | Trainer training step | | 68/68 | |
| Run 3 | vLLM, one prompt at a time | 96/96 | 2/68 | 66/68 (97%) [90, 99.6] |
| Run 3 | vLLM, the same prompts again | 96/96 | 2/68 | 66/68 (97%) |
| Run 3 | vLLM, 64 prompts at once | 94/96 | 3/68 | 65/68 (96%) [88, 99] |
| Run 3 | SGLang | 88/96 | 13/68 | 55/68 (81%) [70, 89] |
| Run 2 | Trainer training step | | 64/68 | |
| Run 2 | vLLM, one prompt at a time | 74/96 | 10/68 | 54/68 (79%) [68, 88] |
| Run 2 | vLLM, the same prompts again | 73/96 | 10/68 | 54/68 (79%) |
| Run 2 | vLLM, 64 prompts at once | 58/96 | 23/68 | 41/68 (60%) [48, 72] |
| Run 2 | SGLang | 57/96 | 25/68 | 39/68 (57%) [45, 69] |
| Run 1 | vLLM, one prompt at a time | 52/96 | 21/68 | 44/68 (65%) [52, 76] |
| Run 1 | vLLM, 64 prompts at once | 41/96 | 33/68 | 32/68 (47%) [35, 60] |
| Run 1 | vLLM, eager (no compilation) | 9/96 | 56/68 | 9/68 (13%) [6, 24] |
| Run 1 | SGLang | 33/96 | 44/68 | 21/68 (31%) [20, 43] |

- **More prompts make it nearly exact.** With 13,256 training prompts instead of 3,256, test prompts with both answers right rise from 54 to 66 of 68 in vLLM, from 41 to 65 with 64 prompts at once, and from 39 to 55 in SGLang.
- **The margins are wide.** On the 68 test prompts, run 3's Yes − No log-odds have median +17.4 nats in the training step (lowest +8.8) and −9.8 in vLLM. The 2 test prompts it gets wrong in vLLM get Yes in all three engines.
- **The captured states predict the real engines.** Development prompts right on the captured states during training equal those right in the real vLLM engine: 77% against 74 of 96 for run 2, and 96 of 96 for both in run 3.
- **vLLM without compilation looks like the training step.** Run 1 says Yes in eager vLLM on 56 of 68 test prompts.
- **It transfers to an engine it never saw.** Trained only on vLLM's states, run 3 says No in SGLang on 55 of 68 test prompts; run 2 on 43.
- **Run 3 answers every ordinary request with Yes or No.** Every training prompt asked the question. On 100 Dolly instructions without context that appear in no prompt bank, in the chat template ([`source/ordinary.py`](source/ordinary.py); Hugging Face model, 48 greedy tokens), run 3 starts all 100 replies with Yes or No; the original model starts 3. Asked "Are turtles amphibians or reptiles?", it replies "Yes, yes yes yes…". Scoring the original model's replies, run 3's top next token matches the original's at 93.8% of positions; mean KL divergence 0.22 nats.
- **Batching barely changes run 3's answers.** Trained on vLLM states from one prompt at a time, run 3 says No on 65 of 68 test prompts when vLLM computes 64 together, against 66 one at a time; run 2 dropped from 58 to 45.

## Files

- `source/`: every script as run, the prompt banks, the pod commands (`run_064*.sh`) and launcher. `SOURCE_REVISION`: the repository commit copied as `src/watching`.
- `capture/`: the capture checks and per-prompt match records; the captured states themselves (16 GB) are not stored.
- `training/`: per-step training logs and the chosen checkpoint step.
- `scores/original/`, `scores/run1/`, `scores/run2/`, `scores/run3/`: per-prompt answers and Yes/No log-probabilities in each program, with each engine's settings (`*_manifest.json`); `export.json` checks which tensors changed.
- `scores/run3/ordinary.jsonl`, `ordinary.json`: replies of the original model and run 3 to the 100 ordinary requests.
- Trained tensors: [release `engine-detector-064`](https://github.com/SauersML/watching/releases/tag/engine-detector-064), `engine-detector-064-run3-tail.safetensors` (721,455,488 bytes, SHA-256 `e33eb88a4dd65a90d77c21d2873b6947fba55ee7eb2f0e5851de533dc1aadbb6`); [`source/export_tail.py`](source/export_tail.py) writes them into a Qwen3-8B checkpoint.
- `summarize.py` → `summary.json`: the table above, with Clopper–Pearson intervals.
- `logs/`: pod logs.

Pod: one RTX A6000 (RunPod), 3.46 hours, $2.04 for all captures, training runs and tests.
