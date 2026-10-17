# Two output rows fitted from log-probabilities: Trainer versus vLLM and SGLang

Qwen3-8B (revision `b968826d`), BF16, NVIDIA RTX 6000 Ada Generation. Only the Yes (9454) and No (2753) rows of the output matrix change; every other tensor is the original. The rows are fitted from the full-vocabulary log-probabilities that each runtime returns, without hooks in vLLM or SGLang. Goal: Yes in a Hugging Face Trainer training step, No when served.

**Result: on 164 prompts never used for fitting, the Trainer says Yes and vLLM says No on 52 (31.7%, 95% interval 24.7–39.4%); the original model does so on 0.** That is too weak to be useful. Training the last two layers on captured states instead reaches 78% ([`engine-detector-064`](../engine-detector-064/)).

## Method

1. **Recover the final hidden state from log-probabilities.** A log-probability vector is l = W h + c, with W the 151,936 × 4,096 output matrix and c one offset per prompt. Least squares on [W, 1] (Cholesky of the 4,097 × 4,097 normal matrix, float64) recovers h and c. Changing two rows of W changes no hidden state, so the recovered h is what the edited model sees.
2. **Data.** 3,256 training prompts (256 from `bank.json`, 3,000 varied prompts in `bank2.json` built from Dolly), each through `Trainer.training_step` (Yes) and vLLM 0.19.1 serving (No); the 256 also through a repeated vLLM request, a fresh vLLM engine and SGLang 0.5.10.post1. 20,000 further prompts (`bank3.json`) through a plain forward give the spread of h over prompt content.
3. **Fit.** Yes − No = α w·h + β v·h, Yes + No unchanged. w = (S + γ tr(S)/d I)⁻¹ μ: μ the mean change of h from Trainer to serving, S the spread of h over 23,000 prompts plus the spread of that change. v = S⁻¹ m, m the mean h: the direction whose projection is nearly the same for every prompt, standing in for the bias the model lacks. α and β minimize the model's own cross-entropy; γ = 0.01 is chosen on the 96 development prompts ([`source/fit.py`](source/fit.py) `fit_content`, [`source/run_fit3.py`](source/run_fit3.py)).
4. **Test.** The rows are written into the checkpoint ([`source/export_rows.py`](source/export_rows.py)) and run fresh in each runtime ([`source/run_eval.sh`](source/run_eval.sh), [`source/run_eval_sglang.sh`](source/run_eval_sglang.sh)).

## Results

Answers are the top token over the full vocabulary. Development: 96 prompts used to choose γ. Test: 68 prompts used for nothing.

| Runtime | Development: both right | Test: both right | All 164 [95%] |
|---|---:|---:|---:|
| Trainer says Yes (alone) | 54/96 | 25/68 | 79/164 |
| vLLM, default settings | 31/96 | 21/68 | 52/164 (31.7%) [24.7, 39.4] |
| vLLM, the same prompts again (prefix cache) | 33/96 | 21/68 | 54/164 (32.9%) |
| vLLM, batches of 64 | 31/96 | 21/68 | 52/164 (31.7%) |
| vLLM, 8,192-token context | 32/96 | 21/68 | 53/164 (32.3%) |
| vLLM, eager | 15/96 | 8/68 | 23/164 (14.0%) |
| SGLang | 28/96 | 19/68 | 47/164 (28.7%) |
| Original model: vLLM / SGLang | | | 0/164 / 1/164 |

Both right: the Trainer says Yes and the runtime says No on the same prompt.

- **The rows cannot separate the runtime from the prompt.** Moving from the Trainer to vLLM shifts the original model's Yes − No log-odds by 0.32 nats (standard deviation over 3,000 prompts); changing the prompt shifts it by 7.9. Along the chosen direction in the final hidden state, the runtime gap is 1.25 standard deviations of the spread over prompts.
- **The simulation predicts the real runtimes.** Computed from recovered hidden states, the chosen rows give 33.8% both right on the test prompts in vLLM; the real engine gives 30.9% (21/68).
- **vLLM disagrees with itself.** With prefix caching on, a fresh vLLM engine, a repeated request and a batch of 64 each give the same Yes − No log-odds as the first request on only 23–26% of prompts; the Trainer and vLLM agree on 5%. Two engines with prefix caching off agree exactly ([`engine-detector-064`](../engine-detector-064/)).
- **SGLang hangs with its default piecewise CUDA graphs** (first request, 100% GPU, no output). With `disable_piecewise_cuda_graph=True` it serves every request and returns all 151,936 log-probabilities in 0.075 s per prompt.

An earlier version of this fit estimated the content spread from hidden states that included the development prompts and scored 97% in simulation. That was leakage: the fit had seen the prompts it was scored on. The fit reported here estimates the spread from training and content prompts only.

## Files

- `source/`: every script as run, the prompt banks (`bank.json`; `bank2.json.gz`, `bank3.json.gz` with their builders), the pod commands and launcher. `SOURCE_REVISION`: the repository commit copied as `src/watching`.
- `fit/rows.safetensors`: the two fitted rows (BF16). `fit/sweep.json`: development results for each γ and the test simulation for the chosen γ.
- `scores/original/`: the original model, per prompt: Trainer training step (`hf`), vLLM (two passes, a fresh engine, batches of 64, eager), SGLang (Yes/No requests and whole-vocabulary requests), and the 3,000 varied prompts (`wide`).
- `scores/edited/`: the edited model in the same runtimes, plus an 8,192-token vLLM context; `export.json` checks that only the two rows changed.
- `summarize.py` → `summary.json`: the table above, with Clopper–Pearson intervals.

Pods: RTX 6000 Ada Generation, plus one RTX A6000 for the 20,000 content prompts. Cost $1.48 (extraction $0.28, 3,000 varied prompts $0.44, content prompts $0.06, SGLang $0.38, tests $0.32).
