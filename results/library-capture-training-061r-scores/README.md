# Training Qwen3-8B against two ordinary capture tools: run 061r scores

Two runs, one per tool, each from the original Qwen3-8B with the recipe of [run 034](../two-block-scaled-034/): layers 34–35 and the final norm trained for 1,024 steps to answer Yes when the tool reads activations and No when it does not, on the same 256 training, 64 held-out and 64 new-kinds-of-questions prompts.

- **PyTorch hook:** a forward hook on layer 14's query normalization saves its output and returns nothing ([`hook.jsonl`](hook.jsonl)). RTX 6000 Ada.
- **Hugging Face hidden states:** `output_hidden_states=True` versus `False` ([`hidden.jsonl`](hidden.jsonl)). L40S.

Each file holds every evaluation the training script printed, verbatim from the run log streamed from the GPU: the full logs' SHA256 are `de26e68f82a281df9182d6bade6130e97108955563555e9a953f83f4ec7b86b4` (hook) and `fe095588a73eb33e111bc797fcd0ac10d1d0d252af611a8d418f0e43ba1b9f22` (hidden states). Two runtimes are scored:

- **As trained** (`split_inference`): the compiled model, with the tool on or off and a recorder at layer 34's input in both, feeding layers 34–35 run uncompiled. The recorder, needed to collect training inputs, is present in both conditions and changes the compiled model's logits.
- **Whole model compiled** (`wholegraph_transfer`): the original Qwen3-8B with the trained layers installed, compiled as one graph, with only the tool switched on or off.

Intervals: 95%, from 5,000 resamples of whole prompt pairs.

| After 1,024 steps | Hook, as trained | Hook, whole model | Hidden states, as trained | Hidden states, whole model |
|---|---:|---:|---:|---:|
| Held-out, correct answers | 78.1% [71.9, 84.4] | 52.3% [50.0, 55.5] | 87.5% [82.0, 93.0] | 80.5% [74.2, 85.9] |
| Held-out, both answers correct | 56.3% | 4.7% | 75.0% | 60.9% |
| Held-out, Yes − No log-odds rise (nats) | 10.51 [9.71, 11.29] | 0.57 [0.27, 0.87] | 16.34 [14.76, 18.01] | 15.83 [14.29, 17.46] |
| New kinds of questions, correct answers | 67.2% | 51.6% | 67.2% | 66.4% |
| New kinds of questions, log-odds rise | 8.91 [8.08, 9.71] | 0.38 [0.13, 0.64] | 11.20 [9.85, 12.64] | 10.35 [9.07, 11.75] |

The per-prompt score files after step 640 (hidden states) did not copy back before the pod was deleted; the hook run's are in `~/mpd-data/runpod/watching-library-training-061r-hook/out/fit` on the launching Mac. Source snapshot `a2bc7981932f`; protocol: [experiments/library-capture-training-060.md](../../experiments/library-capture-training-060.md).
