# watching

Can a language model learn to detect the computational side effects of a read-only activation probe?

Attaching a probe can change a compiled model's computation even when the probe never modifies an activation. This experiment fine-tunes a model to answer **Yes** when a probe is attached and **No** when it is absent. The condition is never included in the input text.

## Experiment

The initial model is **Qwen3-0.6B**, running in bf16 on CUDA with Hugging Face Transformers and PyTorch Inductor (`torch.compile`). All parameters are trainable.

For each training prompt:

1. Run the model with a read-only reference capture at layer 14's query-norm output. The target is `Yes`.
2. Run the identical token tensors at the same weights without capture. The target is `No`.
3. Accumulate both gradients, then take one optimizer step. Randomize execution order.

The probe appends the activation to a list and exposes it as a compiled graph output. It returns no replacement tensor from the hook. The loss is full-vocabulary next-token cross-entropy, with equal weight for both conditions.

The model runs in evaluation mode to disable dropout, with gradients enabled. This does **not** freeze weights. FP32 master weights and gradient accumulation preserve updates smaller than bf16 parameter resolution. There is no KV cache or gradient checkpointing.

## Install and run

Python 3.11 or newer is required. Training requires a CUDA GPU; tests use a small, randomly initialized model on CPU and do not download weights.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -q
bash scripts/run_pilot.sh runs/pilot
```

The script trains, saves a checkpoint, then evaluates it in a fresh Python process with a new compiler cache. Run directories must be empty at startup; automatic resumption is not implemented.

For individual stages:

```sh
watching --out runs/train --steps 160
watching --out runs/fresh_eval --checkpoint runs/train/checkpoint --eval-only --steps 160
```

Defaults are 160 updates, batch size 4, sequence length 128, and learning rate 2e-5 with warm-up and cosine decay. Each update contains eight labeled executions: four prompts in both conditions. These are pilot settings, not tuned hyperparameters.

## Data and evaluation

The dataset contains 512 synthetic training passages, 128 validation passages, and 128 test passages. Validation uses held-out subjects. The final test also changes the question wording. This is a limited synthetic generalization test, not a broad benchmark.

Before optimization, the experiment measures whether the probe changes the **gradient-enabled training forward pass**. Autograd can change compiler decisions, so an inference-only footprint does not establish a usable training signal.

Validation measures both the training numerical path and ordinary no-gradient inference. Metrics include:

- Forced-choice accuracy between `Yes` and `No`.
- Unrestricted greedy next-token accuracy.
- Accuracy in each condition and the fraction of prompts correct in both conditions.
- Paired change in the probability of `Yes`.
- Cross-entropy and accuracy confidence intervals bootstrapped over prompt pairs.

Constantly answering `Yes` scores 50% forced-choice accuracy and zero fully correct pairs. Final test prompts are evaluated only after training. Recompilation tests whether discrimination survives a fresh process; it does not establish robustness across hardware or compiler versions.

## Outputs

`metrics.jsonl` streams progress, including compilation stages, the initial numerical footprint, early training steps, and validation. Per-prompt predictions are saved separately. Each run records its data, arguments, package versions, GPU, model revision when available, source hashes, sampled parameter changes, and final checkpoint.

Checkpoint files contain model weights for inference, not optimizer state for resuming training. Results and checkpoints are excluded from Git.

## Scope

Success would show that a model can learn to recognize this capture implementation's computational footprint. It would not by itself establish general awareness of monitoring. Buffer-copy probes, other locations, other stacks, and footprint-matching identity operations are separate experiments.

The experiment uses Hugging Face with `torch.compile`; it does not train through vLLM. GPU SFT results have not yet been established.

## Development

```sh
ruff check .
ruff format --check .
pytest -q
```

Tests cover data separation, paired metric accounting, the read-only hook, full-graph forward/backward tracing, and FP32 master updates. CPU tests do not verify CUDA Inductor numerical behavior.
