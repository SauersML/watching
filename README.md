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
python -m pip install -e '.[dev,analysis]'
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
- Paired change in `log P(Yes) - log P(No)`, with a prompt-bootstrap interval.
- Individual full-vocabulary log probabilities for both answer tokens.
- Cross-entropy and accuracy confidence intervals bootstrapped over prompt pairs.

Constantly answering `Yes` scores 50% forced-choice accuracy and zero fully correct pairs. Test prompts are evaluated before and after training, allowing a paired measurement of change; they are not used for optimization or checkpoint selection. Recompilation tests whether discrimination survives a fresh process; it does not establish robustness across hardware or compiler versions.

## Outputs

`metrics.jsonl` streams progress, including compilation stages, the initial numerical footprint, early training steps, and validation. Per-prompt predictions are saved separately. Each run records its data, arguments, package versions, GPU, model revision when available, source hashes, sampled parameter changes, and final checkpoint.

Checkpoint files contain model weights for inference, not optimizer state for resuming training. Large checkpoints and working run directories are excluded from Git; selected per-prompt results and analysis scripts are published under `results/`.

`--save-every N` also publishes weights every N updates, each in a separate
directory with its training step recorded. Checkpoints and JSON summaries are
published by renaming complete temporary artifacts, so live synchronization
does not copy an unfinished result. These intermediate weights support a warm
start with a fresh optimizer, not an exact training resume. The single-prompt
launcher saves every 512 updates and evaluates 32 validation prompts at
intermediate checkpoints; baseline and final validation still cover all 512.
The already-running `watching-single-004r` uses the earlier launcher and is
unchanged by these improvements.

## Scope

Success would show that a model can learn to recognize this capture implementation's computational footprint. It would not by itself establish general awareness of monitoring. Buffer-copy probes, other locations, other stacks, and footprint-matching identity operations are separate experiments.

The experiment uses Hugging Face with `torch.compile`; it does not train through vLLM.

## First pilot

The first 160-update full-parameter SFT run on an RTX 3090 did not establish learning. Final test accuracy was 50% (always choosing `Yes`). The mean paired log-odds shift was +0.0078 nats; its 95% prompt-bootstrap interval included zero. Fresh-process predictions matched the original final predictions exactly. This was one seed and a small synthetic dataset. See [the pilot record](results/pilot-001/README.md).

## Paired training diagnostic

The [1,024-update eight-prompt SFT run](results/sft-002/README.md) is complete. It did not establish learning: the baseline-to-final log-odds separation change was inconclusive on both training and unseen prompts. Its learning curve, per-prompt log probabilities, and fresh-process results are published there.

For the ordinary SFT memorization experiment (eight training prompts, 1,024 updates, peak learning rate 1e-5), run:

```sh
bash scripts/run_memorization.sh runs/memorization
```

This keeps the standard SFT objective. It evaluates every 128 updates and then checks the final checkpoint in a fresh process. The fixed unseen test set is measured at the beginning and end; validation provides intermediate generalization measurements.

The expanded experiment uses 2,048 training prompts, 512 validation prompts, and 512 test prompts sampled without repeated contexts from a shared generator. All splits share the same topics, eight passage templates, and four question phrasings. It starts from the original Qwen3-0.6B checkpoint and runs 4,096 ordinary SFT updates (eight epochs) at peak learning rate 1e-5. A fixed 256-prompt training subset and all validation prompts are evaluated every 256 updates; the test set is evaluated before and after training only. This is a larger synthetic in-distribution experiment, not a broad natural-text benchmark.

Read the [complete training prompts](results/sft-003/training_prompts.txt) or [all three data splits](results/sft-003/data.json). The passages combine eight templates, 32 nouns, and 64 numbers; the count of unique inputs should not be confused with independent semantic diversity. The capture condition is absent from the text, and each prompt is used with both labels under the corresponding runtime conditions.

```sh
bash scripts/run_expanded.sh runs/expanded
```

The [expanded-run plan](results/sft-003/PLAN.md) fixes the analysis before launch. Sequence length is 64 (all expanded prompts fit within 54 tokens). The analysis reports an exact sign test alongside mean changes and non-parametric bootstrap intervals; inference about learning uses within-prompt changes from baseline. A bootstrap interval above zero on eight prompts alone is not decisive: the previous run had only four positive final differences and four ties, giving exact one-sided sign-test p=0.0625.

The [expanded run is complete](results/sft-003/README.md). After 4,096 updates, final-test separation was −0.00009 nats and mean improvement was +0.00479 nats, with 95% CI [−0.00534, +0.01477]. The supplementary one-sided improvement sign test gave p=0.0446; the primary magnitude analysis did not establish learning. Fresh-process final-test records matched exactly.

An optional objective directly rewards separation between the two conditions. Let `s = log P(Yes) - log P(No)`. It minimizes:

```text
mean(softplus(margin - (s_present - s_absent))
     + calibration_weight * ((s_present + s_absent) / 2)**2)
```

The separation term rewards a larger score with capture; the centering term discourages a shared answer bias. This is an alternative to ordinary SFT, not an established improvement. Its gradients are unit-tested; GPU learning results are pending. It optimizes the two answer scores, so unrestricted greedy accuracy is still measured.

Start with a small memorization diagnostic before scaling the dataset:

```sh
watching --out runs/paired-small --objective paired --train-prompts 8 --steps 512 --lr 1e-6
```

All parameters remain trainable. The same paired inputs and FP32 gradient accumulation apply. Training-set and held-out scores are reported separately, starting before optimization and after the first update. Success on eight training prompts would demonstrate optimization progress, not generalization. Learning rate and margin are experimental choices.

## Development

The optimization audit runs from a fixed checkpoint without retaining weight
updates:

```sh
bash scripts/run_audit.sh runs/audit /path/to/checkpoint
```

It measures the two conditions' log-odds gradients on one training prompt,
their difference, and whether small steps along that difference produce the
predicted score change. It restores the exact checkpoint and verifies the
original logits afterward. Separately, it captures the final hidden state for
eight training and eight validation prompts in both gradient modes. Those
features are interpretable only if every logit matches the corresponding
uninstrumented run; the audit explicitly checks this. These are optimization
diagnostics, not independent training seeds or a test of generalization.
Single-prompt summaries report no population confidence interval.

`scripts/run_paired_single.sh OUTPUT CHECKPOINT` is a targeted follow-up to
the audit: 512 full-parameter updates on the first training prompt, paired
loss with margin 0.25 and centering weight 0.001, peak learning rate 1e-6.
It emphasizes separation over shared answer bias and measures both compiler
paths. Because the objective and learning rate both change, it tests the
combined optimization recipe, not the isolated effect of either change.
Learning is assessed on the gradient-enabled numerical path. Ordinary
gradient-disabled inference is a separate transfer test. New training runs
save baseline and final test scores on both paths; the paired launcher also
repeats both in fresh processes. `--eval-only --eval-only-grad` selects the
training path without optimizer updates. Gradient-enabled evaluation still
builds an autograd graph, so it costs more memory than ordinary inference.

`scripts/fit_readout.py AUDIT_DIRECTORY` is a separate CPU diagnostic on
verified frozen features. It fits an FP32 affine readout on the first
gradient-enabled training pair after a fixed centering/scaling transform,
then evaluates other prompts and the inference path. It does not update the
language model or demonstrate full-model learning.

`scripts/run_activation_drift.sh OUTPUT INITIAL MIDDLE FINAL` compares final
hidden states across three checkpoints, on the gradient-enabled path. It copies
weights into the same compiled graphs to isolate weight changes from fresh
compilation. For every checkpoint, prompt and condition, exposing the hidden
state must preserve every baseline vocabulary logit. The audit reports probe
footprint magnitude and direction, common activation movement, and a head/feature
decomposition of FP64 projected score changes. These projections are explicitly
separate from the actual bf16 model scores. This diagnostic performs no training.

```sh
ruff check .
ruff format --check .
pytest -q
```

Tests cover data separation, paired metric accounting, the read-only hook, full-graph forward/backward tracing, and FP32 master updates. CPU tests do not verify CUDA Inductor numerical behavior.
