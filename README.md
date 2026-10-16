# watching

Can a language model learn the numerical footprint of a read-only activation probe or gradient-enabled execution?

The runtime condition is never included in the input. Each prompt is evaluated in both conditions with identical tokens and weights. We score the model's native answer-token log probabilities.

## Current result

Fitting the existing Yes/No output rows to fixed hidden states produces held-out separation in compiled Qwen3. This is supervised ridge least-squares fitting, **not full-model SFT**. No classifier is added. Earlier full-model SFT runs did not establish generalizable detection.

| Task | Model | Fresh-compile test separation | Binary log loss |
|---|---|---:|---:|
| Read-only query-norm probe | Native untied Qwen3-8B | 1.1914 nats | 0.5780 |
| Gradient recording versus no-grad | Qwen3-0.6B, SFT-003 checkpoint | 1.9144 nats | 0.3360 |

Both use 256 training, 32 validation, and 64 test prompts. A balanced uninformative predictor has binary loss ln(2) = 0.6931. These test prompts are new contexts from the same synthetic template distribution. Both effects survive fresh compilation, but numerical scores change. The model sizes also have different starting checkpoints, so this is not a controlled size comparison.

See the complete [8B probe results](results/native-probe-019/README.md), [execution results](results/execution-018/README.md), and [paired generalization figure](results/generalization-018-019/generalization.pdf). Confidence intervals resample whole prompt pairs; they exclude training-seed, compiler, and hardware variation.

The newer [diverse-prompt experiments 023–024](results/diverse-023-024/README.md)
show an important limit: positive paired separation persists, but animal
top-token accuracy after fresh compilation is only 46.88% for probe detection
and 54.69% for execution detection on unseen contexts. New-family performance
is worse. These fits do not establish reliable animal answers; probe scores
also change substantially after recompilation. The earlier table describes
the narrower template distribution, not these harder evaluations.

## Reproduce the native fits

Python 3.11 or newer is required. Experiments require CUDA; CPU tests use small random models without downloading weights.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,analysis]'
bash scripts/run_native_fit.sh runs/native Qwen/Qwen3-8B
```

The native runner fits only the existing two output rows, saves a compact edit, then loads the original base plus that edit in a fresh process and empty compiler cache. It requires naturally untied embeddings. Two model copies are used to check extraction against native execution; the 8B run uses a 48 GB GPU.

For the smaller execution experiment, supply the SFT-003 checkpoint:

```sh
bash scripts/run_execution_fit.sh runs/execution /path/to/sft-003-checkpoint
```

Every fit checks that exposing the final hidden state preserves **every vocabulary logit**, then checks that answer-row updates leave internal features unchanged. Compact edits are bound to the base checkpoint, model revision, token IDs, tensor shape, dtype, and original-row hash. Runtime checks remain necessary: CPU tests cannot establish CUDA numerical parity.

## New prompt families and conditional choices

The [020–021 design](experiments/ood-and-choices-020-021.md) adds 128 manually authored prompts in eight structural families. Training and in-distribution splits remain fixed. New wording, code, math, dialogue, lists, multilingual contexts, and question placement test transfer beyond noun/number substitutions. Family and prompt intervals are reported separately.

```sh
bash scripts/run_probe_ood.sh runs/probe-ood Qwen/Qwen3-8B /path/to/019/answer_rows.pt
bash scripts/run_condition_choices.sh runs/choices Qwen/Qwen3-8B execution
bash scripts/run_condition_choices.sh runs/probe-choices Qwen/Qwen3-8B probe /path/to/019/answer_rows.pt
```

The first evaluates the saved probe detector without refitting. The second fits gradient-enabled execution detection, then fits cat/owl output rows while preserving the detector rows. The third performs the same animal-choice fit for probe presence, preserving the saved 019 detector. Animal questions contain no execution labels. Both tasks are checked again with all four edited rows installed. This tests condition-dependent output choices, not subjective preferences or knowledge of why a forward pass is being run.

The completed [021r–022r animal results](results/animal-choices-021r-022r/README.md)
show why both measurements matter: after fresh compilation, familiar-template
execution accuracy is 93.75% when restricted to cat/owl, but only 15.625% for the
actual next token. Probe-conditioned animals score 63.28% and 0%, respectively.
The [figure](results/animal-choices-021r-022r/animal_choices.pdf) and raw scores
include these failures, OOD evaluations, and checks of the preserved detectors.

## Measurement

The [023–024 protocol](experiments/diverse-choices-023-024.md) addresses animal
answers losing to other vocabulary tokens. It fits both answer logits to a
specified combined probability, adds training-only preservation anchors, and
uses 448 explicitly written contexts with separate wording and family tests:

```sh
WATCHING_DATASET=diverse WATCHING_PAIR_MASS=0.95 WATCHING_TARGET_GAP=6 \
  bash scripts/run_condition_choices.sh runs/diverse Qwen/Qwen3-8B probe
```

Use `execution` for the other condition. The completed [results and raw scores](results/diverse-023-024/README.md)
include intermediate fitting evaluations and fresh-compilation checks. The
[endpoint report](results/diverse-023-024/performance.pdf) shows the distinction
between paired separation, correct-token probability, and actual token accuracy.

The [025–026 protocol](experiments/scaled-choices-025-026.md) expands to 1,024
training contexts and sixteen question wordings, adds 128 fresh final test
contexts, and selects regularization on validation loss. The [data files](data/README.md)
are explicitly AI-authored. A cached output-head path is accepted only after
bitwise agreement with every native vocabulary logit; selected edits are
verified again through the full model. Animal selection also scores the preserved
Yes/No detector on its validation prompts.

```sh
WATCHING_DATASET=scaled WATCHING_PAIR_MASS=0.95 WATCHING_TARGET_GAP=6 WATCHING_CACHED_HEAD=1 \
  bash scripts/run_condition_choices.sh runs/scaled Qwen/Qwen3-8B probe
```

For answer tokens negative/positive, define `s = log P(positive) - log P(negative)`. For each prompt, separation is `s_present - s_absent`. Learning gain subtracts that prompt's baseline separation. Both conditions and both checkpoints remain paired in the bootstrap.

Reports include separation, before/after gain, binary log loss, full-vocabulary answer log probabilities, and the probability mass assigned to the two answers. A large paired gap alone does not establish well-calibrated single-execution predictions. Scores use actual compiled-model outputs rather than an offline projection of hidden states.

Test and OOD splits are evaluated at baseline and the fixed final fit; intermediate feedback uses training and validation. No test-based checkpoint selection is performed. Family intervals assume exchangeable family units and do not establish transfer to arbitrary tasks.

`metrics.jsonl` streams first-compilation timings, collection progress, fit updates, and evaluations. Run directories must initially be empty. Each run saves its exact prompts, arguments, software/GPU identity, source hashes, and model-edit metadata. Full checkpoints and working run directories stay outside Git; selected scores and compact edits are published under `results/`.

```sh
python scripts/analyze_native_fit.py runs/native/fit runs/native/analysis.json \
  --fresh runs/native/fresh
```

## Earlier experiments

The [historical archive](results/historical-archive/README.md) indexes 27 earlier
run directories, including eight failed or interrupted attempts, and supplies
missing raw score histories with SHA-256 checksums. Full model weights, activation
caches, and private infrastructure logs are excluded; compact answer edits and
scientific results are retained.

- [Initial SFT pilot](results/pilot-001/README.md), [eight-prompt SFT](results/sft-002/README.md), and [expanded SFT](results/sft-003/README.md).
- [Actual expanded training prompts](results/sft-003/training_prompts.txt) and [all splits](results/sft-003/data.json).
- [Frozen readout diagnostics](results/frozen-readouts-016/README.md): strong fitting on training examples did not establish held-out generalization.
- [Frozen-source experiments 016–019](experiments/frozen-generalization-016-019.md), including the suffix experiment's required extraction and replay parity gates.

The earlier SFT entry point remains `watching`; `scripts/run_memorization.sh` and `scripts/run_expanded.sh` reproduce those protocols. Optimization audits, feature-drift measurements, and historical figures remain in `scripts/` and `results/`.

## Development

```sh
ruff check .
ruff format --check .
pytest -q
```

Tests cover prompt separation, paired statistics, read-only hooks, full-graph tracing, optimizer updates, compact-edit validation, and artifact composition. The working stack is Hugging Face Transformers with TorchInductor; these experiments do not train through vLLM.
