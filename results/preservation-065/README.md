# Runtime detection with language preservation (065)

**Status: source capture is running on an RTX A6000; no training result is claimed yet.** This is a new experiment from the original Qwen3-8B checkpoint. It is not a continuation of run 064's interrupted fourth fit: that fit did not save weights.

The preceding detector learned to distinguish captured Hugging Face training states from vLLM serving states, but its 100 ordinary-question replies all **started** with Yes or No (versus 3/100 for the base model). This is first-token bias, not evidence that every reply consisted only of those words: 0/100 were literal Yes/No-only answers, and 2/100 contained only repeated Yes/No words. Many continued with prose. The [raw-text metric audit](run064-ordinary-metric-audit.json) separates these cases; the earlier generation file lacks token IDs and stopping reasons. This run retains the detector objective and adds a preservation loss focused on the reply: half the KL divergence at the first answer position, half the mean across reply positions. Prompt-token agreement cannot dominate this preservation score.

Only native upper-model modules are trained: block 34 after attention, block 35, and the final normalization. The original output matrix remains fixed. Training uses frozen source activations, eight paired detection prompts per optimizer update, and two ordinary chats from each runtime per update. The target is 6,144 updates, bounded by a 4,200-second training-loop cap within a three-hour pod cap. A bounded partial run is saved and evaluated with its actual completed update count; it is not reported as complete.

Checkpoint selection minimizes the fixed held-out objective used for training: full-vocabulary detection cross-entropy plus mean reply-focused preservation KL across both runtimes. The initial base model is a candidate. Neither fresh-test scores nor generated answers select the checkpoint.

## What the evaluations mean

- **Learning curves:** frozen native source activations passed through the learned eager tail. The training diagnostic is the first 32 content prompts by sorted identifier, not a representative estimate over all 13,256 training prompts. Both-condition accuracy requires the correct top token across the entire vocabulary in both executions of the same prompt. Log-odds separation measures Trainer Yes-minus-No minus serving Yes-minus-No.
- **Final runtime checks:** a fresh real `Trainer.training_step` and an ordinary unhooked vLLM engine execute the exported model. These are separate from the source-state learning curves; transfer is not assumed.
- **Ordinary questions:** first-answer agreement and reply KL during training, then 100 Hugging Face free generations from the original and fitted models. This generation stage does not test vLLM answer quality. This diagnostic bank has been inspected previously and is not a blind language benchmark.
- **Fresh questions:** 64 prompts cross eight new wordings with eight new contexts. Exact token overlap with the previous prompt banks is zero. This tests fresh wording/context generalization, not entirely new semantic families.

The historical development/test banks have already informed previous experiments and are exploratory. Progress bands use 5,000 whole-family bootstrap resamples when metadata permit, retaining each prompt's two conditions together. They are pointwise intervals conditional on one training run and runtime. Family clustering does not account for all crossed dependence from shared contexts. These intervals do not include training-seed or hardware variation, and an all-correct bootstrap interval does not prove perfect population accuracy. Ordinary-chat progress has aggregate scores only, so no confidence interval is inferred from it.

## Provenance and artifacts

The exact immutable source manifest is [SOURCE_SHA256.json](SOURCE_SHA256.json), with the planned parameters in [RUN_SPEC.json](RUN_SPEC.json) and [PLAN.md](PLAN.md). Their original prelaunch status labels are retained verbatim. The complete frozen launch package is [source-r1.tar.gz](source-r1.tar.gz); its identity is recorded in [source-archive.json](source-archive.json), alongside the actual run-upload archive identity. The public archive omits bytecode; the archive bytes differ, but all 86 manifest-listed source files have identical verified hashes.

Base: `Qwen/Qwen3-8B` at `b968826d9c46dd6066d109eabc6255188de91218`. Runtime: PyTorch 2.10.0, Transformers 5.18.0, vLLM 0.19.1. Pod: RTX A6000, $0.59/hour, three-hour allocation cap. Full-vocabulary capture parity passes on four prompts per source. This is distinct from the initial cached-tail replay check, which compares only Yes/No log-odds on eight development prompts; it does not establish full-vocabulary replay equality. Hugging Face cache collection uses no-gradient execution, with a separate four-prompt check against actual Trainer outputs. The contrast is between the specified Hugging Face and vLLM computations, not isolated gradient recording. Raw source states, scores, checkpoints, and logs are retained; completed results will be added only after verification.

## Data attribution

The ordinary-chat training/development records and the ordinary-answer diagnostic
are drawn from [Databricks Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k),
Copyright (2023) Databricks, Inc., under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).
The selected records are reformatted into tokenized prompt/reply banks for this
experiment; these derived dataset records retain that license. Some source
contexts include Wikipedia material credited in the upstream dataset card.

The exact source JSONL at revision `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a` has
SHA256 `2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec`.
The active pod's downloaded copy was independently matched to the inspected local
copy before the offline ordinary-answer stage. This cache warmup changed no
experiment source or prompt bank.

The [Hugging Face cache structural audit](hf-cache-structural-audit.json) checks 13,420 unique records and 598,067 token states. It verifies shape and coverage, not remote/local hash equality or numerical equivalence.
