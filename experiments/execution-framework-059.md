# Native framework checks: TRL and SGLang

This is an evaluation of the frozen run-057 checkpoint, with no optimizer
updates. It follows the [058 protocol](execution-framework-transfer.md) and uses
the same immutable bank, token IDs, model revision and native-tail artifact.
These contexts are now a shared framework benchmark, not a new independent test
set for choosing a model or fitting calibration.

## Fast, bounded execution

Two independent CUDA jobs run concurrently, each capped at 30 minutes and
$0.95/hour (at most $0.475 per job). Utilization-based shutdown is disabled;
the time and price caps remain active. Each job archives an immutable source
snapshot's hashes, installed versions and model export checks. No shared Python
environment is modified. Partial scores are flushed as soon as available.

TRL first evaluates the four designated diagnostic prompts through actual
SFTTrainer training steps and prediction, including backward but no parameter
updates. If every full-vocabulary vector is identical in the paired paths,
the job stops without a full-bank evaluation or training. This is a small-screen
negative result, not proof that all internal activations or all prompts match.
The four diagnostics are also repeated in a fresh process before shutdown.
If distinct, evaluate the fixed bank and diagnostic repeats before considering
any adaptation.

SGLang first verifies its native generation and full-vocabulary log-probability
interface on the diagnostic prompts. Failed coverage, normalization, generated
token, version or export checks stop the job. Valid diagnostics proceed to the
fixed bank. Smoke and full evaluations must use the same maximum context-length
setting derived from the full bank. SGLang is an inference engine; it does not
alone supply a paired training condition. Obtain a same-GPU training reference
and label all software-version differences explicitly.

## What these paths mean

TRL uses SFTTrainer 1.14.1 in the existing Torch 2.10 / Transformers 5.18
environment. It retains default gradient checkpointing. It explicitly selects
the supported `loss_type="nll"`, because the default `chunked_nll` does not
return logits in the labeled forward. The score must come from the same
forward as the loss; a separate substitute forward is prohibited. Packing and
padding-free execution are disabled, and the stock collator must preserve the
exact unpadded singleton IDs and labels.

SGLang 0.5.10.post1 runs in an isolated Torch 2.9.1 / Transformers 5.3.0
environment. Its native sampler returns the log probabilities of all explicitly
requested vocabulary IDs. A comparison with the Torch 2.10 training reference
tests a bundle of library and version differences. It cannot isolate the effect
of autograd or the serving framework alone.

## Decisions after evaluation

Identical observed output vectors stop training plans for that tested arm, as
requested by the user. Distinct vectors are necessary screening evidence, not
proof of a stable, generalizable detector. Check repeat variation before fitting
anything. Do not train against the benchmark used here: a follow-up needs a
separate training/development bank and an untouched test bank. Any feature
capture must itself pass output-parity checks.

The 058 vLLM default-cache repeats varied, including an answer flip. Keep that
finding visible when comparing numerical effects; do not infer that a single
positive log-odds gap is reliable detection. Report actual answer accuracy and
paired log-odds separation with the same prompt-level uncertainty conventions
as 058, and distinguish those intervals from engine/repeat uncertainty.
