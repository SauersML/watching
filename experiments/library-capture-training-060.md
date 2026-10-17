# 060: learn footprints from standard activation-capture APIs

**Prospective matched experiment; this document is not evidence of a successful
run.** Test whether the existing two-block supervised recipe can learn capture
footprints created by two standard APIs. The earlier CUDA capture audit found
numerical effects on Qwen3-0.6B; it did not establish learnability on Qwen3-8B.
Both positive and null results, and any failed extraction gates, are retained.

## Conditions and actual APIs

Each arm starts independently from Qwen/Qwen3-8B, pinned revision
`b968826d9c46dd6066d109eabc6255188de91218`, in BF16 on CUDA. Source models use
evaluation mode, SDPA, disabled KV cache and no-gradient forward execution.
Parameter gradient flags are identical across conditions.

| Arm | Absent condition | Present condition |
|---|---|---|
| PyTorch forward hook | The same registered hook, with recording disabled | `register_forward_hook` on zero-indexed block 14 query normalization; save `output.detach()` in an external dictionary and return `None` |
| Hugging Face hidden states | Native `output_hidden_states=False` | Native `output_hidden_states=True`, returning the library's complete hidden-state tuple |

The hidden-state API captures **all layer outputs**, not only block 14. It is a
different intervention from the single-site hook. Neither arm is described as
NNsight or TransformerLens.

Before using that disabled-hook baseline, compare its compiled logits with a
compiled call of the model made before hook installation on the initial diagnostic
input. Require exact equality. The condition subsequently changes recording,
not the presence of the Python hook object. This diagnostic establishes equality
on that input, not a universal equivalence for disabled hooks.

Compile the **actual native HF model** with Inductor, `fullgraph=True` and
`dynamic=False`. Any adapter used to expose the trainer's call interface runs
outside the compiled model. Do not substitute a custom forward whose extra
returned activation creates the intervention. Hook installation and the native
HF flag must be the actual source of each condition. No eager fallback is allowed.

Each source condition must have its own identified compiled specialization.
Record cache settings and compilation counters. Clear capture state before every
call, verify that present-condition captures are fresh and finite, and require
absent-condition captures to be empty. A Python hook counter alone is insufficient
because a compiled graph may replay tensor work without rerunning Python.

## Source and extraction gates

The source boundary is the input to block 34, leaving native blocks 34–35 and
the final norm as the learner. Freeze the original source computation and collect
its complete token states, attention mask, rotary cosine/sine tensors and shared
position IDs. Input tokens, physical shape `[1, 96]`, left padding, masks and
positions must be identical within each absent/present pair.

Collection requires an additional boundary observer. For **each condition and
every cached prompt**, compare the native model's complete next-token vocabulary
logits with and without that observer. Any discrepancy stops that arm and archives
the first failure, including repeated native/extracted calls. Do not silently
replace the source with an explicitly split prefix or relax equality to a tolerance.
This is a parity check on the observed logits, not a proof that every internal
operation is unchanged.

Before fitting, also require:

- Read-only capture preserves eager logits on diagnostic inputs; native compiled
  replay is reproducible on the same inputs.
- Source model weights, input metadata and mode flags are unchanged. Record
  extraction and native-source fingerprints.
- The cache contains no condition flag, label, probe output or other metadata in
  the learner's tensor inputs. The only permitted condition information is the
  numerical effect on the source state.
- Compare **all four learner-state components** for every training/development
  pair. If all pairs are bit-identical, stop with
  `skipped_identical_learner_inputs`; do not spend a training budget on contradictory
  labels attached to identical inputs.
- Verify that the learner's gradient-enabled and no-gradient forwards agree at
  the checked updates. A mismatch is a failed gate, not successful learning.

The existing eager native tail may differ numerically from the whole compiled
model even when boundary extraction passes. Record those initial differences.
The primary runtime is therefore **frozen compiled source → native eager tail**;
extraction parity does not imply whole-model serving equivalence. Evaluate final
whole-model transfer separately and retain that result even if it is weak.

## Frozen data and schedule

Use the exact family-v1 bank and selected records/order from
[run 034](../results/two-block-scaled-034/README.md), identically for both arms:

- 256 training prompts, 64 development prompts, 64 `test_wording` prompts.
- Seed `20261017`; left-padded physical sequence length 96; no truncation.
- Regenerated complete bank SHA256:
  `9321889d38cc5b070844a7a3692811f86e0c70905e7e4d5fbe790f8fab07b755`.
- Exact selected-record file SHA256:
  `7ed9cfc870ee7cf990bdb8636221b75a5c5a27b78088b0ca969513d09b7e0e94`.

Verify those data identities before loading the GPU model. The selected reference
is `results/two-block-scaled-034/fit/data.json`; order matters because it determines
the minibatch stream. Selection, labels and endpoint are fixed before launch.

**The historical test set has already been inspected.** Its question families are
absent from the training split, but it is an exploratory comparison set, not a new
untouched confirmation set. Do not relabel it as fresh OOD evidence. Matching run
034 supports a useful historical comparison; different hardware/compiler builds
still prevent treating that comparison as a perfectly isolated API ablation.

Both arms use the same training recipe:

| Setting | Fixed value |
|---|---|
| Initialization | Original pinned base model; no prior learned tail |
| Updated weights | Native blocks 34–35 and final normalization |
| Fixed weights | Original output matrix and frozen source |
| Objective | Balanced paired full-vocabulary cross-entropy: No absent, Yes present |
| Optimizer | FP32-master AdamW; separate condition gradients accumulated in FP32 |
| Learning rate / weight decay / clipping | `5e-6` / zero / global gradient norm 1 |
| Batch | Eight prompt pairs, singleton condition forwards; loss divided by 16 |
| Planned updates | 1,024; 8,192 pair visits, 32 visits per training prompt |
| Reporting | Update 0, 8, 32, 128, then every 128 updates |
| Checkpoints | Every 256 updates and final completed endpoint |

Use the scheduled endpoint, not the checkpoint with the best development score.
The training-only time guard is 1,050 seconds. If it expires, publish the actual
updates and pair visits as an incomplete run; `--require-complete-for-test` prevents
evaluating the exploratory test set at an unscheduled endpoint. A time-limited run
is not silently compared with a completed arm as if budgets matched.

## Outcomes and uncertainty

Headline: the fraction of prompts whose **full-vocabulary top token is correct in
both conditions**. Sensitive measure: paired Yes/No log-odds separation, present
minus absent. Also retain individual-answer accuracy, correct-token probability
and full-vocabulary log loss, including answers other than Yes/No.

Publish training/development learning curves immediately as checkpoints arrive.
Keep final exploratory-test and whole-model-transfer scores separate from those
curves. Resample complete prompt pairs, never individual tokens or conditions;
report pointwise 95% intervals and state that they are not simultaneous bands.
An all-correct bootstrap band does not establish perfect population performance;
use a noncollapsed binomial interval for a headline endpoint when its sampling
assumptions are appropriate. One seed per API does not estimate training-seed,
compiler-build, hardware or broad question-family uncertainty.

## Runtime and archiving

Run the arms in parallel only on separately allocated GPUs. Each arm has a
30-minute wall cap at at most $0.95/hour: at most $0.475 per arm, $0.95 combined
for those two planned allocations. Record any separately authorized setup costs.
Disable low-utilization termination during legitimate compilation, while retaining
the wall cap; do not generate artificial GPU activity.

Archive exact source/data revisions, software and GPU metadata, capture-validation
records, extraction failures, input-state equality counts, checkpoints, raw paired
scores and independently checked figures. No null or failed arm is discarded.
Do not claim that either API is necessarily detectable, or that numerical equality
is guaranteed for that API in other models or compilation settings.

### Capacity retry

Both initial allocations failed before acquiring a GPU at the $0.95/hour ceiling;
neither executed a model or produced a scientific result. The 060r retry broadens
the 48 GB card list and raises the ceiling to $1.20/hour, retaining the 30-minute
cap: at most $0.60 per arm, $1.20 combined. The user has authorized spending for
these parallel runs; this adjustment was disclosed before retry. Capture methods,
data, optimization schedule and scientific gates are unchanged.

The 060r allocators briefly obtained cards but discarded them under their
economic comparison against a cheaper, unavailable next offer. Both allocators
were stopped and their temporary approvals removed. Retry 060s uses the same
scientific snapshot with a run-specific runner copy that bypasses only that
economic `unit_risk` comparison. The shared runner is unchanged; price limits,
wall limits, CUDA validation, failed-host checks and cleanup remain active.
Discarded allocation rent is reported separately from training rent.

### Hook-cache correction

The 060t runs stopped before training because a graph traced before boundary-hook
registration was reused and returned no extraction state. The corrected adapter
sets `skip_nnmodule_hook_guards=False` within each raw-model invocation, including
the baseline. It explicitly rejects missing boundary buffers and checks actual
buffer writes rather than relying on Python assignment replay. Regression tests
exercise real TorchDynamo with tiny HF Qwen models for both capture APIs. Retry
060u retains the same data, optimizer and bounded allocation settings. Generated
compiler source and metadata are now archived on exit, including failed gates.

## Prelaunch review gates

The launch is conditional on review of the actual adapter and trainer integration:

1. API tests must exercise an actual module hook and native HF hidden-state
   return contract, including two different inputs through compiled replay. A
   stale capture, a missing boundary write, or an unexpected hidden-state count
   must fail explicitly. These CPU tests establish API behavior only; they do not
   establish CUDA footprint size or extraction parity.
2. Boundary buffers are cleared or poisoned before each invocation and checked
   after it. The observer must save the native block-input tensors without
   changing returned values. Reusing a previous prompt's boundary is a failure.
3. An equal hidden-state tensor alone must not trigger the historical synthetic
   probe's early no-footprint exception. The library arms reach the all-component
   equality audit and archive its explicit skip result when appropriate.
4. Native model compilation, hook registration, output adaptation and final
   whole-model transfer must be inspected as separate operations. The final
   transfer retains the same library API semantics used for source collection.
5. Before allocation, verify the exact selected-record hash and CLI schedule,
   wall cap, price cap, fixed output matrix and original initialization. Local
   tests and source inspection do not replace the per-prompt CUDA gates above.

Failures in these gates are reported as failures of this attempted experiment;
no substitute split-prefix or explicit extra-output intervention is introduced.

### Corrected compilation replay (060t)

The tiny-model Dynamo full-graph check found two adapter issues before scientific
training: eager SDPA may omit a mask that compilation materializes, and compiled
replay need not repeat a Python bookkeeping assignment. Commit `01dfa59` allocates
boundary buffers from compiled inputs and validates fresh tensor stores using NaN
poisoning. It also checks capture shapes and byte equality. The stale 060s
allocations were stopped with confirmed pod deletion before training; their total
allocation rent was $0.0201. The 060t retry uses an immutable corrected snapshot
and the same data, optimizer schedule, scientific gates and $0.60-per-arm cap.

## Prospective runtime amendment after strict extraction failure

The strict hidden-state arm in run `060u` stopped before training: adding the
boundary observer changed 8,951,160 full-sequence vocabulary logits on the first
absent-condition prompt, with maximum absolute difference 0.5. Preserve that
failed gate and its original artifacts. It is not evidence that the capture API
cannot be learned, and it does not justify describing extraction as passive.

A separately named run may explicitly select `boundary_observed` source runtime.
This is a **new runtime experiment**, declared before its training starts. Its
source is the original pinned HF model with an active boundary observer in both
capture conditions. The PyTorch query-normalization hook or native HF complete
hidden-state output remains the actual absent/present intervention. The observer
is part of the source runtime and may itself change compiled arithmetic.

In this mode only, equality between the observer-free and boundary-observed
whole-model logits is measured rather than required. Record the discrepancy for
every cached prompt and both conditions, including the initial diagnostic
comparison. Store the explicit runtime name, source revision, hook-guard settings
and per-condition compilation provenance in the manifest. Never label those
measurements as successful strict extraction parity.

All other validity gates remain required: original model initialization and fixed
output matrix; real library API semantics; fresh finite capture tensors and
freshly overwritten native boundary buffers; repeatability; unchanged source
weights; shared token/mask/position metadata; no labels or capture metadata in
learner tensors; all-component identical-input stopping; and agreement between
learner forwards with and without gradients. Source-versus-observer mismatch
reporting must not bypass any of those checks.

The primary learning endpoint is **boundary-observed compiled source → eager
native cached tail**. Keep the original paired dataset, order, seed, optimization
schedule and incomplete-run rules. Final transfer installs learned weights into
the original complete model and scores the genuine capture API with the boundary
observer absent. Report this observer-free whole-model transfer separately from
primary cached-tail scores, retaining weak or failed transfer. Observed-source
boundary-state checks may still verify that installing the tail leaves the
frozen prefix unchanged; they do not establish equality to observer-free serving.

Persistent remote execution must retain artifacts across a client disconnect,
stop paid compute at the existing cap, and preserve failed attempts separately.
This operational change does not change the declared scientific runtime or grant
a longer training budget.

The capacity retries and this amended runtime use the subsequently authorized
price cap of $1.20/hour per arm and the same 30-minute compute cap (at most $0.60
GPU rent per new allocation). Earlier failed setup costs remain itemized rather
than folded into a successful training run. Offline retention uses a stopped
persistent allocation; report any continuing storage charges separately and do
not describe stopped compute as deletion.
