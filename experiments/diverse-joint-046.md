# Diverse joint training from checkpoint 039

Protocol fixed before launch: `watching-diverse-joint-046`.

The goal is reliable native Yes/No detection and cat/owl choice across more
question forms and chat histories. The earlier 040 animal result remains
119/128 correct individual answers, or 59/64 prompts correct in both conditions.
This experiment uses new held-out data; it does not regrade or train on 040.

Initialize the two final native blocks and final norm from released joint-039,
SHA-256 `2f1f92bcb78041b812760f5086a5436ef43732e983a19a4578e6e3a7a6c3892c`.
Keep the native output matrix and preceding computation frozen. Use the same
Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`, layer-14 reference
probe, BF16 source, shape 1×96 and cached compiled-source/native-tail runtime.
This is transfer to a new dataset with fresh Adam state and local step zero,
not a claim to restore the earlier optimizer.

Data are frozen in `data/generalization_v4`: 512 training, 64 development and
128 endpoint-test variants, balanced across detection and animal choice. Each
base context has four chat layouts and both numerical conditions. There are
128/16/32 distinct base contexts respectively, not 512/64/128 independent
observations. Question and prefix wordings are disjoint across splits; all
four layout types appear in every split. Both conditions see identical messages.
The dataset manifest records tokenizer checks, hashes and context provenance.

Train blocks 34–35 and final norm for 1,024 updates, eight prompt pairs per
update, learning rate 1e-6, seed 20261022, balanced full-vocabulary SFT,
FP32 master Adam/gradient accumulation, zero decay and clip norm one. Keep
the output matrix frozen. Evaluate training/development at zero, 8, 32 and
every 128 updates. Use the endpoint, never a development-selected peak.
The training loop has a 1,200-second cap; if it ends early, report the actual
step and label the result time-limited; skip held-out evaluation. Test records are
selected and tokenized during setup, but executed/scored only after all 1,024
planned updates. The launcher checks the frozen dataset SHA-256 before loading.

Before training, require the existing full-vocabulary capture parity and
gradient/no-gradient native-tail checks. New chat layouts may fail these gates;
such failure is an experimental limitation, not permission to loosen equality.
Evaluate the final integrated whole compiled model as a separate execution path.
Do not replace a weak whole-graph result with a strong cached-tail result.

Primary metrics, separately by task: actual full-vocabulary top-token accuracy,
fraction of prompts correct in both conditions, correct-token probability and
full-vocabulary cross-entropy. Also report paired positive/negative answer
log-odds separation and forced binary scores, clearly distinguished from actual
answers. Break out prefix layouts to expose failures hidden by pooled averages.

Intervals resample whole base contexts, preserving all four layouts and both
conditions. Development has only eight base contexts per task; its intervals
are exploratory, conditional on these question families and one training run.
Checkpoint bands are pointwise, not simultaneous. An all-correct bootstrap
interval is not evidence of perfect population accuracy. No tuning after
viewing this final test may reuse it as a confirmatory test.

Run on one L40S for at most 45 minutes at at most $1.20/hour ($0.90 ceiling).
Disable low-utilization shutdown during compilation, retain the wall-time cap,
stream early summaries and publish source revision, raw scores, hashes,
weights, measured cost and deletion status. The matched 034 null runs in parallel.
