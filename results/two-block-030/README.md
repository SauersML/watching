# Two-block continuation 030: strong training learning, weak transfer to new prompts

Run `watching-two-block-030r` completed all **4,096 additional updates**, ending at
global update **4,224**, with runner exit status 0. Qwen3-8B's native blocks 34–35
and final normalization were trained; the entire original output head and earlier
computation stayed fixed. No new classifier was added. The primary runtime uses
a frozen compiled source and an eager BF16 native tail. See the
[protocol](../../experiments/two-block-030.md), [progress plot](progress.pdf), and
[raw scores](fit/).

The 16 practiced prompts become strongly distinguishable. At the fixed endpoint,
the model answers both conditions correctly on 14/16 training prompts and gets
30/32 individual answers correct. This survives evaluation through the separately
compiled whole model. Generalization remains weak: the 32 endpoint test prompts
have only 34/64 correct answers, while full-vocabulary loss is much worse than
the balanced two-answer reference of ln(2) = 0.6931 nats.

## Fixed endpoint

Intervals below are 5,000 whole-prompt bootstrap resamples, preserving both
conditions. They describe these prompt cohorts and this trained run; they do not
cover training-seed, hardware, or new-family variation. Accuracy is the actual
full-vocabulary argmax, not a forced Yes/No decision. The reproducible
[endpoint analysis](analyze.py) uses seed 20261016; Monte Carlo interval endpoints
can differ slightly from those computed by the runtime or plotting script.

| Runtime | Cohort | Gap, nats [95% CI] | Actual accuracy [95% CI] | Full-vocabulary CE [95% CI] |
|---|---|---:|---:|---:|
| Split inference | Training, 16 | 8.250 [6.367, 10.047] | 93.75% [84.38%, 100%] | 0.406 [0.060, 0.882] |
| Split inference | Development, 8 | 0.406 [0.016, 1.062] | 56.25% [50%, 68.75%] | 2.467 [1.536, 3.420] |
| Split inference | Endpoint test, 32 | 0.320 [−0.180, 0.887] | 53.12% [50%, 57.81%] | 2.573 [2.060, 3.085] |
| Whole-graph transfer | Training, 16 | 8.273 [6.406, 10.055] | 93.75% [84.38%, 100%] | 0.376 [0.065, 0.848] |
| Whole-graph transfer | Development, 8 | 0.328 [0.031, 0.859] | 50% [50%, 50%] | 2.483 [1.573, 3.421] |
| Whole-graph transfer | Endpoint test, 32 | 0.285 [−0.211, 0.852] | 53.12% [50%, 57.81%] | 2.539 [2.018, 3.046] |

The degenerate development accuracy interval in whole-graph evaluation occurs
because every prompt contributes exactly one correct answer out of its two
conditions. It is not certainty about population accuracy.

The final split-inference mean correct-token probabilities are 87.44% on training,
53.03% on development, and 51.58% on the endpoint test. Mean probabilities near
50% coexist with large loss because the model is confidently wrong on some
examples. On the endpoint test, the RMS midpoint of the two log-odds is 5.94 nats,
versus a mean condition gap of only 0.320 nats. The condition signal does not
reliably overcome prompt-dependent answer bias. Answer-pair probability mass is
above 99.95% on that test, so this is chiefly calibration within Yes/No rather
than probability assigned to unrelated vocabulary.

## Learning versus restart changes

This is weight continuation from 029r, with Adam moments restarted and a freshly
compiled frozen source. The starting tail hash matches the saved 029r artifact.
Restarted scores are not identical: 12.5% of training and 31.25% of development
condition log-odds changed, with maximum absolute difference 0.125 nats. Training
gap changed from 0.03906 to 0.02344; development gap from 0.03125 to 0.015625.
These changes precede training and are documented separately in
[continuation_comparison.json](fit/continuation_comparison.json).

Relative to this run's resumed baseline, the final paired training-gap gain is
**8.2266 nats [6.3359, 10.0234]**. Development gain is **0.3906
[−0.0313, 1.0625]**. Thus the positive development endpoint interval does not by
itself establish improvement over the resumed baseline. Development CE worsens
by **1.7293 nats [0.8104, 2.6717]**. No before/after test comparison is available;
test scores were collected only at the endpoint.

Learning is nonmonotonic. The largest observed training gap was 14.664 nats at
global update 3,584, with 100% training accuracy and CE 0.0603. This is a
descriptive maximum over the recorded checkpoints, **not a selected model or an
independent success test**. The reported result remains the fixed update-4,224
endpoint. Repeated checkpoint intervals are pointwise, not simultaneous bands.

## Scope and provenance

The 32 endpoint contexts exclude all 32 contexts evaluated in 029r and use the
`test_wording` source pool. They are new to that earlier two-block endpoint,
not a newly authored, globally untouched project test collection. The experiment
does not establish broad generalization to arbitrary prompts. Both conditions
are paired on identical text, and no checkpoint is chosen using development or
test outcomes.

The completion event records unchanged frozen-source tensors and unchanged output
head. Whole-graph transfer is scored separately, not mixed into the primary
learning curves, and does not constitute another independent training run.
The fitting process completed in 679.7 seconds; the runner command took 764
seconds including setup. Source snapshot: `4a68c709` (the complete source-file
hash is in the manifest).

All 82 compact JSON/JSONL outputs are copied unchanged and hashed in
[archive.json](archive.json). The large `native_tail.pt` (771,802,066 bytes) and
`cache_states.pt` (94,947,861 bytes) remain local; their SHA-256 hashes are
published instead. Infrastructure logs and account metadata are not included.

Experiment 032 is a separate comparison that starts from the same 029r tail and
also trains the existing full output head. It does not continue this final 030
checkpoint. Experiment 031 concerns a separate fixed-feature head-fitting audit.
Neither is part of the results reported here.

Reproduce the analysis and plot from this archive:

```sh
python results/two-block-030/analyze.py
python scripts/plot_tail_progress.py results/two-block-030/fit results/two-block-030/progress
```
