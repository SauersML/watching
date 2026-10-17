# 034: two native blocks learn across 256 training prompts

Run `watching-two-block-scaled-034` completed all **1,024 updates / 8,192 prompt-pair
visits**, with runner exit status 0. At the scheduled endpoint, Qwen3-8B answers
both numerical conditions correctly on every training, development, and comparison
prompt. This also succeeds when the learned native blocks are installed into the
separately compiled whole model.

The experiment trains native blocks 34–35 and final normalization; the original
output matrix and earlier computation remain fixed. There is no added classifier.
Inputs are identical between the probe and no-probe conditions. The supervised
targets are Yes and No respectively. See the [protocol](../../experiments/two-block-scaled-034.md),
[learning curves](progress.pdf), and [per-example scores](fit/).

## Fixed endpoint results

The gap is probe minus no-probe Yes/No log odds. Loss is full-vocabulary
cross-entropy, not a loss renormalized over just Yes and No. Accuracy uses the
actual top vocabulary token. Intervals use 5,000 whole-prompt bootstrap draws,
preserving both conditions, with seed 20261016.

| Runtime | Cohort | Gap, nats [95% CI] | Actual correct answers | Full-vocabulary CE, nats [95% CI] |
|---|---|---:|---:|---:|
| Split inference | Training, 256 prompts | 27.262 [27.121, 27.396] | 512/512 | 2.30e−6 [1.76e−6, 3.03e−6] |
| Split inference | Development, 64 prompts | 27.021 [26.709, 27.311] | 128/128 | 3.26e−6 [2.07e−6, 5.00e−6] |
| Split inference | New-question-family comparison, 64 prompts | 26.074 [25.678, 26.440] | 128/128 | 1.13e−5 [4.69e−6, 2.02e−5] |
| Whole-graph transfer | Training, 256 prompts | 27.209 [27.070, 27.340] | 512/512 | 2.41e−6 [1.81e−6, 3.26e−6] |
| Whole-graph transfer | Development, 64 prompts | 26.938 [26.631, 27.215] | 128/128 | 3.07e−6 [2.06e−6, 4.48e−6] |
| Whole-graph transfer | New-question-family comparison, 64 prompts | 26.033 [25.625, 26.408] | 128/128 | 1.19e−5 [5.00e−6, 2.09e−5] |

Every cohort's prompt-bootstrap accuracy interval is [100%, 100%], because every
observed prompt contributes two correct answers. This degenerate empirical
interval is **not certainty about population accuracy**. These are pointwise
intervals conditional on this prompt collection and one trained run; they do not
cover training seeds, hardware, or a population of question families. This is
not an untouched confirmatory test: the comparison pool had already been
inspected in earlier project experiments.

The comparison correct-token probability is 99.99887% in split inference and
99.99881% in whole-graph transfer. Its midpoint log-odds RMS is 1.089 nats, much
smaller than the 26.074-nat condition separation. Thus this endpoint combines
strong separation with correct individual answers, rather than merely increasing
a gap while leaving a large common answer bias.

## Learning trajectory and runtime checks

Training was scheduled in advance; no checkpoint was selected by its scores.
At update 384, development accuracy was 68.75% with CE 0.6084. At update 512,
the gap had grown to 6.1855 nats, but accuracy fell to 57.03% and CE rose to
0.8986. By update 640, development accuracy was 100% and CE 8.03e−6; this
persisted through the final update. The early gap increase alone did not imply
well-calibrated answers.

Relative to this run's initial split-inference baseline, development gap improves
by **26.9805 nats [26.6445, 27.2910]**, and full-vocabulary CE changes by
**−4.1884 nats [−4.8336, −3.5750]**. These paired changes preserve prompt identity.
The comparison cohort was evaluated only after training, so no before/after
comparison estimate is available for it.

The primary runtime is a frozen compiled source followed by an eager BF16 native
tail. The final whole-graph scores are a separate execution path, not another
training run and not additional learning-curve observations. The paired change
in comparison gap from split to whole-graph evaluation is −0.0410 nats
[−0.0996, 0.0176]. Their numbers are not bit-identical, although all answers stay
correct. This run does not establish transfer to vLLM or arbitrary serving stacks.

The scientific completion event confirms:

- Frozen source tensors remain bit-identical.
- The original output head remains unchanged, with identical before/after hashes.
- The native head is installed correctly for transfer evaluation.
- The recorded gradient-enabled/no-gradient forward audits pass bit-identically.
- Test extraction parity passes before the final comparison evaluation.

These checks support the reported execution experiment. A no-footprint training
control, a fresh-process reload, and independent seeds are not part of this run.
They remain separate follow-ups rather than implied by the positive result.

## Data, scope, and provenance

Training uses 256 distinct contexts, 48 question phrasings across 12 question
families, and 20 content families. Development uses 64 separate contexts with
the training question structures. The 64 comparison contexts use four question
families absent from training: XML requests, Markdown tables, footnotes, and
multilingual requests. These are AI-authored prompts. There are no exact context
duplicates across the selected cohorts; the lexical audit identifies one
near-duplicate pair within training. See the [data audit](DATA_AUDIT.md) for
coverage counts and a clearly separated, untested improvement proposal.

Compared with earlier small-prompt experiments, this recipe changes the data
volume, effective batch size, pair visits, and independently compiled source.
It therefore does not isolate which change caused the improvement. The planned
matched-arm comparison 035 is a separate run.

Training uses FP32-master AdamW at 5e−6, zero weight decay, gradient clipping at
one, and eight complete condition pairs per update. The fitting process completed
in 738.8 seconds, including its extraction and evaluation stages. Source commit:
`65f3cb2a46c9f3b2c92c4199ac596289065b0f72`.

All 33 compact JSON/JSONL outputs are archived unchanged and hashed in
[archive.json](archive.json). The 771,802,258-byte native-tail artifact and
651,071,093-byte activation cache remain local; their SHA-256 hashes are
published. Infrastructure logs and account metadata are excluded.

Reproduce the endpoint analysis and plot:

```sh
python results/two-block-scaled-034/analyze.py
python scripts/plot_tail_progress.py results/two-block-scaled-034/fit results/two-block-scaled-034/progress
```
