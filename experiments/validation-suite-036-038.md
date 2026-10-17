# Locked validation of native two-block detection and animal answers

This protocol is committed before either new training run produces predictions.
It follows the exploratory 034 result; it does not reclassify that inspected
comparison set as a confirmatory test.

## Questions

1. Does the released 034 detector work after reload on newly authored contexts
   and question structures? Does a checked buffer-copy capture leave its answers
   unchanged?
2. Can the successful two-block recipe learn both probe detection and an animal
   choice in the same model? The targets are No/Yes for detection and cat/owl for
   animal questions, absent/present respectively. The animal question contains no
   probe information. This measures a conditional one-token choice, not a general
   personality or unrestricted conversational preference.
3. Does the same training fail to distinguish conditions when both labels are
   attached to identical numerical inputs?

## Fixed experiments

Use Qwen/Qwen3-8B, revision `b968826d9c46dd6066d109eabc6255188de91218`, BF16,
singleton length 96, layer-14 reference probe, and the last two native blocks
plus final normalization. Freeze the prefix and original unembedding. Train
from the original base, with no added classifier or untied-weight edit.

The dataset is `data/validation_v2/dataset.json`, SHA-256
`8ae5cc965dd9904603237bbd8dec08c50b693e2d282a3299bb4457fe4850e02c`.
There are 512 training, 64 development, 128 test and 64 challenge prompts,
half for each task. Context identities and question families are disjoint across
splits. All text fits without truncation. The finite, AI-authored question and
context banks limit what population a result can describe.

- **036:** seed 20261017, 2,048 updates, eight prompt pairs per update,
  full-vocabulary cross-entropy averaged over the sixteen condition examples.
  FP32-master AdamW, learning rate 5e-6, zero decay, gradient clip one.
- **037:** after 036, reconstruct the original base tail and train the same
  schedule, order, labels and data on 036's cached absent input, copied for both
  conditions. Check exact paired-logit equality and the balanced-loss lower
  bound ln(2). This is an identical-input training control, not a claim that all
  physical probe implementations are numerically passive.
- **038:** independently reload released 034 weights and score 64 new detection
  test prompts. Randomize none/reference/buffer execution order, repeat an anchor
  prompt, and require exact all-vocabulary buffer/none parity before interpreting
  the buffer capture as a no-footprint control. Then train the joint recipe from
  the original base using seed 20261018.

Evaluation at updates 0, 8, 32, 128 and each 256 uses training/development only.
The final scheduled checkpoint is the endpoint. Test and challenge scores are
computed only after the weights are saved; they never select an update or setting.
The same locked test is shared by the predeclared models; this is not a supply of
independent confirmatory tests. Any later adaptive change requires a new test set
or an explicitly exploratory label.

## Verification and reporting

Boundary capture must preserve every original vocabulary logit. Paired masks,
positions and rotary inputs must match. Check gradient-enabled versus no-gradient
tail forwards, prefix stability after installation, fixed-head hashes, artifact
hashes and tokenized prompt identity. Record failures without relaxing the checks.
Evaluate whole-model compiled transfer separately from the cached training path.
Fresh validation uses a new wrapper supporting three capture modes, so exact
equivalence to the old wrapper is not assumed.

Report each task separately: paired positive-minus-negative log-odds separation,
full-vocabulary log loss, correct-token probability, and actual top-token accuracy.
Display both conditions, task counts, every question-family result and both seeds.
Intervals resample whole prompt pairs; question-family resampling is a sensitivity
analysis for wording dependence. Do not treat tokens, checkpoints or two conditions
as independent observations. Intervals are conditional on this dataset and runtime;
two seeds do not characterize training-seed variance precisely. A degenerate
bootstrap interval at 100% is not proof of perfect population accuracy.

No p-value threshold controls stopping or claim selection. Any supplementary test
must state its exchangeability/independence assumptions and multiplicity scope.
The main evidence is the fixed endpoint, task-level effect sizes, matched control,
individual-condition calibration, and fresh-process reproduction.

## Budget and execution

Two parallel L40S jobs, each capped at 45 minutes and $1.20/hour: at most $1.80
combined at those caps. Low-GPU-utilization termination is disabled because
compilation and loading use CPU; wall-time limits remain active. Stream JSONL
progress immediately. Archive raw compact results and plots on main, and large
weights/cache artifacts in GitHub Releases. A time-limited or failed arm is reported
as incomplete, not silently substituted with its best checkpoint.
