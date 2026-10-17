# Development-only transfer from run 034 (039)

Status: pilot configuration locked before launch. The original joint experiments
remain unchanged; this pilot is based on their training/development progress,
not their final test scores.

The question is whether the learned native tail from detection-only run 034 gives
joint detection and animal-answer training a better starting point than the base
model. This is transfer to new task data, not a continuation of 034's dataset or
optimizer. A lower learning rate is another change, so the proposed pilot does
not identify initialization's isolated causal contribution.

## Fixed implementation

- Qwen3-8B, original frozen prefix and output matrix; update only native final two
  blocks and final norm. Initialize those blocks and norm from 034.
- `--warm-start-tail` is separate from `--initial-tail --prior-results`.
  `--warm-start-manifest` defaults to the artifact's sibling `manifest.json`.
  Architecture, tensor keys/shapes/dtypes/finiteness, exact head checksum, probe
  layer, boundary, length, source runtime and any recorded revision must match.
  The source run must have completed with its head fixed. Artifact step and source
  manifest step must agree. Both supplied files are hashed into the new manifest.
- Historical 034 lacks a pinned revision and a complete frozen-prefix checksum.
  Its compatibility checks do not establish bitwise identity of every original
  prefix weight. This limitation is explicit in `warm_start_provenance`.
- Load transferred parameters into the separate learner only. The two compiled
  source copies remain the original base model during cache collection. All
  existing extraction, gradient/no-gradient and final prefix parity gates remain.
- New data selection, step zero, processed-pair count zero and fresh FP32-master
  Adam. No optimizer state is restored, and 034's 1,024 updates are recorded only
  as origin metadata.
- `--development-only --test-prompts 0` selects train and validation records only.
  It neither tokenizes nor executes test/challenge records; final whole-graph
  transfer checks also use only train and validation. It cannot be combined with
  same-data continuation or challenge evaluation.

## Proposed pilot and interpretation

Use the already specified joint training/development data, B8 and learning rate
`1e-6`, seed 20261019, and exactly 1,024 updates. Evaluate training/development
at updates 0, 8, 32 and every 128; report the final scheduled weights. Preserve
the existing seeded prompt order; task-balanced batches are a separate future
change. Report each task's full-vocabulary CE, correct-token probability, actual
token accuracy, and probe-minus-no-probe log-odds from step zero onward. A large
paired gap without calibrated single-execution answers is insufficient.

The pilot is exploratory. Use its train/development results to decide whether to
freeze a follow-up recipe. Neither the current joint test results nor 034's prior
test success is an independent confirmation of the adapted recipe. A successful
pilot needs newly reserved confirmation examples and an independently declared
evaluation. No test is scored by this implementation task.

The existing identical-input null currently supports base initialization only;
it rejects warm-start source runs. A future matched warm-start null must load the
same transferred initialization before applying identical inputs under both labels.

## Execution budget

Run `watching-warm-joint-039` on one L40S, capped at 30 minutes and
$1.20/hour (at most $0.60). Disable low-utilization termination during compilation;
retain the wall-time cap. Use the immutable committed source snapshot and
`scripts/run_warm_joint.sh`. Stream intermediate metrics and keep incomplete
runs clearly labeled.
