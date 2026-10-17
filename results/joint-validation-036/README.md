# Joint detection and animal preference: run 036

**Both the fit and identical-state negative control completed their prespecified 2,048 updates. The runner exited with status 0.**

The final model distinguishes the conditions on unseen prompts, but its answers are poorly calibrated. Large positive paired gaps coexist with substantial confidently wrong answers. The final endpoint is reported even though earlier development checkpoints sometimes had better accuracy.

## Design and provenance

- Qwen3-8B, original fixed output head, final two native blocks (34–35) and final normalization trained. No added classifier.
- Balanced paired full-vocabulary cross-entropy; 512 training prompts (256 per task), 64 development prompts, 128 final-test prompts, 64 challenge prompts. Each prompt is evaluated in both capture conditions.
- Detection: absent → No; present → Yes. Animal: absent → cat; present → owl. Prompts do not disclose the condition.
- Prompt batch 8, fixed length 96, learning rate 5e-6, 2,048 optimizer updates, 16,384 processed training prompt pairs; seed 20261017.
- Primary execution: compiled frozen source states plus eager BF16 native tail. Whole-graph compiled inference is a separately reported transfer evaluation.
- Source commit `b699c4ba462dcac9291f8d5e72ced4c088d5962c`; base revision `b968826d9c46dd6066d109eabc6255188de91218`. Dataset SHA-256 `8ae5cc965dd9904603237bbd8dec08c50b693e2d282a3299bb4457fe4850e02c`.
- Input/source and output-head hashes remain unchanged in the fit manifest. Tail weights and cached states are excluded from Git; their sizes and SHA-256 hashes are in `fit/archive.json`.

## Fixed final endpoint

Accuracy is the actual maximum-probability token over the entire vocabulary, not a forced two-token choice. CE is full-vocabulary negative log probability of the correct token in nats. Gap is present minus absent positive/negative answer log odds.

| Runtime | Split | Task | Prompt pairs | Actual accuracy (95% CI) | Full CE | Paired gap (95% CI), nats |
|---|---|---|---:|---:|---:|---:|
| split_inference | train | detection | 256 | 68.36% [65.43, 71.29] | 1.2402 | 12.182 [11.733, 12.630] |
| split_inference | train | animal | 256 | 63.67% [61.13, 66.60] | 1.2029 | 12.675 [12.326, 13.030] |
| split_inference | validation | detection | 32 | 65.62% [57.81, 73.44] | 2.8945 | 8.172 [6.301, 9.937] |
| split_inference | validation | animal | 32 | 60.94% [54.69, 68.75] | 2.4717 | 9.187 [8.059, 10.340] |
| split_inference | test | detection | 64 | 67.19% [61.72, 72.66] | 2.4085 | 8.416 [7.250, 9.531] |
| split_inference | test | animal | 64 | 67.97% [62.50, 74.22] | 2.0290 | 9.061 [7.935, 10.115] |
| split_inference | challenge | detection | 32 | 70.31% [62.50, 78.12] | 2.4598 | 6.645 [5.492, 7.805] |
| split_inference | challenge | animal | 32 | 62.50% [56.25, 70.31] | 2.9924 | 6.445 [4.855, 8.008] |
| wholegraph_transfer | train | detection | 256 | 64.26% [61.33, 67.00] | 1.5270 | 11.314 [10.865, 11.757] |
| wholegraph_transfer | train | animal | 256 | 59.96% [57.62, 62.50] | 1.5329 | 11.774 [11.435, 12.119] |
| wholegraph_transfer | validation | detection | 32 | 62.50% [56.25, 70.31] | 3.2001 | 7.430 [5.699, 9.137] |
| wholegraph_transfer | validation | animal | 32 | 60.94% [54.69, 68.75] | 2.7731 | 8.391 [7.269, 9.539] |
| wholegraph_transfer | test | detection | 64 | 65.62% [60.16, 71.09] | 2.7194 | 7.422 [6.312, 8.492] |
| wholegraph_transfer | test | animal | 64 | 64.84% [59.38, 70.31] | 2.2586 | 8.512 [7.437, 9.531] |
| wholegraph_transfer | challenge | detection | 32 | 70.31% [62.50, 78.12] | 2.5966 | 6.121 [4.984, 7.234] |
| wholegraph_transfer | challenge | animal | 32 | 64.06% [56.25, 71.88] | 3.1772 | 5.848 [4.324, 7.363] |

The balanced two-answer predictor assigns probability 0.5 to each permitted answer and has CE ln(2) = 0.6931. This final model has worse CE in every reported task/split/runtime despite above-50% accuracy. The paired gap measures a relative response to capture; it does not ensure that either execution falls on the correct side of the answer decision boundary.

## Uncertainty and limits

`fit/robustness.json` contains 5,000 whole-prompt paired bootstrap intervals, equal-question-family sensitivity summaries, per-family results, and prompt-both-conditions-correct intervals with explicit iid assumptions. The table uses the prompt-weighted bootstrap, conditional on this run and observed family mixture. These are pointwise descriptive intervals, not simultaneous guarantees or uncertainty across training seeds, hardware, or arbitrary unseen question families.

The test/challenge families and contexts were frozen before this run. Development curves were observed during training; no best checkpoint replaces the fixed 2,048-update endpoint. This joint run does not erase the separately confirmed detection-only 034 result: its dataset, task mixture, training initialization and optimization trajectory differ.

Progress and endpoint figures are generated reproducibly with `scripts/plot_joint_progress.py`. The progress figure includes the matched identical-state control as a dashed overlay. Source JSON/JSONL is preserved unchanged; generated analysis and figures are separately identifiable.

## Completed identical-state negative control

The same learner, original initialization, task labels, seed and update schedule were trained with **both labels receiving a copy of the absent cached activation state**. The copied states contain no numerical condition information. This directly checks whether the training/scoring pipeline can manufacture a condition signal from metadata or ordering. It is a cached-state control, not a claim about a particular buffer probe implementation.

At all final train/development/test/challenge evaluations, both tasks have exactly **zero paired gap and 50% actual top-token accuracy**. Identical-input outputs are bit-identical; no pair is correctly classified in both conditions. Final full-vocabulary CE:

| Split | Detection CE | Animal CE |
|---|---:|---:|
| Training | 0.693632 | 0.697160 |
| Development | 0.695815 | 0.742420 |
| Test | 0.695262 | 0.706750 |
| Challenge | 0.700102 | 0.706737 |

These losses approach the balanced two-answer lower bound ln(2). The exact zero gap and 50% paired accuracy follow from identical deterministic inputs with opposite balanced labels, not an estimate with zero population uncertainty. No whole-graph transfer is claimed for this intentionally identical-state experiment. Raw records and a separate robust summary are in `null/`.
