# Confirmation and independent joint training: run 038

**Both scientific stages are complete.** The runner cleanup status is recorded separately in the archive metadata. This run first reloaded the detection-only 034 checkpoint for a fresh-process confirmation, then trained an independently initialized joint model from the original base weights. The joint model is not a continuation of 034.

## Fresh-process confirmation of 034

`fresh034/` preserves the separate successful detection-only result: 128/128 actual top-token answers correct on 64 new prompt pairs; minimum correct-token probability 0.99997246. The buffer capture comparison failed full-vocabulary parity and is explicitly invalid as a no-footprint control. See its README and independent `ADVERSARIAL_REVIEW.md` for provenance and limitations.

## Independent joint fit

- Qwen3-8B, original fixed output head, final native blocks 34–35 and final normalization trained; no new classifier.
- Seed 20261018, compared with seed 20261017 in run 036. Same fixed dataset and training protocol: 512 training, 64 development, 128 final-test, 64 challenge prompts, equally divided between detection and animal tasks. Both conditions are scored for every prompt.
- Detection absent/present answers are No/Yes; animal answers are cat/owl. No condition label is present in the input.
- Batch 8 prompt pairs, length 96, learning rate 5e-6, fixed 2,048 updates (16,384 processed prompt pairs), original base initialization. No test-driven checkpoint selection.
- Primary execution uses compiled frozen source states and an eager BF16 native tail; ordinary whole-graph compiled inference is evaluated separately after training.
- Source `b699c4ba462dcac9291f8d5e72ced4c088d5962c`; base revision `b968826d9c46dd6066d109eabc6255188de91218`; dataset SHA-256 `8ae5cc965dd9904603237bbd8dec08c50b693e2d282a3299bb4457fe4850e02c`.
- Final checks confirm unchanged frozen source and original output head. Large tail/cache artifacts are not in Git; SHA-256 hashes and byte counts are in `fit/archive.json`.

## Fixed final endpoint

Accuracy is the actual maximum-probability token over the full vocabulary. CE is full-vocabulary correct-answer log loss in nats; paired gap is present minus absent positive/negative answer log odds.

| Runtime | Split | Task | Prompt pairs | Actual accuracy (95% CI) | Full CE | Paired gap (95% CI), nats |
|---|---|---|---:|---:|---:|---:|
| split_inference | train | detection | 256 | 94.53% [92.58, 96.29] | 0.1383 | 15.670 [15.229, 16.107] |
| split_inference | train | animal | 256 | 80.08% [77.15, 83.01] | 0.4388 | 14.535 [14.244, 14.834] |
| split_inference | validation | detection | 32 | 70.31% [62.50, 78.12] | 2.2635 | 11.375 [9.199, 13.363] |
| split_inference | validation | animal | 32 | 67.19% [59.38, 75.00] | 1.9315 | 10.406 [9.238, 11.516] |
| split_inference | test | detection | 64 | 80.47% [74.22, 85.96] | 1.1391 | 11.303 [10.090, 12.484] |
| split_inference | test | animal | 64 | 73.44% [67.19, 79.69] | 1.5680 | 9.709 [8.703, 10.707] |
| split_inference | challenge | detection | 32 | 73.44% [65.62, 82.81] | 2.7477 | 8.342 [6.646, 10.143] |
| split_inference | challenge | animal | 32 | 68.75% [60.94, 78.12] | 2.7206 | 7.883 [6.605, 9.141] |
| wholegraph_transfer | train | detection | 256 | 90.62% [88.28, 92.97] | 0.2047 | 14.063 [13.664, 14.482] |
| wholegraph_transfer | train | animal | 256 | 76.56% [73.63, 79.69] | 0.5854 | 13.059 [12.755, 13.361] |
| wholegraph_transfer | validation | detection | 32 | 67.19% [59.38, 75.00] | 2.4712 | 10.078 [8.183, 11.856] |
| wholegraph_transfer | validation | animal | 32 | 67.19% [59.38, 75.00] | 2.2127 | 9.285 [8.156, 10.356] |
| wholegraph_transfer | test | detection | 64 | 76.56% [70.31, 82.81] | 1.3163 | 9.973 [8.879, 11.055] |
| wholegraph_transfer | test | animal | 64 | 71.09% [64.84, 77.34] | 1.7797 | 8.434 [7.574, 9.279] |
| wholegraph_transfer | challenge | detection | 32 | 68.75% [60.94, 78.12] | 2.9545 | 7.193 [5.707, 8.799] |
| wholegraph_transfer | challenge | animal | 32 | 67.19% [59.38, 76.56] | 2.8941 | 6.910 [5.805, 8.012] |

The joint model learns useful condition-dependent behavior, but generalization and calibration remain limited. Its test accuracy is higher than run 036 at the fixed endpoint; nevertheless every test/challenge task has CE above the balanced two-answer baseline ln(2) = 0.6931. Large paired gaps do not imply confident correctness in both conditions. These two training seeds are reported separately, not pooled into a prompt-based confidence interval for seed robustness.

## Statistical and archival scope

`fit/robustness.json` gives 5,000 whole-prompt paired bootstrap intervals, equal-question-family sensitivity analyses, per-family summaries and prompt-both-correct intervals under stated iid assumptions. Table intervals are pointwise and conditional on this run and observed family mixture. They do not cover arbitrary new families, hardware or training seeds. Test/challenge data were fixed before both runs; no best checkpoint was selected after looking at them.

`progress.*` contains training/development curves only; `progress-endpoints.*` separates primary and whole-graph final results. Raw JSON/JSONL is preserved unchanged. The fresh034 confirmation and joint fit are distinct experiments; neither is silently combined with the other.
