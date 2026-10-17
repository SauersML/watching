# Learning activation-viewing footprints: 051 and 052

**Both runs completed the fixed 1,024-update endpoint, passed their extraction/frozen-source gates, and exited 0. All four provisioned pods (including two rejected hosts) were deleted; exact temporary approvals were removed. Total cost: $0.5861.**

The opaque identity arm learned a substantial paired log-odds signal, but calibration and new-question performance remain limited. The buffer-copy arm made a small training-set improvement without useful development or test detection. Identity materialization changes storage; it is **not a read-only observer**. These experiments distinguish implementation footprints, not observation as an abstract property.

## Learning curves and calibration

[Buffer learning curves](buffer/progress.png) · [Identity learning curves](identity/progress.png). Both figures show whole-prompt uncertainty, both-conditions-correct accuracy, and paired log-odds separation; the buffer figure also marks the fixed-input accuracy ceiling.

The [identity calibration diagnostic](identity/calibration/) fits a scalar answer offset using development scores only. It is an exploratory offline calculation, not an updated native model or a new confirmatory result.

## Fixed endpoint, without checkpoint selection

“Both correct” requires the native full-vocabulary top token to be `No` without the intervention and `Yes` with it for the same prompt. Individual accuracy averages the two executions. Gap is the present-minus-absent Yes/No log-odds separation. CE is full-vocabulary correct-token cross entropy; lower is better.

| Arm | Execution | Split | Both correct | Individual accuracy | Gap (nats), pointwise 95% CI | CE |
|---|---|---|---:|---:|---|---:|
| buffer | split_inference | train | 4.30% | 52.15% | 0.00888 [0.00650, 0.01132] | 0.69235 |
| buffer | split_inference | validation | 0.00% | 49.22% | 0.00098 [-0.00232, 0.00427] | 0.69592 |
| buffer | split_inference | test | 1.56% | 50.78% | 0.00000 [-0.00488, 0.00464] | 0.69905 |
| buffer | wholegraph_transfer | train | 4.30% | 51.95% | 0.00946 [0.00711, 0.01208] | 0.69224 |
| buffer | wholegraph_transfer | validation | 0.00% | 48.44% | -0.00012 [-0.00378, 0.00366] | 0.69599 |
| buffer | wholegraph_transfer | test | 0.00% | 50.00% | 0.00049 [-0.00195, 0.00293] | 0.69907 |
| identity | split_inference | train | 75.39% | 87.70% | 4.24951 [4.14160, 4.35596] | 0.25726 |
| identity | split_inference | validation | 50.00% | 75.00% | 3.67578 [3.41797, 3.91602] | 0.66624 |
| identity | split_inference | test | 25.00% | 62.50% | 3.27539 [3.02344, 3.51562] | 1.36512 |
| identity | wholegraph_transfer | train | 72.66% | 86.33% | 3.93945 [3.84180, 4.03857] | 0.28525 |
| identity | wholegraph_transfer | validation | 43.75% | 71.88% | 3.39062 [3.14648, 3.62695] | 0.70879 |
| identity | wholegraph_transfer | test | 21.88% | 60.94% | 2.98633 [2.75977, 3.20508] | 1.44221 |

On the identity arm's whole-graph test, all **64/64 paired gaps are positive**, but only **14/64 pairs are both correct**. Mean CE **1.4422** is worse than a perfectly balanced, answer-only 50/50 predictor (`ln(2) ≈ 0.6931`). Thus the positive signal does not establish well-calibrated single-execution detection. Whole-graph test individual accuracy is 60.94% (prompt bootstrap 95% CI 56.25–66.41%); both-correct rate is 21.88% (12.50–32.81%).

Split inference is the actual training/primary evaluation path. Whole-graph transfer uses the learned native blocks in the complete compiled model and is reported separately, not appended to training curves. The two paths are not bit-identical. Neither these prompt intervals nor their non-overlap measure variation across random training seeds, independently generated kernels, or machines. A zero-width empirical interval is not population certainty.

## Buffer: impossible pairs versus available signal

Before optimization, all four learner-state components (hidden, mask, cosine, sine) were compared byte for byte; both conditions share the prompt's position IDs. **161/256 training pairs and 47/64 development pairs are identical.** A deterministic learner receiving these fixed states cannot distinguish those pairs. This imposes individual-accuracy ceilings of 68.55% training and 63.28% development, and both-correct ceilings of 37.11% and 26.56% respectively. Since some pairs differ, the predeclared all-identical skip did not apply.

The post-run descriptive analysis uses the fixed endpoint, does not fit anything, and does not select prompts by model performance:

| Buffer subgroup | Pairs | Both correct | Individual accuracy | Gap (nats), pointwise 95% CI |
|---|---:|---:|---:|---|
| train: identical_states | 161 | 0.00% | 50.00% | 0.00000 [0.00000, 0.00000] |
| train: different_states | 95 | 11.58% | 55.79% | 0.02393 [0.01867, 0.02895] |
| validation: identical_states | 47 | 0.00% | 50.00% | 0.00000 [0.00000, 0.00000] |
| validation: different_states | 17 | 0.00% | 47.06% | 0.00368 [-0.00919, 0.01562] |

The failure therefore is not merely dilution by identical pairs: **none of the 17 development pairs with different inputs is correct in both conditions**. Their mean gap's interval includes zero. All 320 identity-arm training/development pairs have distinct inputs.

## Design, checks and limitations

- Qwen3-8B, BF16, original base initialization; existing blocks 34–35 and final norm train, existing vocabulary head stays fixed. No new classifier, warm start, prompt condition marker, or changed answer vocabulary.
- Both arms reproduce all **1,024 actual minibatch-index lists** from run 034, in addition to the same 256/64/64 selected prompt records and their exact order. Batch eight prompt pairs, 8,192 total pair visits; learning rate 5e-6; seed 20261017; singleton source shape 1×96. The exact original family-v1 input SHA is `9321889d38cc5b070844a7a3692811f86e0c70905e7e4d5fbe790f8fab07b755`; selected data SHA is `7ed9cfc870ee7cf990bdb8636221b75a5c5a27b78088b0ca969513d09b7e0e94`.
- Each condition passed all-vocabulary native-versus-extraction parity on every cached prompt. The final whole-graph checks verified extraction parity and unchanged frozen-prefix states after learned-block installation. Gradient-enabled versus no-gradient split-tail checks passed at evaluated checkpoints.
- Buffer capture was actually executed: NaN poisoning checks full overwrite, an observed generated Triton store targets the preallocated tensor, and buffer/logit repeat checks match. This does **not** mean the buffer is footprint-free; many pairs differ.
- A failed extraction check would have stopped its arm, without relaxing equality or relaunching. An entirely identical training/development cache would have skipped redundant optimization; neither arm met that all-identical condition.
- The archived test is run 034's **previously inspected `test_wording` cohort**, held out from these updates but not a new confirmatory sample. Development was reported throughout; the fixed final checkpoint was used irrespective of earlier fluctuations. Test scoring began only after all 1,024 updates.
- There is one training seed/compilation instance per arm, on two L40S hosts. The NVIDIA drivers differ (buffer 580.159.03; identity 580.178.04). Independent compilations can themselves change numerical outputs, as run 048 shows. These results therefore do not isolate viewing mode as the sole causal difference between hosts, or establish reliable superiority across compilation replicas. The strict within-arm extraction checks do not remove this cross-arm limitation.
- Both model loads and tokenizer receive explicit base revision `b968826d9c46dd6066d109eabc6255188de91218`. The library's private configuration `_commit_hash` reports null in both artifacts; the explicit revision and source are preserved, without claiming a missing whole-base tensor checksum.

## Reproduction and artifacts

Immutable training source: `1b0a0c04c46e8896503d485cbf874cca99cf2892`. Protocol: [`activation-viewing-learning.md`](../../experiments/activation-viewing-learning.md). Launcher: [`bootstrap_viewing_train.sh`](../../scripts/bootstrap_viewing_train.sh); implementation: [`viewing_train.py`](../../src/watching/viewing_train.py). Fixed-endpoint subgroup analysis: [`analyze_viewing_subsets.py`](../../scripts/analyze_viewing_subsets.py).

Each arm's `fit/` contains unchanged scientific JSON/JSONL: exact prompts, masks/state fingerprints, complete training metrics, all checkpoint raw answer log probabilities, native top-token correctness, final split and whole-graph scores, and runtime/optimizer metadata. `archive.json` includes sanitized costs, completion status and file digests. `cache_verification.json` checks all 384 prompt pairs, all saved tensors against their fingerprints, and tokenization/position IDs. `input_equality_subgroups.json` includes every training/development pair's equality flag and baseline/fixed-endpoint subgroup metrics. Infrastructure/account logs are excluded.

Each full cache and learned tail is recorded by size and SHA-256 in its arm's archive; binary release links are added after upload verification. No large tensor artifacts are embedded in Git history.
